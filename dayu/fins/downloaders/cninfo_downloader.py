"""巨潮资讯 A 股财报下载器。

该模块只负责与巨潮资讯 HTTP 接口交互，不包含 pipeline 编排、仓储写入或
processed 重建逻辑。当前职责包括：

1. 解析 A 股代码对应的公司基础信息与 orgId。
2. 查询财报公告列表。
3. 下载单个 PDF 原文。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Mapping, Optional, Protocol

import requests

from dayu.fins.ticker_normalization import normalize_ticker

CNINFO_ANNOUNCEMENT_API: Final[str] = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STOCK_JSON_URL: Final[str] = "http://www.cninfo.com.cn/new/data/szse_stock.json"
CNINFO_PDF_BASE_URL: Final[str] = "http://static.cninfo.com.cn/"
CNINFO_STOCK_CACHE_FILENAME: Final[str] = "stock_org.json"
CNINFO_STOCK_CACHE_TTL_SECONDS: Final[int] = 86_400 * 7
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0

REPORT_CATEGORY_MAP: Final[dict[str, str]] = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
}

EXCHANGE_COLUMN_MAP: Final[dict[str, str]] = {
    "SH": "sse",
    "SZ": "szse",
    "BJ": "bj",
}

HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure",
}

_TITLE_HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


class HttpResponseProtocol(Protocol):
    """HTTP 响应最小协议。"""

    headers: Mapping[str, str]
    content: bytes

    def json(self) -> object:
        """返回 JSON 数据。"""

        ...

    def raise_for_status(self) -> None:
        """在状态码异常时抛错。"""

        ...


class HttpSessionProtocol(Protocol):
    """HTTP session 最小协议。"""

    def get(
        self,
        url: str,
        *,
        timeout: float,
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponseProtocol:
        """发起 GET 请求。"""

        ...

    def post(
        self,
        url: str,
        *,
        data: Mapping[str, str],
        timeout: float,
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponseProtocol:
        """发起 POST 请求。"""

        ...


@dataclass(frozen=True)
class CninfoCompanyProfile:
    """巨潮资讯公司档案。"""

    code: str
    exchange: str
    org_id: str
    company_name: str


@dataclass(frozen=True)
class CninfoAnnouncement:
    """巨潮资讯公告摘要。"""

    announcement_id: str
    title: str
    announcement_date: str
    pdf_url: str
    fiscal_year: int
    fiscal_period: str
    report_type: str
    adjunct_size_kb: int


@dataclass(frozen=True)
class DownloadedPdf:
    """已下载 PDF 内容。"""

    content: bytes
    content_type: Optional[str]
    source_url: str


class CninfoDownloader:
    """巨潮资讯下载器。

    Args:
        cache_dir: 本地缓存目录。
        session: 可选 HTTP session。
        timeout_seconds: 默认请求超时秒数。

    Raises:
        ValueError: 参数非法时抛出。
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        session: Optional[HttpSessionProtocol] = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """初始化下载器。

        Args:
            cache_dir: 本地缓存目录。
            session: 可选 HTTP session。
            timeout_seconds: 请求超时秒数。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        self._cache_dir = cache_dir
        self._session = session or requests.Session()
        self._timeout_seconds = timeout_seconds

    def normalize_ticker(self, ticker: str) -> str:
        """规范化 A 股 ticker。

        Args:
            ticker: 原始 ticker。

        Returns:
            6 位 A 股代码。

        Raises:
            ValueError: 输入不是 A 股 ticker 时抛出。
        """

        profile = normalize_ticker(ticker)
        if profile.market != "CN":
            raise ValueError(f"CninfoDownloader 仅支持 A 股 ticker: {ticker}")
        return profile.canonical

    def resolve_company(self, ticker: str) -> CninfoCompanyProfile:
        """解析公司档案。

        Args:
            ticker: A 股 ticker。

        Returns:
            公司档案。

        Raises:
            LookupError: 未找到对应股票时抛出。
            RuntimeError: 远端返回格式非法时抛出。
        """

        normalized_ticker = self.normalize_ticker(ticker)
        stock_records = self._load_stock_records()
        for record in stock_records:
            if record.code == normalized_ticker:
                return record
        raise LookupError(f"巨潮资讯未找到股票代码: {normalized_ticker}")

    def query_announcements(
        self,
        *,
        profile: CninfoCompanyProfile,
        report_type: str,
        start_date: str,
        end_date: str,
    ) -> list[CninfoAnnouncement]:
        """查询指定报告类型的公告。

        Args:
            profile: 公司档案。
            report_type: 报告类型。
            start_date: 查询开始日期。
            end_date: 查询结束日期。

        Returns:
            排序后的公告列表。

        Raises:
            ValueError: 报告类型不支持时抛出。
            RuntimeError: 远端返回格式非法时抛出。
        """

        category = REPORT_CATEGORY_MAP.get(report_type)
        if category is None:
            raise ValueError(f"不支持的 report_type: {report_type}")
        all_announcements: list[CninfoAnnouncement] = []
        page_no = 1
        while True:
            payload = self._build_announcement_payload(
                profile=profile,
                category=category,
                start_date=start_date,
                end_date=end_date,
                page_no=page_no,
            )
            response = self._session.post(
                CNINFO_ANNOUNCEMENT_API,
                data=payload,
                timeout=self._timeout_seconds,
                headers=HEADERS,
            )
            response.raise_for_status()
            payload_json = response.json()
            if not isinstance(payload_json, dict):
                raise RuntimeError("巨潮公告接口返回格式非法")
            announcements_raw = payload_json.get("announcements")
            if not isinstance(announcements_raw, list):
                break
            page_announcements = self._parse_announcements(
                announcements_raw=announcements_raw,
                report_type=report_type,
            )
            if len(page_announcements) == 0:
                break
            all_announcements.extend(page_announcements)
            total_announcement = payload_json.get("totalAnnouncement")
            if not isinstance(total_announcement, int):
                break
            if len(all_announcements) >= total_announcement:
                break
            page_no += 1
        return _deduplicate_announcements(all_announcements)

    def download_pdf(self, pdf_url: str) -> DownloadedPdf:
        """下载 PDF 原文。

        Args:
            pdf_url: PDF 地址。

        Returns:
            下载结果。

        Raises:
            RuntimeError: 下载结果不是合法 PDF 时抛出。
        """

        response = self._session.get(
            pdf_url,
            timeout=self._timeout_seconds,
            headers={"User-Agent": HEADERS["User-Agent"]},
        )
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF-"):
            raise RuntimeError("巨潮返回内容不是合法 PDF")
        content_type = response.headers.get("Content-Type")
        return DownloadedPdf(
            content=content,
            content_type=str(content_type).strip() or None,
            source_url=pdf_url,
        )

    def _load_stock_records(self) -> list[CninfoCompanyProfile]:
        """加载股票档案缓存。

        Args:
            无。

        Returns:
            股票档案列表。

        Raises:
            RuntimeError: 远端返回格式非法时抛出。
        """

        cached_payload = self._load_stock_cache()
        if cached_payload is None:
            cached_payload = self._fetch_stock_records_payload()
            self._save_stock_cache(cached_payload)
        return _parse_stock_records(cached_payload)

    def _fetch_stock_records_payload(self) -> list[dict[str, object]]:
        """从远端拉取股票档案原始载荷。

        Args:
            无。

        Returns:
            原始列表载荷。

        Raises:
            RuntimeError: 远端返回格式非法时抛出。
        """

        response = self._session.get(
            CNINFO_STOCK_JSON_URL,
            timeout=self._timeout_seconds,
            headers=HEADERS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("巨潮股票列表返回格式非法")
        stock_list = payload.get("stockList")
        if not isinstance(stock_list, list):
            raise RuntimeError("巨潮股票列表缺少 stockList")
        normalized_list: list[dict[str, object]] = []
        for item in stock_list:
            if isinstance(item, dict):
                normalized_list.append(item)
        return normalized_list

    def _build_announcement_payload(
        self,
        *,
        profile: CninfoCompanyProfile,
        category: str,
        start_date: str,
        end_date: str,
        page_no: int,
    ) -> dict[str, str]:
        """构建公告查询表单。

        Args:
            profile: 公司档案。
            category: 巨潮 category。
            start_date: 开始日期。
            end_date: 结束日期。
            page_no: 页码。

        Returns:
            表单字典。

        Raises:
            无。
        """

        column = EXCHANGE_COLUMN_MAP.get(profile.exchange, "szse")
        return {
            "pageNum": str(page_no),
            "pageSize": "30",
            "column": column,
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{profile.code},{profile.org_id}",
            "searchkey": "",
            "secid": "",
            "category": category,
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

    def _load_stock_cache(self) -> Optional[list[dict[str, object]]]:
        """从本地读取股票缓存。

        Args:
            无。

        Returns:
            缓存载荷；过期或损坏时返回 `None`。

        Raises:
            无。
        """

        cache_path = self._cache_dir / CNINFO_STOCK_CACHE_FILENAME
        if not cache_path.exists():
            return None
        try:
            mtime = cache_path.stat().st_mtime
            if datetime.now().timestamp() - mtime > CNINFO_STOCK_CACHE_TTL_SECONDS:
                return None
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, list):
            return None
        normalized_payload: list[dict[str, object]] = []
        for item in payload:
            if isinstance(item, dict):
                normalized_payload.append(item)
        return normalized_payload

    def _save_stock_cache(self, payload: list[dict[str, object]]) -> None:
        """写入股票缓存。

        Args:
            payload: 原始列表载荷。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_dir / CNINFO_STOCK_CACHE_FILENAME
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            return

    def _parse_announcements(
        self,
        *,
        announcements_raw: list[object],
        report_type: str,
    ) -> list[CninfoAnnouncement]:
        """解析公告列表。

        Args:
            announcements_raw: 远端原始公告数组。
            report_type: 当前报告类型。

        Returns:
            过滤后的公告列表。

        Raises:
            无。
        """

        parsed_announcements: list[CninfoAnnouncement] = []
        for item in announcements_raw:
            if not isinstance(item, dict):
                continue
            adjunct_type = str(item.get("adjunctType") or "").strip().upper()
            adjunct_url = str(item.get("adjunctUrl") or "").strip()
            if adjunct_type != "PDF" or not adjunct_url:
                continue
            title = _TITLE_HTML_TAG_PATTERN.sub("", str(item.get("announcementTitle") or "")).strip()
            if _should_exclude_announcement_title(title):
                continue
            fiscal_year = _extract_fiscal_year_from_title(title)
            fiscal_period = _resolve_fiscal_period_from_report_type(report_type)
            if fiscal_year is None:
                announcement_date = _parse_announcement_date(item.get("announcementTime"))
                if announcement_date is None:
                    continue
                fiscal_year = int(announcement_date[:4])
                normalized_date = announcement_date
            else:
                announcement_date = _parse_announcement_date(item.get("announcementTime"))
                if announcement_date is None:
                    continue
                normalized_date = announcement_date
            parsed_announcements.append(
                CninfoAnnouncement(
                    announcement_id=str(item.get("announcementId") or "").strip(),
                    title=title,
                    announcement_date=normalized_date,
                    pdf_url=CNINFO_PDF_BASE_URL + adjunct_url,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    report_type=report_type,
                    adjunct_size_kb=_normalize_int(item.get("adjunctSize")),
                )
            )
        return parsed_announcements


def _parse_stock_records(payload: list[dict[str, object]]) -> list[CninfoCompanyProfile]:
    """解析股票档案。

    Args:
        payload: 原始载荷。

    Returns:
        公司档案列表。

    Raises:
        RuntimeError: 远端字段缺失且无法恢复时抛出。
    """

    parsed_records: list[CninfoCompanyProfile] = []
    for item in payload:
        code = str(item.get("code") or "").strip()
        org_id = str(item.get("orgId") or "").strip()
        if not code or not org_id:
            continue
        normalized_ticker = normalize_ticker(code)
        company_name = _extract_company_name(item, fallback=normalized_ticker.canonical)
        exchange = _resolve_exchange_code(normalized_ticker.exchange)
        parsed_records.append(
            CninfoCompanyProfile(
                code=normalized_ticker.canonical,
                exchange=exchange,
                org_id=org_id,
                company_name=company_name,
            )
        )
    return parsed_records


def _extract_company_name(item: dict[str, object], *, fallback: str) -> str:
    """从巨潮股票记录中提取公司简称。

    Args:
        item: 原始股票记录。
        fallback: 回退名称。

    Returns:
        公司名称。

    Raises:
        无。
    """

    candidate_keys: tuple[str, ...] = ("zwjc", "zjc", "ssjc", "secName", "name")
    for key in candidate_keys:
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return fallback


def _resolve_exchange_code(exchange: Optional[str]) -> str:
    """规范化交易所编码。

    Args:
        exchange: 交易所编码。

    Returns:
        `SH/SZ/BJ` 之一，未知时回退 `SZ`。

    Raises:
        无。
    """

    normalized = str(exchange or "").strip().upper()
    if normalized in {"SSE", "SH"}:
        return "SH"
    if normalized in {"SZSE", "SZ"}:
        return "SZ"
    if normalized == "BJ":
        return "BJ"
    return "SZ"


def _should_exclude_announcement_title(title: str) -> bool:
    """判断是否应排除该公告标题。

    Args:
        title: 公告标题。

    Returns:
        是否排除。

    Raises:
        无。
    """

    exclude_keywords: tuple[str, ...] = ("摘要", "英文", "更正", "补充", "致歉", "修订", "已取消")
    return any(keyword in title for keyword in exclude_keywords)


def _extract_fiscal_year_from_title(title: str) -> Optional[int]:
    """从标题中提取财年。

    Args:
        title: 公告标题。

    Returns:
        财年；未提取到时返回 `None`。

    Raises:
        无。
    """

    matched = re.search(r"(20\d{2})年", title)
    if matched is None:
        return None
    return int(matched.group(1))


def _resolve_fiscal_period_from_report_type(report_type: str) -> str:
    """把报告类型映射到项目内 fiscal_period。

    Args:
        report_type: 报告类型。

    Returns:
        fiscal_period。

    Raises:
        ValueError: 报告类型不支持时抛出。
    """

    if report_type == "年报":
        return "FY"
    if report_type == "半年报":
        return "H1"
    if report_type == "一季报":
        return "Q1"
    if report_type == "三季报":
        return "Q3"
    raise ValueError(f"不支持的 report_type: {report_type}")


def _parse_announcement_date(raw_value: object) -> Optional[str]:
    """解析公告日期毫秒时间戳。

    Args:
        raw_value: 原始字段。

    Returns:
        `YYYY-MM-DD`；无法解析时返回 `None`。

    Raises:
        无。
    """

    if not isinstance(raw_value, int):
        return None
    return datetime.fromtimestamp(raw_value / 1000).strftime("%Y-%m-%d")


def _deduplicate_announcements(
    announcements: list[CninfoAnnouncement],
) -> list[CninfoAnnouncement]:
    """按 announcement_id 去重并按日期倒序排序。

    Args:
        announcements: 原始公告列表。

    Returns:
        去重后的公告列表。

    Raises:
        无。
    """

    deduplicated: dict[str, CninfoAnnouncement] = {}
    for announcement in announcements:
        stable_id = announcement.announcement_id or f"{announcement.fiscal_year}:{announcement.fiscal_period}:{announcement.title}"
        previous = deduplicated.get(stable_id)
        if previous is None or announcement.announcement_date > previous.announcement_date:
            deduplicated[stable_id] = announcement
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.announcement_date, item.fiscal_year, item.fiscal_period, item.title),
        reverse=True,
    )


def _normalize_int(value: object) -> int:
    """安全转整数。

    Args:
        value: 原始值。

    Returns:
        转换后的整数；失败时返回 0。

    Raises:
        无。
    """

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


__all__ = [
    "CninfoAnnouncement",
    "CninfoCompanyProfile",
    "CninfoDownloader",
    "DownloadedPdf",
    "REPORT_CATEGORY_MAP",
]
