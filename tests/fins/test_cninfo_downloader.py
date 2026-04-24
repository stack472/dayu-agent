"""CninfoDownloader 单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from dayu.fins.downloaders.cninfo_downloader import CninfoDownloader


class _FakeResponse:
    """测试用响应对象。"""

    def __init__(
        self,
        *,
        json_payload: object | None = None,
        content: bytes = b"",
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """初始化响应对象。

        Args:
            json_payload: JSON 载荷。
            content: 字节内容。
            headers: 响应头。

        Returns:
            无。

        Raises:
            无。
        """

        self._json_payload = json_payload
        self._content = content
        self._headers = headers or {}

    @property
    def headers(self) -> dict[str, str]:
        """返回响应头。"""

        return self._headers

    @property
    def content(self) -> bytes:
        """返回响应体。"""

        return self._content

    def json(self) -> object:
        """返回 JSON 载荷。"""

        return self._json_payload

    def raise_for_status(self) -> None:
        """测试桩中固定成功。"""

        return


class _FakeSession:
    """测试用 HTTP session。"""

    def __init__(self) -> None:
        """初始化响应队列。"""

        self.get_calls: list[str] = []
        self.post_calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        timeout: float,
        headers: Optional[dict[str, str]] = None,
    ) -> _FakeResponse:
        """处理 GET 请求。

        Args:
            url: 请求地址。
            timeout: 超时秒数。
            headers: 请求头。

        Returns:
            固定响应。

        Raises:
            AssertionError: 地址不符合预期时抛出。
        """

        del timeout, headers
        self.get_calls.append(url)
        if url.endswith("szse_stock.json"):
            return _FakeResponse(
                json_payload={
                    "stockList": [
                        {
                            "code": "000001",
                            "orgId": "gssz000001",
                            "zwjc": "平安银行",
                        }
                    ]
                }
            )
        return _FakeResponse(
            content=b"%PDF-1.4 test",
            headers={"Content-Type": "application/pdf"},
        )

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        timeout: float,
        headers: Optional[dict[str, str]] = None,
    ) -> _FakeResponse:
        """处理 POST 请求。

        Args:
            url: 请求地址。
            data: 表单数据。
            timeout: 超时秒数。
            headers: 请求头。

        Returns:
            固定响应。

        Raises:
            AssertionError: 地址不符合预期时抛出。
        """

        del timeout, headers
        self.post_calls.append(url)
        assert data["stock"] == "000001,gssz000001"
        return _FakeResponse(
            json_payload={
                "totalAnnouncement": 1,
                "announcements": [
                    {
                        "announcementId": "ann_1",
                        "announcementTitle": "2024年年度报告",
                        "adjunctUrl": "finalpage/2025-03-28/report.pdf",
                        "adjunctType": "PDF",
                        "adjunctSize": 1024,
                        "announcementTime": 1743091200000,
                    }
                ],
            }
        )


@pytest.mark.unit
def test_resolve_company_uses_cninfo_stock_list(tmp_path: Path) -> None:
    """验证 resolve_company 会解析巨潮股票列表。"""

    downloader = CninfoDownloader(
        cache_dir=tmp_path,
        session=_FakeSession(),
    )

    profile = downloader.resolve_company("000001")

    assert profile.code == "000001"
    assert profile.org_id == "gssz000001"
    assert profile.company_name == "平安银行"
    assert profile.exchange == "SZ"


@pytest.mark.unit
def test_query_announcements_returns_filtered_pdf_records(tmp_path: Path) -> None:
    """验证 query_announcements 会返回标准化公告记录。"""

    session = _FakeSession()
    downloader = CninfoDownloader(
        cache_dir=tmp_path,
        session=session,
    )
    profile = downloader.resolve_company("000001")

    announcements = downloader.query_announcements(
        profile=profile,
        report_type="年报",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    assert len(announcements) == 1
    assert announcements[0].fiscal_year == 2024
    assert announcements[0].fiscal_period == "FY"
    assert announcements[0].pdf_url.endswith("report.pdf")


@pytest.mark.unit
def test_download_pdf_rejects_non_pdf_payload(tmp_path: Path) -> None:
    """验证 download_pdf 会拒绝非 PDF 内容。"""

    class _BadSession(_FakeSession):
        """返回非法 PDF 内容的 session。"""

        def get(
            self,
            url: str,
            *,
            timeout: float,
            headers: Optional[dict[str, str]] = None,
        ) -> _FakeResponse:
            del timeout, headers
            if url.endswith("szse_stock.json"):
                return super().get(url, timeout=1, headers=None)
            return _FakeResponse(content=b"not-pdf", headers={"Content-Type": "text/plain"})

    downloader = CninfoDownloader(
        cache_dir=tmp_path,
        session=_BadSession(),
    )

    with pytest.raises(RuntimeError, match="不是合法 PDF"):
        downloader.download_pdf("http://static.cninfo.com.cn/bad.txt")
