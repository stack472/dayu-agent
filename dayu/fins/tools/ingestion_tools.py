"""财报长事务工具注册模块。

该模块负责将 `download/process` 两类长事务暴露为适合 LLM 调用的
`start/status/cancel` job 工具，并严格控制 schema 与返回数据的认知负担。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from dayu.contracts.fins import (
    FinsCommand,
    FinsCommandName,
    FinsResult,
    UploadFilingCommandPayload,
    UploadFilingsFromCommandPayload,
    UploadMaterialCommandPayload,
)
from dayu.engine.exceptions import ToolArgumentError
from dayu.fins._converters import normalize_optional_text, require_non_empty_text
from dayu.engine.tool_contracts import DupCallSpec
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.base import tool
from dayu.fins.ingestion.job_manager import (
    IngestionJobManager,
    get_or_create_ingestion_job_manager,
)
from dayu.fins.ticker_normalization import normalize_ticker
from dayu.fins.ingestion.service import FinsIngestionService

MODULE = "FINS.INGESTION_TOOLS"
INGESTION_TOOL_TAGS = frozenset({"ingestion"})

_SUPPORTED_DOWNLOAD_FORM_TYPES = [
    "10-K",
    "10-Q",
    "20-F",
    "6-K",
    "8-K",
    "8-K/A",
    "DEF 14A",
    "SC 13D",
    "SC 13D/A",
    "SC 13G",
    "SC 13G/A",
]

_SUPPORTED_DOWNLOAD_MARKETS = frozenset({"US", "CN"})
_JOB_TERMINAL_STATUSES = ["succeeded", "failed", "cancelled"]
_SUPPORTED_FISCAL_PERIODS = ["FY", "H1", "Q1", "Q2", "Q3", "Q4"]
_SUPPORTED_UPLOAD_ACTIONS = ["create", "update", "delete"]
_SUPPORTED_UPLOAD_MATERIAL_FORM_TYPES = [
    "EARNINGS_CALL",
    "EARNINGS_PRESENTATION",
    "CORPORATE_GOVERNANCE",
    "MATERIAL_OTHER",
]


class FinsRuntimeLike(Protocol):
    """上传工具依赖的最小 runtime 协议。"""

    def execute(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult | Any:
        """执行同步财报命令。"""

        ...

def register_ingestion_tools(
    registry: ToolRegistry,
    *,
    service_factory: Callable[[str], FinsIngestionService],
    manager_key: str,
    runtime: FinsRuntimeLike | None = None,
    timeout_budget: float | None = None,
) -> int:
    """注册财报长事务 job 工具。

    Args:
        registry: 工具注册表。
        service_factory: `ticker -> FinsIngestionService` 工厂。
        manager_key: 长事务 job 管理器稳定标识。
        runtime: 可选同步 runtime；提供后额外注册上传相关工具。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 ingestion 工具预留该参数，
            暂未消费。

    Returns:
        新注册的工具数量。

    Raises:
        ValueError: 参数非法时抛出。
    """
    del timeout_budget
    if registry is None:
        raise ValueError("registry 不能为空")
    if service_factory is None:
        raise ValueError("service_factory 不能为空")
    if not str(manager_key).strip():
        raise ValueError("manager_key 不能为空")

    manager = get_or_create_ingestion_job_manager(
        manager_key=manager_key,
        service_factory=service_factory,
    )
    tool_factories = [
        _create_start_download_job_tool,
        _create_get_download_job_status_tool,
        _create_cancel_download_job_tool,
    ]
    if runtime is not None:
        upload_tool_factories = [
            _create_upload_filing_tool,
            _create_prepare_upload_filings_from_tool,
            _create_upload_material_tool,
        ]
        for factory in upload_tool_factories:
            name, func, schema = factory(registry=registry, runtime=runtime)
            registry.register(name, func, schema)
    for factory in tool_factories:
        name, func, schema = factory(registry=registry, manager=manager)
        registry.register(name, func, schema)
    return len(tool_factories) + (3 if runtime is not None else 0)


def _create_start_download_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 启动工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。第一次直接传最自然的写法即可。",
            },
            "form_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": _SUPPORTED_DOWNLOAD_FORM_TYPES,
                },
                "uniqueItems": True,
                "description": "可选表单过滤。只在你明确要缩小下载范围时填写；留空表示下载该公司当前支持的全部表单。",
            },
            "filed_date_from": {
                "type": "string",
                "description": "可选 filed date 下界。只在你明确要限制时间范围时填写；格式 YYYY、YYYY-MM 或 YYYY-MM-DD。",
            },
            "filed_date_to": {
                "type": "string",
                "description": "可选 filed date 上界。只在你明确要限制时间范围时填写；格式 YYYY、YYYY-MM 或 YYYY-MM-DD。",
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否覆盖已有本地下载结果。仅在你明确要重下时设为 true。",
                "default": False,
            },
        },
        "required": ["ticker"],
    }

    @tool(
        registry,
        name="start_financial_filing_download_job",
        description=(
            "启动单个 ticker 的下载任务。拿到返回里的 job.job_id 后，下一步只用状态工具轮询，直到 job.status 进入 succeeded / failed / cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="启动财报下载",
        summary_params=["ticker"],
    )
    def start_financial_filing_download_job(
        ticker: str,
        form_types: Optional[list[str]] = None,
        filed_date_from: Optional[str] = None,
        filed_date_to: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动财报下载 job。

        Args:
            ticker: 股票代码。
            form_types: 可选表单过滤。
            filed_date_from: 可选 filing date 下界。
            filed_date_to: 可选 filing date 上界。
            overwrite: 是否覆盖已有下载结果。

        Returns:
            启动结果，包含 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError(
                "start_financial_filing_download_job",
                "ticker",
                ticker,
                "不能为空",
            ),
        )
        market_profile = normalize_ticker(normalized_ticker)
        if market_profile.market not in _SUPPORTED_DOWNLOAD_MARKETS:
            return _build_not_implemented_start_response(
                ticker=normalized_ticker,
                market=market_profile.market,
            )
        normalized_form_types = _normalize_form_types(form_types)
        request_outcome, snapshot = manager.start_download_job(
            ticker=normalized_ticker,
            form_types=normalized_form_types,
            filed_date_from=normalize_optional_text(filed_date_from),
            filed_date_to=normalize_optional_text(filed_date_to),
            overwrite=bool(overwrite),
        )
        return _build_start_response(
            request_outcome=request_outcome,
            snapshot=snapshot,
            start_tool_name="start_financial_filing_download_job",
            status_tool_name="get_financial_filing_download_job_status",
        )

    return (
        start_financial_filing_download_job.__tool_name__,
        start_financial_filing_download_job,
        start_financial_filing_download_job.__tool_schema__,
    )


def _create_get_download_job_status_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 状态查询工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具返回的 job.job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="get_financial_filing_download_job_status",
        description=(
            "查询下载任务状态。启动后反复调用本工具，直到 job.status 进入 succeeded / failed / cancelled。优先按 next_step.action 决定是继续轮询、停止还是重新启动。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="查询下载状态",
        dup_call=DupCallSpec(
            mode="poll_until_terminal",
            status_path="job.status",
            terminal_values=_JOB_TERMINAL_STATUSES,
        ),
    )
    def get_financial_filing_download_job_status(job_id: str) -> dict[str, Any]:
        """查询下载 job 状态。

        Args:
            job_id: job 标识。

        Returns:
            状态摘要、失败信息与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = require_non_empty_text(
            job_id,
            empty_error=ToolArgumentError(
                "get_financial_filing_download_job_status",
                "job_id",
                job_id,
                "不能为空",
            ),
        )
        snapshot = manager.get_job_snapshot(normalized_job_id)
        return _build_status_response(
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_filing_download_job_status",
            start_tool_name="start_financial_filing_download_job",
        )

    return (
        get_financial_filing_download_job_status.__tool_name__,
        get_financial_filing_download_job_status,
        get_financial_filing_download_job_status.__tool_schema__,
    )


def _create_upload_filing_tool(
    registry: ToolRegistry,
    runtime: FinsRuntimeLike,
) -> tuple[str, Any, Any]:
    """创建单份财报上传工具。

    Args:
        registry: 工具注册表。
        runtime: 财报运行时。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。A股直接传 6 位代码即可，例如 000333、600519。",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "待上传文件的本地绝对路径列表。create/update 必填。",
            },
            "fiscal_year": {
                "type": "integer",
                "description": "财年，例如 2024。",
            },
            "fiscal_period": {
                "type": "string",
                "enum": _SUPPORTED_FISCAL_PERIODS,
                "description": "财期。A股常见为 FY、H1、Q1、Q3。",
            },
            "action": {
                "type": "string",
                "enum": _SUPPORTED_UPLOAD_ACTIONS,
                "description": "可选动作。留空时按稳定 document_id 自动判定 create/update。",
            },
            "amended": {
                "type": "boolean",
                "description": "是否修订版报告。",
                "default": False,
            },
            "filing_date": {
                "type": "string",
                "description": "可选公告日期，格式 YYYY-MM-DD。",
            },
            "report_date": {
                "type": "string",
                "description": "可选报告期末日期，格式 YYYY-MM-DD。",
            },
            "company_id": {
                "type": "string",
                "description": "首次上传且本地尚无公司 meta 时必填。",
            },
            "company_name": {
                "type": "string",
                "description": "首次上传且本地尚无公司 meta 时通常必填。",
            },
            "ticker_aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选别名列表，如 000333.SZ、MIDEA。",
            },
            "infer": {
                "type": "boolean",
                "description": "是否允许在缺少公司名称时尝试推断。",
                "default": False,
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否重置当前 document_id 后再完整重建。",
                "default": False,
            },
        },
        "required": ["ticker", "files", "fiscal_year", "fiscal_period"],
    }

    @tool(
        registry,
        name="upload_financial_filing",
        description=(
            "上传单份本地财报到工作区。适用于 A股/港股/已离线下载的任意市场财报。上传成功后即可继续调用财报读取工具分析。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def upload_financial_filing(
        ticker: str,
        files: list[str],
        fiscal_year: int,
        fiscal_period: str,
        action: Optional[str] = None,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        infer: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """上传单份财报。

        Args:
            ticker: 股票代码。
            files: 本地绝对路径列表。
            fiscal_year: 财年。
            fiscal_period: 财期。
            action: 可选动作。
            amended: 是否修订版。
            filing_date: 可选公告日期。
            report_date: 可选报告期日期。
            company_id: 可选公司主体 ID。
            company_name: 可选公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            infer: 是否允许推断公司名称。
            overwrite: 是否覆盖当前稳定文档。

        Returns:
            上传结果字典。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            TypeError: runtime 未返回同步结果时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError("upload_financial_filing", "ticker", ticker, "不能为空"),
        )
        command = FinsCommand(
            name=FinsCommandName.UPLOAD_FILING,
            payload=UploadFilingCommandPayload(
                ticker=normalized_ticker,
                files=_normalize_upload_paths(tool_name="upload_financial_filing", files=files),
                fiscal_year=int(fiscal_year),
                action=normalize_optional_text(action),
                fiscal_period=require_non_empty_text(
                    fiscal_period,
                    empty_error=ToolArgumentError(
                        "upload_financial_filing",
                        "fiscal_period",
                        fiscal_period,
                        "不能为空",
                    ),
                ),
                amended=bool(amended),
                filing_date=normalize_optional_text(filing_date),
                report_date=normalize_optional_text(report_date),
                company_id=normalize_optional_text(company_id),
                company_name=normalize_optional_text(company_name),
                infer=bool(infer),
                ticker_aliases=tuple(_normalize_optional_string_list(ticker_aliases)),
                overwrite=bool(overwrite),
            ),
            stream=False,
        )
        return _execute_runtime_sync_command(runtime=runtime, command=command)

    return (
        upload_financial_filing.__tool_name__,
        upload_financial_filing,
        upload_financial_filing.__tool_schema__,
    )


def _create_prepare_upload_filings_from_tool(
    registry: ToolRegistry,
    runtime: FinsRuntimeLike,
) -> tuple[str, Any, Any]:
    """创建批量上传脚本生成工具。

    Args:
        registry: 工具注册表。
        runtime: 财报运行时。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。A股直接传 6 位代码即可。",
            },
            "source_dir": {
                "type": "string",
                "description": "待扫描目录的本地绝对路径。",
            },
            "action": {
                "type": "string",
                "enum": _SUPPORTED_UPLOAD_ACTIONS,
                "description": "可选固定动作；留空时脚本内每条命令自动判定 create/update。",
            },
            "output_script": {
                "type": "string",
                "description": "可选输出脚本路径；留空时写到 workspace 根目录。",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否递归扫描子目录。",
                "default": False,
            },
            "amended": {
                "type": "boolean",
                "description": "是否统一按修订版处理。",
                "default": False,
            },
            "filing_date": {
                "type": "string",
                "description": "可选统一公告日期，格式 YYYY-MM-DD。",
            },
            "report_date": {
                "type": "string",
                "description": "可选统一报告期日期，格式 YYYY-MM-DD。",
            },
            "company_id": {
                "type": "string",
                "description": "首次生成并执行上传脚本时用于初始化公司 meta 的主体 ID。",
            },
            "company_name": {
                "type": "string",
                "description": "首次生成并执行上传脚本时用于初始化公司 meta 的公司名称。",
            },
            "infer": {
                "type": "boolean",
                "description": "是否允许推断公司名称与 alias。",
                "default": False,
            },
            "overwrite": {
                "type": "boolean",
                "description": "脚本执行时是否默认透传 overwrite。",
                "default": False,
            },
            "material_forms": {
                "type": "array",
                "items": {"type": "string", "enum": _SUPPORTED_UPLOAD_MATERIAL_FORM_TYPES},
                "description": "可选材料 form 白名单，用于识别目录中的补充材料。",
            },
        },
        "required": ["ticker", "source_dir"],
    }

    @tool(
        registry,
        name="prepare_financial_filings_upload_script",
        description=(
            "扫描本地目录并生成批量上传脚本。适合 A股/港股整批年报、中报、季报与补充材料接入。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def prepare_financial_filings_upload_script(
        ticker: str,
        source_dir: str,
        action: Optional[str] = None,
        output_script: Optional[str] = None,
        recursive: bool = False,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        infer: bool = False,
        overwrite: bool = False,
        material_forms: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """生成批量上传脚本。

        Args:
            ticker: 股票代码。
            source_dir: 待扫描目录。
            action: 可选固定动作。
            output_script: 可选脚本输出路径。
            recursive: 是否递归扫描。
            amended: 是否视为修订版。
            filing_date: 可选统一公告日期。
            report_date: 可选统一报告期日期。
            company_id: 可选公司主体 ID。
            company_name: 可选公司名称。
            infer: 是否允许推断公司名称与 alias。
            overwrite: 是否在脚本中透传 overwrite。
            material_forms: 可选材料 form 白名单。

        Returns:
            脚本生成结果字典。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            TypeError: runtime 未返回同步结果时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError(
                "prepare_financial_filings_upload_script",
                "ticker",
                ticker,
                "不能为空",
            ),
        )
        normalized_source_dir = require_non_empty_text(
            source_dir,
            empty_error=ToolArgumentError(
                "prepare_financial_filings_upload_script",
                "source_dir",
                source_dir,
                "不能为空",
            ),
        )
        normalized_output_script = normalize_optional_text(output_script)
        command = FinsCommand(
            name=FinsCommandName.UPLOAD_FILINGS_FROM,
            payload=UploadFilingsFromCommandPayload(
                ticker=normalized_ticker,
                source_dir=Path(normalized_source_dir),
                action=normalize_optional_text(action),
                output_script=Path(normalized_output_script) if normalized_output_script is not None else None,
                recursive=bool(recursive),
                amended=bool(amended),
                filing_date=normalize_optional_text(filing_date),
                report_date=normalize_optional_text(report_date),
                company_id=normalize_optional_text(company_id),
                company_name=normalize_optional_text(company_name),
                infer=bool(infer),
                overwrite=bool(overwrite),
                material_forms=tuple(_normalize_optional_string_list(material_forms)),
            ),
            stream=False,
        )
        return _execute_runtime_sync_command(runtime=runtime, command=command)

    return (
        prepare_financial_filings_upload_script.__tool_name__,
        prepare_financial_filings_upload_script,
        prepare_financial_filings_upload_script.__tool_schema__,
    )


def _create_upload_material_tool(
    registry: ToolRegistry,
    runtime: FinsRuntimeLike,
) -> tuple[str, Any, Any]:
    """创建材料上传工具。

    Args:
        registry: 工具注册表。
        runtime: 财报运行时。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。A股直接传 6 位代码即可。",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "待上传材料的本地绝对路径列表。create/update 必填。",
            },
            "form_type": {
                "type": "string",
                "enum": _SUPPORTED_UPLOAD_MATERIAL_FORM_TYPES,
                "description": "材料类型，例如 EARNINGS_PRESENTATION。",
            },
            "material_name": {
                "type": "string",
                "description": "材料名称，用于稳定 document_id。",
            },
            "action": {
                "type": "string",
                "enum": _SUPPORTED_UPLOAD_ACTIONS,
                "description": "可选动作。留空时按稳定 document_id 自动判定 create/update。",
            },
            "document_id": {
                "type": "string",
                "description": "可选显式 document_id；必须与稳定规则一致。",
            },
            "internal_document_id": {
                "type": "string",
                "description": "可选显式 internal_document_id；必须与稳定规则一致。",
            },
            "fiscal_year": {
                "type": "integer",
                "description": "可选财年。",
            },
            "fiscal_period": {
                "type": "string",
                "enum": _SUPPORTED_FISCAL_PERIODS,
                "description": "可选财期。",
            },
            "filing_date": {
                "type": "string",
                "description": "可选公告日期，格式 YYYY-MM-DD。",
            },
            "report_date": {
                "type": "string",
                "description": "可选报告期日期，格式 YYYY-MM-DD。",
            },
            "company_id": {
                "type": "string",
                "description": "首次上传且本地尚无公司 meta 时使用的公司主体 ID。",
            },
            "company_name": {
                "type": "string",
                "description": "首次上传且本地尚无公司 meta 时使用的公司名称。",
            },
            "ticker_aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选别名列表。",
            },
            "infer": {
                "type": "boolean",
                "description": "是否允许在缺少公司名称时尝试推断。",
                "default": False,
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否重置当前稳定材料文档后再完整重建。",
                "default": False,
            },
        },
        "required": ["ticker", "files", "form_type", "material_name"],
    }

    @tool(
        registry,
        name="upload_financial_material",
        description=(
            "上传补充材料，如业绩会纪要、演示稿、治理材料。适用于 A股/港股/已离线整理的任意市场材料。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
    )
    def upload_financial_material(
        ticker: str,
        files: list[str],
        form_type: str,
        material_name: str,
        action: Optional[str] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        infer: bool = False,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """上传补充材料。

        Args:
            ticker: 股票代码。
            files: 本地绝对路径列表。
            form_type: 材料类型。
            material_name: 材料名称。
            action: 可选动作。
            document_id: 可选显式 document_id。
            internal_document_id: 可选显式 internal_document_id。
            fiscal_year: 可选财年。
            fiscal_period: 可选财期。
            filing_date: 可选公告日期。
            report_date: 可选报告期日期。
            company_id: 可选公司主体 ID。
            company_name: 可选公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            infer: 是否允许推断公司名称。
            overwrite: 是否覆盖当前稳定文档。

        Returns:
            上传结果字典。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            TypeError: runtime 未返回同步结果时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError("upload_financial_material", "ticker", ticker, "不能为空"),
        )
        command = FinsCommand(
            name=FinsCommandName.UPLOAD_MATERIAL,
            payload=UploadMaterialCommandPayload(
                ticker=normalized_ticker,
                files=_normalize_upload_paths(tool_name="upload_financial_material", files=files),
                action=normalize_optional_text(action),
                form_type=require_non_empty_text(
                    form_type,
                    empty_error=ToolArgumentError(
                        "upload_financial_material",
                        "form_type",
                        form_type,
                        "不能为空",
                    ),
                ),
                material_name=require_non_empty_text(
                    material_name,
                    empty_error=ToolArgumentError(
                        "upload_financial_material",
                        "material_name",
                        material_name,
                        "不能为空",
                    ),
                ),
                document_id=normalize_optional_text(document_id),
                internal_document_id=normalize_optional_text(internal_document_id),
                fiscal_year=fiscal_year,
                fiscal_period=normalize_optional_text(fiscal_period),
                filing_date=normalize_optional_text(filing_date),
                report_date=normalize_optional_text(report_date),
                company_id=normalize_optional_text(company_id),
                company_name=normalize_optional_text(company_name),
                infer=bool(infer),
                ticker_aliases=tuple(_normalize_optional_string_list(ticker_aliases)),
                overwrite=bool(overwrite),
            ),
            stream=False,
        )
        return _execute_runtime_sync_command(runtime=runtime, command=command)

    return (
        upload_financial_material.__tool_name__,
        upload_financial_material,
        upload_financial_material.__tool_schema__,
    )


def _create_cancel_download_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建下载 job 取消工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具或状态工具返回的 job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="cancel_financial_filing_download_job",
        description=(
            "请求取消下载任务。取消不是立即完成的；调用后继续用状态工具确认是否进入 cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="取消下载任务",
    )
    def cancel_financial_filing_download_job(job_id: str) -> dict[str, Any]:
        """请求取消下载 job。

        Args:
            job_id: job 标识。

        Returns:
            取消请求结果、当前 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = require_non_empty_text(
            job_id,
            empty_error=ToolArgumentError(
                "cancel_financial_filing_download_job",
                "job_id",
                job_id,
                "不能为空",
            ),
        )
        cancellation_outcome, snapshot = manager.cancel_job(normalized_job_id)
        return _build_cancel_response(
            cancellation_outcome=cancellation_outcome,
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_filing_download_job_status",
            start_tool_name="start_financial_filing_download_job",
        )

    return (
        cancel_financial_filing_download_job.__tool_name__,
        cancel_financial_filing_download_job,
        cancel_financial_filing_download_job.__tool_schema__,
    )


def _create_start_process_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 启动工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "公司代码。第一次直接传最自然的写法即可。",
            },
            "overwrite": {
                "type": "boolean",
                "description": "是否覆盖已有快照。仅在你明确要重做处理时设为 true。",
                "default": False,
            },
        },
        "required": ["ticker"],
    }

    @tool(
        registry,
        name="start_financial_document_preprocess_job",
        description=(
            "启动单个 ticker 的预处理任务。拿到返回里的 job.job_id 后，下一步只用状态工具轮询，直到 job.status 进入 succeeded / failed / cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="启动文档预处理",
        summary_params=["ticker"],
    )
    def start_financial_document_preprocess_job(
        ticker: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """启动预处理 job。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖已有快照。

        Returns:
            启动结果，包含 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError(
                "start_financial_document_preprocess_job",
                "ticker",
                ticker,
                "不能为空",
            ),
        )
        request_outcome, snapshot = manager.start_process_job(
            ticker=normalized_ticker,
            overwrite=bool(overwrite),
        )
        return _build_start_response(
            request_outcome=request_outcome,
            snapshot=snapshot,
            start_tool_name="start_financial_document_preprocess_job",
            status_tool_name="get_financial_document_preprocess_job_status",
        )

    return (
        start_financial_document_preprocess_job.__tool_name__,
        start_financial_document_preprocess_job,
        start_financial_document_preprocess_job.__tool_schema__,
    )


def _create_get_process_job_status_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 状态查询工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具返回的 job.job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="get_financial_document_preprocess_job_status",
        description=(
            "查询预处理任务状态。启动后反复调用本工具，直到 job.status 进入 succeeded / failed / cancelled。优先按 next_step.action 决定是继续轮询、停止还是重新启动。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="查询预处理状态",
        dup_call=DupCallSpec(
            mode="poll_until_terminal",
            status_path="job.status",
            terminal_values=_JOB_TERMINAL_STATUSES,
        ),
    )
    def get_financial_document_preprocess_job_status(job_id: str) -> dict[str, Any]:
        """查询预处理 job 状态。

        Args:
            job_id: job 标识。

        Returns:
            状态摘要、失败信息与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = require_non_empty_text(
            job_id,
            empty_error=ToolArgumentError(
                "get_financial_document_preprocess_job_status",
                "job_id",
                job_id,
                "不能为空",
            ),
        )
        snapshot = manager.get_job_snapshot(normalized_job_id)
        return _build_status_response(
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_document_preprocess_job_status",
            start_tool_name="start_financial_document_preprocess_job",
        )

    return (
        get_financial_document_preprocess_job_status.__tool_name__,
        get_financial_document_preprocess_job_status,
        get_financial_document_preprocess_job_status.__tool_schema__,
    )


def _create_cancel_process_job_tool(
    registry: ToolRegistry,
    manager: IngestionJobManager,
) -> tuple[str, Any, Any]:
    """创建预处理 job 取消工具。

    Args:
        registry: 工具注册表。
        manager: 长事务 job 管理器。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: schema 构建失败时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "任务 ID。直接使用启动工具或状态工具返回的 job_id。",
            }
        },
        "required": ["job_id"],
    }

    @tool(
        registry,
        name="cancel_financial_document_preprocess_job",
        description=(
            "请求取消预处理任务。取消不是立即完成的；调用后继续用状态工具确认是否进入 cancelled。"
        ),
        parameters=parameters,
        tags=INGESTION_TOOL_TAGS,
        display_name="取消预处理任务",
    )
    def cancel_financial_document_preprocess_job(job_id: str) -> dict[str, Any]:
        """请求取消预处理 job。

        Args:
            job_id: job 标识。

        Returns:
            取消请求结果、当前 job 摘要与下一步建议。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        normalized_job_id = require_non_empty_text(
            job_id,
            empty_error=ToolArgumentError(
                "cancel_financial_document_preprocess_job",
                "job_id",
                job_id,
                "不能为空",
            ),
        )
        cancellation_outcome, snapshot = manager.cancel_job(normalized_job_id)
        return _build_cancel_response(
            cancellation_outcome=cancellation_outcome,
            snapshot=snapshot,
            requested_job_id=normalized_job_id,
            status_tool_name="get_financial_document_preprocess_job_status",
            start_tool_name="start_financial_document_preprocess_job",
        )

    return (
        cancel_financial_document_preprocess_job.__tool_name__,
        cancel_financial_document_preprocess_job,
        cancel_financial_document_preprocess_job.__tool_schema__,
    )

def _normalize_form_types(form_types: Optional[list[str]]) -> Optional[list[str]]:
    """标准化表单数组。

    Args:
        form_types: 原始表单数组。

    Returns:
        去重、排序后的表单数组；为空时返回 `None`。

    Raises:
        ToolArgumentError: 表单值为空白时抛出。
    """

    if form_types is None:
        return None
    normalized_items: list[str] = []
    for item in form_types:
        normalized = str(item or "").strip()
        if not normalized:
            raise ToolArgumentError(
                "start_financial_filing_download_job",
                "form_types",
                form_types,
                "不能包含空白表单类型",
            )
        normalized_items.append(normalized)
    if not normalized_items:
        return None
    return sorted(set(normalized_items))


def _build_status_response(
    *,
    snapshot: Optional[dict[str, Any]],
    requested_job_id: str,
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """构建状态查询返回。

    Args:
        snapshot: job 快照。
        requested_job_id: 请求的 job_id。
        status_tool_name: 对应 status 工具名。
        start_tool_name: 对应 start 工具名（供重试建议）。

    Returns:
        面向 LLM 的极简状态结构。

    Raises:
        无。
    """

    if snapshot is None:
        return {
            "job": None,
            "progress": None,
            "result_summary": None,
            "recent_issues": None,
            "failure": _build_failure(
                code="job_not_found",
                message="找不到这个 job_id，或该任务已过期",
                retryable=False,
            ),
            "next_step": _build_stop_next_step(job_id=requested_job_id),
        }
    return {
        "job": _build_public_job(snapshot),
        "progress": _build_public_progress(snapshot),
        "result_summary": snapshot.get("result_summary"),
        "recent_issues": snapshot.get("recent_issues"),
        "failure": snapshot.get("failure"),
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_start_response(
    *,
    request_outcome: str,
    snapshot: Optional[dict[str, Any]],
    start_tool_name: str,
    status_tool_name: str,
    failure: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构建启动工具返回。

    Args:
        request_outcome: 启动结果枚举。
        snapshot: 可选 job 快照。
        start_tool_name: 启动工具名。
        status_tool_name: 状态查询工具名。
        failure: 可选失败信息。

    Returns:
        面向 LLM 的极简启动结构。

    Raises:
        无。
    """

    if snapshot is None:
        return {
            "request_outcome": request_outcome,
            "job": None,
            "failure": failure,
            "next_step": _build_stop_next_step(job_id=None),
        }
    return {
        "request_outcome": request_outcome,
        "job": _build_public_job(snapshot),
        "failure": failure,
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_cancel_response(
    *,
    cancellation_outcome: str,
    snapshot: Optional[dict[str, Any]],
    requested_job_id: str,
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """构建取消工具返回。

    Args:
        cancellation_outcome: 取消结果枚举。
        snapshot: job 快照。
        requested_job_id: 请求的 job_id。
        status_tool_name: 对应 status 工具名。
        start_tool_name: 对应 start 工具名（供重试建议）。

    Returns:
        面向 LLM 的取消结果结构。

    Raises:
        无。
    """

    # 显式守卫：无论 manager 层返回何种 outcome，snapshot 为 None 时
    # 统一映射为 "job_not_found"，避免内部枚举值泄漏到 LLM。
    if snapshot is None or cancellation_outcome == "not_found":
        return {
            "cancellation_outcome": "job_not_found",
            "job": None,
            "progress": None,
            "result_summary": None,
            "recent_issues": None,
            "failure": _build_failure(
                code="job_not_found",
                message="找不到这个 job_id，或该任务已过期",
                retryable=False,
            ),
            "next_step": _build_stop_next_step(job_id=requested_job_id),
        }
    return {
        "cancellation_outcome": cancellation_outcome,
        "job": _build_public_job(snapshot),
        "progress": _build_public_progress(snapshot),
        "result_summary": snapshot.get("result_summary"),
        "recent_issues": snapshot.get("recent_issues"),
        "failure": snapshot.get("failure"),
        "next_step": _build_next_step(
            snapshot=snapshot,
            status_tool_name=status_tool_name,
            start_tool_name=start_tool_name,
        ),
    }


def _build_public_job(snapshot: dict[str, Any]) -> dict[str, Any]:
    """将内部 job 快照转换为对外 job 摘要。

    Args:
        snapshot: 内部快照。

    Returns:
        对外 job 摘要。

    Raises:
        ValueError: 快照缺失 `job` 时抛出。
    """

    job = snapshot.get("job")
    if not isinstance(job, dict):
        raise ValueError("snapshot.job 缺失")
    return {
        "job_id": job.get("job_id"),
        "job_type": _map_public_job_type(job.get("job_type")),
        "ticker": job.get("ticker"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }


def _build_public_progress(snapshot: dict[str, Any]) -> Optional[dict[str, Any]]:
    """将内部快照转换为对外进度摘要。

    Args:
        snapshot: 内部快照。

    Returns:
        进度结构；缺失时返回 `None`。

    Raises:
        无。
    """

    progress = snapshot.get("progress")
    if not isinstance(progress, dict):
        return None
    return {
        "unit": progress.get("unit"),
        "completed": progress.get("completed"),
        "total": progress.get("total"),
        "percent": progress.get("percent"),
    }


def _build_next_step(
    *,
    snapshot: dict[str, Any],
    status_tool_name: str,
    start_tool_name: str,
) -> dict[str, Any]:
    """根据 job 状态生成下一步建议。

    Args:
        snapshot: job 快照。
        status_tool_name: 对应状态查询工具名。
        start_tool_name: 对应启动工具名（供重试建议）。

    Returns:
        机器可判别的下一步建议。

    Raises:
        ValueError: 快照缺失 `job` 时抛出。
    """

    job = _build_public_job(snapshot)
    result_summary = snapshot.get("result_summary")
    if job["status"] in {"queued", "running", "cancelling"}:
        return {
            "action": "poll_status",
            "tool_name": status_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": 5,
        }
    if job["status"] == "failed":
        return {
            "action": "stop_or_retry",
            "tool_name": start_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": None,
        }
    if _has_failed_units(result_summary):
        return {
            "action": "stop_or_retry",
            "tool_name": start_tool_name,
            "job_id": job["job_id"],
            "suggested_wait_seconds": None,
        }
    return _build_stop_next_step(job_id=str(job["job_id"]))


def _build_stop_next_step(*, job_id: Optional[str]) -> dict[str, Any]:
    """构建停止建议。

    Args:
        job_id: 可选关联 job_id。

    Returns:
        停止建议结构。

    Raises:
        无。
    """

    return {
        "action": "stop",
        "tool_name": None,
        "job_id": job_id,
        "suggested_wait_seconds": None,
    }


def _build_failure(*, code: str, message: str, retryable: bool) -> dict[str, Any]:
    """构建统一失败结构。

    Args:
        code: 错误码。
        message: 错误消息。
        retryable: 是否可重试。

    Returns:
        失败结构。

    Raises:
        无。
    """

    return {
        "code": code,
        "message": message,
        "retryable": retryable,
    }


def _build_not_implemented_start_response(*, ticker: str, market: str) -> dict[str, Any]:
    """构建不支持市场的启动返回。

    Args:
        ticker: 股票代码。
        market: 市场类型。

    Returns:
        `request_outcome=not_implemented` 的极简返回结构。

    Raises:
        无。
    """

    failure = _build_failure(
        code="not_implemented",
        message=f"当前市场暂不支持下载任务：market={market}，ticker={ticker}",
        retryable=False,
    )
    return _build_start_response(
        request_outcome="not_implemented",
        snapshot=None,
        start_tool_name="start_financial_filing_download_job",
        status_tool_name="get_financial_filing_download_job_status",
        failure=failure,
    )


def _normalize_upload_paths(*, tool_name: str, files: list[str]) -> tuple[Path, ...]:
    """校验并标准化上传文件路径列表。

    Args:
        tool_name: 当前工具名。
        files: 原始路径列表。

    Returns:
        标准化后的 ``Path`` 元组。

    Raises:
        ToolArgumentError: 路径列表为空或包含空白项时抛出。
    """

    if len(files) == 0:
        raise ToolArgumentError(tool_name, "files", files, "至少需要一个文件路径")
    normalized_paths: list[Path] = []
    for raw_path in files:
        normalized = require_non_empty_text(
            raw_path,
            empty_error=ToolArgumentError(tool_name, "files", raw_path, "不能包含空白路径"),
        )
        normalized_paths.append(Path(normalized))
    return tuple(normalized_paths)


def _normalize_optional_string_list(values: Optional[list[str]]) -> list[str]:
    """标准化可选字符串列表。

    Args:
        values: 原始字符串列表。

    Returns:
        去空白后的字符串列表；空值返回空列表。

    Raises:
        无。
    """

    if values is None:
        return []
    normalized_values: list[str] = []
    for raw_value in values:
        normalized = normalize_optional_text(raw_value)
        if normalized is None:
            continue
        normalized_values.append(normalized)
    return normalized_values


def _execute_runtime_sync_command(
    *,
    runtime: FinsRuntimeLike,
    command: FinsCommand,
) -> dict[str, Any]:
    """执行同步 runtime 命令并转换为 JSON 结果。

    Args:
        runtime: 财报运行时。
        command: 待执行命令。

    Returns:
        JSON 可序列化结果字典。

    Raises:
        TypeError: runtime 返回的不是同步 ``FinsResult`` 或结果不可序列化时抛出。
    """

    result = runtime.execute(command)
    if not isinstance(result, FinsResult):
        raise TypeError(f"工具要求同步 FinsResult，实际得到 {type(result).__name__}")
    data = result.data
    if not is_dataclass(data):
        raise TypeError(f"FinsResult.data 必须是 dataclass，实际得到 {type(data).__name__}")
    jsonable = asdict(data)
    if not isinstance(jsonable, dict):
        raise TypeError(f"asdict(FinsResult.data) 必须返回 dict，实际得到 {type(jsonable).__name__}")
    return jsonable


def _map_public_job_type(value: Any) -> str:
    """映射对外 job 类型。

    Args:
        value: 内部 job 类型。

    Returns:
        对外稳定枚举值。

    Raises:
        无。
    """

    normalized = str(value or "").strip().lower()
    if normalized == "download":
        return "filing_download"
    if normalized == "process":
        return "document_preprocess"
    return "unknown"


def _has_failed_units(result_summary: Any) -> bool:
    """判断结果摘要中是否包含失败单元。

    Args:
        result_summary: 结果摘要。

    Returns:
        只要任一失败计数大于 0 即返回 `True`。

    Raises:
        无。
    """

    if not isinstance(result_summary, dict):
        return False
    for key, value in result_summary.items():
        if not str(key).endswith("_failed"):
            continue
        try:
            if int(value or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False
