"""Public tool surface for the Big Finance harness.

`default_tools()` is the canonical, ordered tool inventory used in the paper. The
order matters: providers serialize tool definitions into the prompt in registration
order, and changing it can change sampling. Treat the order as part of the
reproducibility contract.
"""

from big_finance_harness.tools.base import Tool, ToolError
from big_finance_harness.tools.edgar_search import EdgarSearchTool
from big_finance_harness.tools.fetch_url import FetchUrlTool
from big_finance_harness.tools.final_answer import FinalAnswerTool
from big_finance_harness.tools.grounding_check import GroundingCheckTool
from big_finance_harness.tools.python_exec import PythonExecTool
from big_finance_harness.tools.web_search import WebSearchTool


def default_tools(*, include_grounding_check: bool = False) -> list[Tool]:
    """Return the canonical tool inventory.

    The default list is the model-facing surface from the paper and its order is
    part of the reproducibility contract — it is unchanged here. ``include_grounding_check``
    opts in the library-callable grounding-check tool (adapted from MiniCheck),
    which compounds the evidence-traceability axis without altering the default
    surface the agent sees.
    """
    tools: list[Tool] = [
        WebSearchTool(),
        EdgarSearchTool(),
        FetchUrlTool(),
        PythonExecTool(),
        FinalAnswerTool(),
    ]
    if include_grounding_check:
        tools.append(GroundingCheckTool())
    return tools


__all__ = [
    "Tool",
    "ToolError",
    "WebSearchTool",
    "EdgarSearchTool",
    "FetchUrlTool",
    "PythonExecTool",
    "FinalAnswerTool",
    "GroundingCheckTool",
    "default_tools",
]
