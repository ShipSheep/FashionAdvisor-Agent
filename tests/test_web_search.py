"""
Tavily Web Search Tool 独立测试。

运行：
    python tests/test_web_search.py

需要：
    TAVILY_ENABLED=true
    TAVILY_API_KEY=...
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from tools.web_search_tool import create_web_search_tool


def main() -> None:
    if not settings.tavily_enabled:
        print("TAVILY_ENABLED=false，跳过联网测试。")
        return

    tool = create_web_search_tool()

    result = tool.invoke(
        {
            "query": "2026年秋冬服装流行色趋势",
            "time_range": "year",
        }
    )

    print(result)


if __name__ == "__main__":
    main()
