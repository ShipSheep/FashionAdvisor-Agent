"""Check local configuration before initializing the knowledge base or UI.

This script does not make external API requests.
"""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def status(ok: bool) -> str:
    return "[OK]" if ok else "[MISSING]"


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(("MODEL_API_KEY", bool(settings.model_api_key), "主 Chat Model"))
    checks.append(("DASHSCOPE_API_KEY", bool(os.getenv("DASHSCOPE_API_KEY")), "Embedding"))

    if settings.tavily_enabled:
        checks.append(("TAVILY_API_KEY", bool(settings.tavily_api_key), "Tavily Web Search"))

    for filename in ("洗涤养护.txt", "颜色选择.txt", "size_chart.json"):
        path = settings.data_directory / filename
        checks.append((filename, path.exists(), str(path.relative_to(PROJECT_ROOT))))

    print("FashionAdvisor Agent setup check\n")
    all_ok = True
    for name, ok, note in checks:
        print(f"{status(ok):10s} {name:24s} {note}")
        all_ok = all_ok and ok

    print("\nKnowledge base initialized:", "yes" if settings.ingested_registry_path.exists() else "no")
    print("\nResult:", "ready" if all_ok else "please complete the missing items above")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
