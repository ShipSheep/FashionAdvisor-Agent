"""不依赖 LLM 的确定性 Tool 测试。"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.size_tool import recommend_size
from tools.weight_tool import convert_weight


def main() -> None:
    print("[1] 体重转换")
    print(
        convert_weight.invoke(
            {
                "value": 160,
                "from_unit": "jin",
                "to_unit": "kg",
            }
        )
    )

    print("\n[2] 尺码推荐")
    print(
        recommend_size.invoke(
            {
                "height_cm": 178,
                "weight": 160,
                "weight_unit": "jin",
                "fit_preference": "regular",
            }
        )
    )


if __name__ == "__main__":
    main()
