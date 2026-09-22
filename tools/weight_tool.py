"""体重单位换算 Tool。"""

from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field


class WeightConvertInput(BaseModel):
    value: float = Field(..., description="需要换算的体重数值，例如 160。")
    from_unit: Literal["jin", "kg"] = Field(
        ..., description="原始单位：jin=斤，kg=公斤。"
    )
    to_unit: Literal["jin", "kg"] = Field(
        ..., description="目标单位：jin=斤，kg=公斤。"
    )


@tool(
    "convert_weight",
    args_schema=WeightConvertInput,
    description=(
        "在中国市斤（jin）和公斤（kg）之间进行体重换算。"
        "当用户明确询问斤/公斤转换，或其他工具需要先完成单位换算时使用。"
    ),
)
def convert_weight(
    value: float,
    from_unit: Literal["jin", "kg"],
    to_unit: Literal["jin", "kg"],
) -> str:
    """1 公斤 = 2 斤。"""

    if value < 0:
        return "体重不能是负数。"

    if from_unit == to_unit:
        result = value
    elif from_unit == "jin" and to_unit == "kg":
        result = value / 2
    elif from_unit == "kg" and to_unit == "jin":
        result = value * 2
    else:
        # Literal 正常情况下已经会限制输入，这个 else 是防御式编程。
        return "不支持该单位转换。"

    return f"{value:g} {from_unit} = {result:.2f} {to_unit}"
