"""
结构化尺码推荐 Tool。

为什么尺码推荐不再主要使用 RAG？
--------------------------------
因为尺码表是典型的结构化规则：身高区间 + 体重区间 -> 尺码。
如果把这类数据全部塞进向量库，再让模型从文本里“猜”尺码，
可控性和可解释性都会变差。

因此这里把原项目的《尺码推荐.txt》转换成 size_chart.json，
Agent 遇到尺码问题时直接调用确定性的 Python Tool。
"""

from functools import lru_cache
import json
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from config import settings


SIZE_ORDER = ["S", "M", "L", "XL", "2XL", "3XL", "4XL", "5XL"]


class SizeRecommendInput(BaseModel):
    height_cm: float = Field(
        ...,
        ge=100,
        le=250,
        description="用户身高，单位厘米，例如 178。",
    )
    weight: float = Field(
        ...,
        gt=0,
        description="用户体重数值，例如 160 或 80。",
    )
    weight_unit: Literal["jin", "kg"] = Field(
        default="jin",
        description="体重单位：jin=斤，kg=公斤。",
    )
    fit_preference: Literal["slim", "regular", "loose"] = Field(
        default="regular",
        description=(
            "穿着偏好：slim=偏修身，regular=正常合身，loose=偏宽松。"
            "如果用户没有说明，使用 regular。"
        ),
    )


@lru_cache(maxsize=1)
def _load_size_chart() -> list[dict]:
    """只在第一次调用时读取 JSON，后续复用内存中的结果。"""
    with settings.size_chart_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _interval_distance(value: float, minimum: float, maximum: float) -> float:
    """计算 value 到闭区间 [minimum, maximum] 的距离。区间内距离为 0。"""
    if value < minimum:
        return minimum - value
    if value > maximum:
        return value - maximum
    return 0.0


def _center_distance(value: float, minimum: float, maximum: float) -> float:
    """value 到区间中心的距离，用于多个尺码区间重叠时进行二次排序。"""
    center = (minimum + maximum) / 2
    return abs(value - center)


def _adjust_size(base_size: str, fit_preference: str) -> str:
    """根据修身/宽松偏好在基础尺码上前后移动一档。"""
    index = SIZE_ORDER.index(base_size)

    if fit_preference == "loose":
        index = min(index + 1, len(SIZE_ORDER) - 1)
    elif fit_preference == "slim":
        index = max(index - 1, 0)

    return SIZE_ORDER[index]


@tool(
    "recommend_size",
    args_schema=SizeRecommendInput,
    description=(
        "根据用户身高、体重以及穿着偏好计算推荐服装尺码。"
        "用户只要是在问‘我应该穿什么码/买什么尺码’，就应该优先调用本工具，"
        "不要让语言模型自行猜测尺码。"
    ),
)
def recommend_size(
    height_cm: float,
    weight: float,
    weight_unit: Literal["jin", "kg"] = "jin",
    fit_preference: Literal["slim", "regular", "loose"] = "regular",
) -> str:
    """执行尺码匹配。"""

    # 统一转成“斤”，因为原始数据就是按斤记录的。
    weight_jin = weight * 2 if weight_unit == "kg" else weight

    chart = _load_size_chart()

    # 每个尺码计算两个分数：
    # 1) range_distance：如果身高/体重超出区间，超出多少；越小越好。
    # 2) center_distance：多个区间同时命中时，离区间中心越近越好。
    candidates = []

    for item in chart:
        height_distance = _interval_distance(
            height_cm,
            item["height_min"],
            item["height_max"],
        )
        weight_distance = _interval_distance(
            weight_jin,
            item["weight_min"],
            item["weight_max"],
        )

        # 身高和体重数值尺度不同，所以做一个简单归一化。
        range_distance = height_distance / 10 + weight_distance / 20

        center_distance = (
            _center_distance(
                height_cm,
                item["height_min"],
                item["height_max"],
            )
            / 10
            + _center_distance(
                weight_jin,
                item["weight_min"],
                item["weight_max"],
            )
            / 20
        )

        candidates.append((range_distance, center_distance, item))

    # 先选“最接近区间”的，再选“最靠近区间中心”的。
    _, _, best_item = min(candidates, key=lambda x: (x[0], x[1]))

    base_size = best_item["size"]
    final_size = _adjust_size(base_size, fit_preference)

    preference_cn = {
        "slim": "偏修身",
        "regular": "正常合身",
        "loose": "偏宽松",
    }[fit_preference]

    return (
        f"根据尺码表计算：用户身高 {height_cm:g} cm，"
        f"体重约 {weight_jin:.1f} 斤；"
        f"基础推荐尺码为 {base_size}。"
        f"用户偏好为“{preference_cn}”，最终建议尺码为 {final_size}。"
        "如果具体商品版型偏大或偏小，仍应以商品详情页的实际尺寸表为准。"
    )
