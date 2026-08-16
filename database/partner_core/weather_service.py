"""
weather_service.py — 天气风险查询服务模块

提供 query_weather_risk 函数，根据地区和生育期判断农业气象风险等级。
数据来源：本地 forecast_weather.csv 数据集。
"""

import os
import pandas as pd
from typing import Dict

# ========== 数据集路径配置 ==========
# 默认数据集文件名，与当前模块位于同一目录
# 项目根目录（core 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "forecast_weather.csv")

# ========== 风险等级常量 ==========
RISK_LOW = "low"
RISK_MIDDLE = "middle"
RISK_HIGH = "high"
RISK_UNKNOWN = "unknown"

# ========== 异常兜底返回值 ===========
_FALLBACK_RESULT: Dict[str, str] = {
    "risk_level": RISK_UNKNOWN,
    "risk_desc": "天气信息暂不可用"
}

# ========== 生育期 → 天气敏感度映射 ==========
# 不同生育期对恶劣天气的敏感程度不同，影响风险等级判定权重
_GROWTH_STAGE_SENSITIVITY: Dict[str, str] = {
    "播种期": "low",
    "出苗期": "low",
    "分蘖期": "low",
    "拔节期": "middle",
    "抽穗期": "high",
    "扬花期": "high",
    "灌浆期": "middle",
    "成熟期": "low",
    "收获期": "middle",
}

# ========== 恶劣天气类型 ==========
# 这些天气类型会显著增加农业风险
_BAD_WEATHER_TYPES = {"中雨", "大雨", "暴雨", "雷阵雨", "阵雨", "小雪", "中雪", "大雪", "暴雪",
                      "沙尘暴", "雾", "强风", "大风", "冰雹"}

# ========== 高温阈值 ==========
_HIGH_TEMP_THRESHOLD = 35  # 摄氏度，超过此值视为高温风险
_LOW_TEMP_THRESHOLD = 5    # 摄氏度，低于此值视为低温风险
_HIGH_AQI_THRESHOLD = 150  # AQI 超过此值视为空气质量风险


def _load_weather_data() -> pd.DataFrame:
    """
    从本地 CSV 数据集加载天气数据。

    Returns:
        pd.DataFrame: 天气数据表

    Raises:
        FileNotFoundError: 数据集文件不存在
        Exception: 解析失败等异常
    """
    if not os.path.exists(_DATASET_PATH):
        raise FileNotFoundError(f"天气数据集未找到: {_DATASET_PATH}")
    df = pd.read_csv(_DATASET_PATH, encoding="utf-8")
    return df


def _extract_temperature(temp_str: str) -> float:
    """
    从 "高温 27℃" / "低温 18℃" 格式的字符串中提取数值。

    Args:
        temp_str: 温度描述字符串

    Returns:
        float: 提取的温度数值，解析失败返回 -999
    """
    try:
        # 提取字符串中的数字部分（支持 "高温 27℃"、"27" 等格式）
        import re
        match = re.search(r"[-+]?\d+\.?\d*", str(temp_str))
        if match:
            return float(match.group())
    except Exception:
        pass
    return -999  # 无效温度标记


def _calculate_risk(region: str, growth_stage: str, weather_df: pd.DataFrame) -> Dict[str, str]:
    """
    根据地区、生育期和天气数据计算风险等级（核心逻辑）。

    风险等级判定规则：
      1. 先根据生育期确定基础敏感度（low/middle/high）
      2. 再根据天气状况（恶劣天气、极端温度、空气质量）调整风险
      3. 综合得出最终风险等级

    Args:
        region: 地区名称，如 "西安"
        growth_stage: 生育期，如 "抽穗期"
        weather_df: 天气数据 DataFrame

    Returns:
        dict: {"risk_level": ..., "risk_desc": ...}
    """
    # ---- 1. 根据生育期确定基础敏感度 ----
    base_sensitivity = _GROWTH_STAGE_SENSITIVITY.get(growth_stage, "middle")

    # ---- 2. 筛选该地区的天气记录 ----
    city_data = weather_df[weather_df["city_name"] == region]

    if city_data.empty:
        # 未找到该地区数据，按未知地区处理
        return {
            "risk_level": RISK_UNKNOWN,
            "risk_desc": f"未找到地区【{region}】的天气数据"
        }

    # ---- 3. 分析天气状况，计算风险因子 ----
    risk_factors = []  # 收集风险因子，用于生成描述

    # 取最新一条记录作为当前天气参考（按日期降序取第一条）
    latest = city_data.iloc[-1]

    # 3a. 天气类型风险
    weather_type = str(latest.get("type", ""))
    if weather_type in _BAD_WEATHER_TYPES:
        risk_factors.append(f"恶劣天气({weather_type})")
        # 恶劣天气直接提升风险等级
        if base_sensitivity == "low":
            base_sensitivity = "middle"
        elif base_sensitivity in ("middle", "high"):
            base_sensitivity = "high"

    # 3b. 高温风险
    high_temp = _extract_temperature(str(latest.get("high", "")))
    if high_temp != -999 and high_temp >= _HIGH_TEMP_THRESHOLD:
        risk_factors.append(f"高温({high_temp}℃)")
        if base_sensitivity == "low":
            base_sensitivity = "middle"
        elif base_sensitivity == "middle":
            base_sensitivity = "high"

    # 3c. 低温风险
    low_temp = _extract_temperature(str(latest.get("low", "")))
    if low_temp != -999 and low_temp <= _LOW_TEMP_THRESHOLD:
        risk_factors.append(f"低温({low_temp}℃)")
        if base_sensitivity == "low":
            base_sensitivity = "middle"
        elif base_sensitivity == "middle":
            base_sensitivity = "high"

    # 3d. 空气质量风险
    aqi_val = latest.get("aqi", 0)
    try:
        aqi_val = float(aqi_val)
    except (ValueError, TypeError):
        aqi_val = 0
    if aqi_val >= _HIGH_AQI_THRESHOLD:
        risk_factors.append(f"空气污染(AQI={int(aqi_val)})")
        if base_sensitivity == "low":
            base_sensitivity = "middle"

    # ---- 4. 组装返回结果 ----
    if risk_factors:
        risk_desc = f"{region}{growth_stage}存在" + "、".join(risk_factors) + "风险"
    else:
        risk_desc = f"{region}{growth_stage}气象条件良好，风险较低"

    return {
        "risk_level": base_sensitivity,
        "risk_desc": risk_desc
    }


def query_weather_risk(region: str, growth_stage: str) -> dict:
    """
    查询指定地区和生育期的天气风险等级。

    Args:
        region: 地区名称，如 "西安"
        growth_stage: 生育期，如 "抽穗期"

    Returns:
        dict: {
            "risk_level": "low" | "middle" | "high" | "unknown",
            "risk_desc": "具体风险描述"
        }

    异常容错：
        天气接口/数据查询失败时，不抛出异常，
        直接返回 {"risk_level": "unknown", "risk_desc": "天气信息暂不可用"}
    """
    try:
        # ---- 参数校验 ----
        if not region or not isinstance(region, str):
            return {**_FALLBACK_RESULT, "risk_desc": "地区参数无效"}
        if not growth_stage or not isinstance(growth_stage, str):
            return {**_FALLBACK_RESULT, "risk_desc": "生育期参数无效"}

        # ---- 加载数据集 ----
        # 【预留位置】后续可替换为接口调用或其他数据源
        weather_df = _load_weather_data()

        # ---- 计算风险等级 ----
        result = _calculate_risk(region, growth_stage, weather_df)
        return result

    except Exception as e:
        # 所有异常统一兜底，不向外抛出
        return _FALLBACK_RESULT


# ========== 模块自测 ==========
if __name__ == "__main__":
    # 测试正常查询
    print("=== 测试1：查询北京市抽穗期风险 ===")
    print(query_weather_risk("北京市", "抽穗期"))

    print("\n=== 测试2：查询西安市灌浆期风险 ===")
    print(query_weather_risk("西安市", "灌浆期"))

    print("\n=== 测试3：查询不存在的地区 ===")
    print(query_weather_risk("火星市", "播种期"))

    print("\n=== 测试4：参数为空 ===")
    print(query_weather_risk("", "抽穗期"))

    print("\n=== 测试5：数据集异常兜底（模拟） ===")
    # 修改本模块全局变量 _DATASET_PATH，使 _load_weather_data 抛出 FileNotFoundError
    import sys
    _self = sys.modules[__name__]
    _original_path = _self._DATASET_PATH
    _self._DATASET_PATH = "/nonexistent/path.csv"
    print(query_weather_risk("北京市", "抽穗期"))
    _self._DATASET_PATH = _original_path
