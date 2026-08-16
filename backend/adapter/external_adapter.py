# adapter/external_adapter.py
# 【对接适配层：外部模块变更仅修改此处，请勿修改上层业务】
# 本文件集中封装所有外部依赖（模型、天气、存储）。
# 当前对接对方模块（weather_service, database）。
# 知识库由7号直接对接6号前端，后端不再处理。

import sys
import os
import importlib.util
from typing import List, Dict, Any, Optional
from PIL import Image

# ========== 路径设置 ==========
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATABASE_DIR = os.path.join(_PROJECT_ROOT, "database")

# ========== 动态导入对方模块（避免命名冲突） ==========
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_partner_weather = _load_module(
    "partner_weather",
    os.path.join(_DATABASE_DIR, "partner_core", "weather_service.py")
)
_partner_database = _load_module(
    "partner_database",
    os.path.join(_DATABASE_DIR, "partner_core", "database.py")
)


def init() -> None:
    """
    适配层初始化，服务启动时调用。
    验证天气数据和数据库可用性。
    """
    print("[适配层] 初始化：对接对方模块...")

    csv_path = os.path.join(_DATABASE_DIR, "data", "forecast_weather.csv")
    if os.path.exists(csv_path):
        print(f"[适配层] 天气数据已就绪: {csv_path}")
    else:
        print(f"[适配层] 天气数据文件未找到: {csv_path}")


def predict_image(pil_img: Image.Image) -> List[Dict[str, Any]]:
    """
    调用模型推理，返回Top-3结果。
    对方无模型推理模块，仍使用Mock实现。
    """
    from config import DEVICE
    from mock.mock_modules import mock_inference
    return mock_inference.predict_image(pil_img, DEVICE)


def query_weather_risk(region: str, growth_stage: str) -> Dict[str, str]:
    """
    查询天气风险（使用对方 weather_service 模块）。
    自动处理城市名兼容："西安" → "西安市" 等。
    :param region: 地区名称，如 "西安"
    :param growth_stage: 生育期，如 "抽穗期"
    :return: {"risk_level": "low/middle/high/unknown", "risk_desc": "..."}
    """
    result = _partner_weather.query_weather_risk(region, growth_stage)
    if result.get("risk_level") != "unknown":
        return result

    if not region.endswith("市"):
        result2 = _partner_weather.query_weather_risk(region + "市", growth_stage)
        if result2.get("risk_level") != "unknown":
            return result2

    if region.endswith("市"):
        result3 = _partner_weather.query_weather_risk(region[:-1], growth_stage)
        if result3.get("risk_level") != "unknown":
            return result3

    return result


def save_record(record: dict) -> dict:
    """
    保存诊断记录到对方 SQLite 数据库，并返回包含 id 的完整记录。
    字段映射: field → field_name, region → location
    """
    partner_record = {
        "field_name": record.get("field", record.get("field_name", "")),
        "location": record.get("region", record.get("location", "")),
        "predicted_class": record.get("predicted_class", ""),
        "confidence": record.get("confidence", 0.0),
        "is_reliable": record.get("is_reliable", False),
        "weather_risk": record.get("weather_risk", ""),
    }

    _partner_database.save_record(partner_record)

    all_records = _partner_database.get_all_records()
    if all_records:
        last = max(all_records, key=lambda x: x.get("id", 0))
        record["id"] = last["id"]
        record["created_at"] = last["created_at"]

    return record


def get_all_records() -> List[Dict]:
    """
    获取全部历史诊断记录。
    字段映射: field_name → field, location → region
    """
    records = _partner_database.get_all_records()
    mapped = []
    for r in records:
        mapped.append({
            "id": r.get("id"),
            "field": r.get("field_name", ""),
            "region": r.get("location", ""),
            "predicted_class": r.get("predicted_class", ""),
            "confidence": r.get("confidence", 0.0),
            "is_reliable": r.get("is_reliable", False),
            "weather_risk": r.get("weather_risk", ""),
            "created_at": r.get("created_at", ""),
        })
    return mapped


def get_statistics() -> Dict[str, Any]:
    """
    获取诊断统计数据。
    基于对方数据库的全部记录计算完整统计。
    """
    records = get_all_records()

    by_class = {}
    by_reliable = {"reliable": 0, "unreliable": 0}
    by_risk_level = {}

    for r in records:
        cls = r.get("predicted_class", "unknown")
        by_class[cls] = by_class.get(cls, 0) + 1

        if r.get("is_reliable"):
            by_reliable["reliable"] += 1
        else:
            by_reliable["unreliable"] += 1

        risk = r.get("weather_risk", "unknown")
        by_risk_level[risk] = by_risk_level.get(risk, 0) + 1

    return {
        "total_diagnoses": len(records),
        "by_class": by_class,
        "by_reliable": by_reliable,
        "by_risk_level": by_risk_level,
        "recent_high_risk": [],
        "daily_trend": [],
    }