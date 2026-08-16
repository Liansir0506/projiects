"""
api.py — 天气预报与病虫害风险预测 REST API

所有接口均返回 JSON 格式数据，统一响应结构：
{
    "code": 200,          # 状态码：200成功, 400参数错误, 500服务异常
    "message": "success", # 描述信息
    "data": { ... }       # 业务数据
}

接口列表：
  GET  /api/weather/forecast       天气预报 + 病虫害风险预测（核心接口）
  GET  /api/weather/risk           天气风险等级查询
  GET  /api/weather/cities         获取支持的城市列表
  GET  /api/pest/knowledge         病虫害知识库查询
  POST /api/record                 保存诊断记录
  GET  /api/records                获取全部诊断记录
  GET  /api/statistics             获取统计信息
  GET  /api/health                 健康检查
"""

import os
import sys
import json
import re
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import weather_service
import database

app = Flask(__name__)
CORS(app)  # 允许跨域访问

# ========== 预加载数据 ==========
_weather_df = None
_weather_cities = []
_knowledge = None


def load_data():
    """预加载天气数据和病虫害知识库"""
    global _weather_df, _weather_cities, _knowledge

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast_weather.csv")
    if os.path.exists(csv_path):
        _weather_df = pd.read_csv(csv_path, encoding="utf-8")
        _weather_cities = sorted(_weather_df["city_name"].unique().tolist())

    formatted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_formatted.txt")
    if os.path.exists(formatted_path):
        from generate_weather_pest_report import parse_formatted_file, extract_disease_pest_knowledge
        sentences = parse_formatted_file(formatted_path)
        _knowledge = extract_disease_pest_knowledge(sentences)


def _extract_temp(s):
    """从 '高温 23℃' 中提取数值"""
    m = re.search(r"[-+]?\d+\.?\d*", str(s))
    return float(m.group()) if m else -999


# ========== 统一响应格式 ==========

def success(data=None, message="success"):
    """成功响应"""
    return jsonify({"code": 200, "message": message, "data": data})


def error(code=400, message="error", data=None):
    """错误响应"""
    return jsonify({"code": code, "message": message, "data": data}), code


# ========== 核心业务逻辑 ==========

def _predict_pests(city: str) -> dict:
    """
    核心预测逻辑：输入城市 → 返回未来20天天气预报 + 病虫害风险预测
    """
    if _weather_df is None or _knowledge is None:
        return None

    city_df = _weather_df[_weather_df["city_name"] == city].copy()
    if city_df.empty:
        return None

    # 去重，取未来20天
    city_df = city_df.sort_values("ymd").drop_duplicates(subset=["ymd"], keep="last").tail(20)
    city_df["high_val"] = city_df["high"].apply(_extract_temp)
    city_df["low_val"] = city_df["low"].apply(_extract_temp)

    # 每日天气
    forecast = []
    for _, row in city_df.iterrows():
        forecast.append({
            "date": str(row["ymd"]),
            "week": str(row.get("week", "")),
            "high": row["high_val"],
            "low": row["low_val"],
            "type": str(row.get("type", "")),
            "aqi": int(row.get("aqi", 0)) if pd.notna(row.get("aqi")) else 0,
            "wind_direction": str(row.get("fx", "")),
            "wind_level": str(row.get("fl", "")),
            "notice": str(row.get("notice", "")),
        })

    # 天气概况
    recent = city_df.tail(7)
    all_valid = city_df[city_df["high_val"] != -999]
    avg_high = round(all_valid["high_val"].mean(), 1) if len(all_valid) else 0
    avg_low = round(all_valid["low_val"].mean(), 1) if len(all_valid) else 0
    max_high = all_valid["high_val"].max() if len(all_valid) else 0
    min_low = all_valid["low_val"].min() if len(all_valid) else 0
    rain_days = int(len(city_df[city_df["type"].str.contains("雨", na=False)]))
    recent_rain = int(len(recent[recent["type"].str.contains("雨", na=False)]))
    avg_temp = round((avg_high + avg_low) / 2, 1)

    weather_summary = {
        "avg_high": avg_high,
        "avg_low": avg_low,
        "avg_temp": avg_temp,
        "max_high": max_high,
        "min_low": min_low,
        "rain_days": rain_days,
        "recent_rain_days": recent_rain,
        "total_days": len(forecast),
    }

    # 病虫害风险预测
    has_rain = recent_rain > 0
    has_heavy_rain = recent_rain >= 3
    predictions = []

    for name, info in _knowledge.items():
        conditions = info["weather_conditions"]
        if not conditions:
            continue

        all_text = " ".join(conditions)
        match_details = []

        # 温度匹配
        temp_matched = False
        temp_ranges = re.findall(r"(\d+)[～\-—到至](\d+)℃", all_text)
        for lo, hi in temp_ranges:
            lo_f, hi_f = float(lo), float(hi)
            span = hi_f - lo_f
            if lo_f <= avg_temp <= hi_f:
                if span <= 15:
                    temp_matched = True
                    match_details.append(f"均温{avg_temp}℃在适温{lo}～{hi}℃")
                    break
                elif span <= 25:
                    match_details.append(f"均温{avg_temp}℃在参考温度{lo}～{hi}℃范围")
                    break
        if not temp_matched:
            for t in re.findall(r"(\d+)℃", all_text):
                if abs(float(t) - avg_temp) <= 3:
                    temp_matched = True
                    match_details.append(f"均温{avg_temp}℃接近参考温度{t}℃")
                    break

        # 降雨匹配
        rain_matched = False
        if "雨" in all_text and has_rain:
            rain_matched = True
            match_details.append(f"近7天{recent_rain}天有雨")
        if ("阴雨" in all_text or "连阴雨" in all_text) and has_heavy_rain:
            rain_matched = True
            match_details.append(f"近7天{recent_rain}天降雨符合连阴雨条件")

        # 高温高湿
        high_temp_humid = False
        if ("高温" in all_text or "高湿" in all_text) and avg_temp >= 25:
            high_temp_humid = True
            match_details.append(f"均温{avg_temp}℃符合高温高湿条件")

        # 低温
        low_temp_match = False
        if ("低温" in all_text or "寒流" in all_text) and min_low <= 10:
            low_temp_match = True
            match_details.append(f"最低温{min_low}℃符合低温条件")

        # 干旱
        dry_match = False
        if ("干燥" in all_text or "干旱" in all_text) and recent_rain == 0:
            dry_match = True
            match_details.append("近7天无降雨，天气干燥")

        # 反向条件：需高湿但实际干燥
        needs_moisture = any(kw in all_text for kw in ["雨", "高湿", "湿度", "阴雨", "连阴雨", "露", "水流"])
        is_actually_dry = recent_rain == 0 and not has_rain

        match_count = sum([temp_matched, rain_matched, high_temp_humid, low_temp_match, dry_match])
        if needs_moisture and is_actually_dry:
            match_count -= 1

        if match_count >= 2:
            risk_level = "high"
        elif match_count == 1:
            risk_level = "middle"
        else:
            continue  # 低风险不返回

        predictions.append({
            "name": name,
            "type": info["type"],
            "type_cn": "病害" if info["type"] == "Disease" else "虫害",
            "risk_level": risk_level,
            "risk_reason": "；".join(match_details) if match_details else "气象条件部分吻合",
            "weather_conditions": conditions[:3],
            "symptoms": info["symptoms"][:4],
            "approaches": info["approaches"][:3],
            "drugs": info["drugs"][:3],
        })

    # 按风险等级排序
    risk_order = {"high": 0, "middle": 1}
    predictions.sort(key=lambda x: risk_order.get(x["risk_level"], 9))

    return {
        "city": city,
        "weather": weather_summary,
        "forecast": forecast,
        "predictions": {
            "total": len(predictions),
            "high_risk_count": len([p for p in predictions if p["risk_level"] == "high"]),
            "middle_risk_count": len([p for p in predictions if p["risk_level"] == "middle"]),
            "items": predictions,
        },
    }


# ============================================================
# API 路由
# ============================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """健康检查"""
    return success({
        "status": "running",
        "cities_loaded": len(_weather_cities),
        "knowledge_loaded": len(_knowledge) if _knowledge else 0,
    })


@app.route("/api/weather/cities", methods=["GET"])
def api_cities():
    """获取支持的城市列表"""
    return success({
        "total": len(_weather_cities),
        "cities": _weather_cities,
    })


@app.route("/api/weather/forecast", methods=["GET"])
def api_forecast():
    """
    核心接口：天气预报 + 病虫害风险预测

    参数：
      city (必填) - 城市名称，如"西安市"、"北京市"

    返回：
      {
        "code": 200,
        "data": {
          "city": "西安市",
          "weather": { "avg_high", "avg_low", "avg_temp", "max_high", "min_low", "rain_days", "recent_rain_days", "total_days" },
          "forecast": [ { "date", "week", "high", "low", "type", "aqi", "wind_direction", "wind_level", "notice" } ],
          "predictions": {
            "total": 9,
            "high_risk_count": 2,
            "middle_risk_count": 7,
            "items": [ { "name", "type", "type_cn", "risk_level", "risk_reason", "weather_conditions", "symptoms", "approaches", "drugs" } ]
          }
        }
      }
    """
    city = request.args.get("city", "").strip()
    if not city:
        return error(400, "缺少必填参数: city")

    result = _predict_pests(city)
    if result is None:
        return error(404, f"未找到城市【{city}】的天气数据")

    return success(result)


@app.route("/api/weather/risk", methods=["GET"])
def api_weather_risk():
    """
    天气风险等级查询（调用 weather_service 模块）

    参数：
      region (必填) - 地区名称
      growth_stage (必填) - 生育期，如"抽穗期"

    返回：
      { "code": 200, "data": { "risk_level": "high"|"middle"|"low"|"unknown", "risk_desc": "描述" } }
    """
    region = request.args.get("region", "").strip()
    growth_stage = request.args.get("growth_stage", "").strip()

    if not region:
        return error(400, "缺少必填参数: region")
    if not growth_stage:
        return error(400, "缺少必填参数: growth_stage")

    result = weather_service.query_weather_risk(region, growth_stage)
    return success(result)


@app.route("/api/pest/knowledge", methods=["GET"])
def api_pest_knowledge():
    """
    病虫害知识库查询

    参数（均可选，用于筛选）：
      type          - 类型筛选: "Disease" | "Pest"
      name          - 名称搜索（模糊匹配）
      weather_only  - 是否仅返回有气象关联的: "true" | "false"（默认false）

    返回：
      { "code": 200, "data": { "total": 98, "items": [ ... ] } }
    """
    type_filter = request.args.get("type", "").strip()
    name_filter = request.args.get("name", "").strip()
    weather_only = request.args.get("weather_only", "false").lower() == "true"

    if _knowledge is None:
        return error(500, "知识库未加载")

    items = []
    for name, info in _knowledge.items():
        if type_filter and info["type"] != type_filter:
            continue
        if name_filter and name_filter not in name:
            continue
        if weather_only and not info["weather_conditions"]:
            continue

        items.append({
            "name": name,
            "type": info["type"],
            "type_cn": "病害" if info["type"] == "Disease" else "虫害",
            "weather_conditions": info["weather_conditions"],
            "symptoms": info["symptoms"],
            "approaches": info["approaches"],
            "drugs": info["drugs"],
        })

    return success({
        "total": len(items),
        "items": items,
    })


@app.route("/api/record", methods=["POST"])
def api_save_record():
    """
    保存诊断记录

    请求体（JSON）：
      {
        "field_name": "小麦叶片",
        "location": "西安",
        "predicted_class": "锈病",
        "confidence": 0.92,
        "is_reliable": true,
        "weather_risk": "high"
      }

    返回：
      { "code": 200, "data": null }
    """
    if not request.is_json:
        return error(400, "请求体必须为JSON格式")

    record = request.get_json()
    required_fields = ["field_name", "location", "predicted_class"]
    for field in required_fields:
        if field not in record or not record[field]:
            return error(400, f"缺少必填字段: {field}")

    database.save_record(record)
    return success(message="记录保存成功")


@app.route("/api/records", methods=["GET"])
def api_get_records():
    """
    获取全部诊断记录

    返回：
      { "code": 200, "data": { "total": 5, "items": [ { "id", "field_name", "location", ... } ] } }
    """
    records = database.get_all_records()
    return success({
        "total": len(records),
        "items": records,
    })


@app.route("/api/statistics", methods=["GET"])
def api_statistics():
    """
    获取统计信息

    返回：
      { "code": 200, "data": { "total_diagnose": 10 } }
    """
    stats = database.get_statistics()
    return success(stats)


# ========== 错误处理 ==========

@app.errorhandler(404)
def not_found(e):
    return error(404, "接口不存在")


@app.errorhandler(405)
def method_not_allowed(e):
    return error(405, "请求方法不允许")


@app.errorhandler(500)
def internal_error(e):
    return error(500, "服务器内部错误")


# ========== 启动 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("  天气预报与病虫害风险预测 API")
    print("=" * 60)
    print("\n预加载数据...")
    load_data()
    print(f"  城市: {len(_weather_cities)} 个")
    print(f"  病虫害知识: {len(_knowledge) if _knowledge else 0} 种")

    print(f"\nAPI 文档:")
    print(f"  GET  /api/health              健康检查")
    print(f"  GET  /api/weather/cities      城市列表")
    print(f"  GET  /api/weather/forecast     天气+病虫害预测 (参数: city)")
    print(f"  GET  /api/weather/risk         天气风险查询 (参数: region, growth_stage)")
    print(f"  GET  /api/pest/knowledge       病虫害知识库 (参数: type, name, weather_only)")
    print(f"  POST /api/record               保存诊断记录")
    print(f"  GET  /api/records              获取诊断记录")
    print(f"  GET  /api/statistics           统计信息")

    print(f"\n启动服务: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
