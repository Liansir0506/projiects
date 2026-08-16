"""
app.py — 天气预报与病虫害风险预测系统

核心流程：输入城市 → 展示未来天气 → 自动预测可能发生的病虫害
"""

import os
import sys
import json
import re
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import weather_service
import database

app = Flask(__name__)

# ========== 预加载数据 ==========
_weather_df = None
_weather_cities = []
_knowledge = None


def load_data():
    global _weather_df, _weather_cities, _knowledge
    csv_path = os.path.join(_PROJECT_ROOT, "data", "forecast_weather.csv")
    if os.path.exists(csv_path):
        _weather_df = pd.read_csv(csv_path, encoding="utf-8")
        _weather_cities = sorted(_weather_df["city_name"].unique().tolist())

    formatted_path = os.path.join(_PROJECT_ROOT, "data", "train_formatted.txt")
    if os.path.exists(formatted_path):
        from generate_weather_pest_report import parse_formatted_file, extract_disease_pest_knowledge
        sentences = parse_formatted_file(formatted_path)
        _knowledge = extract_disease_pest_knowledge(sentences)


def _extract_temp(s):
    """从 '高温 23℃' 中提取数值"""
    m = re.search(r"[-+]?\d+\.?\d*", str(s))
    return float(m.group()) if m else -999


# ========== 核心预测逻辑 ==========

# 气象关键词
WEATHER_KEYWORDS = [
    "温度", "气温", "高温", "低温", "温限", "适温", "最适",
    "湿度", "高湿", "低湿", "相对湿度",
    "雨", "阴雨", "连阴雨", "暴雨", "暴风雨", "大雨", "中雨", "小雨", "阵雨",
    "雨日", "雨湿", "风雨", "露滴", "水流", "淹水", "串灌", "漫灌",
    "干燥", "干旱", "日照", "光照", "通风",
    "台风", "洪涝", "寒流", "冷", "暖",
    "气候", "气象", "天气", "环境", "℃", "度", "pH",
]


def predict_pests_for_city(city: str) -> dict:
    """
    输入城市名，返回：未来天气预报 + 病虫害风险预测
    """
    if _weather_df is None or _knowledge is None:
        return {"error": "数据未加载"}

    # ---- 1. 获取该城市未来天气预报 ----
    city_df = _weather_df[_weather_df["city_name"] == city].copy()
    if city_df.empty:
        return {"error": f"未找到城市【{city}】的天气数据"}

    # 按日期去重，只取未来20天
    city_df = city_df.sort_values("ymd").drop_duplicates(subset=["ymd"], keep="last").tail(20)
    city_df["high_val"] = city_df["high"].apply(_extract_temp)
    city_df["low_val"] = city_df["low"].apply(_extract_temp)

    days = []
    for _, row in city_df.iterrows():
        days.append({
            "date": str(row["ymd"]),
            "week": str(row.get("week", "")),
            "high": row["high_val"],
            "low": row["low_val"],
            "type": str(row.get("type", "")),
            "aqi": int(row.get("aqi", 0)) if pd.notna(row.get("aqi")) else 0,
            "fx": str(row.get("fx", "")),
            "fl": str(row.get("fl", "")),
            "notice": str(row.get("notice", "")),
        })

    # ---- 2. 计算天气概况（按最近7天重点分析） ----
    recent = city_df.tail(7)  # 最近7天
    all_valid = city_df[city_df["high_val"] != -999]

    # 全期概况
    avg_high = round(all_valid["high_val"].mean(), 1) if len(all_valid) else 0
    avg_low = round(all_valid["low_val"].mean(), 1) if len(all_valid) else 0
    max_high = all_valid["high_val"].max() if len(all_valid) else 0
    min_low = all_valid["low_val"].min() if len(all_valid) else 0
    rain_days = int(len(city_df[city_df["type"].str.contains("雨", na=False)]))
    avg_temp = round((avg_high + avg_low) / 2, 1)

    # 最近7天降雨天数（用于风险预测的核心指标）
    recent_rain = int(len(recent[recent["type"].str.contains("雨", na=False)]))

    weather_summary = {
        "avg_high": avg_high, "avg_low": avg_low,
        "max_high": max_high, "min_low": min_low,
        "avg_temp": avg_temp,
        "rain_days": rain_days,
        "recent_rain_days": recent_rain,
        "total_days": len(days),
    }

    # ---- 3. 基于天气预测病虫害风险 ----
    # 使用近7天降雨天数作为短期预警指标
    has_rain = recent_rain > 0
    has_heavy_rain = recent_rain >= 3

    predictions = []

    for name, info in _knowledge.items():
        conditions = info["weather_conditions"]
        if not conditions:
            continue

        all_text = " ".join(conditions)
        risk_level = "低风险"
        risk_reasons = []
        match_details = []

        # 温度匹配（温度区间跨度>15℃视为过宽，不作为主要判定依据）
        temp_matched = False
        temp_ranges = re.findall(r"(\d+)[～\-—到至](\d+)℃", all_text)
        for lo, hi in temp_ranges:
            lo_f, hi_f = float(lo), float(hi)
            span = hi_f - lo_f
            if lo_f <= avg_temp <= hi_f:
                if span <= 15:
                    # 适温区间较窄，权重高
                    temp_matched = True
                    match_details.append(f"均温{avg_temp}℃在适温{lo}～{hi}℃")
                    break
                elif span <= 25:
                    # 适温区间较宽，给半权重（只在已有其他条件时作为加分项）
                    match_details.append(f"均温{avg_temp}℃在参考温度{lo}～{hi}℃范围")
                    break
        # 也检查单点温度（温差3℃以内算强匹配）
        if not temp_matched:
            single_temps = re.findall(r"(\d+)℃", all_text)
            for t in single_temps:
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

        # 低温（仅当最低温≤10℃且病害明确提到低温/寒流才触发）
        low_temp_match = False
        if ("低温" in all_text or "寒流" in all_text) and min_low <= 10:
            low_temp_match = True
            match_details.append(f"最低温{min_low}℃符合低温条件")

        # 干旱
        dry_match = False
        if ("干燥" in all_text or "干旱" in all_text) and recent_rain == 0:
            dry_match = True
            match_details.append("近7天无降雨，天气干燥")

        # 反向条件：病害需要高湿/降雨但天气干燥，降低风险
        needs_moisture = any(kw in all_text for kw in ["雨", "高湿", "湿度", "阴雨", "连阴雨", "露", "水流"])
        is_actually_dry = recent_rain == 0 and not has_rain

        # 综合判定
        match_count = sum([temp_matched, rain_matched, high_temp_humid, low_temp_match, dry_match])

        # 如果病害明确需要高湿/降雨但当前干燥，降低一档风险
        if needs_moisture and is_actually_dry:
            match_count -= 1
        if match_count >= 2:
            risk_level = "高风险"
        elif match_count == 1:
            risk_level = "中等风险"

        if risk_level != "低风险":
            predictions.append({
                "name": name,
                "type": info["type"],
                "type_cn": "病害" if info["type"] == "Disease" else "虫害",
                "risk_level": risk_level,
                "risk_reason": "；".join(match_details) if match_details else "气象条件部分吻合",
                "conditions": conditions[:3],
                "symptoms": info["symptoms"][:4],
                "approaches": info["approaches"][:3],
                "drugs": info["drugs"][:3],
            })

    # 按风险等级排序
    risk_order = {"高风险": 0, "中等风险": 1}
    predictions.sort(key=lambda x: risk_order.get(x["risk_level"], 9))

    return {
        "city": city,
        "weather": weather_summary,
        "forecast": days,
        "predictions": predictions,
        "total_diseases": len(predictions),
        "high_risk_count": len([p for p in predictions if p["risk_level"] == "高风险"]),
        "mid_risk_count": len([p for p in predictions if p["risk_level"] == "中等风险"]),
    }


# ========== HTML ==========
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>天气预报与病虫害风险预测</title>
<style>
  :root {
    --bg: #0a1628; --card: #12223b; --card2: #1a2d4a; --border: #1e3a5f;
    --accent: #4fc3f7; --green: #66bb6a; --warn: #ffa726; --danger: #ef5350;
    --text: #e8eaf6; --text2: #78909c;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:"Microsoft YaHei","PingFang SC",sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

  .hero { background:linear-gradient(135deg,#0d2137 0%,#0a1628 50%,#1a0e2e 100%); padding:40px 20px 30px; text-align:center; }
  .hero h1 { font-size:32px; background:linear-gradient(90deg,#4fc3f7,#81d4fa,#66bb6a); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .hero p { color:var(--text2); margin-top:8px; font-size:15px; }

  .search-box { max-width:600px; margin:24px auto 0; display:flex; gap:0; }
  .search-box input { flex:1; padding:14px 20px; border:2px solid var(--border); border-right:none; border-radius:12px 0 0 12px; background:var(--card); color:var(--text); font-size:16px; outline:none; }
  .search-box input::placeholder { color:var(--text2); }
  .search-box input:focus { border-color:var(--accent); }
  .search-box button { padding:14px 32px; background:linear-gradient(135deg,#4fc3f7,#29b6f6); color:#000; border:none; border-radius:0 12px 12px 0; font-size:16px; font-weight:bold; cursor:pointer; transition:all .2s; }
  .search-box button:hover { opacity:.9; }
  .suggestions { max-width:600px; margin:0 auto; position:relative; }
  .suggestions ul { position:absolute; top:0; left:0; right:0; background:var(--card); border:1px solid var(--border); border-radius:8px; list-style:none; max-height:240px; overflow-y:auto; z-index:100; display:none; }
  .suggestions ul.show { display:block; }
  .suggestions li { padding:10px 20px; cursor:pointer; color:var(--text2); font-size:14px; }
  .suggestions li:hover { background:var(--card2); color:var(--text); }

  .container { max-width:1200px; margin:0 auto; padding:24px 20px; }

  .section { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:24px; margin-bottom:20px; }
  .section h2 { font-size:20px; color:var(--accent); margin-bottom:16px; display:flex; align-items:center; gap:8px; }

  /* 天气概况卡片 */
  .overview-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; }
  .overview-item { background:var(--card2); border-radius:12px; padding:16px; text-align:center; }
  .overview-item .val { font-size:28px; font-weight:bold; }
  .overview-item .val.hot { color:var(--danger); }
  .overview-item .val.cold { color:var(--accent); }
  .overview-item .val.rain { color:#42a5f5; }
  .overview-item .lbl { font-size:12px; color:var(--text2); margin-top:4px; }

  /* 每日天气 */
  .forecast-scroll { display:flex; gap:10px; overflow-x:auto; padding:8px 0; }
  .forecast-card { min-width:120px; background:var(--card2); border-radius:12px; padding:12px; text-align:center; flex-shrink:0; }
  .forecast-card .f-date { color:var(--text2); font-size:12px; }
  .forecast-card .f-week { color:var(--text2); font-size:11px; }
  .forecast-card .f-icon { font-size:28px; margin:6px 0; }
  .forecast-card .f-type { color:var(--warn); font-size:13px; font-weight:bold; }
  .forecast-card .f-temp { margin-top:6px; }
  .forecast-card .f-high { color:var(--danger); font-size:16px; font-weight:bold; }
  .forecast-card .f-low { color:var(--accent); font-size:14px; }
  .forecast-card .f-aqi { color:var(--text2); font-size:11px; margin-top:4px; }
  .forecast-card .f-wind { color:var(--text2); font-size:11px; }
  .forecast-card.rain { border:1px solid rgba(66,165,245,0.3); }

  /* 风险总览 */
  .risk-summary { display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }
  .risk-card { flex:1; min-width:200px; background:var(--card2); border-radius:12px; padding:20px; text-align:center; border-top:3px solid var(--border); }
  .risk-card.high { border-top-color:var(--danger); }
  .risk-card.mid { border-top-color:var(--warn); }
  .risk-card .count { font-size:36px; font-weight:bold; }
  .risk-card.high .count { color:var(--danger); }
  .risk-card.mid .count { color:var(--warn); }
  .risk-card .desc { color:var(--text2); font-size:13px; margin-top:4px; }

  /* 病虫害预测卡片 */
  .pest-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px; }
  .pest-card { background:var(--card2); border-radius:12px; padding:16px; border-left:4px solid var(--border); transition:transform .2s; }
  .pest-card:hover { transform:translateY(-2px); }
  .pest-card.high-risk { border-left-color:var(--danger); }
  .pest-card.mid-risk { border-left-color:var(--warn); }
  .pest-card .header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .pest-card .name { font-size:17px; font-weight:bold; }
  .pest-card .badges { display:flex; gap:6px; }
  .badge { display:inline-block; padding:3px 10px; border-radius:10px; font-size:12px; font-weight:bold; }
  .badge-disease { background:rgba(239,83,80,.15); color:var(--danger); }
  .badge-pest { background:rgba(255,167,38,.15); color:var(--warn); }
  .badge-high { background:rgba(239,83,80,.2); color:var(--danger); }
  .badge-mid { background:rgba(255,167,38,.2); color:var(--warn); }
  .pest-card .reason { color:var(--accent); font-size:13px; margin-bottom:10px; padding:8px 12px; background:rgba(79,195,247,.06); border-radius:8px; }
  .pest-card .detail { font-size:12px; color:var(--text2); line-height:1.8; }
  .pest-card .detail span { color:var(--text); }

  .empty { text-align:center; padding:40px; color:var(--text2); }
  .loading { text-align:center; padding:60px; color:var(--text2); }
  .spinner { display:inline-block; width:32px; height:32px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .weather-icon-晴::before { content:"☀️"; }
  .weather-icon-多云::before { content:"⛅"; }
  .weather-icon-阴::before { content:"☁️"; }
  .weather-icon-小雨::before { content:"🌦️"; }
  .weather-icon-中雨::before { content:"🌧️"; }
  .weather-icon-大雨::before { content:"🌧️"; }
  .weather-icon-暴雨::before { content:"⛈️"; }
  .weather-icon-阵雨::before { content:"🌦️"; }
  .weather-icon-雷阵雨::before { content:"⛈️"; }
  .weather-icon-小雪::before { content:"🌨️"; }
  .weather-icon-中雪::before { content:"❄️"; }
  .weather-icon-大雪::before { content:"❄️"; }
  .weather-icon-雾::before { content:"🌫️"; }
  .weather-icon-霾::before { content:"🌫️"; }
  .weather-icon-扬沙::before { content:"🌪️"; }
</style>
</head>
<body>

<div class="hero">
  <h1>🌾 天气预报与病虫害风险预测</h1>
  <p>输入城市，查看未来天气，自动预测可能发生的病虫害</p>
  <div class="search-box">
    <input id="city-input" type="text" placeholder="输入城市名称，如：西安、北京市、成都市..." autocomplete="off">
    <button onclick="searchCity()">查询预测</button>
  </div>
  <div class="suggestions"><ul id="city-suggestions"></ul></div>
</div>

<div class="container" id="result-area">
  <div class="empty">👆 请在上方输入城市名称开始查询</div>
</div>

<script>
const allCities = {{ cities | safe }};

// ========== 自动补全 ==========
const input = document.getElementById('city-input');
const sugList = document.getElementById('city-suggestions');

input.addEventListener('input', function() {
  const v = this.value.trim();
  if (!v) { sugList.classList.remove('show'); return; }
  const matches = allCities.filter(c => c.includes(v)).slice(0, 10);
  if (!matches.length) { sugList.classList.remove('show'); return; }
  sugList.innerHTML = matches.map(c => `<li onclick="pickCity('${c}')">${c}</li>`).join('');
  sugList.classList.add('show');
});

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') searchCity();
});

document.addEventListener('click', function(e) {
  if (!e.target.closest('.suggestions')) sugList.classList.remove('show');
});

function pickCity(name) {
  input.value = name;
  sugList.classList.remove('show');
  searchCity();
}

// ========== 天气图标映射 ==========
const weatherIcons = {
  '晴':'☀️','多云':'⛅','阴':'☁️',
  '小雨':'🌦️','中雨':'🌧️','大雨':'🌧️','暴雨':'⛈️','阵雨':'🌦️','雷阵雨':'⛈️',
  '小雪':'🌨️','中雪':'❄️','大雪':'❄️',
  '雾':'🌫️','霾':'🌫️','扬沙':'🌪️','沙尘暴':'🌪️',
};

// ========== 主查询 ==========
async function searchCity() {
  const city = input.value.trim();
  if (!city) return;
  sugList.classList.remove('show');

  const area = document.getElementById('result-area');
  area.innerHTML = '<div class="loading"><div class="spinner"></div><p style="margin-top:16px">正在分析天气与病虫害风险...</p></div>';

  try {
    const resp = await fetch('/api/predict?city=' + encodeURIComponent(city));
    const data = await resp.json();

    if (data.error) {
      area.innerHTML = `<div class="empty">❌ ${data.error}</div>`;
      return;
    }

    renderResult(data);
  } catch(e) {
    area.innerHTML = '<div class="empty" style="color:var(--danger);">查询失败，请重试</div>';
  }
}

// ========== 渲染结果 ==========
function renderResult(data) {
  const area = document.getElementById('result-area');
  let html = '';

  // ---- 天气概况 ----
  html += `<div class="section">
    <h2>📍 ${data.city} · 未来${data.weather.total_days}天天气概况</h2>
    <div class="overview-grid">
      <div class="overview-item"><div class="val hot">${data.weather.avg_high}℃</div><div class="lbl">平均高温</div></div>
      <div class="overview-item"><div class="val cold">${data.weather.avg_low}℃</div><div class="lbl">平均低温</div></div>
      <div class="overview-item"><div class="val">${data.weather.avg_temp}℃</div><div class="lbl">平均气温</div></div>
      <div class="overview-item"><div class="val rain">${data.weather.rain_days}天</div><div class="lbl">降雨天数</div></div>
      <div class="overview-item"><div class="val hot">${data.weather.max_high}℃</div><div class="lbl">最高温</div></div>
      <div class="overview-item"><div class="val cold">${data.weather.min_low}℃</div><div class="lbl">最低温</div></div>
    </div>
  </div>`;

  // ---- 每日天气 ----
  html += `<div class="section">
    <h2>📅 每日天气预报</h2>
    <div class="forecast-scroll">`;
  data.forecast.forEach(d => {
    const isRain = d.type.includes('雨');
    const icon = weatherIcons[d.type] || '🌡️';
    html += `<div class="forecast-card${isRain ? ' rain' : ''}">
      <div class="f-date">${d.date.slice(5)}</div>
      <div class="f-week">${d.week}</div>
      <div class="f-icon">${icon}</div>
      <div class="f-type">${d.type}</div>
      <div class="f-temp"><span class="f-high">${d.high}°</span> / <span class="f-low">${d.low}°</span></div>
      <div class="f-wind">${d.fx} ${d.fl}</div>
      <div class="f-aqi">AQI ${d.aqi}</div>
    </div>`;
  });
  html += `</div></div>`;

  // ---- 病虫害风险预测 ----
  html += `<div class="section">
    <h2>🔬 病虫害风险预测</h2>`;

  if (data.predictions.length === 0) {
    html += `<div class="empty">✅ 根据当前天气预报，暂无明显病虫害风险</div>`;
  } else {
    html += `<div class="risk-summary">
      <div class="risk-card high"><div class="count">${data.high_risk_count}</div><div class="desc">高风险病虫害</div></div>
      <div class="risk-card mid"><div class="count">${data.mid_risk_count}</div><div class="desc">中等风险病虫害</div></div>
      <div class="risk-card"><div class="count">${data.total_diseases}</div><div class="desc">总计风险项</div></div>
    </div>`;

    html += `<div class="pest-grid">`;
    data.predictions.forEach(p => {
      const riskClass = p.risk_level === '高风险' ? 'high-risk' : 'mid-risk';
      const typeBadge = p.type === 'Disease' ? 'badge-disease' : 'badge-pest';
      const riskBadge = p.risk_level === '高风险' ? 'badge-high' : 'badge-mid';
      const icon = p.type === 'Disease' ? '🦠' : '🐛';

      html += `<div class="pest-card ${riskClass}">
        <div class="header">
          <span class="name">${icon} ${p.name}</span>
          <div class="badges">
            <span class="badge ${typeBadge}">${p.type_cn}</span>
            <span class="badge ${riskBadge}">${p.risk_level}</span>
          </div>
        </div>
        <div class="reason">⚡ ${p.risk_reason}</div>
        <div class="detail">`;

      if (p.conditions.length) {
        html += `<div>🌡 <span>气象诱因：</span>${p.conditions.map(c => c.length > 50 ? c.substring(0,50)+'...' : c).join('；')}</div>`;
      }
      if (p.symptoms.length) {
        html += `<div>🔍 <span>典型症状：</span>${p.symptoms.join('、')}</div>`;
      }
      if (p.approaches.length) {
        html += `<div>🛡 <span>防治方法：</span>${p.approaches.join('、')}</div>`;
      }
      if (p.drugs.length) {
        html += `<div>💊 <span>推荐用药：</span>${p.drugs.join('、')}</div>`;
      }

      html += `</div></div>`;
    });
    html += `</div>`;
  }
  html += `</div>`;

  area.innerHTML = html;
}
</script>
</body>
</html>
"""


# ========== API ==========

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        cities=json.dumps(_weather_cities, ensure_ascii=False),
    )


@app.route("/api/predict")
def api_predict():
    """核心API：输入城市 → 返回天气+病虫害预测"""
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "请输入城市名称"})
    result = predict_pests_for_city(city)
    return jsonify(result)


# ========== 启动 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("  天气预报与病虫害风险预测系统")
    print("=" * 60)
    print("\n预加载数据...")
    load_data()
    print(f"  城市: {len(_weather_cities)} 个")
    print(f"  病虫害知识: {len(_knowledge) if _knowledge else 0} 种")

    print(f"\n启动服务: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
