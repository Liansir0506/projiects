"""
generate_weather_pest_report.py

将 NER 标注知识（train_formatted.txt）中的病虫害-气象条件知识，
与天气预报数据（forecast_weather.csv）关联，
生成"天气条件 → 病虫害风险"关联报告。
"""

import os
import re
import pandas as pd

# ========== 路径配置 ==========
# 项目根目录（scripts 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATTED_FILE = os.path.join(_PROJECT_ROOT, "data", "train_formatted.txt")
WEATHER_FILE = os.path.join(_PROJECT_ROOT, "data", "forecast_weather.csv")
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "output", "weather_pest_report.txt")

# ========== 气象关键词 ==========
WEATHER_KEYWORDS = [
    "温度", "气温", "高温", "低温", "温限", "适温", "最适",
    "湿度", "高湿", "低湿", "相对湿度",
    "雨", "阴雨", "连阴雨", "暴雨", "暴风雨", "大雨", "中雨", "小雨", "阵雨",
    "雨日", "雨湿", "风雨", "露滴", "水流", "淹水", "串灌", "漫灌",
    "干燥", "干旱", "日照", "光照", "通风",
    "台风", "洪涝", "寒流", "冷", "暖",
    "气候", "气象", "天气", "环境",
    "℃", "度",
    "pH",
]

# ========== 实体类型中文名 ==========
TYPE_CN = {
    "Disease":  "病害",
    "Approach": "防治方法",
    "Symptom":  "症状",
    "Area":     "地区",
    "Drug":     "药物",
    "Rice":     "水稻品种",
    "Pest":     "虫害",
}


# ============================================================
# 第一步：从 train_formatted.txt 解析句子和实体
# ============================================================

def parse_formatted_file(filepath: str) -> list[dict]:
    """
    从格式化后的 train_formatted.txt 中解析句子和实体。

    文件格式：
    【句子 N】完整原文
      ┌────────┬─────────────┐
      │ 实体文本 │  实体类型     │
      ├────────┼─────────────┤
      │ 稻曲病   │ Disease（病害）│
      └────────┴─────────────┘
    或：
      （无实体标注）

    返回：[{"text": "完整原文", "entities": [{"text": "稻曲病", "type": "Disease"}, ...]}]
    """
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 按句子标记切分
    # 匹配 【句子 N】后面紧跟的文本
    sentence_blocks = re.split(r"(?=【句子 \d+】)", content)

    for block in sentence_blocks:
        block = block.strip()
        if not block:
            continue

        # 提取句子文本
        text_match = re.match(r"【句子 \d+】(.+)", block, re.DOTALL)
        if not text_match:
            continue

        text = text_match.group(1).strip().split("\n")[0].strip()

        # 提取实体
        entities = []
        # 匹配表格行：│ 实体文本 │ Disease（病害） │
        entity_rows = re.findall(r"│\s*(\S+)\s*│\s*(\w+)（[^）]*）\s*│", block)
        for entity_text, entity_type in entity_rows:
            # 跳过表头行
            if entity_text == "实体文本" or entity_type == "实体类型":
                continue
            entities.append({"text": entity_text, "type": entity_type})

        if text:
            sentences.append({"text": text, "entities": entities})

    return sentences


# ============================================================
# 第二步：提取病虫害-气象条件知识
# ============================================================

def extract_disease_pest_knowledge(sentences: list[dict]) -> dict:
    """
    从句子中提取每种病虫害对应的气象条件知识。

    返回结构：
    {
        "稻曲病": {
            "type": "Disease",
            "weather_conditions": ["日平均气温在25～28℃...", ...],
            "symptoms": ["黄白色小球", ...],
            "approaches": ["选用抗病品种", ...],
            "drugs": ["苯菌灵", ...],
        },
    }
    """
    knowledge = {}
    # 追踪当前段落正在描述的病虫害
    current_diseases = []

    for sent in sentences:
        entities = sent["entities"]
        text = sent["text"]

        # 提取各类实体
        disease_ents = [e for e in entities if e["type"] in ("Disease", "Pest")]
        symptom_ents = [e for e in entities if e["type"] == "Symptom"]
        approach_ents = [e for e in entities if e["type"] == "Approach"]
        drug_ents = [e for e in entities if e["type"] == "Drug"]

        # 如果本句包含病害/虫害实体，更新追踪
        if disease_ents:
            for e in disease_ents:
                name = e["text"]
                if name not in knowledge:
                    knowledge[name] = {
                        "type": e["type"],
                        "weather_conditions": [],
                        "symptoms": [],
                        "approaches": [],
                        "drugs": [],
                    }
            current_diseases = [e["text"] for e in disease_ents]

        # 将当前句子的信息归入追踪中的病害
        if current_diseases:
            # 判断是否涉及气象条件
            is_weather_related = any(kw in text for kw in WEATHER_KEYWORDS)

            if is_weather_related:
                for name in current_diseases:
                    if name in knowledge:
                        knowledge[name]["weather_conditions"].append(text)

            # 归入症状
            for e in symptom_ents:
                for name in current_diseases:
                    if name in knowledge and e["text"] not in knowledge[name]["symptoms"]:
                        knowledge[name]["symptoms"].append(e["text"])

            # 归入防治方法
            for e in approach_ents:
                for name in current_diseases:
                    if name in knowledge and e["text"] not in knowledge[name]["approaches"]:
                        knowledge[name]["approaches"].append(e["text"])

            # 归入药物
            for e in drug_ents:
                for name in current_diseases:
                    if name in knowledge and e["text"] not in knowledge[name]["drugs"]:
                        knowledge[name]["drugs"].append(e["text"])

    return knowledge


# ============================================================
# 第三步：天气数据处理
# ============================================================

def extract_temp(temp_str: str) -> float:
    """从 '高温 27℃' 格式提取数值"""
    match = re.search(r"[-+]?\d+\.?\d*", str(temp_str))
    return float(match.group()) if match else -999


def load_weather_data(filepath: str) -> pd.DataFrame:
    """加载天气 CSV 数据"""
    df = pd.read_csv(filepath, encoding="utf-8")
    df["high_temp"] = df["high"].apply(extract_temp)
    df["low_temp"] = df["low"].apply(extract_temp)
    return df


def get_city_weather_summary(df: pd.DataFrame, city: str) -> dict:
    """获取某城市近7天天气概况"""
    city_df = df[df["city_name"] == city]
    if city_df.empty:
        return None

    # 取最后7条记录（去重后）
    recent = city_df.drop_duplicates(subset=["ymd"]).tail(7)

    return {
        "city": city,
        "avg_high": round(recent["high_temp"].mean(), 1),
        "avg_low": round(recent["low_temp"].mean(), 1),
        "max_high": recent["high_temp"].max(),
        "min_low": recent["low_temp"].min(),
        "rain_days": len(recent[recent["type"].str.contains("雨", na=False)]),
        "weather_types": recent["type"].value_counts().to_dict(),
        "avg_aqi": round(recent["aqi"].mean(), 0),
    }


# ============================================================
# 第四步：根据天气条件评估病虫害风险
# ============================================================

def assess_pest_risk(weather_summary: dict, knowledge: dict) -> list[dict]:
    """
    根据当前天气条件，评估各病虫害的发生风险。

    风险判定规则：
    - 高风险：温度在适温区间 + 有降雨（2个以上条件匹配）
    - 中等风险：1个条件匹配
    - 低风险：无明显匹配
    """
    risks = []
    if not weather_summary:
        return risks

    avg_temp = (weather_summary["avg_high"] + weather_summary["avg_low"]) / 2
    rain_days = weather_summary["rain_days"]
    has_rain = rain_days > 0
    has_heavy_rain = rain_days >= 3

    for disease_name, info in knowledge.items():
        conditions = info["weather_conditions"]
        if not conditions:
            continue

        risk_level = "低风险"
        risk_reason = []

        all_text = " ".join(conditions)

        # ---- 温度匹配 ----
        temp_matched = False
        temp_ranges = re.findall(r"(\d+)[～\-—到至](\d+)℃", all_text)
        for lo, hi in temp_ranges:
            lo_f, hi_f = float(lo), float(hi)
            if lo_f <= avg_temp <= hi_f:
                temp_matched = True
                risk_reason.append(f"当前均温{avg_temp:.0f}℃在适温{lo}～{hi}℃区间")
                break

        # ---- 降雨匹配 ----
        rain_matched = False
        if "雨" in all_text and has_rain:
            rain_matched = True
            risk_reason.append(f"近期{rain_days}天降雨，符合雨湿条件")
        if ("阴雨" in all_text or "连阴雨" in all_text) and has_heavy_rain:
            rain_matched = True
            risk_reason.append(f"连续{rain_days}天降雨，符合连阴雨诱病条件")

        # ---- 高温高湿匹配 ----
        high_temp_humid = False
        if ("高温" in all_text or "高湿" in all_text) and avg_temp >= 25:
            high_temp_humid = True
            risk_reason.append(f"均温{avg_temp:.0f}℃，符合高温高湿发病条件")

        # ---- 低温匹配 ----
        low_temp_match = False
        if ("低温" in all_text or "寒流" in all_text) and weather_summary["min_low"] <= 15:
            low_temp_match = True
            risk_reason.append(f"最低温{weather_summary['min_low']}℃，符合低温诱病条件")

        # ---- 综合判定 ----
        match_count = sum([temp_matched, rain_matched, high_temp_humid, low_temp_match])
        if match_count >= 2:
            risk_level = "高风险"
        elif match_count == 1:
            risk_level = "中等风险"

        if risk_level != "低风险":
            risks.append({
                "name": disease_name,
                "type": "病害" if info["type"] == "Disease" else "虫害",
                "risk_level": risk_level,
                "risk_reason": "；".join(risk_reason),
                "weather_conditions": conditions,
                "symptoms": info["symptoms"],
                "approaches": info["approaches"],
                "drugs": info["drugs"],
            })

    risk_order = {"高风险": 0, "中等风险": 1, "低风险": 2}
    risks.sort(key=lambda x: risk_order.get(x["risk_level"], 9))
    return risks


# ============================================================
# 第五步：生成格式化报告
# ============================================================

def format_report(knowledge: dict, weather_df: pd.DataFrame) -> str:
    """生成完整关联报告"""
    lines = []

    lines.append("=" * 90)
    lines.append("  天气条件与病虫害风险关联报告")
    lines.append("  数据来源：NER病虫害知识库 + forecast_weather.csv 天气预报数据")
    lines.append("=" * 90)

    # ============ Part 1: 病虫害-气象条件知识库 ============
    lines.append("")
    lines.append("━" * 90)
    lines.append("  第一部分：病虫害-气象条件知识库")
    lines.append("━" * 90)

    weather_related = {k: v for k, v in knowledge.items() if v["weather_conditions"]}
    lines.append(f"\n  共 {len(knowledge)} 种病虫害，其中 {len(weather_related)} 种与气象条件明确关联\n")

    for idx, (name, info) in enumerate(weather_related.items(), 1):
        type_label = TYPE_CN.get(info["type"], info["type"])
        lines.append(f"  【{idx}】{name}（{type_label}）")
        lines.append(f"  {'─' * 60}")

        if info["weather_conditions"]:
            lines.append(f"  气象诱因：")
            for cond in info["weather_conditions"]:
                lines.append(f"    · {cond}")

        if info["symptoms"]:
            lines.append(f"  典型症状：{'、'.join(info['symptoms'][:5])}")

        if info["approaches"]:
            lines.append(f"  防治方法：{'、'.join(info['approaches'][:5])}")

        if info["drugs"]:
            lines.append(f"  推荐用药：{'、'.join(info['drugs'][:5])}")

        lines.append("")

    # ============ Part 2: 各城市风险预警 ============
    lines.append("")
    lines.append("━" * 90)
    lines.append("  第二部分：各城市当前天气 → 病虫害风险预警")
    lines.append("━" * 90)

    cities = weather_df["city_name"].unique().tolist()

    for city in cities:
        summary = get_city_weather_summary(weather_df, city)
        if not summary:
            continue

        risks = assess_pest_risk(summary, knowledge)
        if not risks:
            continue

        high_risks = [r for r in risks if r["risk_level"] == "高风险"]
        mid_risks = [r for r in risks if r["risk_level"] == "中等风险"]

        lines.append(f"\n  ┌{'─' * 70}┐")
        lines.append(f"  │ {city}")
        lines.append(f"  ├{'─' * 70}┤")

        weather_desc = (
            f"近7日 高温均{summary['avg_high']}℃ / 低温均{summary['avg_low']}℃ | "
            f"降雨{summary['rain_days']}天 | 均AQI {int(summary['avg_aqi'])}"
        )
        lines.append(f"  │ 天气概况: {weather_desc}")

        if high_risks:
            lines.append(f"  ├{'─' * 70}┤")
            lines.append(f"  │ ★ 高风险预警（{len(high_risks)}项）")
            for r in high_risks:
                lines.append(f"  │   ● {r['type']}：{r['name']}")
                lines.append(f"  │     原因：{r['risk_reason']}")
                if r["drugs"]:
                    lines.append(f"  │     建议用药：{'、'.join(r['drugs'][:3])}")
                lines.append(f"  │")

        if mid_risks:
            lines.append(f"  ├{'─' * 70}┤")
            lines.append(f"  │ ▲ 中等风险（{len(mid_risks)}项）")
            for r in mid_risks:
                lines.append(f"  │   ● {r['type']}：{r['name']}")
                lines.append(f"  │     原因：{r['risk_reason']}")

        lines.append(f"  └{'─' * 70}┘")

    # ============ Part 3: 温度-病虫害速查表 ============
    lines.append("")
    lines.append("━" * 90)
    lines.append("  第三部分：温度-病虫害速查表")
    lines.append("━" * 90)
    lines.append("")
    lines.append("  根据当前均温，快速查找可能发生的病虫害：\n")

    # 预处理：为每个病害提取温度区间
    disease_temp_zones = {}
    for name, info in weather_related.items():
        all_text = " ".join(info["weather_conditions"])
        temp_ranges = re.findall(r"(\d+)[～\-—到至](\d+)℃", all_text)
        if temp_ranges:
            # 取第一个温度范围作为主要分类依据
            lo, hi = float(temp_ranges[0][0]), float(temp_ranges[0][1])
            mid = (lo + hi) / 2
            disease_temp_zones[name] = {
                "mid": mid,
                "lo": lo,
                "hi": hi,
                "type": info["type"],
                "info": info,
            }

    # 按温度区间分类
    zones = [
        ("≤15℃（低温区）", lambda m: m <= 15),
        ("15～22℃（中低温区）", lambda m: 15 < m <= 22),
        ("22～28℃（中温区）", lambda m: 22 < m <= 28),
        ("28～35℃（高温区）", lambda m: 28 < m <= 35),
        (">35℃（极端高温区）", lambda m: m > 35),
    ]

    for zone_name, zone_check in zones:
        lines.append(f"  【{zone_name}】")
        found = False
        for name, data in sorted(disease_temp_zones.items(), key=lambda x: x[1]["mid"]):
            if zone_check(data["mid"]):
                type_label = TYPE_CN.get(data["type"], data["type"])
                lines.append(f"    · {name}（{type_label}，适温{data['lo']:.0f}～{data['hi']:.0f}℃）")
                found = True
        if not found:
            lines.append("    （暂无记录）")
        lines.append("")

    # ============ Part 4: 降雨-病虫害速查表 ============
    lines.append("━" * 90)
    lines.append("  第四部分：降雨条件-病虫害速查表")
    lines.append("━" * 90)
    lines.append("")

    rain_categories = [
        ("连阴雨/多雨天气", ["连阴雨", "阴雨", "多雨", "雨日多", "连续阴雨"]),
        ("暴风雨/台风", ["暴风雨", "暴雨", "台风", "洪涝"]),
        ("一般降雨（小雨/中雨/阵雨）", ["雨", "降雨"]),
        ("干燥少雨天气", ["干燥", "干旱", "少雨", "日照多"]),
    ]

    for cat_name, keywords in rain_categories:
        lines.append(f"  【{cat_name}】")
        found = False
        for name, info in weather_related.items():
            all_text = " ".join(info["weather_conditions"])
            if any(kw in all_text for kw in keywords):
                type_label = TYPE_CN.get(info["type"], info["type"])
                lines.append(f"    · {name}（{type_label}）")
                found = True
        if not found:
            lines.append("    （暂无记录）")
        lines.append("")

    # ============ Part 5: 湿度-病虫害速查表 ============
    lines.append("━" * 90)
    lines.append("  第五部分：湿度-病虫害速查表")
    lines.append("━" * 90)
    lines.append("")

    humidity_categories = [
        ("高湿（相对湿度≥90%）", ["湿度高", "高湿", "相对湿度", "湿度大", "饱和"]),
        ("中低湿度", ["干燥", "干旱"]),
    ]

    for cat_name, keywords in humidity_categories:
        lines.append(f"  【{cat_name}】")
        found = False
        for name, info in weather_related.items():
            all_text = " ".join(info["weather_conditions"])
            if any(kw in all_text for kw in keywords):
                type_label = TYPE_CN.get(info["type"], info["type"])
                lines.append(f"    · {name}（{type_label}）")
                found = True
        if not found:
            lines.append("    （暂无记录）")
        lines.append("")

    lines.append("=" * 90)
    lines.append("  报告结束")
    lines.append("=" * 90)

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  天气-病虫害关联报告生成器")
    print("=" * 60)

    # 1. 解析格式化后的 NER 数据
    print("\n[1/3] 解析 NER 格式化数据...")
    sentences = parse_formatted_file(FORMATTED_FILE)
    print(f"  解析完成：{len(sentences)} 个句子")

    # 2. 提取病虫害气象知识
    print("[2/3] 提取病虫害-气象条件知识...")
    knowledge = extract_disease_pest_knowledge(sentences)
    weather_related_count = sum(1 for v in knowledge.values() if v["weather_conditions"])
    print(f"  提取完成：{len(knowledge)} 种病虫害，其中 {weather_related_count} 种与气象关联")

    # 3. 加载天气数据
    print("[3/3] 加载天气预报数据...")
    weather_df = load_weather_data(WEATHER_FILE)
    unique_cities = weather_df["city_name"].nunique()
    print(f"  加载完成：{len(weather_df)} 条天气记录，覆盖 {unique_cities} 个城市")

    # 4. 生成报告
    print("\n生成关联报告...")
    report = format_report(knowledge, weather_df)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存至: {OUTPUT_FILE}")
    print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
