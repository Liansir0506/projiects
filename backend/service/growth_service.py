# service/growth_service.py
# 小麦长势评估服务（专家规则引擎 + 加权评分法）
# 方案一：零数据依赖，基于农业专家规则，可解释性强，完全复用现有模块

import json
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置文件路径
_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "knowledge", "growth_rules", "growth_stages.json"
)
_SUGGESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "knowledge", "growth_rules", "intervention_suggestions.json"
)

# 维度中文名映射
DIMENSION_NAMES = {
    "temperature": "温度适宜度",
    "soil_moisture": "土壤水分条件",
    "soil_fertility": "土壤肥力条件",
    "sunlight": "光照条件",
    "farming_operation": "农事操作匹配度",
    "pest_risk": "病虫害风险",
    "visual_condition": "视觉苗情"
}

# 维度单位
DIMENSION_UNITS = {
    "temperature": "℃",
    "soil_moisture": "%田间持水量",
    "soil_fertility": "肥力指数",
    "sunlight": "小时/天",
    "farming_operation": "完成度%",
    "pest_risk": "风险指数",
    "visual_condition": "视觉评分"
}


class GrowthService:
    """小麦长势评估服务类"""

    _rules_cache = None
    _suggestions_cache = None

    @classmethod
    def _load_rules(cls) -> Dict[str, Any]:
        """加载评分规则配置（带缓存）"""
        if cls._rules_cache is None:
            with open(_RULES_PATH, "r", encoding="utf-8") as f:
                cls._rules_cache = json.load(f)
        return cls._rules_cache

    @classmethod
    def _load_suggestions(cls) -> Dict[str, Any]:
        """加载干预建议库（带缓存）"""
        if cls._suggestions_cache is None:
            with open(_SUGGESTIONS_PATH, "r", encoding="utf-8") as f:
                cls._suggestions_cache = json.load(f)
        return cls._suggestions_cache

    @staticmethod
    def _match_score(value: float, rules: List[Dict]) -> Tuple[int, str]:
        """
        根据输入值匹配评分规则区间，返回(得分, 标签)
        rules 格式: [{"min": x, "max": y, "score": z, "label": "..."}, ...]
        """
        for rule in rules:
            if rule["min"] <= value < rule["max"]:
                return rule["score"], rule["label"]
        # 超出所有区间，取最近边界
        if value < rules[0]["min"]:
            return rules[0]["score"], rules[0]["label"]
        return rules[-1]["score"], rules[-1]["label"]

    @classmethod
    def evaluate(
        cls,
        growth_stage: str,
        temperature: float,
        soil_moisture: float,
        soil_fertility: float,
        sunlight: float,
        farming_operation: float = 70.0,
        pest_risk: float = 20.0,
        visual_condition: Optional[float] = None,
        region: str = "",
        field: str = "",
        future_weather: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        执行完整的长势评估流程

        参数:
            growth_stage: 生育期（tillering/reviving/jointing/heading/filling）
            temperature: 日均温(℃)
            soil_moisture: 土壤湿度(%田间持水量)
            soil_fertility: 土壤肥力指数(0-100)
            sunlight: 光照时长(小时/天)
            farming_operation: 农事操作完成度(0-100)，默认70
            pest_risk: 病虫害风险指数(0-100，越低越好)，默认20
            visual_condition: 视觉苗情评分(0-100)，可选
            region: 地区
            field: 地块名称
            future_weather: 未来7天天气，用于趋势预判，可选

        返回:
            {
                "evaluation_id": "...",
                "created_at": "...",
                "growth_stage": "...",
                "total_score": 85,
                "grade": "壮苗",
                "grade_code": "strong",
                "dimension_scores": {...},
                "weakness_analysis": {...},
                "intervention_suggestions": [...],
                "trend_prediction": {...}
            }
        """
        logger.info(f"=== 开始长势评估 生育期={growth_stage} ===")

        rules = cls._load_rules()
        suggestions = cls._load_suggestions()

        # 1. 校验生育期
        stages = rules["growth_stages"]
        if growth_stage not in stages:
            available = ", ".join(stages.keys())
            raise ValueError(f"不支持的生育期: {growth_stage}，可选: {available}")

        stage_config = stages[growth_stage]
        stage_name = stage_config["name"]
        weights = stage_config["weights"]
        scoring_rules = stage_config["scoring_rules"]

        # 2. 构建输入值字典
        inputs = {
            "temperature": temperature,
            "soil_moisture": soil_moisture,
            "soil_fertility": soil_fertility,
            "sunlight": sunlight,
            "farming_operation": farming_operation,
            "pest_risk": pest_risk,
        }
        if visual_condition is not None:
            inputs["visual_condition"] = visual_condition

        # 3. 计算各维度得分
        dimension_scores = {}
        total_score = 0.0
        total_weight = 0.0

        for dim, value in inputs.items():
            if dim not in scoring_rules:
                continue
            score, label = cls._match_score(value, scoring_rules[dim]["rules"])
            weight = weights.get(dim, 0)
            dimension_scores[dim] = {
                "name": DIMENSION_NAMES.get(dim, dim),
                "value": value,
                "unit": DIMENSION_UNITS.get(dim, ""),
                "score": score,
                "weight": weight,
                "label": label,
                "weighted_score": round(score * weight, 2)
            }
            total_score += score * weight
            total_weight += weight

        # 归一化（防止权重和不为1）
        if total_weight > 0:
            total_score = round(total_score / total_weight, 1)
        else:
            total_score = 0.0

        # 4. 判断长势等级
        grade_code, grade_name, grade_desc = cls._determine_grade(
            total_score, dimension_scores, rules["grade_mapping"]
        )

        # 5. 短板分析（找出得分最低的2个维度）
        weakness = cls._analyze_weakness(dimension_scores)

        # 6. 匹配干预建议
        intervention = cls._match_suggestions(
            growth_stage, dimension_scores, grade_code, suggestions
        )

        # 7. 短期趋势预判
        trend = cls._predict_trend(
            growth_stage, dimension_scores, future_weather, rules
        )

        # 8. 组装结果
        evaluation_id = cls._generate_id()
        result = {
            "evaluation_id": evaluation_id,
            "created_at": datetime.now().isoformat(),
            "region": region,
            "field": field,
            "growth_stage": growth_stage,
            "growth_stage_name": stage_name,
            "total_score": total_score,
            "grade": grade_name,
            "grade_code": grade_code,
            "grade_description": grade_desc,
            "dimension_scores": dimension_scores,
            "weakness_analysis": weakness,
            "intervention_suggestions": intervention,
            "trend_prediction": trend
        }

        logger.info(f"评估完成: 总分={total_score}, 等级={grade_name}")
        return result

    @staticmethod
    def _determine_grade(
        total_score: float,
        dimension_scores: Dict,
        grade_mapping: Dict
    ) -> Tuple[str, str, str]:
        """
        判断长势等级
        特殊：旺苗(overgrowth)需要满足条件：总分高 + 肥力高 + 温度偏低
        """
        # 先检查旺苗条件
        overgrowth = grade_mapping.get("overgrowth", {})
        if (total_score >= overgrowth.get("min", 90) and
            dimension_scores.get("soil_fertility", {}).get("score", 0) >= 85 and
            dimension_scores.get("temperature", {}).get("score", 100) <= 60):
            return (
                "overgrowth",
                overgrowth.get("label", "旺苗"),
                overgrowth.get("description", "")
            )

        # 正常等级映射
        for code, info in grade_mapping.items():
            if code == "overgrowth":
                continue
            if info["min"] <= total_score < info["max"]:
                return code, info["label"], info["description"]

        return "normal", "正常苗", "长势一般"

    @staticmethod
    def _analyze_weakness(dimension_scores: Dict) -> Dict[str, Any]:
        """分析短板：找出得分最低的维度"""
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]["score"]
        )
        weakest = sorted_dims[0]
        second_weakest = sorted_dims[1] if len(sorted_dims) > 1 else None

        return {
            "primary_weakness": {
                "dimension": weakest[0],
                "name": weakest[1]["name"],
                "score": weakest[1]["score"],
                "value": weakest[1]["value"],
                "label": weakest[1]["label"]
            },
            "secondary_weakness": {
                "dimension": second_weakest[0],
                "name": second_weakest[1]["name"],
                "score": second_weakest[1]["score"],
                "value": second_weakest[1]["value"],
                "label": second_weakest[1]["label"]
            } if second_weakest else None,
            "summary": f"主要短板为{weakest[1]['name']}（{weakest[1]['score']}分），{weakest[1]['label']}"
        }

    @classmethod
    def _match_suggestions(
        cls,
        growth_stage: str,
        dimension_scores: Dict,
        grade_code: str,
        suggestions: Dict
    ) -> List[Dict[str, Any]]:
        """根据短板维度和等级匹配干预建议"""
        result = []
        stage_suggestions = suggestions.get("suggestions", {}).get(growth_stage, {})

        # 按维度得分从低到高排序，为每个低分维度匹配建议
        sorted_dims = sorted(
            dimension_scores.items(),
            key=lambda x: x[1]["score"]
        )

        for dim_key, dim_info in sorted_dims:
            if dim_info["score"] >= 70:
                continue  # 得分>=70不需要干预建议
            dim_sugs = stage_suggestions.get(dim_key, {})

            # 判断是偏低还是偏高
            if dim_key in ("temperature", "soil_moisture", "soil_fertility", "sunlight"):
                # 这些维度：得分低可能是值偏低或偏高
                optimal = dimension_scores[dim_key].get("label", "")
                if "低" in optimal or "不足" in optimal or "干旱" in optimal or "寡照" in optimal:
                    sug_type = "low"
                elif "高" in optimal or "过" in optimal or "旺长" in optimal or "偏湿" in optimal:
                    sug_type = "high"
                else:
                    sug_type = "low"
            elif dim_key == "pest_risk":
                sug_type = "high"  # 风险高需要防治
            else:
                sug_type = "low"

            sug = dim_sugs.get(sug_type)
            if sug:
                result.append({
                    "dimension": dim_key,
                    "dimension_name": dim_info["name"],
                    "current_score": dim_info["score"],
                    "title": sug["title"],
                    "actions": sug["actions"]
                })

        # 旺苗特殊建议
        if grade_code == "overgrowth":
            overgrowth_sug = suggestions.get("general_suggestions", {}).get("overgrowth")
            if overgrowth_sug:
                result.append({
                    "dimension": "overall",
                    "dimension_name": "综合旺长防控",
                    "current_score": dimension_scores.get("soil_fertility", {}).get("score", 0),
                    "title": overgrowth_sug["title"],
                    "actions": overgrowth_sug["actions"]
                })

        # 严重弱苗特殊建议
        if grade_code in ("weak", "very_weak"):
            weak_sug = suggestions.get("general_suggestions", {}).get("weak_seedling")
            if weak_sug:
                result.append({
                    "dimension": "overall",
                    "dimension_name": "弱苗转化升级",
                    "current_score": 0,
                    "title": weak_sug["title"],
                    "actions": weak_sug["actions"]
                })

        return result

    @staticmethod
    def _predict_trend(
        growth_stage: str,
        dimension_scores: Dict,
        future_weather: Optional[List[Dict]],
        rules: Dict
    ) -> Dict[str, Any]:
        """
        短期趋势预判（基于未来7天天气）
        future_weather 格式: [{"date": "...", "temp_avg": 15, "rain": 5, "sunlight": 6}, ...]
        """
        if not future_weather:
            return {
                "available": False,
                "summary": "未提供未来天气数据，无法进行趋势预判",
                "advice": "建议接入天气模块后获取7天预报进行趋势分析"
            }

        # 简单趋势预判：根据未来温度和降水评估
        avg_temp = sum(d.get("temp_avg", 0) for d in future_weather) / len(future_weather)
        total_rain = sum(d.get("rain", 0) for d in future_weather)
        avg_sun = sum(d.get("sunlight", 0) for d in future_weather) / len(future_weather)

        stage_config = rules["growth_stages"].get(growth_stage, {})
        temp_optimal = stage_config.get("scoring_rules", {}).get("temperature", {}).get("optimal_range", [10, 20])
        moisture_optimal = stage_config.get("scoring_rules", {}).get("soil_moisture", {}).get("optimal_range", [60, 75])

        trend_score = 0
        factors = []

        # 温度趋势
        if temp_optimal[0] <= avg_temp <= temp_optimal[1]:
            trend_score += 1
            factors.append(f"未来平均气温{avg_temp}℃，处于适宜区间")
        elif avg_temp < temp_optimal[0]:
            factors.append(f"未来平均气温{avg_temp}℃，偏低，可能影响生长速度")
        else:
            factors.append(f"未来平均气温{avg_temp}℃，偏高，注意旺长或高温逼熟")

        # 降水趋势
        if 10 <= total_rain <= 50:
            trend_score += 1
            factors.append(f"未来7天降水{total_rain}mm，墒情有望改善")
        elif total_rain < 10:
            factors.append(f"未来7天降水仅{total_rain}mm，注意防旱")
        else:
            factors.append(f"未来7天降水{total_rain}mm，注意排涝和病害")

        # 光照趋势
        if avg_sun >= 5:
            trend_score += 1
            factors.append(f"未来平均光照{avg_sun}小时/天，光照充足")
        else:
            factors.append(f"未来平均光照{avg_sun}小时/天，光照偏少")

        if trend_score >= 2:
            outlook = "乐观"
            summary = "未来天气条件总体有利，长势有望持续向好"
        elif trend_score == 1:
            outlook = "一般"
            summary = "未来天气条件一般，需关注不利因素影响"
        else:
            outlook = "偏谨慎"
            summary = "未来天气存在不利因素，建议提前做好应对准备"

        return {
            "available": True,
            "days": len(future_weather),
            "avg_temperature": round(avg_temp, 1),
            "total_rainfall": round(total_rain, 1),
            "avg_sunlight": round(avg_sun, 1),
            "outlook": outlook,
            "summary": summary,
            "factors": factors,
            "advice": "请根据天气变化及时调整农事操作"
        }

    @staticmethod
    def _generate_id() -> str:
        """生成评估ID：G + 日期 + 4位随机数"""
        import random
        import string
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = ''.join(random.choices(string.digits, k=4))
        return f"G{date_str}{suffix}"

    @classmethod
    def get_available_stages(cls) -> List[Dict[str, str]]:
        """获取所有支持的生育期列表"""
        rules = cls._load_rules()
        return [
            {"code": code, "name": info["name"], "description": info["description"]}
            for code, info in rules["growth_stages"].items()
        ]

    @classmethod
    def get_stage_weights(cls, growth_stage: str) -> Dict[str, Any]:
        """获取指定生育期的评分维度和权重"""
        rules = cls._load_rules()
        stage = rules["growth_stages"].get(growth_stage)
        if not stage:
            return {}
        return {
            "stage": growth_stage,
            "name": stage["name"],
            "weights": stage["weights"],
            "dimensions": [
                {
                    "code": dim,
                    "name": DIMENSION_NAMES.get(dim, dim),
                    "unit": DIMENSION_UNITS.get(dim, ""),
                    "weight": stage["weights"].get(dim, 0),
                    "optimal_range": stage["scoring_rules"][dim]["optimal_range"]
                }
                for dim in stage["weights"].keys()
            ]
        }
