# service/diagnose_service.py
# 业务逻辑层：实现完整的诊断流程
# 本层只调用 adapter 层，对底层实现（Mock或真实）无感知，后期无需修改
# 返回格式已按 /api/v1/ 契约对齐

import logging
import random
import string
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image

from config import CONFIDENCE_THRESHOLD, CLASS_NAMES, CLASS_MAPPING
import adapter.external_adapter as adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnoseService:
    """诊断服务类"""

    @staticmethod
    def _generate_diagnosis_id() -> str:
        """
        生成诊断ID，格式：D + 日期(8位) + 随机4位数字
        示例：D202608150001
        """
        date_str = datetime.now().strftime("%Y%m%d")
        random_suffix = ''.join(random.choices(string.digits, k=4))
        return f"D{date_str}{random_suffix}"

    @staticmethod
    def _translate_predictions(top3: List[Dict]) -> List[Dict]:
        """
        将模型输出的英文类别翻译为中文，并添加 class_id
        输入：[{"class_name": "Yellow Rust", "confidence": 0.91}, ...]
        输出：[{"class_id": 15, "class_name": "小麦条锈病", "display_name": "小麦条锈病", "confidence": 0.91}, ...]
        """
        predictions = []
        # 构建英文名 → class_id 的映射（按 CLASS_MAPPING 的顺序）
        en_to_id = {en: idx + 1 for idx, en in enumerate(CLASS_MAPPING.keys())}

        for item in top3:
            en_name = item.get("class_name", "Unknown")
            confidence = item.get("confidence", 0.0)
            # 获取中文名，如果找不到则用英文名本身
            cn_name = CLASS_MAPPING.get(en_name, en_name)
            class_id = en_to_id.get(en_name, 0)

            predictions.append({
                "class_id": class_id,
                "class_name": cn_name,
                "display_name": cn_name,  # 与 class_name 保持一致，均为中文
                "confidence": confidence
            })

        return predictions

    @staticmethod
    def diagnose(image: Image.Image, region: str, field: str, growth_stage: str) -> Dict[str, Any]:
        """
        执行完整诊断流程

        步骤：
        1. 模型推理获取Top-3（模型返回英文类别名）
        2. 将英文类别翻译为中文，添加 class_id
        3. 判断最高置信度是否达标
        4. 达标则查询知识库和天气风险
        5. 组装响应（格式对齐 /api/v1/ 契约）
        6. 保存记录

        返回格式示例：
        {
            "diagnosis_id": "D202608150001",
            "created_at": "2026-08-15T10:30:08+08:00",
            "predictions": [...],
            "reliable": true,
            "knowledge": {...},
            "weather_risk": {...}
        }
        """
        logger.info("=== 开始诊断 ===")

        # ===== 1. 模型推理 =====
        try:
            top3 = adapter.predict_image(image)
            logger.info(f"模型原始输出: {top3}")
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise RuntimeError("模型推理失败，请稍后重试") from e

        if not top3:
            raise RuntimeError("模型未返回有效结果")

        # ===== 2. 翻译预测结果为中文 =====
        predictions = DiagnoseService._translate_predictions(top3)
        logger.info(f"翻译后输出: {predictions}")

        # 取最高置信度结果
        top1 = predictions[0]
        top1_class = top1["class_name"]          # 中文名，如"小麦条锈病"
        top1_class_en = top3[0]["class_name"]    # 英文名，如"Yellow Rust"
        top1_conf = top1["confidence"]
        reliable = top1_conf >= CONFIDENCE_THRESHOLD

        # ===== 3. 生成诊断ID和创建时间 =====
        diagnosis_id = DiagnoseService._generate_diagnosis_id()
        created_at = datetime.now().isoformat()

        knowledge = None
        weather_risk = None

        # ===== 4. 置信度达标则查询天气风险 =====
        if reliable:
            logger.info(f"置信度达标 ({top1_conf:.2f} >= {CONFIDENCE_THRESHOLD})")

            # 查询天气风险
            try:
                weather_raw = adapter.query_weather_risk(region, growth_stage)
                weather_risk = {
                    "level": weather_raw.get("risk_level", "unknown"),
                    "summary": weather_raw.get("risk_desc", "天气信息暂不可用"),
                    "daily": []
                }
                logger.info(f"天气查询成功: {weather_risk}")
            except Exception as e:
                logger.error(f"天气查询失败: {e}")
                weather_risk = {
                    "level": "unknown",
                    "summary": "天气信息暂时无法获取，请关注实时天气。",
                    "daily": []
                }
        else:
            logger.info(f"置信度不达标 ({top1_conf:.2f} < {CONFIDENCE_THRESHOLD})")

        # ===== 5. 组装响应（对齐 /api/v1/ 契约） =====
        response = {
            "diagnosis_id": diagnosis_id,
            "created_at": created_at,
            "predictions": predictions,
            "reliable": reliable,
            "threshold": CONFIDENCE_THRESHOLD,
            "knowledge": knowledge,
            "weather_risk": weather_risk,
            # 以下字段保留作为兼容（前端可能用到）
            "region": region,
            "field": field,
            "growth_stage": growth_stage,
        }

        # ===== 6. 保存记录（无论可靠与否） =====
        try:
            record = {
                "diagnosis_id": diagnosis_id,      # 新增：使用诊断ID
                "field": field,
                "region": region,
                "growth_stage": growth_stage,
                "predicted_class": top1_class,      # 存储中文名
                "predicted_class_en": top1_class_en,  # 存储英文名（方便后续追溯）
                "confidence": top1_conf,
                "is_reliable": reliable,
                "weather_risk": weather_risk.get("level") if weather_risk else None,
                "model_version": "mock_v1",
            }
            # 保存并获取完整记录（包含 id 和 created_at）
            saved_record = adapter.save_record(record)
            if saved_record:
                # 如果 adapter 返回了记录，可以补充更多信息
                logger.info(f"记录保存成功，ID: {saved_record.get('id')}")
        except Exception as e:
            logger.error(f"保存记录失败: {e}")
            # 保存失败不影响主流程

        return response

    @staticmethod
    def get_all_records() -> List[Dict]:
        """获取所有历史诊断记录"""
        return adapter.get_all_records()

    @staticmethod
    def get_statistics() -> Dict[str, Any]:
        """
        获取诊断统计数据
        返回格式已对齐 /api/v1/statistics 契约
        """
        try:
            stats = adapter.get_statistics()
            return {
                "total_diagnoses": stats.get("total_diagnoses", 0),
                "by_class": stats.get("by_class", {}),
                "by_reliable": stats.get("by_reliable", {"reliable": 0, "unreliable": 0}),
                "by_risk_level": stats.get("by_risk_level", {}),
                "recent_high_risk": stats.get("recent_high_risk", []),
                "daily_trend": stats.get("daily_trend", [])
            }
        except Exception as e:
            logger.error(f"获取统计数据失败: {e}")
            return {
                "total_diagnoses": 0,
                "by_class": {},
                "by_reliable": {"reliable": 0, "unreliable": 0},
                "by_risk_level": {},
                "recent_high_risk": [],
                "daily_trend": []
            }

    @staticmethod
    def get_classes() -> List[str]:
        """获取所有类别名称（英文）"""
        return CLASS_NAMES