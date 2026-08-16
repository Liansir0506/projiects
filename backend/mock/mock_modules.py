# mock/mock_modules.py
# 【临时Mock实现，待7号/8号/4号交付后替换】
# 严格模拟下游模块的接口行为，使后端可在无真实模块时独立运行

import random
import datetime
from typing import List, Dict
from PIL import Image

# 导入类别列表，保证Mock返回的类别在有效范围内
from config import CLASS_NAMES


class MockInference:
    """模拟4号-模型推理模块"""
    # ===== 新增：类变量，用于控制是否强制低置信度 =====
    force_low_confidence = False

    # ===== 新增结束 =====

    @staticmethod
    def predict_image(pil_img: Image.Image, device: str) -> List[Dict]:
        """模拟Top-3输出，随机选择类别，并分配置信度"""
        # 从 CLASS_NAMES 中随机选3个不同类别
        classes = random.sample(CLASS_NAMES, 3)

        # ===== 修改：使用类变量控制，或20%随机 =====
        if MockInference.force_low_confidence or random.random() < 0.2:
            confidences = [0.50, 0.30, 0.20]
        else:
            confidences = [0.85, 0.10, 0.05]
        # ===== 修改结束 =====

        results = []
        for i in range(3):
            results.append({
                "class_name": classes[i],
                "confidence": confidences[i]
            })
        # 按置信度降序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results


class MockKnowledge:
    """模拟7号-农业知识库模块"""
    @staticmethod
    def query_pest_knowledge(class_name: str) -> Dict:
        """返回固定格式的知识信息"""
        return {
            "disease_name": class_name,
            "symptom": "叶片出现黄色至褐色斑点，严重时病斑连片，导致叶片枯死。",
            "prevention": "选用抗病品种；合理轮作；发病初期可使用三唑类、嘧菌酯等药剂按说明喷雾防治。"
        }


class MockWeather:
    """模拟8号-天气风险模块"""
    @staticmethod
    def query_weather_risk(region: str, growth_stage: str) -> Dict:
        """返回固定低风险，也可随机生成不同等级"""
        # 可扩展随机逻辑，此处为演示统一返回low
        return {
            "risk_level": "low",
            "risk_desc": f"当前{region}地区天气良好，温度湿度适宜，对小麦{growth_stage or '生长期'}影响较小。"
        }


class MockDatabase:
    """模拟8号-数据库存储模块（内存存储）"""
    def __init__(self):
        self.records = []
        self.id_counter = 1

    def save_record(self, record: dict) -> dict:
        """保存记录，自动添加id和创建时间，并返回完整记录"""
        record_copy = record.copy()
        record_copy["id"] = self.id_counter
        self.id_counter += 1
        record_copy["created_at"] = datetime.datetime.now().isoformat()
        self.records.append(record_copy)
        return record_copy  # 返回保存后的完整记录

    def get_all_records(self) -> List[Dict]:
        return self.records

    def get_statistics(self) -> Dict:
        # 计算各类别数量
        by_class = {}
        by_reliable = {"reliable": 0, "unreliable": 0}
        by_risk_level = {}

        for r in self.records:
            cls = r.get("predicted_class", "unknown")
            by_class[cls] = by_class.get(cls, 0) + 1

            if r.get("is_reliable"):
                by_reliable["reliable"] += 1
            else:
                by_reliable["unreliable"] += 1

            risk = r.get("weather_risk", "unknown")
            by_risk_level[risk] = by_risk_level.get(risk, 0) + 1

        return {
            "total_diagnoses": len(self.records),
            "by_class": by_class,
            "by_reliable": by_reliable,
            "by_risk_level": by_risk_level,
            "recent_high_risk": [],  # 注意字段名是 recent_high_risk
            "daily_trend": []
        }


# 创建单例对象供适配层调用
mock_inference = MockInference()
mock_knowledge = MockKnowledge()
mock_weather = MockWeather()
mock_database = MockDatabase()