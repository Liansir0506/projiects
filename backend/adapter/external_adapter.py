# adapter/external_adapter.py
# 【对接适配层：外部模块变更仅修改此处，请勿修改上层业务】
# 本文件集中封装所有外部依赖（模型、知识库、天气、存储）。
# 当前默认使用Mock实现，后期替换为真实模块时仅需修改本文件。

import sys
import os
# 将项目根目录加入路径，确保能导入mock模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from PIL import Image

# 导入Mock模块（临时）
from mock.mock_modules import mock_inference, mock_knowledge, mock_weather, mock_database

# 全局模型句柄（真实模型加载后使用）
_model = None


def init() -> None:
    """
    适配层初始化，服务启动时调用。
    当前Mock模式仅打印信息；后期替换为真实模型加载。
    """
    global _model
    print("[适配层] 初始化：使用Mock模拟模式")
    # 【后期替换示例】
    # from inference import load_model  # 假设4号提供该函数
    # _model = load_model(device=config.DEVICE)
    # print("[适配层] 真实模型加载完成")


def predict_image(pil_img: Image.Image) -> List[Dict[str, Any]]:
    """
    调用模型推理，返回Top-3结果。
    :param pil_img: PIL Image对象（RGB模式）
    :return: [{"class_name": str, "confidence": float}, ...]
    """
    # 获取设备（可从config读取）
    from config import DEVICE
    # 【临时Mock实现】
    return mock_inference.predict_image(pil_img, DEVICE)
    # 【后期替换为真实推理】
    # global _model
    # return real_inference(_model, pil_img)


def query_knowledge(class_name: str) -> Dict[str, str]:
    """查询知识库"""
    # 【临时Mock实现】
    return mock_knowledge.query_pest_knowledge(class_name)
    # 【后期替换】
    # return real_knowledge.query(class_name)


def query_weather_risk(region: str, growth_stage: str) -> Dict[str, str]:
    """查询天气风险"""
    # 【临时Mock实现】
    return mock_weather.query_weather_risk(region, growth_stage)
    # 【后期替换】
    # return real_weather.query(region, growth_stage)


def save_record(record: dict) -> dict:
    """保存诊断记录，并返回包含 id 和 created_at 的完整记录"""
    # 【临时Mock实现】
    return mock_database.save_record(record)
    # 【后期替换】
    # return real_database.save(record)


def get_all_records() -> List[Dict]:
    """获取全部历史记录"""
    return mock_database.get_all_records()
    # 【后期替换】
    # return real_database.get_all()


def get_statistics() -> Dict[str, int]:
    """获取统计数据"""
    return mock_database.get_statistics()
    # 【后期替换】
    # return real_database.statistics()