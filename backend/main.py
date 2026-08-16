# main.py
# 路由层：定义所有HTTP接口，负责参数接收、校验、调用业务层、返回响应
# 接口路径统一前缀：/api/v1/ （1号契约最终版）
# 全局返回格式：{"code": 0, "msg": "success", "data": {}}

import sys
import os
import io
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, Query, Path
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

# 确保项目根目录在 sys.path 中（兼容直接运行）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MAX_IMAGE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    CLASS_NAMES,
    CLASS_MAPPING,
    CONFIDENCE_THRESHOLD
)
from service.diagnose_service import DiagnoseService
import adapter.external_adapter as adapter

# ===== 创建FastAPI应用 =====
app = FastAPI(
    title="🌾 小麦病虫害智能诊断系统 API",
    description="基于 /api/v1 前缀的统一接口，全局 code/msg/data 返回格式",
    version="2.0.0"
)


# ===== 启动事件（加载模型等） =====
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化适配层（加载模型、连接服务等）"""
    print("正在初始化适配层...")
    adapter.init()
    print("适配层初始化完成")


# ===== 全局响应封装函数 =====
def success_response(data=None, msg="success"):
    """成功响应：code=0"""
    return {"code": 0, "msg": msg, "data": data or {}}


def error_response(code: int, msg: str):
    """错误响应：code非0，HTTP状态码仍为200"""
    return {"code": code, "msg": msg, "data": {}}


# ===== 1. 健康检查 =====
@app.get("/api/v1/health", tags=["系统"])
async def health():
    """健康检查接口，返回服务运行状态"""
    return success_response({
        "status": "running",
        "service": "wheat-disease-backend",
        "version": "2.0.0"
    })


# ===== 2. 病害类别列表 =====
@app.get("/api/v1/classes", tags=["系统"])
async def get_classes():
    """返回15类病害中英文名称列表，class_id 从1开始递增"""
    classes_list = []
    for idx, (en_name, cn_name) in enumerate(CLASS_MAPPING.items(), start=1):
        classes_list.append({
            "class_id": idx,
            "class_name": cn_name,
            "class_name_en": en_name
        })
    return success_response({"classes": classes_list})


# ===== 3. 核心诊断接口 =====
@app.post("/api/v1/diagnoses", tags=["诊断"])
async def diagnose(
    image: UploadFile = File(..., description="小麦图片文件"),
    region: str = Form(..., description="地区"),
    growth_stage: str = Form(..., description="小麦生育期"),
    field: str = Form(None, description="地块名称（可选）")
):
    """
    核心诊断接口：上传小麦图片，返回病虫害识别结果
    """
    # 1. 必填参数校验
    if not region or not growth_stage:
        return error_response(1001, "region 和 growth_stage 为必填参数")

    # 2. 图片格式校验
    filename = image.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return error_response(1002, f"不支持的图片格式，仅支持 {', '.join(ALLOWED_EXTENSIONS)}")

    # 3. 读取并校验文件大小
    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        return error_response(1003, f"图片大小超过 {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB 限制")

    # 4. 尝试解析为PIL图片
    try:
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        _ = pil_image.size  # 验证图片有效性
    except Exception as e:
        return error_response(1004, f"图片无法解析，请确认是有效的图片文件：{str(e)}")

    # 5. 图片质检通过，构建 image_quality 对象
    image_quality = {
        "status": "success",
        "message": "图片质检通过"
    }

    # 6. 调用业务层执行推理
    try:
        result = DiagnoseService.diagnose(pil_image, region, field, growth_stage)

        # 7. 组装响应数据
        data = {
            "diagnosis_id": result.get("diagnosis_id"),
            "created_at": result.get("created_at"),
            "region": region,
            "growth_stage": growth_stage,
            "field": field,
            "image_quality": image_quality,
            "predictions": result.get("predictions", []),
            "reliable": result.get("reliable", False),
            "threshold": CONFIDENCE_THRESHOLD,
            "knowledge": result.get("knowledge"),
            "weather_risk": result.get("weather_risk"),
            "record_saved": True
        }

        return success_response(data)

    except Exception as e:
        return error_response(5001, f"模型推理失败：{str(e)}")


# ===== 4. 历史记录列表 =====
@app.get("/api/v1/diagnoses", tags=["历史"])
async def get_records():
    """获取所有历史诊断记录，字段已对齐契约"""
    try:
        records = DiagnoseService.get_all_records()
        formatted_records = []
        for r in records:
            formatted_records.append({
                "diagnosis_id": r.get("id"),
                "created_at": r.get("created_at"),
                "region": r.get("region"),
                "field": r.get("field"),
                "growth_stage": r.get("growth_stage"),
                "predicted_class": r.get("predicted_class"),
                "confidence": r.get("confidence"),
                "reliable": r.get("is_reliable", False),
                "risk_level": r.get("weather_risk")
            })
        return success_response({"records": formatted_records})
    except Exception as e:
        return error_response(5002, f"获取历史记录失败：{str(e)}")


# ===== 5. 单条诊断详情 =====
@app.get("/api/v1/diagnoses/{diagnosis_id}", tags=["历史"])
async def get_record_detail(
    diagnosis_id: int = Path(..., description="诊断记录ID")
):
    """获取单条诊断详情"""
    try:
        records = DiagnoseService.get_all_records()
        for r in records:
            if r.get("id") == diagnosis_id:
                return success_response({
                    "diagnosis_id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "region": r.get("region"),
                    "field": r.get("field"),
                    "growth_stage": r.get("growth_stage"),
                    "predicted_class": r.get("predicted_class"),
                    "confidence": r.get("confidence"),
                    "reliable": r.get("is_reliable", False),
                    "risk_level": r.get("weather_risk")
                })
        return error_response(4001, f"未找到 diagnosis_id 为 {diagnosis_id} 的记录")
    except Exception as e:
        return error_response(5003, f"获取记录详情失败：{str(e)}")


# ===== 6. 数据统计 =====
@app.get("/api/v1/statistics", tags=["统计"])
async def get_statistics():
    """获取诊断统计数据"""
    try:
        stats = DiagnoseService.get_statistics()
        return success_response({
            "total_diagnoses": stats.get("total_diagnoses", 0),
            "by_class": stats.get("by_class", {}),
            "by_reliable": stats.get("by_reliable", {"reliable": 0, "unreliable": 0}),
            "by_risk_level": stats.get("by_risk_level", {}),
            "recent_high_risk": stats.get("recent_high_risk", []),
            "daily_trend": stats.get("daily_trend", [])
        })
    except Exception as e:
        return error_response(5004, f"获取统计数据失败：{str(e)}")


# ===== 7. （调试接口）低置信度开关 =====
@app.post("/debug/set-low-confidence", tags=["调试"])
async def set_low_confidence(enable: bool = True):
    """
    调试接口：控制是否强制返回低置信度
    - enable=true  : 后续诊断强制返回低置信度（confidence ≈ 0.5）
    - enable=false : 恢复随机模式（20%概率低置信度）
    """
    from mock.mock_modules import MockInference
    MockInference.force_low_confidence = enable
    return success_response({
        "force_low_confidence": enable,
        "message": f"低置信度强制模式已{'开启' if enable else '关闭'}"
    })


# ===== 启动服务 =====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)