# test/test_api.py
# 接口自测脚本，Mock状态下可直接运行（需先启动服务）
# 使用requests库调用所有接口，验证基本功能

import requests
# import json
import os

BASE_URL = "http://localhost:8000"

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print(f"/health: {resp.status_code} {resp.json()}")

def test_classes():
    resp = requests.get(f"{BASE_URL}/classes")
    print(f"/classes: {resp.status_code} {resp.json()}")

def test_predict():
    # 若当前目录存在 test.jpg 则使用，否则生成一个简单的彩色图片用于测试
    img_path = "test.jpg"
    if not os.path.exists(img_path):
        print("未找到 test.jpg，将生成一个临时测试图片...")
        from PIL import Image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(img_path)
        print("已生成 test.jpg")
    with open(img_path, 'rb') as f:
        files = {'file': (img_path, f, 'image/jpeg')}
        data = {'region': '西安', 'field': '试验田1号', 'growth_stage': '抽穗期'}
        resp = requests.post(f"{BASE_URL}/predict", files=files, data=data)
    print(f"/predict: {resp.status_code} {resp.json()}")

def test_records():
    resp = requests.get(f"{BASE_URL}/records")
    print(f"/records: {resp.status_code} {resp.json()}")

def test_statistics():
    resp = requests.get(f"{BASE_URL}/statistics")
    print(f"/statistics: {resp.status_code} {resp.json()}")

def test_weather_risk():
    params = {'region': '西安', 'growth_stage': '抽穗期'}
    resp = requests.get(f"{BASE_URL}/weather-risk", params=params)
    print(f"/weather-risk: {resp.status_code} {resp.json()}")

if __name__ == "__main__":
    print("=== 开始接口自测 ===")
    test_health()
    test_classes()
    test_predict()
    test_records()
    test_statistics()
    test_weather_risk()
    print("=== 测试完成 ===")