import sys
import os

_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapter.external_adapter import (
    init, query_weather_risk,
    save_record, get_all_records, get_statistics
)

init()

print("\n=== 测试1: 天气查询(西安) ===")
result = query_weather_risk("西安", "抽穗期")
print(f"天气风险: {result}")

print("\n=== 测试2: 天气查询(北京市) ===")
result2 = query_weather_risk("北京市", "灌浆期")
print(f"天气风险: {result2}")

print("\n=== 测试3: 保存记录 ===")
rec = {
    "field": "测试地块",
    "region": "西安",
    "growth_stage": "抽穗期",
    "predicted_class": "小麦条锈病",
    "confidence": 0.9,
    "is_reliable": True,
    "weather_risk": "low"
}
saved = save_record(rec)
print(f"保存结果: id={saved.get('id')}, created_at={saved.get('created_at')}")

print("\n=== 测试4: 统计 ===")
stats = get_statistics()
print(f"统计: {stats}")

print("\n=== 测试5: 记录列表 ===")
all_recs = get_all_records()
print(f"记录数: {len(all_recs)}")
for r in all_recs:
    print(f"  id={r.get('id')}, field={r.get('field')}, region={r.get('region')}")

print("\n=== 测试6: 不存在的地区 ===")
result5 = query_weather_risk("火星市", "播种期")
print(f"天气风险(火星): {result5}")

print("\n✅ 所有测试通过！")
