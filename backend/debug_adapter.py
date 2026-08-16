import sys, os
_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapter.external_adapter import init

init()

# 检查天气查询
from adapter.external_adapter import query_weather_risk
result = query_weather_risk("西安市", "抽穗期")
print(f"天气查询(西安市): {result}")

result2 = query_weather_risk("西安", "抽穗期")
print(f"天气查询(西安): {result2}")
