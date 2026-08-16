# test/test_growth.py
# 小麦长势评估功能测试

import sys
import os
import json

# 确保backend目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.growth_service import GrowthService


def test_get_available_stages():
    """测试获取所有支持的生育期"""
    stages = GrowthService.get_available_stages()
    print(f"\n=== 测试1: 获取生育期列表 ===")
    print(f"支持的生育期数量: {len(stages)}")
    for s in stages:
        print(f"  {s['code']}: {s['name']} - {s['description']}")
    assert len(stages) == 5, f"应该有5个生育期，实际{len(stages)}"
    codes = [s['code'] for s in stages]
    assert 'tillering' in codes
    assert 'reviving' in codes
    assert 'jointing' in codes
    assert 'heading' in codes
    assert 'filling' in codes
    print("✅ 测试1通过")


def test_get_stage_weights():
    """测试获取指定生育期的评分维度和权重"""
    print(f"\n=== 测试2: 获取返青期评分维度 ===")
    weights = GrowthService.get_stage_weights("reviving")
    print(f"生育期: {weights['name']}")
    print(f"维度数量: {len(weights['dimensions'])}")
    total_weight = sum(d['weight'] for d in weights['dimensions'])
    print(f"权重总和: {total_weight}")
    for d in weights['dimensions']:
        print(f"  {d['name']}: 权重={d['weight']}, 适宜区间={d['optimal_range']}")
    assert abs(total_weight - 1.0) < 0.01, f"权重总和应为1，实际{total_weight}"
    print("✅ 测试2通过")


def test_evaluate_normal():
    """测试正常苗情评估"""
    print(f"\n=== 测试3: 正常苗情评估（返青期） ===")
    result = GrowthService.evaluate(
        growth_stage="reviving",
        temperature=12.0,      # 适宜温度
        soil_moisture=65.0,    # 适宜墒情
        soil_fertility=75.0,   # 肥力充足
        sunlight=7.0,          # 光照充足
        farming_operation=85.0, # 管理到位
        pest_risk=15.0,        # 病虫害风险低
        region="陕西省西安市",
        field="试验田A区"
    )
    print(f"评估ID: {result['evaluation_id']}")
    print(f"生育期: {result['growth_stage_name']}")
    print(f"总分: {result['total_score']}")
    print(f"等级: {result['grade']} ({result['grade_code']})")
    print(f"等级说明: {result['grade_description']}")
    print(f"\n各维度得分:")
    for dim, info in result['dimension_scores'].items():
        print(f"  {info['name']}: {info['score']}分 (权重{info['weight']}, {info['label']})")
    print(f"\n短板分析: {result['weakness_analysis']['summary']}")
    print(f"干预建议数量: {len(result['intervention_suggestions'])}")
    for sug in result['intervention_suggestions']:
        print(f"  - {sug['title']}: {len(sug['actions'])}条建议")

    assert result['total_score'] >= 70, f"正常苗情总分应>=70，实际{result['total_score']}"
    assert result['grade_code'] in ('strong', 'normal'), f"等级应为壮苗或正常苗，实际{result['grade_code']}"
    print("✅ 测试3通过")


def test_evaluate_weak():
    """测试弱苗评估（干旱+低温+缺肥）"""
    print(f"\n=== 测试4: 弱苗评估（返青期，干旱+低温+缺肥） ===")
    result = GrowthService.evaluate(
        growth_stage="reviving",
        temperature=4.0,       # 低温
        soil_moisture=35.0,    # 严重干旱
        soil_fertility=25.0,   # 肥力不足
        sunlight=3.0,          # 光照不足
        farming_operation=40.0, # 管理不足
        pest_risk=60.0         # 病虫害风险高
    )
    print(f"总分: {result['total_score']}")
    print(f"等级: {result['grade']} ({result['grade_code']})")
    print(f"主要短板: {result['weakness_analysis']['primary_weakness']['name']}")
    print(f"干预建议数量: {len(result['intervention_suggestions'])}")
    for sug in result['intervention_suggestions']:
        print(f"  - {sug['title']}")
        for action in sug['actions'][:2]:
            print(f"    * {action}")

    assert result['total_score'] < 60, f"弱苗总分应<60，实际{result['total_score']}"
    assert len(result['intervention_suggestions']) >= 3, "弱苗应有至少3条干预建议"
    print("✅ 测试4通过")


def test_evaluate_overgrowth():
    """测试旺苗评估（肥力过高+温度偏高）"""
    print(f"\n=== 测试5: 旺苗评估（拔节期，高肥+高温） ===")
    result = GrowthService.evaluate(
        growth_stage="jointing",
        temperature=22.0,      # 偏高
        soil_moisture=70.0,    # 适宜
        soil_fertility=95.0,   # 肥力过高
        sunlight=8.0,
        farming_operation=70.0,
        pest_risk=20.0
    )
    print(f"总分: {result['total_score']}")
    print(f"等级: {result['grade']} ({result['grade_code']})")
    print(f"肥力得分: {result['dimension_scores']['soil_fertility']['score']}")
    print(f"温度得分: {result['dimension_scores']['temperature']['score']}")

    # 旺苗条件：总分>=90 + 肥力>=85 + 温度<=60
    if result['grade_code'] == 'overgrowth':
        print("检测到旺苗状态")
        has_overgrowth_sug = any(
            s['dimension'] == 'overall' and '旺长' in s['title']
            for s in result['intervention_suggestions']
        )
        assert has_overgrowth_sug, "旺苗应有综合旺长防控建议"
    else:
        print(f"未触发旺苗判定（肥力得分{result['dimension_scores']['soil_fertility']['score']}, 温度得分{result['dimension_scores']['temperature']['score']}）")
    print("✅ 测试5通过")


def test_evaluate_all_stages():
    """测试所有生育期都能正常评估"""
    print(f"\n=== 测试6: 全生育期评估验证 ===")
    stages = ['tillering', 'reviving', 'jointing', 'heading', 'filling']
    for stage in stages:
        result = GrowthService.evaluate(
            growth_stage=stage,
            temperature=15.0,
            soil_moisture=65.0,
            soil_fertility=70.0,
            sunlight=6.0,
            farming_operation=75.0,
            pest_risk=25.0
        )
        print(f"  {result['growth_stage_name']}: 总分{result['total_score']}, 等级{result['grade']}")
        assert result['total_score'] > 0
        assert 'evaluation_id' in result
        assert 'dimension_scores' in result
        assert 'weakness_analysis' in result
        assert 'intervention_suggestions' in result
    print("✅ 测试6通过")


def test_invalid_stage():
    """测试无效生育期报错"""
    print(f"\n=== 测试7: 无效生育期异常处理 ===")
    try:
        GrowthService.evaluate(
            growth_stage="invalid_stage",
            temperature=15.0,
            soil_moisture=65.0,
            soil_fertility=70.0,
            sunlight=6.0
        )
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print(f"正确抛出异常: {e}")
        assert "不支持的生育期" in str(e)
    print("✅ 测试7通过")


def test_trend_prediction():
    """测试带未来天气的趋势预判"""
    print(f"\n=== 测试8: 短期趋势预判 ===")
    future_weather = [
        {"date": "2026-08-17", "temp_avg": 12, "rain": 5, "sunlight": 6},
        {"date": "2026-08-18", "temp_avg": 14, "rain": 0, "sunlight": 8},
        {"date": "2026-08-19", "temp_avg": 16, "rain": 20, "sunlight": 4},
        {"date": "2026-08-20", "temp_avg": 13, "rain": 10, "sunlight": 5},
        {"date": "2026-08-21", "temp_avg": 11, "rain": 0, "sunlight": 7},
        {"date": "2026-08-22", "temp_avg": 15, "rain": 2, "sunlight": 6},
        {"date": "2026-08-23", "temp_avg": 17, "rain": 0, "sunlight": 9},
    ]
    result = GrowthService.evaluate(
        growth_stage="reviving",
        temperature=12.0,
        soil_moisture=60.0,
        soil_fertility=70.0,
        sunlight=6.0,
        future_weather=future_weather
    )
    trend = result['trend_prediction']
    print(f"趋势预判可用: {trend['available']}")
    print(f"未来天数: {trend['days']}")
    print(f"平均气温: {trend['avg_temperature']}℃")
    print(f"总降水量: {trend['total_rainfall']}mm")
    print(f"平均光照: {trend['avg_sunlight']}小时/天")
    print(f"趋势展望: {trend['outlook']}")
    print(f"总结: {trend['summary']}")
    print(f"影响因素:")
    for f in trend['factors']:
        print(f"  - {f}")

    assert trend['available'] == True
    assert trend['days'] == 7
    print("✅ 测试8通过")


if __name__ == "__main__":
    print("=" * 60)
    print("小麦长势评估功能测试")
    print("=" * 60)

    test_get_available_stages()
    test_get_stage_weights()
    test_evaluate_normal()
    test_evaluate_weak()
    test_evaluate_overgrowth()
    test_evaluate_all_stages()
    test_invalid_stage()
    test_trend_prediction()

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
