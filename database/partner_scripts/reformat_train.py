"""
reformat_train.py — 将 NER 逐字标注的 train.txt 整理为人类可读格式

输入格式（逐字标注）：
    病 O
    粒 O
    稻 B-Disease
    曲 I-Disease
    病 I-Disease

输出格式（整句 + 实体标注表格）：
    【句子1】病粒以穗的中下部为主，上部次之；...
    ┌──────────┬────────────┐
    │ 实体文本   │ 实体类型    │
    ├──────────┼────────────┤
    │ 稻曲病     │ Disease    │
    └──────────┴────────────┘
"""

import os

# ========== 路径配置 ==========
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(_PROJECT_ROOT, "data", "train.txt")
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "data", "train_formatted.txt")

# ========== 实体类型中文映射 ==========
TYPE_CN = {
    "Disease":  "病害",
    "Approach": "防治方法",
    "Symptom":  "症状",
    "Area":     "地区",
    "Drug":     "药物",
    "Rice":     "水稻品种",
    "Pest":     "虫害",
}


def parse_ner_file(filepath: str) -> list[dict]:
    """
    解析 NER 标注文件，返回句子列表。
    每个句子结构：
    {
        "text": "完整文本",
        "entities": [{"text": "稻曲病", "type": "Disease", "start": 3, "end": 6}, ...]
    }
    """
    sentences = []
    current_chars = []   # 当前句子的字符列表
    current_labels = []  # 当前句子的标签列表

    def _flush():
        """将当前累积的字符和标签组装为一个句子"""
        if not current_chars:
            return

        text = "".join(current_chars)
        entities = []
        i = 0
        while i < len(current_labels):
            label = current_labels[i]
            if label.startswith("B-"):
                entity_type = label[2:]
                start = i
                j = i + 1
                # 连续的 I- 标签属于同一实体
                while j < len(current_labels) and current_labels[j] == f"I-{entity_type}":
                    j += 1
                entities.append({
                    "text": text[start:j],
                    "type": entity_type,
                    "start": start,
                    "end": j,
                })
                i = j
            else:
                i += 1

        sentences.append({"text": text, "entities": entities})
        current_chars.clear()
        current_labels.clear()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.strip() == "":
                # 空行表示句子分隔
                _flush()
            else:
                parts = line.strip().split()
                if len(parts) >= 2:
                    current_chars.append(parts[0])
                    current_labels.append(parts[1])
                elif len(parts) == 1:
                    current_chars.append(parts[0])
                    current_labels.append("O")

    # 文件末尾可能没有空行，也要 flush
    _flush()
    return sentences


def format_sentences(sentences: list[dict]) -> str:
    """
    将句子列表格式化为可读文本。
    含实体的句子用表格展示，无实体的句子只显示原文。
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  NER 标注数据 — 可读格式")
    lines.append(f"  共 {len(sentences)} 个句子")
    lines.append("=" * 80)
    lines.append("")

    # 统计实体类型分布
    type_count = {}
    total_entities = 0
    for s in sentences:
        for e in s["entities"]:
            t = e["type"]
            type_count[t] = type_count.get(t, 0) + 1
            total_entities += 1

    lines.append("【统计概览】")
    lines.append(f"  总句数：{len(sentences)}")
    lines.append(f"  总实体数：{total_entities}")
    lines.append(f"  实体类型分布：")
    for t, cnt in sorted(type_count.items(), key=lambda x: -x[1]):
        cn = TYPE_CN.get(t, t)
        lines.append(f"    {t}（{cn}）：{cnt} 个")
    lines.append("")
    lines.append("-" * 80)

    for idx, sent in enumerate(sentences, 1):
        # 句子编号 + 原文
        lines.append(f"")
        lines.append(f"【句子 {idx}】{sent['text']}")

        if not sent["entities"]:
            lines.append("  （无实体标注）")
        else:
            # 绘制实体表格
            entities = sent["entities"]
            # 计算列宽
            max_text = max(len(e["text"]) for e in entities)
            max_type = max(len(f"{e['type']}（{TYPE_CN.get(e['type'], e['type'])}）") for e in entities)
            col1_w = max(max_text, 6) + 2   # "实体文本" = 4，留点余量
            col2_w = max(max_type, 8) + 2

            # 表头
            lines.append(f"  ┌{'─' * col1_w}┬{'─' * col2_w}┐")
            lines.append(f"  │{'实体文本'.center(col1_w)}│{'实体类型'.center(col2_w)}│")
            lines.append(f"  ├{'─' * col1_w}┼{'─' * col2_w}┤")

            for e in entities:
                cn = TYPE_CN.get(e["type"], e["type"])
                type_str = f"{e['type']}（{cn}）"
                lines.append(f"  │{e['text'].center(col1_w)}│{type_str.center(col2_w)}│")

            lines.append(f"  └{'─' * col1_w}┴{'─' * col2_w}┘")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  文件结束")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    print(f"正在读取: {INPUT_FILE}")
    sentences = parse_ner_file(INPUT_FILE)
    print(f"解析完成，共 {len(sentences)} 个句子")

    result = format_sentences(sentences)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"已输出至: {OUTPUT_FILE}")
    print(f"文件大小: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
