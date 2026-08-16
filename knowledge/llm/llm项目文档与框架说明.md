---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 4c23526d26232c099b5cf898a4d8e386_c000973d991011f1a98a525400f8a581
    ReservedCode1: jEIfHB/J9pemvBnNA7d8z+QCBX9srzUhBwGO4VDg54pKJtyQ+ZBT/w7tNvIVZwIZCh1EV4HpBkKJVE0JKr1sc3rVDZ2Iw5knSTALLE9ZVDSX3OO/YzMzlqOALVt5G+wX6wUjuO0ZMPi5wHlb1EpOtZPrMSmxVK3V0gV74MsBP+NN4Nfvj6aD9WnS1/k=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 4c23526d26232c099b5cf898a4d8e386_c000973d991011f1a98a525400f8a581
    ReservedCode2: jEIfHB/J9pemvBnNA7d8z+QCBX9srzUhBwGO4VDg54pKJtyQ+ZBT/w7tNvIVZwIZCh1EV4HpBkKJVE0JKr1sc3rVDZ2Iw5knSTALLE9ZVDSX3OO/YzMzlqOALVt5G+wX6wUjuO0ZMPi5wHlb1EpOtZPrMSmxVK3V0gV74MsBP+NN4Nfvj6aD9WnS1/k=
---



# LLM 项目文档与框架说明

> 项目路径：`E:\itcast\05-toumanfen\03all_code\000code\llm`
> 文档生成日期：2026-08-16
> 项目类型：基于 LlamaIndex + Chroma 的本地知识库 RAG（检索增强生成）问答系统

---

## 1. 项目概述

本项目是一个**面向农业害虫/病害知识的中文 RAG 问答系统**，采用"本地文档知识库 + 大模型生成"的架构：

- **知识库构建**：将 `chroma/docs` 目录下的 PDF / DOCX / TXT / MD / DOC 文档解析、分块、向量化后持久化到 Chroma 向量数据库，同时缓存全量文本节点供 BM25 关键词检索。
- **混合检索**：向量检索（语义相似）+ BM25 关键词检索（中文 jieba 分词）按权重 `ALPHA` 融合排序。
- **生成回答**：调用阿里云百炼（DashScope）OpenAI 兼容接口（`qwen-plus` 对话 + `text-embedding-v3` 向量），基于检索上下文流式生成答案。
- **联网补充**：集成博查 Web Search，当知识库命中时可附加联网结果作为补充上下文；知识库未命中时降级为纯联网回答。
- **服务形态**：提供 CLI（`ingest` / `chat` / `ask`）与 FastAPI Web 服务（`/api/v1/kb/search`）两种入口，支持 docs 目录自动增量入库。

当前知识库中已入库的文档为小麦病虫害相关资料（虫害一栏表、虫害信息、主要虫害防治措施文件等），属农业病虫害咨询场景。

---

## 2. 目录结构

```
llm/
├── agent/                          # 智能体/聊天引擎层
│   ├── __init__.py
│   └── chat_engine.py              # 流式聊天：检索→拼装Prompt→百炼流式生成
├── chroma/                         # 数据目录（运行时生成）
│   ├── docs/                       # ① 原始文档目录（用户放置 pdf/txt/md/docx/doc）
│   │   ├── 主要虫害防措施文件.txt
│   │   ├── 虫害一栏表.md
│   │   ├── 虫害一栏表.pdf
│   │   └── 虫害信息.docx
│   ├── index_storage/              # ② Chroma 向量索引持久化目录（含 nodes.pkl）
│   │   ├── chroma.sqlite3
│   │   ├── nodes.pkl               # 全量节点缓存（供 BM25 检索）
│   │   └── <collection>/           # Chroma HNSW 索引二进制文件
│   └── md5_files/                  # ③ MD5 文件存储目录（去重/溯源）
│       ├── md5_index.json          # MD5 台账（md5 → 原始名/存储路径/大小/时间）
│       └── <md5>.<ext>             # 按内容 MD5 命名的文件副本
├── chroma_loader/                  # 文档解析层
│   ├── __init__.py
│   └── document_loader.py          # 多格式解析：pdf/txt/md/docx/doc → Document
├── core/                           # 核心业务层
│   ├── __init__.py
│   ├── bocha_search.py             # 博查 Web Search 联网搜索
│   ├── hybrid_retriever.py         # 混合检索器（向量+BM25 加权融合）
│   └── ingest.py                   # 文档入库：解析→去重→分块→向量化→持久化
├── docs/                           # 空目录（备用）
├── logger/                         # 日志目录（运行时生成）
│   ├── all.log                     # 全量日志
│   ├── debug.log / info.log / warning.log / error.log   # 分等级日志
├── prompt/                         # Prompt 资源目录
│   ├── system_prompt.txt           # 系统提示词（角色定义+回答格式约束）
│   └── prompt_template             # 上下文拼装模板（{context} + {query}）
├── server/                         # Web 服务层
│   ├── __init__.py
│   └── app.py                      # FastAPI 知识库检索服务（含自动增量更新）
├── util/                           # 工具层
│   ├── __init__.py
│   ├── config.py                   # 配置模块（主入口，读取 api_keys.yaml）
│   ├── load_tool.py                # 旧版 YAML 加载工具
│   ├── logger_pathlib.py           # loguru 日志配置（pathlib 路径版）
│   ├── md5_store.py                # MD5 去重存储 + 台账
│   └── path_tool.py                # 旧版路径工具（基于 cwd，已基本废弃）
├── yaml/                           # 配置文件目录
│   ├── api_keys.yaml               # ★ 主配置文件（全项目唯一生效的模型配置入口）
│   ├── agent.yaml                  # 空配置文件（占位）
│   ├── chroma.yaml                 # 旧配置（部分字段，当前未使用）
│   └── rag_dashscope.yaml          # 备用副本配置（当前未使用）
├── main.py                         # CLI 入口（ingest/chat/ask）
├── requirements.txt                # Python 依赖清单
└── __init__.py
```

---

## 3. 模块划分与职责

| 模块 | 核心文件 | 职责 |
|------|---------|------|
| **入口层** | `main.py` | CLI 命令分发：`ingest` 入库、`chat` 交互聊天、`ask "问题"` 单次提问 |
| **聊天引擎** | `agent/chat_engine.py` | 检索结果格式化为带来源标注的上下文，组装 Prompt，调用百炼模型流式生成；维护多轮对话历史 |
| **文档解析** | `chroma_loader/document_loader.py` | 支持 pdf（pymupdf→pypdf 兜底）、txt/md（多编码探测）、docx（段落+表格）、doc（Word COM）；扫描目录时过滤 Office 临时文件（`~$`） |
| **入库模块** | `core/ingest.py` | 主流程：MD5 去重 → 解析 → `SentenceSplitter` 分块 → 写入 Chroma → 缓存 nodes.pkl；提供 `has_new_docs()` 轻量增量检测 |
| **混合检索** | `core/hybrid_retriever.py` | 向量检索（Chroma）与 BM25 检索各取 TOP_K×3 候选，内部归一化后按 `ALPHA` 加权融合排序取 TOP_K |
| **联网搜索** | `core/bocha_search.py` | 调用博查 Web Search API，将结果格式化为「标题+链接+摘要」文本供模型参考 |
| **Web 服务** | `server/app.py` | FastAPI 接口：健康检查、检索问答；启动时增量入库 + 后台每 60s 轮询 docs 目录自动更新；查询前即时检测新文档 |
| **配置模块** | `util/config.py` | 基于 `__file__` 定位项目根目录（不依赖 cwd），读取 yaml 配置并导出全部配置常量，自动创建数据目录 |
| **MD5 去重** | `util/md5_store.py` | 按文件内容 MD5 去重：以 `<md5><扩展名>` 命名存档，维护 `md5_index.json` 台账，支持溯源与增量判断 |
| **日志模块** | `util/logger_pathlib.py` | loguru 配置：控制台彩色输出 + all.log 全量 + 按等级分文件输出（10MB 轮转，保留 7 天） |
| **Prompt 资源** | `prompt/` | 系统提示词（中文回答、格式规范、示例）与上下文拼装模板 |

---

## 4. 核心代码逻辑

### 4.1 整体数据流

```
┌─────────────────────────────── 入库流程（ingest） ───────────────────────────────┐
│                                                                                  │
│  scan_docs() ──► md5_store.store_file() ──► document_loader.load_file()          │
│  扫描文档目录       (MD5去重，重复跳过)        解析为 Document(+md5元数据)        │
│                                                                                  │
│  SentenceSplitter ──► VectorStoreIndex ──► Chroma(PERSIST_DIR) ──► nodes.pkl    │
│  分块(512/64)         向量化+写库                 索引持久化            节点缓存   │
└──────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── 问答流程（chat/ask/search） ──────────────────────┐
│                                                                                  │
│  HybridRetriever.retrieve(query)                                                 │
│    ├─ 向量检索(Chroma) ──┐                                                       │
│    └─ BM25检索(jieba分词) ┴─► 归一化 ──► ALPHA加权融合 ──► TOP_K 片段            │
│                                                                                  │
│  format_context(nodes) ──► 带来源标注的上下文                                    │
│  bocha_search(query)      （可选）联网结果附加为补充上下文                        │
│  PROMPT_TEMPLATE.format(context, query) ──► 组装 user prompt                    │
│  build_llm() ──► OpenAILike(qwen-plus) ──► stream_chat 流式输出                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 入库流程（`core/ingest.py`）

1. `scan_docs()` 递归扫描 `DOCS_DIR` 下所有支持格式（pdf/txt/md/markdown/docx/doc），排除 `~$` 开头的 Office 临时文件。
2. 对每个文件计算内容 MD5，经 `md5_store.store_file()` 去重：
   - 台账中已存在 → 跳过（不重复解析入库）；
   - 新文件 → 以 `<md5><扩展名>` 命名复制到 `MD5_DIR` 并登记台账。
3. 用 `SentenceSplitter(chunk_size=512, chunk_overlap=64)` 将 Document 切分为节点，metadata 中写入 `md5`。
4. `load_or_create_index(nodes)` 构建 `VectorStoreIndex` 写入 Chroma（collection 名 `rag_docs`）。
5. `_cache_nodes()` 将新旧节点合并后覆盖写回 `nodes.pkl`（BM25 检索的数据源）。
6. `has_new_docs()` 提供轻量增量检测：只算新文件 MD5 与台账比对，不做完整解析。

### 4.3 混合检索（`core/hybrid_retriever.py`）

- 候选集：向量检索器与 BM25 检索器各取 `TOP_K * 3`（至少 10 条）候选。
- 归一化：各自内部 `score / max_score`，消除向量分与 BM25 分的量纲差异。
- 融合：`fused_score = ALPHA × vector_score + (1 - ALPHA) × bm25_score`，默认 `ALPHA = 0.5`。
- 排序：按融合分降序取 `TOP_K`（默认 5）条返回。
- 中文分词：优先使用 jieba `cut_for_search` 搜索模式；未安装时降级为空白分词并告警。

### 4.4 问答生成（`agent/chat_engine.py`）

1. 加载 `system_prompt.txt`（系统提示词）与 `prompt_template`（上下文模板），路径基于 `__file__` 定位，不依赖 cwd。
2. 检索片段经 `format_context()` 格式化为 `[序号] 来源: 文件名 (md5: xxx)\n内容` 形式，来源信息随上下文透传给模型。
3. 多轮聊天时，历史（最近 10 轮）以 User/Assistant 消息追加进 messages。
4. `build_llm()` 构建 `OpenAILike`（阿里百炼兼容端点），`stream_chat()` 逐 token 产出回答。

### 4.5 Web 服务（`server/app.py`）

基于 FastAPI + uvicorn 的 HTTP 知识库检索服务，统一返回 `{"code", "status", "msg", "data"}` 四段式结构。

#### 接口一览

| 方法 | 路由 | 功能 | 请求体 | 响应 data |
|------|------|------|--------|-----------|
| GET | `/` | 服务状态探测 | 无 | `null`（msg 为"服务器加载完成"） |
| GET | `/api/v1/kb/health` | 健康检查（索引就绪状态 + 模型信息） | 无 | `{"model", "embed_model"}`；未就绪时 code=1 |
| POST | `/api/v1/kb/search` | 知识库检索问答（含联网补充） | `{"query": "..."}` | `{"answer", "sources"[], "web_search"}` |

#### 请求参数（POST /api/v1/kb/search）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 检索问题，如 `"小麦蚜虫是什么病害"`；空字符串返回 code=1 |

#### 响应格式

- 成功：`{"code": "0", "status": "success", "msg": "success", "data": {...}}`
- 业务失败（参数/索引未就绪）：`{"code": "1", "status": "fail", "msg": "...", "data": null}`
- 服务异常：`{"code": "2", "status": "fail", "msg": "检索异常: <错误信息>", "data": null}`

`data` 字段明细：

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 大模型生成的回答文本 |
| `sources` | array | 命中片段来源列表，每项含 `file_name`（文档名）、`md5`（内容指纹）、`score`（融合相似度，保留 4 位小数）；知识库无命中时缺省 |
| `web_search` | bool | 是否附加了博查联网搜索结果 |

#### 启动方式

```bash
# 在 000code 目录下运行
python -m uvicorn llm.server.app:app --host 0.0.0.0 --port 8000
```

启动时自动完成三件事：① 若开启 `AUTO_UPDATE` 先执行一次增量入库；② 后台任务每 `AUTO_UPDATE_INTERVAL`（默认 60s）轮询 docs 目录，发现新文档自动入库并重置检索器；③ 检索器懒加载——首次请求时构建（索引不存在则报错提示先 ingest）。

- **自动更新机制**：
  - 启动时（lifespan）先执行一次增量入库；
  - 后台任务每 `AUTO_UPDATE_INTERVAL`（默认 60s）轮询 docs 目录，发现新文档即增量入库并重置检索器；
  - 每次查询前轻量检测新文档，保证手动放文件后无需等待轮询即可命中。
- **联网降级策略**：博查搜索失败不阻塞主流程；知识库无命中且联网无结果时返回提示文案。

### 4.6 文档解析细节（`chroma_loader/document_loader.py`）

| 格式 | 解析方式 | 备注 |
|------|---------|------|
| txt / md | 自动探测编码 utf-8 → gbk → gb18030 → latin-1 | 兜底 errors=ignore |
| pdf | pymupdf（fitz）优先，失败退 pypdf | 提取纯文本 |
| docx | python-docx：段落 + 表格（表格行内单元格以 `\|` 连接） | |
| doc | Word COM（win32com） | 需本机安装 Word |

---

## 5. 配置文件说明

### 5.1 主配置：`yaml/api_keys.yaml`

全项目**唯一生效**的模型配置入口（`util/config.py` 统一读取），包含：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DASHSCOPE_API_KEY` | （实际密钥） | 阿里云百炼 API Key，聊天 + Embedding 共用 |
| `CHAT_BASE_URL` / `CHAT_MODEL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` / `qwen-plus` | 对话模型（OpenAI 兼容端点） |
| `EMBED_BASE_URL` / `EMBED_MODEL` | 同上 / `text-embedding-v3` | Embedding 模型 |
| `EMBEDDING_DIM` | 1024 | 向量维度 |
| `TEMPERATURE` | 0.3 | 采样温度 |
| `BOCHA_API_KEY` | （实际密钥） | 博查 Web Search Key |
| `BOCHA_SEARCH_COUNT` | 5 | 博查返回条数 |
| `BOCHA_FRESHNESS` | `noLimit` | 搜索时间范围（oneDay/oneWeek/oneMonth/oneYear/noLimit） |
| `DOCS_DIR` | `chroma/docs` | 原始文档目录 |
| `PERSIST_DIR` | `chroma/index_storage` | Chroma 索引持久化目录 |
| `MD5_DIR` | `chroma/md5_files` | MD5 文件存储目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 512 / 64 | 分块大小与重叠 |
| `ALPHA` | 0.5 | 混合检索权重（向量占比，BM25 为 1-ALPHA） |
| `TOP_K` | 5 | 最终返回片段数 |
| `AUTO_UPDATE` | true | 是否自动增量入库 |
| `AUTO_UPDATE_INTERVAL` | 60 | 轮询间隔（秒） |

### 5.2 其他配置文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `yaml/chroma.yaml` | 未使用（旧配置） | 仅含部分检索/分块参数与测试字段 `lol` |
| `yaml/rag_dashscope.yaml` | 未使用（备用副本） | 与 api_keys.yaml 内容基本一致的副本 |
| `yaml/agent.yaml` | 空文件 | 占位 |
| `util/load_tool.py` | 遗留工具 | 旧版 YAML 加载函数（与 config.py 并存，新代码不使用） |
| `util/path_tool.py` | 遗留工具 | 旧版基于 cwd 的路径工具，新代码已改为基于 `__file__` |

> 注意：配置文件中包含真实的 API Key 明文，请勿将项目目录提交到公开仓库或泄露。

---

## 6. 依赖清单（requirements.txt）

核心框架为 **LlamaIndex 0.14.x** + **ChromaDB**，运行环境为 Python 3.14（conda env: agentchat）：

- **LLM 框架**：llama-index-core、llama-index-embeddings-openai、llama-index-llms-openai-like、llama-index-retrievers-bm25、llama-index-vector-stores-chroma
- **向量数据库**：chromadb 1.5.9
- **Web 服务**：fastapi、uvicorn
- **数据/校验**：pydantic 2.13.4
- **日志/配置**：loguru 0.7.3、PyYAML 6.0.3
- **文档解析**：pymupdf、pypdf、python-docx、pywin32
- **检索/分词**：jieba（BM25 中文分词）
- **网络**：requests、openai（OpenAI 兼容客户端）

---

## 7. 使用方式

```bash
# 在 000code 目录下运行（模块化导入依赖项目根目录）

# 1. 文档入库（首次使用必须执行）
python -m llm.main ingest

# 2. 交互式流式聊天
python -m llm.main chat

# 3. 单次提问（流式输出）
python -m llm.main ask "小麦蚜虫是什么病害"

# 4. 启动 FastAPI Web 服务（0.0.0.0:8000）
python -m uvicorn llm.server.app:app --host 0.0.0.0 --port 8000
```

### 使用流程

1. 将需要检索的文档（pdf/txt/md/docx/doc）放入 `chroma/docs` 目录；
2. 执行 `ingest` 入库（或启动 Web 服务由自动更新机制代劳）；
3. 通过 CLI `chat` / `ask` 或 Web 接口 `/api/v1/kb/search` 发起问答。

### API 调用示例

```bash
curl -X POST http://127.0.0.1:8000/api/v1/kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "小麦蚜虫是什么病害"}'
```

返回结构：

```json
{
  "code": "0",
  "status": "success",
  "msg": "success",
  "data": {
    "answer": "经知识库检索得：...",
    "sources": [
      {"file_name": "虫害信息.docx", "md5": "ae370620...", "score": 0.82}
    ],
    "web_search": true
  }
}
```

---

## 8. 对话结果示例

以下示例基于项目实际的问答链路构造（示例问题"小麦赤霉病怎么防治"为运行日志中的真实查询，命中来源与 MD5 均取自实际索引台账），演示 CLI 与 Web 两种入口的完整交互。

### 8.1 CLI 入口示例（`python -m llm.main ask`）

```
$ python -m llm.main ask "小麦赤霉病怎么防治"

检索到 3 个相关片段：
  [1] 小麦主要病虫害知识库.md (md5=c886185a13422996cceab96ccc5c29e8, score=0.912)
  [2] 小麦主要病虫害知识库.txt (md5=0ab3b6286aeabd4beba97de4c559ab72, score=0.887)
  [3] 小麦主要病虫害知识库.pdf (md5=e444d1c42e7565c2f04db80ac9977116, score=0.854)

回答：经知识库检索得：
小麦赤霉病是由禾谷镰刀菌（Fusarium graminearum）等多种镰刀菌引起的真菌性穗部病害，是我国小麦最重要的病害之一，素有"小麦癌症"之称，其产生的 DON 毒素（呕吐毒素）会严重威胁人畜健康。该病主要危害穗部，扬花期侵染后初期在颖壳上出现水渍状褐色小斑，后期病穗产生黑色小颗粒（子囊壳），病粒皱缩、灰白或粉红色，称"赤霉粒"。防治上需坚持"见花打药"原则，在小麦齐穗至扬花初期（扬花 5%~10%）喷药，可选用氰烯菌酯、戊唑醇、咪鲜胺、丙硫菌唑、氟唑菌酰羟胺等药剂并注意轮换用药；同时配合农业防治，选用抗（耐）赤霉病品种、清除田间病残体、深翻灭茬、合理排灌降低田间湿度；遇阴雨天气时，雨后 3~5 天内及时补喷。
经博查联网搜索得：
博查搜索补充了赤霉病防治药剂的具体用量与最佳防治时期等联网信息（示例）。

$
```

交互式聊天（`python -m llm.main chat`）输出结构与上例相同，区别在于：① 每轮先打印检索到的引用片段（文件名 + md5 + 融合分）；② 回答逐 token 流式输出；③ 多轮对话历史（最近 10 轮）会以 User/Assistant 消息追加进模型上下文，支持追问。

### 8.2 Web 入口示例（`POST /api/v1/kb/search`）

```bash
curl -X POST http://127.0.0.1:8000/api/v1/kb/search \
  -H "Content-Type: application/json" \
  -d '{"query": "小麦赤霉病怎么防治"}'
```

真实响应（结构示例，知识库命中 + 联网搜索补充）：

```json
{
  "code": "0",
  "status": "success",
  "msg": "success",
  "data": {
    "answer": "经知识库检索得：\n小麦赤霉病是由禾谷镰刀菌（Fusarium graminearum）等多种镰刀菌引起的真菌性穗部病害，是我国小麦最重要的病害之一，素有\"小麦癌症\"之称。该病主要危害穗部，扬花期侵染后颖壳出现水渍状褐色小斑，后期病穗产生黑色小颗粒（子囊壳），病粒皱缩、灰白或粉红色。防治坚持\"见花打药\"，齐穗至扬花初期（扬花 5%~10%）选用氰烯菌酯、戊唑醇、咪鲜胺等药剂喷防，并配合抗病品种、清除病残体等农业防治措施，雨后 3~5 天内及时补喷。\n经博查联网搜索得：\n博查搜索补充了赤霉病防治的药剂用量与防治时期等联网信息（示例）。",
    "sources": [
      {
        "file_name": "小麦主要病虫害知识库.md",
        "md5": "c886185a13422996cceab96ccc5c29e8",
        "score": 0.9120
      },
      {
        "file_name": "小麦主要病虫害知识库.txt",
        "md5": "0ab3b6286aeabd4beba97de4c559ab72",
        "score": 0.8870
      },
      {
        "file_name": "小麦主要病虫害知识库.pdf",
        "md5": "e444d1c42e7565c2f04db80ac9977116",
        "score": 0.8540
      }
    ],
    "web_search": true
  }
}
```

> 说明：上述 `sources` 中的文件名与 MD5 取自实际 MD5 台账（`chroma/md5_files/md5_index.json`），与 docs 目录中"虫害一栏表.md / 虫害信息.docx"等当前文件名对应同一批入库内容（入库后源文件被重命名）。`answer` 中"经知识库检索得 / 经博查联网搜索得"格式由 `prompt/system_prompt.txt` 约束。`score` 为 `ALPHA=0.5` 下向量分与 BM25 分归一化融合后的相似度。

### 8.3 异常响应示例

```json
// 索引未初始化时（未执行 ingest）
{"code": "1", "status": "fail", "msg": "知识库索引未初始化，请先 ingest 入库", "data": null}

// query 为空
{"code": "1", "status": "fail", "msg": "query 不能为空", "data": null}

// 知识库与联网均无结果
{"code": "0", "status": "success", "msg": "success",
 "data": {"answer": "知识库中没有检索到相关内容，联网搜索也未获取到结果，请换一个问题试试。"}}

// 服务内部异常
{"code": "2", "status": "fail", "msg": "检索异常: <异常详情>", "data": null}
```

## 9. 关键设计要点与注意事项

1. **路径自洽**：所有模块基于 `__file__` 定位项目根目录，不依赖当前工作目录（旧版 `path_tool.py` 的 cwd 方案已废弃），`python -m` 方式运行必须从 `000code` 目录执行以保证模块导入。
2. **去重与增量**：MD5 台账实现内容级去重（同一内容不同文件名只入库一次）；`AUTO_UPDATE` 开启时新文档自动入库，查询前也有即时检测。
3. **来源可溯源**：每个入库文件记录 `md5` 元数据，检索结果返回文件名 + md5 + 相似度分数，回答引用可追溯。
4. **降级链**：文档解析 pymupdf→pypdf；BM25 分词 jieba→空白分词；联网搜索失败不阻塞主流程；知识库无命中时降级纯联网回答。
5. **Embedding 参数**：`build_embed_model()` 使用 OpenAIEmbedding 的占位模型名通过构造校验，以 `model_name` 覆盖为百炼实际模型；`embed_batch_size=10` 适配百炼兼容端点 batch 限制。
6. **配置收敛**：模型参数与 Key 全部收敛于 `yaml/api_keys.yaml`，禁止散落硬编码；`chroma.yaml` / `rag_dashscope.yaml` 为遗留文件，建议后续清理。

---

## 10. 当前知识库状态

- 已入库文档（`chroma/docs`）：虫害一栏表（md/pdf）、虫害信息（docx）、主要虫害防治措施文件（txt），共 4 个源文件（其中 md 与 pdf 内容对应同一文档，MD5 台账会分别记录）。
- 索引已构建：`chroma/index_storage` 下存在 Chroma SQLite + HNSW 二进制索引及 `nodes.pkl` 节点缓存。
- 日志记录至 2026-08-16 09:09，项目最近有实际运行。

