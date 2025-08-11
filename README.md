### RAG-ASR 使用说明

一个用于 ASR 候选文本纠错/润色的简易 RAG 流水线：
- 输入 `query`（ASR 候选文本）
- 从关键词数据库检索最相似的 Top-K 关键词
- 将 `query` 与关键词构造成 Prompt，交给 LLM 生成更好的文本

本仓库已实现：
- 检索器：`smith_waterman`（默认）、`bm25`、`embedding`
- LLM：`openai`/兼容、`ollama`
- 数据库格式：`.txt`、`.csv`、`.jsonl`

---

### 安装

建议使用 Python 3.9+，先创建虚拟环境：

```bash
cd /Users/zslin/Documents/mypaper/rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Smith-Waterman（默认检索器）需要 `biopython`
- `bm25` 检索需要：`rank-bm25`
- `embedding` 检索需要：`sentence-transformers`、`numpy`、`torch`

如果你需要可选检索器：
```bash
pip install rank-bm25 sentence-transformers numpy torch
```

---

### 数据库格式

支持三种格式，均为“每条记录一个关键词”。
- `.txt`：每行一个关键词（推荐）
- `.csv`：包含 `keyword` 列，或默认读取第一列
- `.jsonl`：每行一个 JSON 对象，包含 `keyword` 字段

示例：`data/keywords.txt`
```
weather
show me weather
fuxing road
复兴路
天下第一
天气
路线导航
地铁站
最近的医院
取消导航
```

---

### 快速体验（仅检索与 Prompt 预览，不调用 LLM）

```bash
python rag_asr_main.py \
  --query "我想去付兴路呀" \
  --database ./data/keywords.txt \
  --retriever smith_waterman \
  --top-k 5 \
  --dry-run \
  --show-scores
```

- `--dry-run`：仅打印检索结果与构造的 Prompt，不调用任何 LLM
- `--show-scores`：打印检索到的关键词与相似度分数

---

### 真实调用（OpenAI 或兼容）

```bash
export OPENAI_API_KEY=YOUR_KEY
python rag_asr_main.py \
  --query "wo xiang qu fuxing lu ya" \
  --database ./data/keywords.txt \
  --retriever smith_waterman \
  --top-k 5 \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --show-scores
```

OpenAI 兼容服务（自定义 `--base-url` 和 `--api-key`）：
```bash
python rag_asr_main.py \
  --query "我想去付兴路呀" \
  --database ./data/keywords.txt \
  --retriever bm25 \
  --llm-provider openai \
  --llm-model qwen2.5 \
  --base-url http://localhost:8000/v1 \
  --api-key sk-xxxx
```

---

### 真实调用（Ollama 本地）

确保本地 `ollama` 服务可用并已拉取模型：
```bash
ollama run qwen2:latest
```
然后运行：
```bash
python rag_asr_main.py \
  --query "show me the wether" \
  --database ./data/keywords.txt \
  --retriever embedding \
  --llm-provider ollama \
  --llm-model qwen2:latest \
  --show-scores
```

---

### 命令行参数

- `--query`：ASR 候选文本（必填）
- `--database`：关键词数据库路径，默认 `./data/keywords.txt`
- `--retriever`：检索器，`smith_waterman|bm25|embedding`（默认 `smith_waterman`）
- `--top-k`：检索返回的关键词数量，默认 `5`
- `--max-keywords-in-prompt`：插入到 Prompt 的关键词最大个数，可用于截断
- `--show-scores`：显示检索分数
- `--dry-run`：只展示 Prompt，不调用 LLM

- `--llm-provider`：`openai|openai_compat|ollama`（`openai_compat` 与 `openai` 用法一致）
- `--llm-model`：LLM 模型名称（例如 `gpt-4o-mini` 或 `qwen2:latest`）
- `--api-key`：OpenAI 或兼容服务的 API Key（也可使用环境变量 `OPENAI_API_KEY`）
- `--base-url`：OpenAI 兼容或 Ollama 的 Base URL（Ollama 默认 `http://localhost:11434`）
- `--embedding-model`：向量检索使用的模型名称（默认 `sentence-transformers/paraphrase-MiniLM-L6-v2`）

---

### 项目结构

- `rag_asr_main.py`：命令行入口，负责参数解析、检索、Prompt构造与调用 LLM
- `retrievers/`：可插拔检索器
  - `base.py`：检索器抽象基类 `Retriever`
  - `smith_waterman_retriever.py`：字符级 Smith-Waterman 检索（依赖 `biopython`）
  - `bm25_retriever.py`：BM25 检索（依赖 `rank-bm25`）
  - `embedding_retriever.py`：向量检索（依赖 `sentence-transformers`、`numpy`、`torch`）
- `llm/`：可插拔 LLM 客户端
  - `base.py`：LLM 抽象基类 `LLMClient`
  - `openai_client.py`：OpenAI/兼容 客户端
  - `ollama_client.py`：Ollama 客户端
- `prompting/prompt_builder.py`：Prompt 构造逻辑（可自定义）
- `data/keywords.txt`：示例关键词库
- `similarity/Smith_Waterman.py`：现有 Smith-Waterman 实现

---

### 如何修改或新增“检索器”

最简单方式是新增一个文件，例如 `retrievers/my_retriever.py`，并实现 `Retriever` 接口：

```python
# retrievers/my_retriever.py
from typing import List
from .base import Retriever

class MyRetriever(Retriever):
    def __init__(self, keywords: List[str]):
        super().__init__(keywords)
        # 在此做初始化（加载模型、索引等）

    def score(self, query: str, keyword: str) -> float:
        # 返回一个分数（越大越相似）
        return 0.0

    # 可选：如果需要一次性对全库更高效检索，可以重载 search()
    # def search(self, query: str, top_k: int = 5):
    #     return [("keywordA", 0.9), ("keywordB", 0.8)][:top_k]
```

然后在 `rag_asr_main.py` 中的 `get_retriever()` 增加一个分支（或直接用与已有一致的命名规则、懒加载方式）：

```python
# rag_asr_main.py 内 get_retriever()
if name in ('my', 'my_retriever'):
    module = importlib.import_module('retrievers.my_retriever')
    RetrieverClass = getattr(module, 'MyRetriever')
    return RetrieverClass(keywords)
```

之后即可用：
```bash
python rag_asr_main.py --query "..." --retriever my --database ./data/keywords.txt
```

---

### 如何修改或新增“LLM 客户端”

新增一个文件，例如 `llm/my_client.py`，实现 `LLMClient` 接口：

```python
# llm/my_client.py
from .base import LLMClient

class MyClient(LLMClient):
    def __init__(self, model: str, **kwargs):
        self.model = model
        # 在此初始化你的客户端（API、SDK、本地模型等）

    def generate(self, prompt: str) -> str:
        # 根据 prompt 返回生成文本
        return ""
```

然后在 `rag_asr_main.py` 中的 `get_llm_client()` 增加分支：

```python
# rag_asr_main.py 内 get_llm_client()
if provider in ('myllm',):
    module = importlib.import_module('llm.my_client')
    ClientClass = getattr(module, 'MyClient')
    return ClientClass(model=model, **kwargs)
```

运行时即可切换：
```bash
python rag_asr_main.py --query "..." --llm-provider myllm --llm-model your_model
```

---

### 如何修改 Prompt

编辑 `prompting/prompt_builder.py` 中的 `build_prompt(query, keywords)`，即可自定义 Prompt 的格式与策略。例如加入更详细的指令、输出格式约束等。

---

### 常见问题

- 运行时报依赖错误：按需安装可选包（BM25/Embedding）或确认已安装 `biopython`。
- 向量检索首次运行较慢：`sentence-transformers` 会首次下载模型，注意网络环境。
- Ollama 连接失败：确认本地服务 `http://localhost:11434` 可访问，并已拉取相应模型。

---

### 许可

MIT 