---
title: LlamaIndex 学习笔记
description: LlamaIndex 是一个强大的工具，用于构建和部署由大型语言模型（LLM）驱动的应用程序。本笔记旨在帮助初学者快速了解其核心概念和用法。
date: 2025-08-19
---

## 1. LlamaIndex 是什么？

LlamaIndex 是一个用于 LLM 应用程序的**数据框架**，旨在帮助开发者注入、结构化并访问私有或特定领域的数据。它通过**上下文增强（Context Augmentation）**技术，将您的数据安全地提供给大型语言模型（LLM），从而让 LLM 能够基于这些数据进行准确的回答。

最常见的上下文增强技术是**检索增强生成（Retrieval-Augmented Generation, RAG）**，它在推理时将从您的数据中检索到的相关上下文与用户的查询一起提供给 LLM。LlamaIndex 为构建从原型到生产的各种 RAG 应用提供了全面的工具。

## 2. 核心组件

LlamaIndex 提供了一系列可组合的模块，使得构建 LLM 应用变得更加简单：

- **数据连接器（Data Connectors）**: 从各种数据源（如 API、PDF、SQL 数据库等）中提取数据，并将其转换为 LlamaIndex 支持的 `Document` 对象。
- **文档（Documents）/ 节点（Nodes）**: `Document` 是数据的通用容器，而 `Node` 是 `Document` 的分块，是 LlamaIndex 中最小的数据单元。这种分块处理有助于实现更精确和高效的数据检索。
- **数据索引（Data Indexes）**: 将 `Node` 对象结构化，以便于快速检索。最常见的索引是 `VectorStoreIndex`，它为每个 `Node` 生成向量嵌入。
- **检索器（Retrievers）**: 定义如何根据用户查询高效地从索引中检索相关的 `Node`。
- **响应合成器（Response Synthesizers）**: 根据用户查询和检索到的上下文（文本块），利用 LLM 生成最终的自然语言响应。
- **引擎（Engines）**: 提供与数据进行自然语言交互的端到端接口。
  - **查询引擎（Query Engines）**: 用于单次问答（Q&A）的强大接口。
  - **聊天引擎（Chat Engines）**: 用于多轮对话的交互式接口。
- **代理（Agents）**: 由 LLM 驱动的自动化决策者，可以动态地使用各种工具（如查询引擎、其他 API）来完成更复杂的任务。

## 3. 深入理解 RAG：LlamaIndex 的核心

RAG 是 LlamaIndex 应用的核心，它包含两个主要阶段：

### 3.1 索引阶段 (Indexing Phase)

这是构建知识库的过程，目的是让数据为查询做好准备。

1. **加载数据**: 使用**数据连接器**从不同的数据源加载数据。
2. **解析与分块**: LlamaIndex 将加载的 `Documents` 解析成一系列 `Nodes`（文本块）。
3. **生成嵌入与索引**: 对每个 `Node`，LlamaIndex 会使用一个嵌入模型（如 OpenAI 的 `text-embedding-ada-002`）来生成向量嵌入，然后将这些 `Node` 和它们的嵌入存储在**数据索引**中。

这个流程可以概括为：**Data Sources → Data Connectors → Documents → Nodes → Knowledge Base (Index)**

### 3.2 查询阶段 (Querying Phase)

当用户提出查询时，RAG 管道会从知识库中检索相关信息，并将其传递给 LLM 以生成响应。

1. **查询嵌入**: LlamaIndex 会使用相同的嵌入模型为用户的查询生成一个向量嵌入。
2. **检索上下文**: **检索器**使用查询的向量在数据索引中进行相似度搜索，找出最相关的 `Nodes`（上下文）。
3. **后处理 (Optional)**: **Node Postprocessors** 可以对检索到的 `Node` 列表进行转换、过滤或重新排序。
4. **合成响应**: **响应合成器**将用户的原始查询和检索到的上下文一起发送给 LLM，由 LLM 生成最终的回答。

这个流程可以概括为：**Knowledge Base → Retriever → Node Postprocessors → Response Synthesizer → Final Response**

## 4. 主要用例

LlamaIndex 可以用于构建多种应用：

- **问答系统（RAG）**: 基于您的文档或数据进行提问和回答。
- **聊天机器人**: 创建能与您的数据进行对话的机器人。
- **文档理解与数据提取**: 从非结构化文档中提取结构化信息。
- **自主代理**: 构建能够执行研究、采取行动的智能代理。
- **多模态应用**: 结合文本、图像等多种数据类型。
- **模型微调**: 在您的数据上微调模型以提升性能。

## 5. 实战案例

### 5.1 案例一：基础对话（无 RAG）- 模型依赖通用知识

本案例展示**传统的直接对话模式**，模型只能依赖训练时学到的通用知识来回答问题，无法访问外部知识库或最新信息。

**技术特点**:

- 仅依赖模型内置知识
- 无法获取特定领域信息
- 对新概念或专业术语可能回答不准确
- 无法追溯信息来源

**应用场景**: 通用对话、基础常识问答、不需要特定领域知识的任务

**操作流程:**

1. **环境准备**:

   首先，确保您已安装所有必需的 Python 库。

   ```bash
   pip install llama-index-llms-huggingface torch accelerate
   ```

2. **创建并运行脚本**:

   假设您已将模型下载至 `./models/qwen/Qwen1.5-1.8B-Chat` 目录。将以下代码保存为 `basic_chat.py` 文件，它将直接加载本地模型进行对话。

   ```python
   # basic_chat.py
   from llama_index.core.llms import ChatMessage
   from llama_index.llms.huggingface import HuggingFaceLLM

   # --- 1. 指定本地模型路径 ---
   # 请确保此路径下已包含预先下载的模型文件
   model_local_path = "./models/qwen/Qwen1.5-1.8B-Chat"
   print(f"正在从本地路径加载模型: {model_local_path}")

   # --- 2. 初始化 LLM ---
   llm = HuggingFaceLLM(
       model_name=model_local_path,
       tokenizer_name=model_local_path,
       model_kwargs={"trust_remote_code": True},
       tokenizer_kwargs={"trust_remote_code": True}
   )
   print("模型初始化完成。")

   # --- 3. 执行对话 ---
   rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
   print("\n--- 模型回答 ---")
   print(rsp)
   ```

3. **执行**:

   在终端中运行此脚本。

   ```bash
   python basic_chat.py
   ```

### 5.2 案例二：RAG 应用 - 基于知识库的精准问答

本案例展示**RAG（检索增强生成）技术**的强大能力，通过构建本地知识库，模型可以基于特定文档内容给出准确、具体、可追溯的回答。

**技术优势**:

- 基于真实文档内容回答
- 支持特定领域和专业知识
- 答案准确性和相关性更高
- 可以引用具体的文档来源
- 支持知识库实时更新

**应用场景**: 企业知识库问答、技术文档检索、专业领域咨询、客服系统

**操作流程:**

1. **环境准备**:

   首先，确保您已安装所有必需的 Python 库。

   ```bash
   pip install llama-index-embeddings-huggingface llama-index-llms-huggingface torch accelerate
   ```

2. **准备模型与数据**:

   - **模型**: 确保您已预先下载了语言模型（LLM）和嵌入模型（Embedding Model），并分别存放在以下路径：
     - `./models/qwen/Qwen1.5-1.8B-Chat`
     - `./models/damo/sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2`
   - **数据**: 在项目根目录下创建一个 `data` 文件夹，并将您的知识库文档（如 `.txt`, `.md` 文件）放入其中。

3. **创建并运行脚本**:

   将以下代码保存为 `rag_chat.py` 文件。该脚本将加载本地模型和数据，构建一个基于 RAG 的查询引擎，并回答您的问题。

   ```python
   # rag_chat.py
   from llama_index.embeddings.huggingface import HuggingFaceEmbedding
   from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
   from llama_index.llms.huggingface import HuggingFaceLLM

   # --- 1. 定义模型和数据路径 ---
   llm_model_path = "./models/qwen/Qwen1.5-1.8B-Chat"
   embedding_model_path = "./models/damo/sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2"
   data_path = "./data"
   print("模型和数据路径已配置。")

   # --- 2. 配置全局设置 (Settings) ---
   print("\n正在配置嵌入模型和语言模型...")
   # 配置嵌入模型
   embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)
   Settings.embed_model = embed_model

   # 配置语言模型
   llm = HuggingFaceLLM(
       model_name=llm_model_path,
       tokenizer_name=llm_model_path,
       model_kwargs={"trust_remote_code": True},
       tokenizer_kwargs={"trust_remote_code": True}
   )
   Settings.llm = llm
   print("全局设置配置完成。")

   # --- 3. 加载数据并构建索引 ---
   documents = SimpleDirectoryReader(data_path).load_data()
   index = VectorStoreIndex.from_documents(documents)
   print("索引构建完成。")

   # --- 4. 创建查询引擎并提问 ---
   query_engine = index.as_query_engine()
   print("查询引擎已就绪。")

   rsp = query_engine.query("xtuner是什么？")
   print("\n--- 模型回答 ---")
   print(rsp)
   ```

4. **执行**:

   在终端中运行此脚本。

   ```bash
   python rag_chat.py
   ```

**核心差异对比**:

| 对比维度       | 案例一（无 RAG）                   | 案例二（RAG）                  |
| -------------- | ---------------------------------- | ------------------------------ |
| **知识来源**   | 仅依赖模型训练时的通用知识         | 本地知识库文档 + 模型知识      |
| **回答准确性** | 对通用知识较好，专业领域可能不准确 | 基于文档内容，准确性显著提升   |
| **信息时效性** | 截止到模型训练时间                 | 可实时更新知识库               |
| **可追溯性**   | 无法追溯信息来源                   | 可显示引用的具体文档片段       |
| **专业领域**   | 表现一般，可能出现幻觉             | 表现优秀，基于真实文档         |
| **部署复杂度** | 简单，只需加载模型                 | 复杂，需要构建索引和向量库     |
| **资源消耗**   | 较低                               | 较高（需要嵌入模型和向量存储） |

**实际效果对比**:

- 对于"xtuner 是什么？"这类问题：
  - 案例一：可能回答不准确或表示不知道
  - 案例二：如果知识库中有相关文档，将给出准确且详细的回答

### 5.3 案例三：集成持久化向量数据库 ChromaDB

案例二中的向量索引是存储在内存里的，程序每次重启都需要重新构建。此案例将展示如何使用 ChromaDB 将索引持久化到本地磁盘，实现“一次构建，多次使用”。

**技术优势**:

- **持久化存储**: 索引存储在磁盘上，重启程序无需重新计算。
- **高效加载**: 再次运行时直接加载现有索引，启动速度快。
- **可扩展性**: 支持更大的数据集，不受内存限制。

**操作流程:**

1. **环境准备**:

   安装 ChromaDB 及其与 LlamaIndex 集成的相关库。

   ```bash
   pip install llama-index-vector-stores-chroma chromadb
   ```

2. **准备模型与数据**:

   与案例二的要求相同，确保所需的模型和数据文件已放置在正确目录下。

3. **创建并运行脚本**:

   将以下代码保存为 `rag_persistent.py`。该脚本会智能地处理索引：

   - **首次运行**：加载文档，创建索引，并将其保存到本地的 `chroma_db` 文件夹中。
   - **再次运行**：直接从 `chroma_db` 文件夹加载现有索引，跳过耗时的创建过程。

   ```python
   # rag_persistent.py
   import chromadb
   import os
   from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
   from llama_index.embeddings.huggingface import HuggingFaceEmbedding
   from llama_index.llms.huggingface import HuggingFaceLLM
   from llama_index.vector_stores.chroma import ChromaVectorStore

   # --- 1. 定义所有路径 ---
   llm_model_path = "./models/qwen/Qwen1.5-1.8B-Chat"
   embedding_model_path = "./models/damo/sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2"
   data_path = "./data"
   chroma_db_path = "./chroma_db"
   collection_name = "rag_knowledge_base"

   # --- 2. 配置全局模型 ---
   Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_path)
   Settings.llm = HuggingFaceLLM(
       model_name=llm_model_path,
       tokenizer_name=llm_model_path,
       model_kwargs={"trust_remote_code": True},
       tokenizer_kwargs={"trust_remote_code": True}
   )

   # --- 3. 初始化 ChromaDB 并加载/创建索引 ---
   db = chromadb.PersistentClient(path=chroma_db_path)
   chroma_collection = db.get_or_create_collection(collection_name)
   vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

   # 检查索引是否已在数据库中存在
   if chroma_collection.count() == 0:
       print(f"知识库 '{collection_name}' 为空，正在从头构建...")
       documents = SimpleDirectoryReader(data_path).load_data()
       storage_context = StorageContext.from_defaults(vector_store=vector_store)
       index = VectorStoreIndex.from_documents(
           documents, storage_context=storage_context
       )
       print("新索引已成功构建并保存。")
   else:
       print(f"已找到现有知识库 '{collection_name}'，正在直接加载...")
       index = VectorStoreIndex.from_vector_store(
           vector_store=vector_store,
       )
       print("现有索引加载完成。")

   # --- 4. 创建查询引擎并提问 ---
   query_engine = index.as_query_engine()
   rsp = query_engine.query("xtuner是什么？")
   print("\n--- 模型回答 ---")
   print(rsp)
   ```

4. **执行**:

   在终端中运行此脚本。

   ```bash
   python rag_persistent.py
   ```

   > **提示**: 首次运行时，您会看到构建索引的日志。当您再次运行此脚本时，它会跳过构建步骤，直接加载索引，速度会快很多。

通过这个案例，我们学到了如何通过集成向量数据库来提升 RAG 应用的效率和实用性。

## 6. 三种实现模式对比

| 技术方案       | 案例一（基础对话） | 案例二（RAG 应用）    | 案例三（持久化 RAG）    |
| -------------- | ------------------ | --------------------- | ----------------------- |
| **知识来源**   | 模型内置知识       | 本地知识库 + 模型知识 | 持久化知识库 + 模型知识 |
| **存储方式**   | 无需存储           | 内存向量索引          | 磁盘持久化向量库        |
| **启动效率**   | 快速               | 每次重建索引          | 一次构建，多次复用      |
| **适用场景**   | 通用对话、原型验证 | 专业问答、实时构建    | 生产环境、大规模应用    |
| **技术复杂度** | 低                 | 中等                  | 高                      |
