# 5分钟上手指南

本指南将在5分钟内带您体验 LangChain ClickZetta 的核心功能。

## 🎯 目标

完成本指南后，您将能够：
- 建立ClickZetta连接
- 执行自然语言SQL查询
- 创建向量存储并进行相似性搜索
- 使用键值存储保存数据

## 📋 前提条件

- 已安装 `langchain-clickzetta`
- 已获得 ClickZetta 连接参数
- （可选）灵积DashScope API密钥

## 🚀 第一步：建立连接

```python
from langchain_clickzetta import ClickZettaEngine

# 创建ClickZetta引擎
engine = ClickZettaEngine(
    service="your-service",
    instance="your-instance",
    workspace="your-workspace",
    schema="your-schema",
    username="your-username",
    password="your-password",
    vcluster="your-vcluster"
)

# 测试连接
results, columns = engine.execute_query("SELECT CURRENT_TIMESTAMP as now")
print(f"连接成功！当前时间: {results[0]['now']}")
```

## 🤖 第二步：自然语言SQL查询

```python
from langchain_clickzetta import ClickZettaSQLChain
from langchain_community.llms import Tongyi

# 初始化大语言模型
llm = Tongyi(
    dashscope_api_key="your-dashscope-api-key",
    model_name="qwen-plus"
)

# 创建SQL链
sql_chain = ClickZettaSQLChain.from_engine(
    engine=engine,
    llm=llm,
    return_sql=True
)

# 用自然语言查询数据库
response = sql_chain.invoke({
    "query": "显示数据库中所有的表"
})

print("AI回答:", response["result"])
print("生成的SQL:", response["sql_query"])
```

## 🔍 第三步：向量存储与相似性搜索

```python
from langchain_clickzetta import ClickZettaVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    dashscope_api_key="your-dashscope-api-key",
    model="text-embedding-v4"
)

# 创建向量存储
vector_store = ClickZettaVectorStore(
    engine=engine,
    embedding=embeddings,
    table_name="quickstart_vectors"
)

# 添加一些文档
documents = [
    Document(page_content="云器ClickZetta是新一代云原生湖仓一体化平台"),
    Document(page_content="LangChain是构建AI应用的开发框架"),
    Document(page_content="向量搜索可以实现语义相似性检索"),
    Document(page_content="ClickZetta支持实时数据分析和处理")
]

# 将文档添加到向量存储
vector_store.add_documents(documents)
print("✅ 文档已添加到向量存储")

# 进行相似性搜索
query = "什么是ClickZetta？"
results = vector_store.similarity_search(query, k=2)

print(f"\n搜索查询: {query}")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
```

## 💾 第四步：键值存储

```python
from langchain_clickzetta import ClickZettaStore

# 创建键值存储
store = ClickZettaStore(
    engine=engine,
    table_name="quickstart_store"
)

# 存储一些键值对
data = [
    ("user:123", b"张三"),
    ("config:app", b'{"theme": "dark", "language": "zh"}'),
    ("cache:result", b"计算结果缓存数据")
]

store.mset(data)
print("✅ 数据已存储")

# 检索数据
keys = ["user:123", "config:app", "cache:result"]
values = store.mget(keys)

for key, value in zip(keys, values):
    if value:
        print(f"{key}: {value.decode('utf-8')}")
```

## 🎨 第五步：混合搜索（向量+全文）

```python
from langchain_clickzetta import ClickZettaHybridStore, ClickZettaUnifiedRetriever

# 创建混合存储（单表支持向量+全文索引）
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embedding=embeddings,
    table_name="quickstart_hybrid",
    text_analyzer="ik"  # 中文分词器
)

# 添加中文文档
chinese_docs = [
    Document(page_content="人工智能正在改变世界，深度学习是其核心技术"),
    Document(page_content="云计算提供了可扩展的计算资源"),
    Document(page_content="大数据分析帮助企业做出更好的决策"),
    Document(page_content="机器学习算法可以从数据中学习模式")
]

hybrid_store.add_documents(chinese_docs)

# 创建统一检索器
retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",  # 混合搜索
    alpha=0.5,  # 向量搜索和全文搜索的权重平衡
    k=3
)

# 执行混合搜索
query = "AI和机器学习"
results = retriever.invoke(query)

print(f"\n混合搜索查询: {query}")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
```

## 💬 第六步：聊天历史

```python
from langchain_clickzetta import ClickZettaChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 创建聊天历史管理
chat_history = ClickZettaChatMessageHistory(
    engine=engine,
    session_id="user_demo",
    table_name="quickstart_chat"
)

# 添加对话消息
chat_history.add_message(HumanMessage(content="你好，我想了解ClickZetta"))
chat_history.add_message(AIMessage(content="您好！ClickZetta是云器科技推出的新一代云原生湖仓一体化平台，具有10倍性能提升的特点。"))
chat_history.add_message(HumanMessage(content="它有什么特色功能？"))
chat_history.add_message(AIMessage(content="ClickZetta的特色包括：1）增量计算引擎 2）统一存储计算 3）实时数据处理 4）云原生架构。"))

print("✅ 对话历史已保存")

# 检索对话历史
messages = chat_history.messages
print(f"\n对话历史 (共{len(messages)}条消息):")
for msg in messages:
    speaker = "用户" if msg.__class__.__name__ == "HumanMessage" else "AI"
    print(f"{speaker}: {msg.content}")
```

## 🏆 完成！

恭喜！您已经在5分钟内体验了 LangChain ClickZetta 的主要功能：

✅ **数据库连接** - 建立了与ClickZetta的连接
✅ **AI SQL查询** - 使用自然语言查询数据库
✅ **向量搜索** - 实现了语义相似性检索
✅ **键值存储** - 存储和检索结构化数据
✅ **混合搜索** - 结合向量和全文搜索
✅ **聊天历史** - 管理对话记忆

## 🚀 下一步

现在您已经掌握了基础用法，建议深入学习：

### 功能进阶
- [SQL查询详细指南](sql-queries.md) - 深入了解AI驱动的SQL查询
- [向量存储优化](vector-storage.md) - 高级向量搜索技术
- [存储服务最佳实践](storage-services.md) - 企业级存储解决方案

### 实际应用
- [构建RAG应用](../tutorials/rag-application.md) - 端到端检索增强生成
- [中文AI应用开发](../tutorials/chinese-ai-app.md) - 中文场景优化
- [企业级部署](../tutorials/enterprise-deployment.md) - 生产环境部署

### API参考
- [完整API文档](../api/) - 详细的接口说明
- [示例代码库](../examples/) - 更多实用示例

## 💡 实用提示

1. **性能优化**: 在生产环境中使用连接池
2. **安全性**: 使用环境变量管理API密钥
3. **监控**: 启用日志记录以便调试
4. **扩展性**: 考虑表分区和索引优化

## ❓ 遇到问题？

- 查看 [常见问题](../tutorials/debugging.md)
- 访问 [GitHub Issues](https://github.com/yunqiqiliang/langchain-clickzetta/issues)
- 参与 [社区讨论](https://github.com/yunqiqiliang/langchain-clickzetta/discussions)