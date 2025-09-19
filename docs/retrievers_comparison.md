# ClickZetta检索器对比说明

## 两种混合检索实现

LangChain-ClickZetta提供了两种混合检索实现，分别适用于不同的使用场景：

### 1. ClickZettaHybridRetriever (跨表混合检索)

**位置**: `langchain_clickzetta.retrievers.ClickZettaHybridRetriever`

**特点**:
- 使用两个分离的表：向量表 + 全文搜索表
- 分别从两个表获取结果，然后在应用层合并
- 兼容现有的 `ClickZettaVectorStore` 和 `ClickZettaFullTextRetriever`

**表结构**:
```sql
-- 向量表
CREATE TABLE vector_table (
    id String,
    content String,
    metadata String,
    embedding vector(float, 1024),
    INDEX embedding_idx (embedding) USING VECTOR
)

-- 全文搜索表
CREATE TABLE fulltext_table (
    id String,
    content String,
    metadata String,
    INDEX content_fts (content) USING INVERTED
)
```

**使用场景**:
- 已有项目迁移
- 需要独立管理向量和全文数据
- 分步实施混合搜索

**代码示例**:
```python
from langchain_clickzetta import ClickZettaHybridRetriever

# 需要分别创建向量存储和全文检索器
vector_store = ClickZettaVectorStore(...)
fulltext_retriever = ClickZettaFullTextRetriever(...)

# 创建跨表混合检索器
hybrid_retriever = ClickZettaHybridRetriever.from_engines(
    engine=engine,
    vector_store=vector_store,
    alpha=0.5
)
```

### 2. ClickZettaUnifiedRetriever (真正的混合检索)

**位置**: `langchain_clickzetta.hybrid_store.ClickZettaUnifiedRetriever`

**特点**:
- 使用单个表同时支持向量索引和倒排索引
- 真正的混合检索，在数据库层面优化
- 更高性能，更符合ClickZetta设计理念

**表结构**:
```sql
-- 统一混合表
CREATE TABLE hybrid_table (
    id String,
    content String,
    metadata String,
    embedding vector(float, 1024),
    INDEX content_fts (content) USING INVERTED PROPERTIES ('analyzer' = 'unicode'),
    INDEX embedding_idx (embedding) USING VECTOR PROPERTIES ('distance.function' = 'cosine_distance')
)
```

**使用场景**:
- 新项目推荐使用
- 追求最佳性能
- 充分利用ClickZetta原生能力

**代码示例**:
```python
from langchain_clickzetta import ClickZettaHybridStore, ClickZettaUnifiedRetriever

# 创建统一混合存储
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embeddings=embeddings,
    table_name="hybrid_docs"
)

# 创建统一检索器
unified_retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",  # "vector", "fulltext", "hybrid"
    alpha=0.5
)
```

## 功能对比

| 特性 | ClickZettaHybridRetriever | ClickZettaUnifiedRetriever |
|------|---------------------------|----------------------------|
| 表结构 | 两个独立表 | 单个混合表 |
| 性能 | 中等（跨表查询） | 高（单表查询） |
| 存储开销 | 高（数据重复） | 低（数据统一） |
| 实现复杂度 | 高（需要合并逻辑） | 低（数据库原生） |
| LangChain兼容性 | 完全兼容 | 完全兼容 |
| ClickZetta特性利用 | 部分 | 完全 |
| 推荐场景 | 兼容性优先 | 性能优先 |

## 推荐使用

### 新项目 ✅
推荐使用 `ClickZettaUnifiedRetriever` + `ClickZettaHybridStore`：
- 更好的性能
- 更少的存储开销
- 更简单的维护

### 现有项目迁移 📝
可以从 `ClickZettaHybridRetriever` 开始，逐步迁移到 `ClickZettaUnifiedRetriever`

### API设计原则
两种实现都完全兼容LangChain标准接口，确保无缝切换。