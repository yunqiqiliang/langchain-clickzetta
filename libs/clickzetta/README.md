# LangChain ClickZetta Integration

An integration package connecting ClickZetta and LangChain.

LangChain integration for ClickZetta, providing SQL queries, vector storage, and full-text search capabilities.

## Features

- **SQL Queries**: Natural language to SQL conversion and execution
- **Vector Storage**: Efficient vector storage and similarity search
- **Full-text Search**: Advanced text search capabilities with inverted index
- **Chat History**: Persistent conversation memory
- **Hybrid Search**: Combine vector and full-text search
- **True Hybrid Store**: Single table with both vector and inverted indexes (ClickZetta native)

## Installation

```bash
pip install langchain-clickzetta
```

## Quick Start

### Basic Setup

```python
from langchain_clickzetta import ClickZettaEngine

# Create engine
engine = ClickZettaEngine(
    service="your-service",
    instance="your-instance",
    workspace="your-workspace",
    schema="your-schema",
    username="your-username",
    password="your-password",
    vcluster="your-vcluster"
)
```

### Vector Storage

```python
from langchain_clickzetta import ClickZettaVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

# Setup embeddings
embeddings = DashScopeEmbeddings(
    dashscope_api_key="your-api-key",
    model="text-embedding-v4"
)

# Create vector store
vector_store = ClickZettaVectorStore(
    engine=engine,
    embeddings=embeddings,
    table_name="my_vectors"
)

# Add documents
texts = ["Hello world", "LangChain is great"]
vector_store.add_texts(texts)

# Search
results = vector_store.similarity_search("greeting", k=2)
```

### True Hybrid Search

```python
from langchain_clickzetta import ClickZettaHybridStore, ClickZettaUnifiedRetriever

# Create hybrid store (single table with vector + full-text indexes)
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embeddings=embeddings,
    table_name="hybrid_docs"
)

# Add documents
hybrid_store.add_texts([
    "ClickZetta is a high-performance analytics database",
    "LangChain enables building applications with LLMs"
])

# Create unified retriever
retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",  # "vector", "fulltext", or "hybrid"
    alpha=0.5  # Balance between vector and full-text search
)

# Search with hybrid approach
results = retriever.get_relevant_documents("analytics database")
```

### SQL Chain

```python
from langchain_clickzetta import ClickZettaSQLChain
from langchain_community.llms import Tongyi

llm = Tongyi(dashscope_api_key="your-api-key")

sql_chain = ClickZettaSQLChain.from_engine(
    engine=engine,
    llm=llm
)

result = sql_chain.invoke({"query": "How many tables are there?"})
print(result["result"])
```

## Documentation

For more detailed documentation, see the main repository README and examples.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This package is released under the MIT License.