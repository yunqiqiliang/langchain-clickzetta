# ClickZetta LangChain Examples

This directory contains comprehensive examples demonstrating how to use the `langchain-clickzetta` integration package.

## Quick Start

### 1. Setup Environment

Make sure you have your ClickZetta connection configured in a `.env` file:

```bash
# ClickZetta Connection Configuration
CLICKZETTA_SERVICE=your-service-url
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_VCLUSTER=your-vcluster

# DashScope API Configuration (for LLM and embeddings)
DASHSCOPE_API_KEY=your-dashscope-api-key
```

### 2. Setup Demo Data (First Time Only)

Before running the examples, set up the demo tables and data:

```bash
python setup_demo_data.py
```

This creates sample tables (`demo_customers`, `demo_orders`, `demo_products`) with realistic business data for SQL Chain demonstrations.

### 3. Run Examples

```bash
# Run basic usage examples
python basic_usage.py

# Run advanced RAG system example
python advanced_rag.py
```

### 4. Clean Up (Optional)

To clean up test data after running examples:

```bash
python cleanup_example.py
```

## Available Examples

### 📁 `basic_usage.py`
Demonstrates all core LangChain ClickZetta features:

1. **SQL Chain** - Natural language to SQL conversion with real demo data
2. **Vector Store** - Document embedding storage and similarity search
3. **Chat Message History** - Persistent conversation memory
4. **Full-Text Search** - Advanced text search with inverted indexes
5. **True Hybrid Search** - Single-table vector + full-text search

### 📁 `advanced_rag.py`
Shows a complete Retrieval-Augmented Generation (RAG) system:

- **Knowledge Base Setup** - Vector + full-text document storage
- **Intelligent Q&A** - Context-aware question answering
- **Hybrid Retrieval** - Combines multiple search strategies
- **Conversation Analysis** - Chat history management
- **Search Demonstrations** - Different retrieval approaches

### 📁 `setup_demo_data.py`
Creates sample business data for SQL Chain demonstrations:

- **Customer data** (10 customers with demographics and spending)
- **Order data** (10 orders with products and status)
- **Product catalog** (10 products with ratings and categories)

### 📁 `storage_example.py`
Comprehensive demonstration of ClickZetta storage services:

- **Key-Value Store** - Basic persistent storage with LangChain BaseStore
- **Document Store** - Document storage with metadata support
- **File Store** - Binary file storage with MIME type handling
- **Volume Store** - Native ClickZetta Volume integration for file storage
- **Integration Patterns** - Caching, session storage, and LangChain patterns

### 📁 `cleanup_example.py`
Utility to clean up all test data:

- Drops all example tables
- Attempts to clean up test indexes
- Provides clean environment for fresh runs

## Features Demonstrated

### 🔍 **SQL Chain**
```python
sql_chain = ClickZettaSQLChain.from_engine(
    engine=engine,
    llm=llm,
    table_names=["demo_customers", "demo_orders", "demo_products"]
)

result = sql_chain.invoke({
    "query": "Show me the top 5 customers by total spent"
})
```

### 🧠 **Vector Store**
```python
vector_store = ClickZettaVectorStore(
    engine=engine,
    embeddings=embeddings,
    table_name="example_vectors"
)

# Add documents and search
ids = vector_store.add_documents(documents)
results = vector_store.similarity_search("query", k=3)
```

### 💬 **Chat History**
```python
chat_history = ClickZettaChatMessageHistory(
    engine=engine,
    session_id="user_session_123",
    table_name="chat_history"
)

chat_history.add_message(HumanMessage(content="Hello!"))
messages = chat_history.messages
```

### 🔍 **Full-Text Search**
```python
fulltext_retriever = ClickZettaFullTextRetriever(
    engine=engine,
    table_name="documents",
    search_type="phrase"  # or "boolean", "natural"
)

results = fulltext_retriever.invoke("search query")
```

### 🚀 **Hybrid Search**
```python
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embeddings=embeddings,
    table_name="hybrid_docs",
    text_analyzer="ik",  # Chinese text analyzer
    distance_metric="cosine"
)

unified_retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",  # "vector", "fulltext", or "hybrid"
    alpha=0.5  # Balance between vector and full-text
)
```

### 💾 **Storage Services**
```python
# Key-Value Store
kv_store = ClickZettaStore(
    engine=engine,
    table_name="my_cache"
)
kv_store.mset([("key1", b"value1"), ("key2", b"value2")])
values = kv_store.mget(["key1", "key2"])

# Document Store with Metadata
doc_store = ClickZettaDocumentStore(
    engine=engine,
    table_name="documents"
)
doc_store.store_document("doc1", "content", {"author": "user", "type": "text"})
content, metadata = doc_store.get_document("doc1")

# File Store for Binary Data
file_store = ClickZettaFileStore(
    engine=engine,
    table_name="files"
)
file_store.store_file("model.bin", binary_data, "application/octet-stream")
content, mime_type = file_store.get_file("model.bin")

# Volume Store (Native ClickZetta Volume)
volume_store = ClickZettaUserVolumeStore(
    engine=engine,
    subdirectory="langchain_data"
)
volume_store.mset([("config.json", b'{"setting": "value"}')])
config = volume_store.mget(["config.json"])[0]
```

## Expected Output

### SQL Chain Success
```
Question: Show me the top 5 customers by total spent from the demo_customers table
SQL Query: SELECT first_name, last_name, email, total_spent FROM demo_customers ORDER BY total_spent DESC LIMIT 5
Answer: The top 5 customers by total spent are:
1. Eve Davis – eve.davis@email.com – $4,500.00
2. Henry Anderson – henry.anderson@email.com – $3,450.90
...
```

### Vector Search Results
```
Query: What is ClickZetta?
Similar documents:
  1. ClickZetta is a high-performance cloud-native analytics database...
     Similarity: 0.888
```

### Hybrid Search Results
```
True hybrid search for: 'database analytics'
Results (single table with vector + inverted indexes):
  1. ClickZetta是高性能云原生分析数据库
     Hybrid Score: 0.500 (Vector: 1.0, Full-text: 0.0)
```

### Storage Services Results
```
=== ClickZetta Storage Services Demo ===

1. Basic Key-Value Store
✓ Stored 4 key-value pairs
Retrieved values:
  user:123:profile: {"name": "Alice", "age": 30}
  config:app:theme: dark

2. Document Store with Metadata
✓ Stored 3 documents with metadata
Document doc_001:
  Content: ClickZetta is a cloud-native analytics database...
  Metadata: {'category': 'technology', 'author': 'Tech Team'}

3. File Store for Binary Data
✓ Stored 3 files
File models/embeddings.bin:
  Size: 1600 bytes, MIME Type: application/octet-stream

4. Volume-based File Storage with ClickZetta Volume
✓ Files stored in ClickZetta Volume
Retrieving files:
  config.json: 38 bytes
    Content: {"app": "langchain", "version": "1.0"}
  ✓ Volume storage demonstrates ClickZetta's native file capabilities
```

## Architecture Notes

### ClickZetta Integration Features

1. **True Hybrid Storage**: Single table with both vector and inverted indexes
2. **Global Index Naming**: Unique index names to avoid ClickZetta conflicts
3. **Automatic Schema Detection**: Workspace.schema.table naming support
4. **Distance Metrics**: Cosine, Euclidean, Manhattan for vector search
5. **Text Analyzers**: Unicode, IK (Chinese), and other analyzers
6. **Error Handling**: Graceful degradation with informative error messages
7. **LangChain Storage Services**: Complete BaseStore implementation with multiple backends
8. **Native Volume Integration**: Leverages ClickZetta's Volume capabilities for file storage

### Performance Optimizations

- **Columnar Storage**: Optimized for ClickZetta's architecture
- **Batch Operations**: Efficient bulk document insertion
- **Index Building**: Automatic index creation and building
- **Connection Pooling**: Reuses database connections
- **Metadata Filtering**: Efficient filtering in hybrid searches
- **Storage Backend Options**: Table-based storage for transactional consistency
- **Volume File Operations**: Native PUT/GET operations for efficient file handling
- **Binary Encoding**: Base64 encoding for safe binary data storage

## Troubleshooting

### Common Issues

1. **"Table not found"** - Run `setup_demo_data.py` first
2. **"Index already exists"** - Run `cleanup_example.py` to reset
3. **"Connection failed"** - Check your `.env` file configuration
4. **"No results found"** - Ensure demo data was inserted successfully
5. **"Volume storage requires permissions"** - Ensure proper Volume access rights
6. **"Storage service errors"** - Check table creation permissions and schema access

### Environment Variables

Make sure all required environment variables are set in your `.env` file. You can copy from the ClickZetta console or use the MCP connection configuration.

### Index Conflicts

If you encounter index naming conflicts, run the cleanup script:

```bash
python cleanup_example.py
```

This will drop all test tables and attempt to clean up any orphaned indexes.

## Integration with Your Application

These examples can serve as templates for integrating ClickZetta with your own LangChain applications:

1. **Copy the connection setup** from any example
2. **Adapt the table names** to your schema
3. **Customize the prompts** for your domain
4. **Add your embedding model** (examples use DashScope)
5. **Implement error handling** for production use

## Next Steps

- Explore the [LangChain ClickZetta documentation](../README.md)
- Check out the [ClickZetta product documentation](https://yunqi.tech/documents)
- Build your own RAG applications using these patterns
- Contribute improvements to the langchain-clickzetta package

## Support

For issues specific to these examples:
- Check the troubleshooting section above
- Review the error messages for specific guidance
- Ensure your ClickZetta instance supports the required features

For general ClickZetta support:
- Visit [ClickZetta Documentation](https://yunqi.tech/documents)
- Contact ClickZetta support for instance-specific issues