# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of LangChain ClickZetta integration
- ClickZettaEngine for database connections and query execution
- ClickZettaVectorStore for vector storage and similarity search
- ClickZettaSQLChain for natural language to SQL conversion
- ClickZettaChatMessageHistory for conversation persistence
- ClickZettaFullTextRetriever for full-text search with inverted index
- ClickZettaHybridRetriever for cross-table hybrid search
- ClickZettaHybridStore for true single-table hybrid search
- ClickZettaUnifiedRetriever for unified search interface
- Comprehensive real-world integration tests
- Support for DashScope embeddings and LLM services
- Complete documentation and examples

### Features
- **SQL Queries**: Natural language to SQL conversion and execution
- **Vector Storage**: Efficient vector storage with HNSW index and similarity search
- **Full-text Search**: Advanced text search with inverted index and Chinese tokenization
- **Chat History**: Persistent conversation memory storage
- **Hybrid Search**: Cross-table search combining vector and full-text
- **True Hybrid Store**: Single table with both vector and inverted indexes
- **Multi-language Support**: Unicode tokenization for Chinese and mixed content
- **Real-time Indexing**: Automatic index building for new data
- **Flexible Distance Metrics**: Support for cosine, euclidean, and manhattan distance
- **Type Safety**: Complete type annotations with mypy validation

### Technical Highlights
- Complete LangChain VectorStore interface compatibility
- ClickZetta-native hybrid search implementation
- 100% real-environment test coverage
- Support for both sync operations
- Comprehensive error handling and logging
- Production-ready connection management

## [0.1.0] - 2024-09-19

### Added
- Initial release of langchain-clickzetta
- Core integration components for ClickZetta database
- Vector storage and search capabilities
- SQL query chain implementation
- Full-text search with inverted index
- Chat message history storage
- Hybrid search functionality
- Real-world integration testing
- Complete documentation and examples

### Dependencies
- langchain-core >= 0.1.0
- clickzetta-connector-python >= 0.8.92
- clickzetta-zettapark-python >= 0.1.3
- sqlalchemy >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.20.0 (< 2.0.0 for compatibility)
- pydantic >= 2.0.0

### Supported Python Versions
- Python 3.9+
- Python 3.10
- Python 3.11

### Testing
- Unit tests for all core components
- Integration tests with real ClickZetta UAT environment
- Integration tests with real DashScope services
- 100% test coverage for public APIs