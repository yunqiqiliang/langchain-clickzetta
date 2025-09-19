# ClickZetta LangChain Integration Documentation

This directory contains detailed documentation for the ClickZetta LangChain integration.

## 📚 Documentation Index

### Design Documents
- **[STORAGE_IMPROVEMENTS.md](./STORAGE_IMPROVEMENTS.md)** - Storage services implementation details and improvements

### API Documentation
- **[../README.md](../README.md)** - Main library README with usage examples
- **[../examples/README.md](../examples/README.md)** - Examples and tutorials

### Project Documentation (Root Level)
- **[../../README.md](../../README.md)** - Project overview and getting started
- **[../../docs/LANGCHAIN_COMPLIANCE.md](../../docs/LANGCHAIN_COMPLIANCE.md)** - LangChain interface compliance details
- **[../../CHANGELOG.md](../../CHANGELOG.md)** - Version history and changes
- **[../../CONTRIBUTING.md](../../CONTRIBUTING.md)** - Development guidelines
- **[../../DEVELOPMENT.md](../../DEVELOPMENT.md)** - Development setup and workflow

## 🏗️ Architecture Overview

The ClickZetta LangChain integration provides:

### Core Components
1. **Engine** (`langchain_clickzetta.engine`) - Database connection and query execution
2. **Vector Stores** (`langchain_clickzetta.vectorstores`) - Vector similarity search
3. **Storage Services** (`langchain_clickzetta.stores`) - Persistent key-value storage
4. **SQL Chains** (`langchain_clickzetta.sql_chain`) - Natural language to SQL
5. **Retrievers** (`langchain_clickzetta.retrievers`) - Document retrieval systems

### Storage Services Architecture

#### Table-Based Storage
- **ClickZettaStore** - Key-value storage using SQL tables
- **ClickZettaDocumentStore** - Document storage with metadata (inherits from ClickZettaStore)

#### Volume-Based Storage
- **ClickZettaFileStore** - Binary file storage using ClickZetta Volume
- **ClickZettaUserVolumeStore** - User-specific volume storage
- **ClickZettaTableVolumeStore** - Table-specific volume storage
- **ClickZettaNamedVolumeStore** - Named volume storage

## 🧪 Testing Structure

### Unit Tests (`../tests/unit_tests/`)
- Interface compatibility tests
- Parameter validation tests
- Component isolation tests

### Integration Tests (`../tests/integration_tests/`)
- End-to-end functionality tests
- LangChain standard usage pattern tests
- Real database connection tests

## 📋 Key Features

### ✅ LangChain Compatibility
- Full `BaseStore` interface implementation
- Synchronous and asynchronous method support
- Standard LangChain usage patterns
- Type-safe operations

### ✅ ClickZetta Native Features
- MERGE INTO operations for atomic UPSERT
- Volume storage for binary data
- SQL-queryable document storage
- Proper error handling and logging

### ✅ Performance Optimizations
- Batch operations for multiple keys
- Atomic transactions using MERGE INTO
- Efficient prefix-based key filtering
- Connection pooling and reuse

## 🔗 Related Links

- [ClickZetta Documentation](https://docs.clickzetta.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Project Repository](https://github.com/your-org/langchain-clickzetta)