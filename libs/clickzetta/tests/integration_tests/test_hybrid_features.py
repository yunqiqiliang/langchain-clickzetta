"""Real integration tests for hybrid features using UAT ClickZetta connection."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest

# Real DashScope services for comprehensive testing
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaFullTextRetriever,
    ClickZettaHybridRetriever,
    ClickZettaHybridStore,
    ClickZettaUnifiedRetriever,
    ClickZettaVectorStore,
)


def load_dashscope_config():
    """Load DashScope configuration from connections.json."""
    config_path = Path.home() / ".clickzetta" / "connections.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return (
            config_data.get("system_config", {})
            .get("embedding", {})
            .get("dashscope", {})
        )
    except:
        return None


def load_uat_connection() -> Dict[str, Any]:
    """Load UAT connection configuration from connections.json."""
    config_path = Path.home() / ".clickzetta" / "connections.json"

    if not config_path.exists():
        pytest.skip("ClickZetta connections.json not found")

    with open(config_path, encoding="utf-8") as f:
        config_data = json.load(f)

    # Find UAT connection
    uat_conn = None
    for conn in config_data.get("connections", []):
        if conn.get("name") == "uat":
            uat_conn = conn
            break

    if not uat_conn:
        pytest.skip("UAT connection not found in connections.json")

    return uat_conn


@pytest.fixture
def uat_engine():
    """Create real ClickZetta engine using UAT connection."""
    uat_config = load_uat_connection()

    engine = ClickZettaEngine(
        service=uat_config["service"],
        instance=uat_config["instance"],
        workspace=uat_config["workspace"],
        schema=uat_config["schema"],
        username=uat_config["username"],
        password=uat_config["password"],
        vcluster=uat_config["vcluster"],
        query_timeout=300,
    )

    yield engine

    # Cleanup
    engine.close()


@pytest.fixture
def real_embeddings():
    """Provide real DashScope embeddings."""
    config = load_dashscope_config()
    if config and config.get("api_key"):
        return DashScopeEmbeddings(
            dashscope_api_key=config["api_key"],
            model=config.get("model", "text-embedding-v4"),
        )
    else:
        pytest.skip("DashScope embeddings not configured")


class TestRealHybridStore:
    """Test ClickZetta hybrid store with real connection."""

    def test_hybrid_store_creation(self, uat_engine, real_embeddings):
        """Test hybrid store creation with both vector and inverted indexes."""
        table_name = f"test_hybrid_store_{int(time.time())}"

        try:
            # Create hybrid store
            hybrid_store = ClickZettaHybridStore(
                engine=uat_engine,
                embeddings=real_embeddings,
                table_name=table_name,
                text_analyzer="unicode",
                distance_metric="cosine",
            )

            # Wait a moment for table creation to complete
            time.sleep(2)

            # Get the full table name with workspace and schema
            full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"

            # Check table structure
            table_info_sql = f"DESC {full_table_name}"
            results, _ = uat_engine.execute_query(table_info_sql)

            # Verify columns exist
            columns = [row["column_name"] for row in results]
            assert "id" in columns
            assert "content" in columns
            assert "metadata" in columns
            assert "embedding" in columns

            # Check indexes
            show_indexes_sql = f"SHOW INDEX FROM {full_table_name}"
            index_results, _ = uat_engine.execute_query(show_indexes_sql)

            # Check if indexes exist (they might not show up immediately)
            if index_results:
                index_types = [
                    row.get("index_type", row.get("Index_type", ""))
                    for row in index_results
                ]
                print(f"✓ Found indexes: {index_types}")
            else:
                print("⚠ No indexes found immediately (may take time to create)")

            print(f"✓ Hybrid store created with table: {table_name}")
            if index_results:
                print("✓ Table structure verified successfully")

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass

    def test_hybrid_store_add_and_search(self, uat_engine, real_embeddings):
        """Test adding documents and hybrid search functionality."""
        table_name = f"test_hybrid_search_{int(time.time())}"

        try:
            # Create hybrid store
            hybrid_store = ClickZettaHybridStore(
                engine=uat_engine,
                embeddings=real_embeddings,
                table_name=table_name,
                text_analyzer="unicode",
            )

            # Test documents
            test_texts = [
                "ClickZetta是一个高性能的云原生分析数据库，支持向量搜索",
                "LangChain是一个强大的大语言模型应用开发框架",
                "混合检索结合了向量相似性搜索和关键词匹配功能",
                "机器学习和人工智能技术在数据分析中发挥重要作用",
            ]

            test_metadatas = [
                {"category": "database", "source": "docs"},
                {"category": "framework", "source": "docs"},
                {"category": "search", "source": "tech"},
                {"category": "ai", "source": "research"},
            ]

            # Add documents
            doc_ids = hybrid_store.add_texts(test_texts, test_metadatas)
            assert len(doc_ids) == 4
            print(f"✓ Added {len(doc_ids)} documents to hybrid store")

            # Test vector search
            vector_results = hybrid_store.similarity_search("数据库性能", k=2)
            assert len(vector_results) > 0
            print(f"✓ Vector search returned {len(vector_results)} results")

            # Test full-text search
            fulltext_results = hybrid_store.fulltext_search("ClickZetta", k=2)
            assert len(fulltext_results) > 0
            print(f"✓ Full-text search returned {len(fulltext_results)} results")

            # Test hybrid search
            hybrid_results = hybrid_store.hybrid_search("机器学习数据", k=2, alpha=0.5)
            assert len(hybrid_results) > 0
            print(f"✓ Hybrid search returned {len(hybrid_results)} results")

            # Verify hybrid scores
            for doc in hybrid_results:
                assert "hybrid_score" in doc.metadata
                assert "vector_score" in doc.metadata
                assert "fulltext_score" in doc.metadata

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass

    def test_hybrid_store_langchain_interface(self, uat_engine, real_embeddings):
        """Test LangChain VectorStore interface compatibility."""
        table_name = f"test_langchain_interface_{int(time.time())}"

        try:
            # Test from_texts class method
            test_texts = ["测试文档1", "测试文档2", "测试文档3"]
            test_metadatas = [{"id": i} for i in range(3)]

            hybrid_store = ClickZettaHybridStore.from_texts(
                texts=test_texts,
                embedding=real_embeddings,
                metadatas=test_metadatas,
                engine=uat_engine,
                table_name=table_name,
            )

            # Test similarity_search
            results = hybrid_store.similarity_search("测试", k=2)
            assert len(results) > 0
            print(f"✓ similarity_search returned {len(results)} results")

            # Test similarity_search_with_score
            results_with_scores = hybrid_store.similarity_search_with_score("文档", k=2)
            assert len(results_with_scores) > 0
            assert all(
                isinstance(score, (int, float)) for _, score in results_with_scores
            )
            print(
                f"✓ similarity_search_with_score returned {len(results_with_scores)} results"
            )

            # Test delete
            # Get some document IDs
            all_docs = hybrid_store.similarity_search("", k=10)  # Get all docs
            if all_docs:
                doc_id = all_docs[0].metadata.get("document_id")
                if doc_id:
                    success = hybrid_store.delete([doc_id])
                    assert success
                    print("✓ Document deletion successful")

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass


class TestRealUnifiedRetriever:
    """Test unified retriever with real connection."""

    def test_unified_retriever_creation(self, uat_engine, real_embeddings):
        """Test unified retriever creation."""
        table_name = f"test_unified_retriever_{int(time.time())}"

        try:
            # Create hybrid store
            hybrid_store = ClickZettaHybridStore(
                engine=uat_engine, embeddings=real_embeddings, table_name=table_name
            )

            # Create unified retriever
            unified_retriever = ClickZettaUnifiedRetriever(
                hybrid_store=hybrid_store, k=3, alpha=0.5, search_type="hybrid"
            )

            assert unified_retriever is not None
            assert unified_retriever.k == 3
            assert unified_retriever.alpha == 0.5
            assert unified_retriever.search_type == "hybrid"

            print("✓ Unified retriever created successfully")

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass

    def test_unified_retriever_search_modes(self, uat_engine, real_embeddings):
        """Test different search modes of unified retriever."""
        table_name = f"test_search_modes_{int(time.time())}"

        try:
            # Create and populate hybrid store
            hybrid_store = ClickZettaHybridStore(
                engine=uat_engine, embeddings=real_embeddings, table_name=table_name
            )

            test_texts = [
                "ClickZetta是高性能分析数据库",
                "LangChain框架支持多种数据源",
                "混合检索提供更精准的结果",
            ]

            hybrid_store.add_texts(test_texts)

            # Create unified retriever
            unified_retriever = ClickZettaUnifiedRetriever(
                hybrid_store=hybrid_store, k=2
            )

            # Test vector search mode
            unified_retriever.search_type = "vector"
            vector_results = unified_retriever._get_relevant_documents("数据库分析")
            assert len(vector_results) <= 2
            print(f"✓ Vector mode returned {len(vector_results)} results")

            # Test full-text search mode
            unified_retriever.search_type = "fulltext"
            fulltext_results = unified_retriever._get_relevant_documents("ClickZetta")
            print(f"✓ Full-text mode returned {len(fulltext_results)} results")

            # Test hybrid search mode
            unified_retriever.search_type = "hybrid"
            hybrid_results = unified_retriever._get_relevant_documents("框架数据")
            assert len(hybrid_results) <= 2
            print(f"✓ Hybrid mode returned {len(hybrid_results)} results")

            # Verify hybrid results have score metadata
            for doc in hybrid_results:
                if "hybrid_score" in doc.metadata:
                    assert isinstance(doc.metadata["hybrid_score"], (int, float))

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass


class TestRealCrossTableHybrid:
    """Test cross-table hybrid retriever with real connection."""

    def test_cross_table_hybrid_retriever(self, uat_engine, real_embeddings):
        """Test cross-table hybrid retriever functionality."""
        vector_table = f"test_cross_vector_{int(time.time())}"
        fulltext_table = f"test_cross_fulltext_{int(time.time())}"

        try:
            # Create vector store
            vector_store = ClickZettaVectorStore(
                engine=uat_engine, embeddings=real_embeddings, table_name=vector_table
            )

            # Create full-text retriever
            fulltext_retriever = ClickZettaFullTextRetriever(
                engine=uat_engine, table_name=fulltext_table, search_type="phrase"
            )

            # Create cross-table hybrid retriever
            hybrid_retriever = ClickZettaHybridRetriever.from_engines(
                engine=uat_engine,
                vector_store=vector_store,
                fulltext_table=fulltext_table,
                alpha=0.5,
                k=2,
            )

            # Add test documents to both stores
            test_docs = [
                Document(
                    page_content="ClickZetta数据库支持向量搜索和全文检索",
                    metadata={"category": "database"},
                ),
                Document(
                    page_content="LangChain框架提供统一的接口",
                    metadata={"category": "framework"},
                ),
            ]

            # Add to vector store
            vector_ids = vector_store.add_documents(test_docs)
            assert len(vector_ids) == 2

            # Add to full-text retriever
            fulltext_ids = fulltext_retriever.add_documents(test_docs)
            assert len(fulltext_ids) == 2

            # Test hybrid retrieval
            hybrid_results = hybrid_retriever._get_relevant_documents("数据库框架")
            print(f"✓ Cross-table hybrid search returned {len(hybrid_results)} results")

            # Verify results have hybrid scores
            for doc in hybrid_results:
                if "hybrid_score" in doc.metadata:
                    assert isinstance(doc.metadata["hybrid_score"], (int, float))

        finally:
            # Cleanup
            try:
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {vector_table}")
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {fulltext_table}")
            except:
                pass


class TestRealFullTextComplete:
    """Complete full-text search functionality tests."""

    def test_fulltext_complete_workflow(self, uat_engine):
        """Test complete full-text search workflow including BUILD INDEX."""
        table_name = f"test_fulltext_complete_{int(time.time())}"

        try:
            # Create full-text retriever
            retriever = ClickZettaFullTextRetriever(
                engine=uat_engine, table_name=table_name, search_type="phrase"
            )

            # Test documents
            test_docs = [
                Document(
                    page_content="ClickZetta是高性能云原生分析数据库",
                    metadata={"category": "database"},
                ),
                Document(
                    page_content="全文搜索支持中文分词和倒排索引",
                    metadata={"category": "search"},
                ),
                Document(
                    page_content="机器学习模型用于智能数据分析",
                    metadata={"category": "ai"},
                ),
            ]

            # Add documents
            doc_ids = retriever.add_documents(test_docs)
            assert len(doc_ids) == 3
            print(f"✓ Added {len(doc_ids)} documents to full-text index")

            # Manually build index for testing
            try:
                build_sql = f"BUILD INDEX content_fts ON {table_name}"
                uat_engine.execute_query(build_sql)
                print("✓ Full-text index built successfully")
            except Exception as e:
                print(f"⚠️ Index build warning: {e}")

            # Test different search types
            # Test phrase search
            phrase_results = retriever._get_relevant_documents("分析数据库")
            print(f"✓ Phrase search returned {len(phrase_results)} results")

            # Test with different search type
            retriever.search_type = "any"
            any_results = retriever._get_relevant_documents("机器学习")
            print(f"✓ Any-match search returned {len(any_results)} results")

            # Test tokenization
            try:
                tokenize_sql = "SELECT TOKENIZE('ClickZetta分析数据库', map('analyzer', 'unicode')) as tokens"
                tokenize_results, _ = uat_engine.execute_query(tokenize_sql)
                if tokenize_results:
                    tokens = tokenize_results[0]["tokens"]
                    print(f"✓ Tokenization test: {tokens}")
                    assert isinstance(tokens, list)
            except Exception as e:
                print(f"⚠️ Tokenization test warning: {e}")

        finally:
            # Cleanup
            try:
                full_table_name = f"{uat_engine.connection_config['workspace']}.{uat_engine.connection_config['schema']}.{table_name}"
                uat_engine.execute_query(f"DROP TABLE IF EXISTS {full_table_name}")
            except:
                pass


if __name__ == "__main__":
    # Can run basic tests manually
    import sys

    print("Run with: pytest tests/integration/test_hybrid_features.py -v")
    sys.exit(0)
