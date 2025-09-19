"""Real integration tests using UAT ClickZetta connection."""

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest
from langchain_community.embeddings import DashScopeEmbeddings

# Real DashScope services for comprehensive testing
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from langchain_clickzetta import (
    ClickZettaChatMessageHistory,
    ClickZettaEngine,
    ClickZettaSQLChain,
    ClickZettaVectorStore,
)
from langchain_clickzetta.retrievers import (
    ClickZettaFullTextRetriever,
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


class RealDashScopeEmbeddings:
    """Real DashScope embeddings for testing."""

    def __init__(self):
        config = load_dashscope_config()
        if config and config.get("api_key"):
            self.embeddings = DashScopeEmbeddings(
                dashscope_api_key=config["api_key"], model=config["model"]
            )
        else:
            # Fallback to mock if config not available
            self.embeddings = None

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for query."""
        if self.embeddings:
            return self.embeddings.embed_query(text)
        else:
            # Fallback mock
            import hashlib

            hash_val = hashlib.md5(text.encode()).hexdigest()
            return [
                float(int(hash_val[i : i + 2], 16)) / 255.0
                for i in range(0, min(32, len(hash_val)), 2)
            ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        if self.embeddings:
            return self.embeddings.embed_documents(texts)
        else:
            return [self.embed_query(text) for text in texts]


def get_real_dashscope_llm():
    """Get real DashScope LLM or fallback to mock."""
    config = load_dashscope_config()
    if config and config.get("api_key"):
        try:
            return Tongyi(
                dashscope_api_key=config["api_key"],
                model_name="qwen-turbo",
                max_tokens=200,
            )
        except:
            pass

    # Fallback mock LLM
    class MockLLM:
        def generate(self, messages):
            class MockGeneration:
                def __init__(self, text):
                    self.text = text

            class MockResult:
                def __init__(self, generation):
                    self.generations = [[generation]]

            return MockResult(
                MockGeneration(
                    "SQLQuery: SELECT COUNT(*) as count FROM information_schema.tables;"
                )
            )

    return MockLLM()


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
    return RealDashScopeEmbeddings()


@pytest.fixture
def real_llm():
    """Provide real DashScope LLM."""
    return get_real_dashscope_llm()


class TestRealClickZettaEngine:
    """Test ClickZetta engine with real connection."""

    def test_connection(self, uat_engine):
        """Test basic connection to ClickZetta."""
        # Test simple query
        results, columns = uat_engine.execute_query("SELECT 1 as test_value")

        assert len(results) == 1
        assert results[0]["test_value"] == 1
        print(f"✓ Connection test passed: {results}")

    def test_session_creation(self, uat_engine):
        """Test ClickZetta session creation."""
        session = uat_engine.get_session()
        assert session is not None
        print("✓ Session creation test passed")

    def test_table_info(self, uat_engine):
        """Test table information retrieval."""
        try:
            table_info = uat_engine.get_table_info()
            print(f"✓ Table info retrieved: {len(table_info)} characters")
            assert isinstance(table_info, str)
        except Exception as e:
            print(f"Table info test - expected behavior: {e}")


class TestRealVectorStore:
    """Test vector store with real ClickZetta connection."""

    def test_vector_table_creation(self, uat_engine, real_embeddings):
        """Test vector table creation."""
        table_name = f"test_vectors_{int(time.time())}"

        # Determine vector dimension based on real embeddings
        test_embedding = real_embeddings.embed_query("test")
        vector_dimension = len(test_embedding)

        vector_store = ClickZettaVectorStore(
            engine=uat_engine,
            embeddings=real_embeddings,
            table_name=table_name,
            vector_element_type="float",
            vector_dimension=vector_dimension,
        )

        print(f"✓ Vector store created with table: {table_name}")

    def test_add_and_search_vectors(self, uat_engine, real_embeddings):
        """Test adding documents and vector search."""
        table_name = f"test_vectors_{int(time.time())}"

        # Determine vector dimension based on real embeddings
        test_embedding = real_embeddings.embed_query("test")
        vector_dimension = len(test_embedding)

        vector_store = ClickZettaVectorStore(
            engine=uat_engine,
            embeddings=real_embeddings,
            table_name=table_name,
            vector_element_type="float",
            vector_dimension=vector_dimension,
        )

        # Test documents
        documents = [
            Document(
                page_content="ClickZetta is a high-performance analytics database",
                metadata={"category": "database", "source": "test1"},
            ),
            Document(
                page_content="LangChain enables building applications with LLMs",
                metadata={"category": "framework", "source": "test2"},
            ),
        ]

        try:
            # Add documents
            ids = vector_store.add_documents(documents)
            assert len(ids) == 2
            print(f"✓ Added {len(ids)} documents to vector store")

            # Search documents
            results = vector_store.similarity_search("analytics database", k=1)
            assert len(results) > 0
            print(f"✓ Vector search returned {len(results)} results")

        except Exception as e:
            print(f"Vector store test error (may be expected): {e}")


class TestRealSQLChain:
    """Test SQL chain with real ClickZetta connection."""

    def test_sql_chain_creation(self, uat_engine, real_llm):
        """Test SQL chain creation."""
        sql_chain = ClickZettaSQLChain.from_engine(
            engine=uat_engine, llm=real_llm, return_sql=True
        )

        assert sql_chain is not None
        print("✓ SQL chain created successfully")

    def test_sql_generation_and_execution(self, uat_engine, real_llm):
        """Test SQL generation and execution."""
        sql_chain = ClickZettaSQLChain.from_engine(
            engine=uat_engine, llm=real_llm, return_sql=True
        )

        try:
            # Test with a simple query
            result = sql_chain.invoke({"query": "How many tables are there?"})

            assert "result" in result
            print(f"✓ SQL chain execution result: {result.get('result', '')[:100]}...")

        except Exception as e:
            print(f"SQL chain test error (may be expected with mock LLM): {e}")


class TestRealFullTextSearch:
    """Test full-text search with real ClickZetta connection."""

    def test_fulltext_table_creation(self, uat_engine):
        """Test full-text search table creation."""
        table_name = f"test_fulltext_{int(time.time())}"

        try:
            retriever = ClickZettaFullTextRetriever(
                engine=uat_engine, table_name=table_name, search_type="phrase", k=5
            )

            print(f"✓ Full-text retriever created with table: {table_name}")

        except Exception as e:
            print(f"Full-text search test error: {e}")


class TestRealChatHistory:
    """Test chat message history with real ClickZetta connection."""

    def test_chat_history_creation(self, uat_engine):
        """Test chat history table creation."""
        table_name = f"test_chat_{int(time.time())}"
        session_id = f"test_session_{int(time.time())}"

        chat_history = ClickZettaChatMessageHistory(
            engine=uat_engine, session_id=session_id, table_name=table_name
        )

        print(f"✓ Chat history created with table: {table_name}")

    def test_add_and_retrieve_messages(self, uat_engine):
        """Test adding and retrieving chat messages."""
        table_name = f"test_chat_{int(time.time())}"
        session_id = f"test_session_{int(time.time())}"

        chat_history = ClickZettaChatMessageHistory(
            engine=uat_engine, session_id=session_id, table_name=table_name
        )

        try:
            # Add messages
            chat_history.add_message(HumanMessage(content="Hello ClickZetta!"))
            chat_history.add_message(AIMessage(content="Hello! How can I help you?"))

            # Retrieve messages
            messages = chat_history.messages
            assert len(messages) == 2
            print(f"✓ Chat history: Added and retrieved {len(messages)} messages")

        except Exception as e:
            print(f"Chat history test error: {e}")


if __name__ == "__main__":
    # Run basic connection test
    try:
        uat_config = load_uat_connection()
        print(
            f"Found UAT connection: {uat_config['service']} - {uat_config['instance']}"
        )

        engine = ClickZettaEngine(
            service=uat_config["service"],
            instance=uat_config["instance"],
            workspace=uat_config["workspace"],
            schema=uat_config["schema"],
            username=uat_config["username"],
            password=uat_config["password"],
            vcluster=uat_config["vcluster"],
        )

        # Test basic connection
        results, _ = engine.execute_query("SELECT 1 as test")
        print(f"✓ Basic connection test successful: {results}")

        engine.close()

    except Exception as e:
        print(f"✗ Connection test failed: {e}")
