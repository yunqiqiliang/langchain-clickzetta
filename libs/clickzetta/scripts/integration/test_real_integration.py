#!/usr/bin/env python3
"""Quick real integration test for ClickZetta LangChain package."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "langchain_clickzetta"))

from langchain_core.documents import Document

from langchain_clickzetta import ClickZettaEngine, ClickZettaVectorStore


class SimpleEmbeddings:
    """Simple mock embeddings for testing."""

    def embed_query(self, text: str) -> list[float]:
        """Generate simple hash-based embedding."""
        import hashlib
        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Generate 16-dimensional embedding from hash
        embedding = []
        for i in range(0, min(32, len(hash_val)), 2):
            val = int(hash_val[i:i+2], 16) / 255.0
            embedding.append(val)
        # Pad to 16 dimensions
        while len(embedding) < 16:
            embedding.append(0.0)
        return embedding[:16]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]


def load_uat_connection():
    """Load UAT connection from config file."""
    config_path = Path.home() / ".clickzetta" / "connections.json"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None

    try:
        with open(config_path, encoding='utf-8') as f:
            config_data = json.load(f)

        # Find UAT connection
        for conn in config_data.get("connections", []):
            if conn.get("name") == "uat":
                return conn

        print("❌ UAT connection not found in config")
        return None

    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def test_basic_connection():
    """Test basic ClickZetta connection."""
    print("🔄 Testing basic ClickZetta connection...")

    uat_config = load_uat_connection()
    if not uat_config:
        return False

    try:
        engine = ClickZettaEngine(
            service=uat_config["service"],
            instance=uat_config["instance"],
            workspace=uat_config["workspace"],
            schema=uat_config["schema"],
            username=uat_config["username"],
            password=uat_config["password"],
            vcluster=uat_config["vcluster"]
        )

        # Test simple query
        results, columns = engine.execute_query("SELECT 1 as test_value, 'hello' as message")

        if results and len(results) > 0:
            print(f"✅ Connection successful! Result: {results[0]}")
            return True
        else:
            print("❌ No results returned")
            return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        try:
            engine.close()
        except:
            pass


def test_vector_store():
    """Test vector store functionality."""
    print("🔄 Testing ClickZetta vector store...")

    uat_config = load_uat_connection()
    if not uat_config:
        return False

    try:
        engine = ClickZettaEngine(
            service=uat_config["service"],
            instance=uat_config["instance"],
            workspace=uat_config["workspace"],
            schema=uat_config["schema"],
            username=uat_config["username"],
            password=uat_config["password"],
            vcluster=uat_config["vcluster"]
        )

        embeddings = SimpleEmbeddings()
        table_name = f"langchain_test_vectors_{int(time.time())}"

        # Create vector store
        vector_store = ClickZettaVectorStore(
            engine=engine,
            embeddings=embeddings,
            table_name=table_name,
            vector_element_type="float",
            vector_dimension=16
        )

        print(f"✅ Vector store created with table: {table_name}")

        # Test adding documents
        documents = [
            Document(
                page_content="ClickZetta is a high-performance cloud-native analytics database",
                metadata={"category": "database", "source": "test"}
            ),
            Document(
                page_content="LangChain is a framework for developing applications with LLMs",
                metadata={"category": "framework", "source": "test"}
            )
        ]

        ids = vector_store.add_documents(documents)
        print(f"✅ Added {len(ids)} documents with IDs: {ids}")

        # Test similarity search
        results = vector_store.similarity_search("analytics database", k=1)
        if results:
            print(f"✅ Vector search successful! Found: {results[0].page_content[:50]}...")
            return True
        else:
            print("❌ Vector search returned no results")
            return False

    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False
    finally:
        try:
            engine.close()
        except:
            pass


def test_table_operations():
    """Test basic table operations."""
    print("🔄 Testing table operations...")

    uat_config = load_uat_connection()
    if not uat_config:
        return False

    try:
        engine = ClickZettaEngine(
            service=uat_config["service"],
            instance=uat_config["instance"],
            workspace=uat_config["workspace"],
            schema=uat_config["schema"],
            username=uat_config["username"],
            password=uat_config["password"],
            vcluster=uat_config["vcluster"]
        )

        # Test getting table info
        table_info = engine.get_table_info()
        if table_info:
            print(f"✅ Table info retrieved ({len(table_info)} chars)")
        else:
            print("⚠️  No table info returned (may be expected)")

        # Test current instance ID
        results, _ = engine.execute_query("SELECT CURRENT_INSTANCE_ID() as instance_id")
        if results:
            print(f"✅ Current instance ID: {results[0]['instance_id']}")
        else:
            print("⚠️  Could not get instance ID")

        return True

    except Exception as e:
        print(f"❌ Table operations test failed: {e}")
        return False
    finally:
        try:
            engine.close()
        except:
            pass


def main():
    """Run all integration tests."""
    print("=" * 50)
    print("ClickZetta LangChain Integration Tests")
    print("=" * 50)

    tests = [
        ("Basic Connection", test_basic_connection),
        ("Table Operations", test_table_operations),
        ("Vector Store", test_vector_store),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")

        print("-" * 30)

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
