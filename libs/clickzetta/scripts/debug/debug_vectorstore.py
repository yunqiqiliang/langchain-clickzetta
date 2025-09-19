#!/usr/bin/env python3
"""Debug vector store initialization issue."""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


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


def test_vector_store_creation():
    """Test just the vector store creation without table initialization."""
    print("Testing vector store creation...")

    try:
        import json
        from pathlib import Path

        # Load UAT connection
        config_path = Path.home() / ".clickzetta" / "connections.json"
        with open(config_path, encoding='utf-8') as f:
            config_data = json.load(f)

        uat_config = None
        for conn in config_data.get("connections", []):
            if conn.get("name") == "uat":
                uat_config = conn
                break

        if not uat_config:
            print("❌ UAT connection not found")
            return False

        # Create engine
        engine = ClickZettaEngine(**uat_config)
        embeddings = SimpleEmbeddings()

        print("✅ Engine and embeddings created")

        # Try to import vector store class first
        try:
            from langchain_clickzetta.vectorstores import ClickZettaVectorStore
            print("✅ Vector store class imported successfully")
        except Exception as e:
            print(f"❌ Failed to import vector store class: {e}")
            return False

        # Try to create vector store - modify to not call table initialization
        print("Attempting to create vector store instance...")

        # Create vector store using proper constructor
        try:
            vector_store = ClickZettaVectorStore(
                engine=engine,
                embeddings=embeddings,
                table_name="test_table",
                vector_dimension=16,
                vector_element_type="float"
            )
            print("✅ Vector store object created with constructor")

            # Test if we can access embeddings
            test_embedding = vector_store.embeddings.embed_query("test")
            print(f"✅ Embedding test successful: {len(test_embedding)} dimensions")

        except Exception as e:
            print(f"❌ Failed to create/test vector store: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vector_store_creation()
    print("✅ Test completed" if success else "❌ Test failed")
