"""Example demonstrating ClickZetta storage services for LangChain."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_clickzetta import (
    ClickZettaDocumentStore,
    ClickZettaEngine,
    ClickZettaFileStore,
    ClickZettaStore,
    ClickZettaUserVolumeStore,
)


def main():
    """Demonstrate ClickZetta storage services."""
    # Initialize ClickZetta engine
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster")
    )

    print("=== ClickZetta Storage Services Demo ===\n")

    # Example 1: Basic Key-Value Store
    print("1. Basic Key-Value Store")
    print("-" * 25)

    kv_store = ClickZettaStore(
        engine=engine,
        table_name="example_kv_store"
    )

    try:
        # Store some key-value pairs
        test_data = [
            ("user:123:profile", b'{"name": "Alice", "age": 30}'),
            ("user:124:profile", b'{"name": "Bob", "age": 25}'),
            ("config:app:theme", b"dark"),
            ("cache:expensive_computation", b"42"),
        ]

        kv_store.mset(test_data)
        print(f"✓ Stored {len(test_data)} key-value pairs")

        # Retrieve values
        keys_to_get = ["user:123:profile", "config:app:theme", "nonexistent:key"]
        values = kv_store.mget(keys_to_get)

        print("Retrieved values:")
        for key, value in zip(keys_to_get, values):
            if value:
                print(f"  {key}: {value.decode('utf-8')}")
            else:
                print(f"  {key}: (not found)")

        # List keys with prefix
        print("\\nKeys with 'user:' prefix:")
        for key in kv_store.yield_keys(prefix="user:"):
            print(f"  {key}")

    except Exception as e:
        print(f"✗ Key-Value Store Error: {e}")

    print("\\n")

    # Example 2: Document Store
    print("2. Document Store with Metadata")
    print("-" * 30)

    doc_store = ClickZettaDocumentStore(
        engine=engine,
        table_name="example_doc_store"
    )

    try:
        # Store documents with metadata
        documents = [
            {
                "id": "doc_001",
                "content": "ClickZetta is a cloud-native analytics database that provides high-performance data processing capabilities.",
                "metadata": {"category": "technology", "author": "Tech Team", "tags": ["database", "analytics"]}
            },
            {
                "id": "doc_002",
                "content": "LangChain enables developers to build applications powered by large language models.",
                "metadata": {"category": "ai", "author": "AI Team", "tags": ["llm", "framework"]}
            },
            {
                "id": "doc_003",
                "content": "Vector databases enable semantic search and similarity matching for AI applications.",
                "metadata": {"category": "ai", "author": "AI Team", "tags": ["vector", "search"]}
            }
        ]

        for doc in documents:
            doc_store.store_document(doc["id"], doc["content"], doc["metadata"])

        print(f"✓ Stored {len(documents)} documents with metadata")

        # Retrieve documents
        for doc_id in ["doc_001", "doc_002", "nonexistent_doc"]:
            result = doc_store.get_document(doc_id)
            if result:
                content, metadata = result
                print(f"\\nDocument {doc_id}:")
                print(f"  Content: {content[:50]}...")
                print(f"  Metadata: {metadata}")
            else:
                print(f"\\nDocument {doc_id}: (not found)")

        # List all document IDs
        print("\\nAll document IDs:")
        for doc_id in doc_store.yield_keys():
            print(f"  {doc_id}")

    except Exception as e:
        print(f"✗ Document Store Error: {e}")

    print("\\n")

    # Example 3: File Store for Binary Data
    print("3. File Store for Binary Data")
    print("-" * 28)

    file_store = ClickZettaFileStore(
        engine=engine,
        volume_type="user",
        subdirectory="example_files"
    )

    try:
        # Store different types of files
        files_to_store = [
            {
                "path": "models/embeddings.bin",
                "content": b"\\x00\\x01\\x02\\x03" * 100,  # Simulated binary data
                "mime_type": "application/octet-stream"
            },
            {
                "path": "configs/app.json",
                "content": b'{"database": "clickzetta", "version": "1.0"}',
                "mime_type": "application/json"
            },
            {
                "path": "images/logo.png",
                "content": b"\\x89PNG\\r\\n\\x1a\\n" + b"fake_png_data" * 50,  # Simulated PNG
                "mime_type": "image/png"
            }
        ]

        for file_info in files_to_store:
            file_store.store_file(
                file_info["path"],
                file_info["content"],
                file_info["mime_type"]
            )

        print(f"✓ Stored {len(files_to_store)} files")

        # Retrieve files
        for file_path in ["models/embeddings.bin", "configs/app.json", "nonexistent.txt"]:
            result = file_store.get_file(file_path)
            if result:
                content, mime_type = result
                print(f"\\nFile {file_path}:")
                print(f"  Size: {len(content)} bytes")
                print(f"  MIME Type: {mime_type}")
                if mime_type == "application/json":
                    print(f"  Content: {content.decode('utf-8')}")
            else:
                print(f"\\nFile {file_path}: (not found)")

        # List files
        print("\\nAll stored files:")
        files_list = file_store.list_files()
        for file_path, file_size, mime_type in files_list:
            print(f"  {file_path} ({file_size} bytes, {mime_type})")

        # List files with prefix
        print("\\nFiles in 'models/' directory:")
        models_files = file_store.list_files("models/")
        for file_path, file_size, _ in models_files:
            print(f"  {file_path} ({file_size} bytes)")

    except Exception as e:
        print(f"✗ File Store Error: {e}")

    print("\\n")

    # Example 4: Volume-based File Storage (ClickZetta Volume)
    print("4. Volume-based File Storage with ClickZetta Volume")
    print("-" * 48)

    try:
        # ClickZetta Volume provides native file storage capabilities
        volume_store = ClickZettaUserVolumeStore(
            engine=engine,
            subdirectory="langchain_examples"
        )

        # Store some files using Volume storage
        test_files = [
            ("config.json", b'{"app": "langchain", "version": "1.0"}'),
            ("model.bin", b"\\x00\\x01\\x02\\x03" * 25),  # Simulated binary model
            ("README.txt", b"This is a README file stored in ClickZetta Volume"),
        ]

        print(f"Storing {len(test_files)} files in User Volume...")
        for filename, content in test_files:
            volume_store.mset([(filename, content)])
        print("✓ Files stored in ClickZetta Volume")

        # Retrieve files
        print("\\nRetrieving files:")
        filenames = [f[0] for f in test_files]
        retrieved = volume_store.mget(filenames)
        for filename, content in zip(filenames, retrieved):
            if content:
                print(f"  {filename}: {len(content)} bytes")
                if filename.endswith('.json') or filename.endswith('.txt'):
                    print(f"    Content: {content.decode('utf-8')}")
            else:
                print(f"  {filename}: (not found)")

        print("\\n✓ Volume storage demonstrates ClickZetta's native file capabilities")

    except Exception as e:
        print(f"✗ Volume Storage Error: {e}")
        print("Note: Volume storage requires proper ClickZetta permissions")

    print("\\n")

    # Example 5: Storage Integration Patterns
    print("5. Integration with LangChain Patterns")
    print("-" * 35)

    try:
        # Pattern 1: Caching expensive computations
        cache_store = ClickZettaStore(engine=engine, table_name="computation_cache")

        def expensive_computation(input_data: str) -> str:
            """Simulate an expensive computation."""
            import hashlib  # noqa: E402
            cache_key = f"computation:{hashlib.md5(input_data.encode()).hexdigest()}"

            # Check cache first
            cached = cache_store.mget([cache_key])
            if cached[0]:
                print(f"  ✓ Cache hit for input: {input_data}")
                return cached[0].decode('utf-8')

            # Simulate computation
            result = f"processed_{input_data}_result"

            # Store in cache
            cache_store.mset([(cache_key, result.encode('utf-8'))])
            print(f"  ✓ Computed and cached result for: {input_data}")

            return result

        # Test caching
        inputs = ["data1", "data2", "data1"]  # data1 appears twice
        for input_data in inputs:
            result = expensive_computation(input_data)
            print(f"    Input: {input_data} -> Output: {result}")

        # Pattern 2: Session data storage
        session_store = ClickZettaStore(engine=engine, table_name="user_sessions")

        def store_user_session(user_id: str, session_data: dict):
            """Store user session data."""
            import json  # noqa: E402
            session_key = f"session:{user_id}"
            session_json = json.dumps(session_data).encode('utf-8')
            session_store.mset([(session_key, session_json)])

        def get_user_session(user_id: str) -> dict:
            """Retrieve user session data."""
            import json  # noqa: E402
            session_key = f"session:{user_id}"
            session_data = session_store.mget([session_key])
            if session_data[0]:
                return json.loads(session_data[0].decode('utf-8'))
            return {}

        # Test session storage
        store_user_session("user_123", {"preferences": {"theme": "dark"}, "last_login": "2024-01-01"})
        user_session = get_user_session("user_123")
        print(f"\\n  User session data: {user_session}")

    except Exception as e:
        print(f"✗ Integration Pattern Error: {e}")

    # Cleanup
    try:
        engine.close()
        print("\\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\\nError closing connection: {e}")

    print("\\n=== Storage Services Demo Complete ===")


if __name__ == "__main__":
    main()
