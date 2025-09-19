"""Basic usage examples for langchain-clickzetta."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from langchain_clickzetta import (
    ClickZettaChatMessageHistory,
    ClickZettaEngine,
    ClickZettaHybridStore,
    ClickZettaSQLChain,
    ClickZettaUnifiedRetriever,
    ClickZettaVectorStore,
)


def main():
    """Run basic usage examples."""
    # Initialize ClickZetta engine
    # ClickZetta requires exactly 7 connection parameters
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster")
    )

    # Initialize DashScope components (recommended for Chinese/English support)
    api_key = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-api-key")
    llm = Tongyi(dashscope_api_key=api_key, temperature=0)
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model="text-embedding-v4"
    )

    print("=== ClickZetta LangChain Examples ===\n")

    # Example 1: SQL Chain
    print("1. SQL Chain Example")
    print("-" * 20)

    # Set up demo tables for SQL Chain (run once)
    workspace = engine.connection_config['workspace']
    schema = engine.connection_config['schema']

    sql_chain = ClickZettaSQLChain.from_engine(
        engine=engine,
        llm=llm,
        return_sql=True,
        top_k=5,
        table_names=[
            f"{workspace}.{schema}.demo_customers",
            f"{workspace}.{schema}.demo_orders",
            f"{workspace}.{schema}.demo_products"
        ]
    )

    try:
        result = sql_chain.invoke({
            "query": "Show me the top 5 customers by total spent from the demo_customers table"
        })
        print("Question: Show me the top 5 customers by total spent from the demo_customers table")
        print(f"SQL Query: {result.get('sql_query', 'N/A')}")
        print(f"Answer: {result['result']}")
    except Exception as e:
        print(f"SQL Chain Error: {e}")

    print("\n")

    # Example 2: Vector Store
    print("2. Vector Store Example")
    print("-" * 20)

    vector_store = ClickZettaVectorStore(
        engine=engine,
        embeddings=embeddings,
        table_name="example_vectors",
        vector_element_type="float"  # ClickZetta supports: float, int, tinyint
    )

    # Sample documents
    documents = [
        Document(
            page_content="ClickZetta is a high-performance cloud-native analytics database designed for big data processing.",
            metadata={"source": "docs", "category": "database"}
        ),
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "docs", "category": "framework"}
        ),
        Document(
            page_content="Vector databases enable semantic search and similarity matching for AI applications.",
            metadata={"source": "docs", "category": "ai"}
        )
    ]

    try:
        # Add documents
        ids = vector_store.add_documents(documents)
        print(f"Added {len(ids)} documents to vector store")

        # Search for similar documents
        query = "What is ClickZetta?"
        results = vector_store.similarity_search(query, k=2)

        print(f"\nQuery: {query}")
        print("Similar documents:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
            print(f"     Metadata: {doc.metadata}")

        # Search with scores
        results_with_scores = vector_store.similarity_search_with_score(query, k=2)
        print("\nSimilarity scores:")
        for doc, score in results_with_scores:
            print(f"  Score: {score:.3f} - {doc.page_content[:50]}...")

    except Exception as e:
        print(f"Vector Store Error: {e}")

    print("\n")

    # Example 3: Chat History
    print("3. Chat Message History Example")
    print("-" * 30)

    chat_history = ClickZettaChatMessageHistory(
        engine=engine,
        session_id="example_session_001",
        table_name="example_chat_history"
    )

    try:
        # Add some conversation
        messages_to_add = [
            HumanMessage(content="Hello! Can you help me understand ClickZetta?"),
            AIMessage(content="Of course! ClickZetta is a cloud-native analytics database. What would you like to know?"),
            HumanMessage(content="How does it integrate with LangChain?"),
            AIMessage(content="ClickZetta integrates with LangChain through this package, providing SQL chains, vector storage, and more!")
        ]

        for msg in messages_to_add:
            chat_history.add_message(msg)

        print("Added conversation to chat history")

        # Retrieve conversation
        stored_messages = chat_history.messages
        print(f"\nConversation history ({len(stored_messages)} messages):")
        for i, message in enumerate(stored_messages, 1):
            msg_type = "Human" if isinstance(message, HumanMessage) else "AI"
            print(f"  {i}. {msg_type}: {message.content}")

        # Get recent messages
        recent_messages = chat_history.get_messages_by_count(2)
        print("\nLast 2 messages:")
        for message in recent_messages:
            msg_type = "Human" if isinstance(message, HumanMessage) else "AI"
            print(f"  {msg_type}: {message.content}")

    except Exception as e:
        print(f"Chat History Error: {e}")

    print("\n")

    # Example 4: Full-text Search
    print("4. Full-text Search Example")
    print("-" * 25)

    from langchain_clickzetta.retrievers import ClickZettaFullTextRetriever

    fulltext_retriever = ClickZettaFullTextRetriever(
        engine=engine,
        table_name="example_documents",
        search_type="phrase",
        k=3
    )

    try:
        # Add documents for full-text search
        fulltext_retriever.add_documents(documents)
        print("Added documents to full-text search index")

        # Search documents
        query = "analytics database"
        results = fulltext_retriever.invoke(query)

        print(f"\nFull-text search for: '{query}'")
        print("Results:")
        for i, doc in enumerate(results, 1):
            relevance = doc.metadata.get('relevance_score', 0.0)
            print(f"  {i}. (Score: {relevance:.3f}) {doc.page_content[:80]}...")

    except Exception as e:
        print(f"Full-text Search Error: {e}")

    print("\n")

    # Example 5: True Hybrid Search (Single Table)
    print("5. True Hybrid Search Example")
    print("-" * 28)

    try:
        # Create true hybrid store (single table with vector + inverted indexes)
        hybrid_store = ClickZettaHybridStore(
            engine=engine,
            embeddings=embeddings,
            table_name="example_hybrid_docs",
            text_analyzer="ik",  # Chinese text analyzer
            distance_metric="cosine"
        )

        # Add documents to hybrid store
        hybrid_documents = [
            Document(page_content="ClickZetta是高性能云原生分析数据库", metadata={"lang": "zh"}),
            Document(page_content="LangChain framework enables building LLM applications", metadata={"lang": "en"}),
            Document(page_content="向量数据库支持语义搜索和相似性匹配", metadata={"lang": "zh"}),
        ]

        hybrid_store.add_documents(hybrid_documents)
        print("Added documents to hybrid store")

        # Create unified retriever
        unified_retriever = ClickZettaUnifiedRetriever(
            hybrid_store=hybrid_store,
            search_type="hybrid",  # "vector", "fulltext", or "hybrid"
            alpha=0.5,  # Balance between vector and full-text search
            k=3
        )

        query = "database analytics"
        results = unified_retriever.invoke(query)

        print(f"\nTrue hybrid search for: '{query}'")
        print("Results (single table with vector + inverted indexes):")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
            print(f"     Metadata: {doc.metadata}")

    except Exception as e:
        print(f"Hybrid Search Error: {e}")

    # Optional cleanup (uncomment to clean up test data)
    # cleanup_test_objects(engine)

    # Cleanup
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")

    print("\n=== Examples completed ===")


def cleanup_test_objects(engine):
    """Clean up test objects created during examples."""
    print("\n6. Cleaning up test objects...")
    print("-" * 30)

    # Tables created in this example
    test_tables = [
        "example_vectors",
        "example_chat_history",
        "example_documents",
        "example_hybrid_docs"
    ]

    workspace = engine.connection_config['workspace']
    schema = engine.connection_config['schema']

    for table_name in test_tables:
        full_table_name = f"{workspace}.{schema}.{table_name}"
        try:
            # Directly truncate table
            truncate_sql = f"TRUNCATE TABLE {full_table_name}"
            engine.execute_query(truncate_sql)
            print(f"✓ Cleaned table: {full_table_name}")

        except Exception as e:
            # If truncate fails, try drop
            try:
                drop_sql = f"DROP TABLE IF EXISTS {full_table_name}"
                engine.execute_query(drop_sql)
                print(f"✓ Dropped table: {full_table_name}")
            except Exception as drop_e:
                print(f"✗ Error cleaning table {full_table_name}: {e} / {drop_e}")

    print("✓ Test object cleanup completed")


if __name__ == "__main__":
    main()
