"""Advanced RAG (Retrieval-Augmented Generation) example using ClickZetta."""

import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_clickzetta import (
    ClickZettaChatMessageHistory,
    ClickZettaEngine,
    ClickZettaHybridStore,
    ClickZettaSQLChain,
    ClickZettaUnifiedRetriever,
    ClickZettaVectorStore,
)


class ClickZettaRAGSystem:
    """Advanced RAG system using ClickZetta as the backend."""

    def __init__(self, engine: ClickZettaEngine, llm: Any, embeddings: Any):
        """Initialize the RAG system."""
        self.engine = engine
        self.llm = llm
        self.embeddings = embeddings

        # Initialize components
        self.vector_store = ClickZettaVectorStore(
            engine=engine,
            embeddings=embeddings,
            table_name="rag_documents"
        )

        self.sql_chain = ClickZettaSQLChain.from_engine(
            engine=engine,
            llm=llm,
            return_sql=False,
            top_k=10
        )

        # Initialize hybrid store (single table with vector + inverted indexes)
        self.hybrid_store = ClickZettaHybridStore(
            engine=engine,
            embeddings=embeddings,
            table_name="rag_hybrid_docs",
            text_analyzer="ik",
            distance_metric="cosine"
        )

        self.hybrid_retriever = ClickZettaUnifiedRetriever(
            hybrid_store=self.hybrid_store,
            search_type="hybrid",
            alpha=0.6,  # Favor vector search slightly
            k=5
        )

        # RAG prompt template
        self.rag_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer the user's question accurately and comprehensively.

Context from knowledge base:
{context}

Database query result (if applicable):
{sql_result}

User Question: {question}

Instructions:
1. Use the provided context to answer the question
2. If database results are available, incorporate them into your answer
3. Be specific and cite relevant information from the context
4. If you cannot answer based on the context, say so clearly
5. Provide a helpful and well-structured response

Answer:""")

        # Setup the RAG chain
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Create the RAG processing chain."""
        # Parallel retrieval: get documents and potentially run SQL
        retrieval_chain = RunnableParallel({
            "context": self._format_docs,
            "sql_result": self._try_sql_query,
            "question": RunnablePassthrough()
        })

        # Complete RAG chain
        return retrieval_chain | self.rag_prompt | self.llm | StrOutputParser()

    def _format_docs(self, question: str) -> str:
        """Retrieve and format relevant documents."""
        try:
            docs = self.hybrid_retriever.invoke(question)
            if not docs:
                return "No relevant documents found."

            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                hybrid_score = doc.metadata.get('hybrid_score', 0.0)
                source = doc.metadata.get('source', 'Unknown')
                formatted_docs.append(
                    f"Document {i} (Score: {hybrid_score:.3f}, Source: {source}):\n{doc.page_content}\n"
                )

            return "\n".join(formatted_docs)
        except Exception as e:
            return f"Error retrieving documents: {e}"

    def _try_sql_query(self, question: str) -> str:
        """Try to answer using SQL if the question seems data-related."""
        # Simple heuristics to determine if question might need SQL
        sql_keywords = ['count', 'how many', 'total', 'sum', 'average', 'maximum', 'minimum',
                       'users', 'orders', 'sales', 'revenue', 'data', 'records', 'table']

        question_lower = question.lower()
        if any(keyword in question_lower for keyword in sql_keywords):
            try:
                result = self.sql_chain.invoke({"query": question})
                return result.get("result", "")
            except Exception as e:
                return f"SQL query error: {e}"

        return "No SQL query attempted."

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the hybrid knowledge base."""
        return self.hybrid_store.add_documents(documents)

    def ask(self, question: str, session_id: str = None) -> dict[str, Any]:
        """Ask a question and get an answer using RAG."""
        # Store conversation if session_id provided
        chat_history = None
        if session_id:
            chat_history = ClickZettaChatMessageHistory(
                engine=self.engine,
                session_id=session_id,
                table_name="rag_conversations"
            )
            chat_history.add_message(HumanMessage(content=question))

        try:
            # Get answer using RAG
            answer = self.rag_chain.invoke(question)

            # Store AI response if session provided
            if chat_history:
                chat_history.add_message(AIMessage(content=answer))

            # Get retrieved context for transparency
            context_docs = self.hybrid_retriever.invoke(question)
            sql_result = self._try_sql_query(question)

            return {
                "answer": answer,
                "context_documents": len(context_docs),
                "sql_executed": "No SQL query attempted." not in sql_result,
                "session_id": session_id
            }

        except Exception as e:
            error_msg = f"Error processing question: {e}"
            if chat_history:
                chat_history.add_message(AIMessage(content=error_msg))
            return {
                "answer": error_msg,
                "context_documents": 0,
                "sql_executed": False,
                "session_id": session_id
            }


def main():
    """Run the advanced RAG example."""
    print("=== Advanced ClickZetta RAG System ===\n")

    # Initialize components
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
    llm = Tongyi(dashscope_api_key=api_key, temperature=0.1)
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model="text-embedding-v4"
    )

    # Create RAG system
    rag_system = ClickZettaRAGSystem(engine, llm, embeddings)

    # Sample knowledge base documents
    knowledge_base = [
        Document(
            page_content="""
            ClickZetta Architecture Overview:
            ClickZetta is a cloud-native MPP (Massively Parallel Processing) analytics database
            that separates storage and compute. It features:
            - Distributed columnar storage with advanced compression
            - Vector index support for AI/ML workloads
            - Real-time and batch processing capabilities
            - Auto-scaling compute resources
            - SQL and REST API interfaces
            """,
            metadata={"source": "architecture_guide", "category": "technical", "version": "2024.1"}
        ),
        Document(
            page_content="""
            ClickZetta Performance Optimization:
            To achieve optimal performance in ClickZetta:
            1. Use columnar data formats (Parquet recommended)
            2. Implement proper partitioning strategies
            3. Utilize vector indexes for similarity search
            4. Enable result caching for frequently accessed data
            5. Use appropriate compression algorithms
            6. Optimize query patterns for parallel execution
            """,
            metadata={"source": "performance_guide", "category": "optimization", "version": "2024.1"}
        ),
        Document(
            page_content="""
            LangChain Integration Features:
            The langchain-clickzetta package provides:
            - ClickZettaSQLChain for natural language to SQL conversion
            - ClickZettaVectorStore for embedding storage and similarity search
            - ClickZettaChatMessageHistory for conversation persistence
            - ClickZettaFullTextRetriever for advanced text search
            - ClickZettaHybridRetriever combining vector and full-text search
            - Support for metadata filtering and custom distance metrics
            """,
            metadata={"source": "integration_guide", "category": "langchain", "version": "0.1.0"}
        ),
        Document(
            page_content="""
            Vector Search Capabilities:
            ClickZetta supports advanced vector operations:
            - Multiple distance metrics (cosine, euclidean, manhattan)
            - High-dimensional vector storage and indexing
            - Approximate nearest neighbor (ANN) search
            - Metadata filtering for vector queries
            - Batch vector operations for efficiency
            - Integration with popular embedding models
            """,
            metadata={"source": "vector_guide", "category": "ai", "version": "2024.1"}
        ),
        Document(
            page_content="""
            Use Cases and Applications:
            ClickZetta is ideal for:
            - Real-time analytics and reporting
            - Machine learning feature stores
            - IoT data processing and analysis
            - Customer 360 analytics
            - Financial risk modeling
            - Recommendation systems with vector similarity
            - Log analysis and monitoring
            - Data lakehouse architectures
            """,
            metadata={"source": "use_cases", "category": "business", "version": "2024.1"}
        )
    ]

    print("1. Setting up knowledge base...")
    try:
        doc_ids = rag_system.add_documents(knowledge_base)
        print(f"✓ Added {len(doc_ids)} documents to knowledge base")
    except Exception as e:
        print(f"✗ Error setting up knowledge base: {e}")
        return

    print("\n2. Testing RAG system with various questions...\n")

    # Test questions covering different aspects
    test_questions = [
        "What is ClickZetta and what are its key features?",
        "How can I optimize performance in ClickZetta?",
        "What LangChain integration features are available?",
        "Explain vector search capabilities in ClickZetta",
        "What are some common use cases for ClickZetta?",
        "How many different distance metrics are supported for vector search?"
    ]

    session_id = "advanced_rag_demo"

    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 50)

        try:
            result = rag_system.ask(question, session_id=session_id)

            print(f"Answer: {result['answer']}")
            print(f"Context documents used: {result['context_documents']}")
            print(f"SQL executed: {result['sql_executed']}")
            print()

        except Exception as e:
            print(f"Error: {e}\n")

    print("\n3. Conversation History Analysis")
    print("-" * 30)

    # Analyze conversation history
    try:
        chat_history = ClickZettaChatMessageHistory(
            engine=engine,
            session_id=session_id,
            table_name="rag_conversations"
        )

        messages = chat_history.messages
        print(f"Total conversation turns: {len(messages)}")

        # Show last few exchanges
        recent_messages = chat_history.get_messages_by_count(6)  # Last 3 Q&A pairs
        print("\nLast 3 question-answer pairs:")
        for i in range(0, len(recent_messages), 2):
            if i + 1 < len(recent_messages):
                human_msg = recent_messages[i]
                ai_msg = recent_messages[i + 1]
                print(f"\nQ: {human_msg.content}")
                print(f"A: {ai_msg.content[:200]}..." if len(ai_msg.content) > 200 else f"A: {ai_msg.content}")

    except Exception as e:
        print(f"Error analyzing conversation: {e}")

    print("\n4. Advanced Search Demonstration")
    print("-" * 35)

    # Demonstrate different search strategies
    search_queries = [
        ("performance optimization", "Vector + Full-text hybrid search"),
        ("machine learning", "Semantic vector search"),
        ("SQL conversion", "Keyword-based full-text search")
    ]

    for query, strategy in search_queries:
        print(f"\nQuery: '{query}' using {strategy}")
        try:
            docs = rag_system.hybrid_retriever.invoke(query)
            for doc in docs[:2]:  # Show top 2 results
                score_info = f"Hybrid: {doc.metadata.get('hybrid_score', 0):.3f}"
                print(f"  - {score_info} | {doc.page_content[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")

    # Optional cleanup (uncomment to clean up test data)
    # cleanup_rag_objects(engine)

    # Cleanup
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")

    print("\n=== Advanced RAG Demo Complete ===")


def cleanup_rag_objects(engine):
    """Clean up RAG test objects."""
    print("\n5. Cleaning up RAG test objects...")
    print("-" * 35)

    # Tables created in RAG example
    rag_tables = [
        "rag_documents",
        "rag_hybrid_docs",
        "rag_conversations"
    ]

    workspace = engine.connection_config['workspace']
    schema = engine.connection_config['schema']

    for table_name in rag_tables:
        full_table_name = f"{workspace}.{schema}.{table_name}"
        try:
            # Check if table exists
            check_sql = f"SHOW TABLES LIKE '{table_name}'"
            results, _ = engine.execute_query(check_sql)

            if results:
                # Truncate table (faster than DROP for large tables)
                truncate_sql = f"TRUNCATE TABLE {full_table_name}"
                engine.execute_query(truncate_sql)
                print(f"✓ Cleaned table: {full_table_name}")
            else:
                print(f"- Table not found: {full_table_name}")

        except Exception as e:
            print(f"✗ Error cleaning table {full_table_name}: {e}")

    print("✓ RAG object cleanup completed")


if __name__ == "__main__":
    main()
