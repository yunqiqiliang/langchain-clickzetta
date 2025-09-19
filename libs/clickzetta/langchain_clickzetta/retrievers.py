"""ClickZetta retrievers for full-text search and document retrieval."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaFullTextRetriever(BaseRetriever):
    """ClickZetta full-text search retriever.

    This retriever uses ClickZetta's full-text search capabilities
    to find relevant documents based on text queries.
    """

    engine: ClickZettaEngine = Field(exclude=True)
    table_name: str = "langchain_documents"
    content_column: str = "content"
    metadata_column: str = "metadata"
    id_column: str = "id"
    search_type: str = "phrase"  # "phrase", "boolean", "natural"
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        engine: ClickZettaEngine,
        table_name: str = "langchain_documents",
        content_column: str = "content",
        metadata_column: str = "metadata",
        id_column: str = "id",
        search_type: str = "phrase",
        k: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta full-text retriever.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the table containing documents
            content_column: Name of the content column
            metadata_column: Name of the metadata column
            id_column: Name of the ID column
            search_type: Type of full-text search ("phrase", "boolean", "natural")
            k: Number of documents to retrieve
            **kwargs: Additional arguments
        """
        # Ensure table name includes workspace and schema if not already specified
        if table_name.count(".") == 0:
            # No dots - add workspace.schema
            table_name = f"{engine.connection_config['workspace']}.{engine.connection_config['schema']}.{table_name}"
        elif table_name.count(".") == 1:
            # One dot - assume it's schema.table, add workspace
            table_name = f"{engine.connection_config['workspace']}.{table_name}"
        # else: Two or more dots - use as is (supports workspace.schema.table)

        super().__init__(
            engine=engine,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            id_column=id_column,
            search_type=search_type,
            k=k,
            **kwargs,
        )

        # Initialize full-text search table if needed
        self._create_fulltext_index_if_not_exists()

    def _create_fulltext_index_if_not_exists(self) -> None:
        """Create full-text search table if it doesn't exist."""
        # Create the table first without index due to ClickZetta syntax limitations
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.id_column} String,
            {self.content_column} String,
            {self.metadata_column} String
        )
        """

        try:
            self.engine.execute_query(create_table_sql)
            logger.info(
                f"Full-text search table '{self.table_name}' created or verified"
            )

            # Try to create inverted index separately (optional)
            self._create_fulltext_index_optional()

        except Exception as e:
            logger.error(f"Failed to create full-text search table: {e}")
            raise

    def _create_fulltext_index_optional(self) -> None:
        """Try to create full-text index separately - failures are non-fatal."""
        try:
            # Generate unique index name based on table name to avoid global conflicts
            import hashlib

            table_hash = hashlib.md5(self.table_name.encode()).hexdigest()[:8]
            index_name = f"{self.content_column}_fts_{table_hash}"

            # Always try to create index, let ClickZetta handle duplicates
            index_sql = f"""
            CREATE INVERTED INDEX {index_name} ON TABLE {self.table_name}({self.content_column})
            PROPERTIES('analyzer'='unicode')
            """
            self.engine.execute_query(index_sql)
            logger.info(
                f"Full-text index '{index_name}' created for table '{self.table_name}'"
            )

            # Build index for existing data (only if creation succeeded)
            try:
                build_sql = f"BUILD INDEX {index_name} ON {self.table_name}"
                self.engine.execute_query(build_sql)
                logger.info(f"Full-text index '{index_name}' built for existing data")
            except Exception as build_error:
                logger.warning(f"Failed to build inverted index: {build_error}")

        except Exception as e:
            # Check if it's actually an "already exists" error
            if "AlreadyExist" in str(e) or "already exist" in str(e).lower():
                logger.info(f"Full-text index '{index_name}' already exists")
                # For existing indexes, skip BUILD to avoid orphaned index issues
            else:
                logger.warning(f"Could not create full-text index: {e}")

    def _index_exists(self, index_name: str) -> bool:
        """Check if an index exists globally."""
        try:
            # ClickZetta uses global index names, so check by DESC INDEX
            check_sql = f"DESC INDEX {index_name}"
            results, columns = self.engine.execute_query(check_sql)

            # If DESC INDEX succeeds, the index exists
            # Also verify it belongs to our table
            for row in results:
                if isinstance(row, dict):
                    if (
                        row.get("info_name") == "table_name"
                        and row.get("info_value") in self.table_name
                    ):
                        return True

            return False
        except Exception:
            # If DESC INDEX fails, index doesn't exist
            return False

    def _parse_metadata(self, metadata_str: str) -> dict[str, Any]:
        """Parse metadata from JSON string."""
        try:
            return json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            return {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use

        Returns:
            List of relevant documents
        """
        # Build ClickZetta full-text search query based on search type
        escaped_query = query.replace("'", "''")  # Escape single quotes

        if self.search_type == "phrase":
            # Use match_phrase for exact phrase matching
            search_condition = f"match_phrase({self.content_column}, '{escaped_query}', map('analyzer', 'unicode'))"
        elif self.search_type == "boolean":
            # Use match_all for boolean-like matching (all terms must match)
            search_condition = f"match_all({self.content_column}, '{escaped_query}', map('analyzer', 'unicode'))"
        elif self.search_type == "natural":
            # Use match_any for natural language search (any term can match)
            search_condition = f"match_any({self.content_column}, '{escaped_query}', map('analyzer', 'unicode'))"
        else:
            # Default to match_phrase for phrase search
            search_condition = f"match_phrase({self.content_column}, '{escaped_query}', map('analyzer', 'unicode'))"

        # Full-text search query - ClickZetta doesn't have score() function, use constant
        search_sql = f"""
        SELECT
            {self.id_column},
            {self.content_column},
            {self.metadata_column},
            1.0 AS relevance_score
        FROM {self.table_name}
        WHERE {search_condition}
        LIMIT {self.k}
        """

        try:
            results, _ = self.engine.execute_query(search_sql)

            # Convert results to Documents
            documents = []
            for row in results:
                content = row[self.content_column]
                metadata = self._parse_metadata(row[self.metadata_column])

                # Add relevance score to metadata
                metadata["relevance_score"] = row.get("relevance_score", 0.0)
                metadata["document_id"] = row[self.id_column]

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            logger.debug(
                f"Found {len(documents)} relevant documents for query: {query}"
            )
            return documents

        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            # Return empty list on failure
            return []

    def add_documents(
        self, documents: list[Document], ids: list[str] | None = None
    ) -> list[str]:
        """Add documents to the full-text search index.

        Args:
            documents: List of documents to add
            ids: Optional list of document IDs

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Generate IDs if not provided
        if ids is None:
            import uuid

            ids = [str(uuid.uuid4()) for _ in documents]

        # Prepare batch insert
        values = []
        for i, doc in enumerate(documents):
            content = doc.page_content.replace("'", "''")
            metadata_str = json.dumps(doc.metadata, ensure_ascii=False).replace(
                "'", "''"
            )

            values.append(f"('{ids[i]}', '{content}', '{metadata_str}')")

        # Batch insert
        insert_sql = f"""
        INSERT INTO {self.table_name}
        ({self.id_column}, {self.content_column}, {self.metadata_column})
        VALUES {', '.join(values)}
        """

        try:
            self.engine.execute_query(insert_sql)
            logger.info(f"Added {len(documents)} documents to full-text search index")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents from the full-text search index.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if deletion was successful
        """
        if not ids:
            return False

        ids_str = "', '".join(ids)
        delete_sql = (
            f"DELETE FROM {self.table_name} WHERE {self.id_column} IN ('{ids_str}')"
        )

        try:
            self.engine.execute_query(delete_sql)
            logger.info(f"Deleted {len(ids)} documents from full-text search index")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False


class ClickZettaHybridRetriever(BaseRetriever):
    """Hybrid retriever combining full-text search and vector similarity.

    This retriever combines the results from both full-text search and
    vector similarity search to provide comprehensive document retrieval.
    """

    fulltext_retriever: ClickZettaFullTextRetriever = Field(exclude=True)
    vector_store: Any = Field(exclude=True)  # ClickZettaVectorStore
    alpha: float = (
        0.5  # Weight for combining scores (0.0 = only full-text, 1.0 = only vector)
    )
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        fulltext_retriever: ClickZettaFullTextRetriever,
        vector_store: Any,  # ClickZettaVectorStore
        alpha: float = 0.5,
        k: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            fulltext_retriever: Full-text search retriever
            vector_store: Vector store for similarity search
            alpha: Weight for combining scores (0.0-1.0)
            k: Number of documents to retrieve
            **kwargs: Additional arguments
        """
        super().__init__(
            fulltext_retriever=fulltext_retriever,
            vector_store=vector_store,
            alpha=alpha,
            k=k,
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Get documents using hybrid search.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use

        Returns:
            List of relevant documents
        """
        # Get results from both retrievers
        # Retrieve more documents from each to ensure diversity after fusion
        retrieval_k = max(self.k * 2, 10)

        # Full-text search
        fulltext_retriever_temp = ClickZettaFullTextRetriever(
            engine=self.fulltext_retriever.engine,
            table_name=self.fulltext_retriever.table_name,
            content_column=self.fulltext_retriever.content_column,
            metadata_column=self.fulltext_retriever.metadata_column,
            id_column=self.fulltext_retriever.id_column,
            search_type=self.fulltext_retriever.search_type,
            k=retrieval_k,
        )
        fulltext_docs = fulltext_retriever_temp._get_relevant_documents(
            query, run_manager=run_manager
        )

        # Vector similarity search
        vector_docs_with_scores = self.vector_store.similarity_search_with_score(
            query, k=retrieval_k
        )
        vector_docs = [doc for doc, score in vector_docs_with_scores]

        # Create score mappings with explicit float conversion
        fulltext_scores = {}
        for doc in fulltext_docs:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            relevance_score = doc.metadata.get("relevance_score", 0.0)
            # Normalize fulltext score (assuming it's already 0-1 range) and ensure float
            fulltext_scores[doc_id] = float(relevance_score)

        vector_scores = {}
        for doc, score in vector_docs_with_scores:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            # Vector scores are similarity scores (higher is better) - ensure float
            vector_scores[doc_id] = float(score)

        # Combine documents and calculate hybrid scores
        all_docs = {}
        for doc in fulltext_docs:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            all_docs[doc_id] = doc

        for doc in vector_docs:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            all_docs[doc_id] = doc

        # Calculate hybrid scores
        hybrid_scores = []
        for doc_id, doc in all_docs.items():
            fulltext_score = float(fulltext_scores.get(doc_id, 0.0))
            vector_score = float(vector_scores.get(doc_id, 0.0))

            # Normalize scores to 0-1 range if needed
            if fulltext_scores:
                max_fulltext = float(max(fulltext_scores.values()))
                if max_fulltext > 0:
                    fulltext_score = fulltext_score / max_fulltext

            if vector_scores:
                max_vector = float(max(vector_scores.values()))
                if max_vector > 0:
                    vector_score = vector_score / max_vector

            # Calculate hybrid score
            hybrid_score = (1 - self.alpha) * fulltext_score + self.alpha * vector_score

            # Add hybrid score to metadata
            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["fulltext_score"] = fulltext_score
            doc.metadata["vector_score"] = vector_score

            hybrid_scores.append((doc, hybrid_score))

        # Sort by hybrid score and return top k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in hybrid_scores[: self.k]]

        logger.debug(f"Hybrid search returned {len(final_docs)} documents")
        return final_docs

    @classmethod
    def from_engines(
        cls,
        engine: ClickZettaEngine,
        vector_store: Any,  # ClickZettaVectorStore
        fulltext_table: str = "langchain_documents",
        alpha: float = 0.5,
        k: int = 4,
        **kwargs: Any,
    ) -> ClickZettaHybridRetriever:
        """Create hybrid retriever from ClickZetta engine and vector store.

        Args:
            engine: ClickZetta database engine
            vector_store: Vector store for similarity search
            fulltext_table: Table name for full-text search
            alpha: Weight for combining scores
            k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            ClickZettaHybridRetriever instance
        """
        fulltext_retriever = ClickZettaFullTextRetriever(
            engine=engine,
            table_name=fulltext_table,
            k=k * 2,  # Retrieve more for fusion
            **kwargs,
        )

        return cls(
            fulltext_retriever=fulltext_retriever,
            vector_store=vector_store,
            alpha=alpha,
            k=k,
        )
