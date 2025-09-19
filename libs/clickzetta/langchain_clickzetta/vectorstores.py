"""ClickZetta vector store implementation for LangChain."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaVectorStore(VectorStore):
    """ClickZetta vector store for embedding storage and similarity search.

    This implementation uses ClickZetta's vector index capabilities for
    efficient similarity search and supports metadata filtering.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        embeddings: Embeddings,
        table_name: str = "langchain_vectors",
        embedding_column: str = "embedding",
        content_column: str = "content",
        metadata_column: str = "metadata",
        id_column: str = "id",
        distance_metric: str = "cosine",
        vector_dimension: int | None = None,
        vector_element_type: str = "float",  # ClickZetta supports: tinyint, int, float
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta vector store.

        Args:
            engine: ClickZetta database engine
            embeddings: Embedding model to use
            table_name: Name of the table to store vectors
            embedding_column: Name of the embedding column
            content_column: Name of the content column
            metadata_column: Name of the metadata column
            id_column: Name of the ID column
            distance_metric: Distance metric for similarity search
            vector_dimension: Dimension of the embedding vectors
            vector_element_type: Element type for vectors (float, int, tinyint)
            **kwargs: Additional arguments
        """
        super().__init__()
        self.engine = engine
        self._embeddings = embeddings
        # Ensure table name includes workspace and schema if not already specified
        if table_name.count(".") == 0:
            # No dots - add workspace.schema
            self.table_name = f"{engine.connection_config['workspace']}.{engine.connection_config['schema']}.{table_name}"
        elif table_name.count(".") == 1:
            # One dot - assume it's schema.table, add workspace
            self.table_name = f"{engine.connection_config['workspace']}.{table_name}"
        else:
            # Two or more dots - use as is (supports workspace.schema.table)
            self.table_name = table_name
        self.embedding_column = embedding_column
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.id_column = id_column
        self.distance_metric = distance_metric
        self.vector_dimension = vector_dimension
        self.vector_element_type = vector_element_type

        # Initialize table if it doesn't exist
        self._create_table_if_not_exists()

    @property
    def embeddings(self) -> Embeddings:
        """Access the query embedding object."""
        return self._embeddings

    def _create_table_if_not_exists(self) -> None:
        """Create the vector table if it doesn't exist."""
        # Get vector dimension if not provided
        if self.vector_dimension is None:
            try:
                # Embed a test string to get dimension
                test_embedding = self._embeddings.embed_query("test")
                self.vector_dimension = len(test_embedding)
                logger.info(f"Detected vector dimension: {self.vector_dimension}")
            except Exception as e:
                logger.warning(f"Could not detect vector dimension: {e}")
                self.vector_dimension = 1536  # Default for OpenAI embeddings

        # Map distance metric to ClickZetta function name
        distance_function_map = {
            "cosine": "cosine_distance",
            "euclidean": "l2_distance",
            "l2": "l2_distance",
            "manhattan": "l1_distance",
        }
        distance_function_map.get(
            self.distance_metric.lower(), "cosine_distance"
        )

        # Map vector element types to ClickZetta types and scalar types
        element_type_map = {
            "float": ("float", "f32"),
            "int": ("int", "f32"),  # Use f32 for index even with int data
            "tinyint": ("tinyint", "i8"),
        }

        column_type, scalar_type = element_type_map.get(
            self.vector_element_type.lower(), ("float", "f32")
        )

        # Create table with vector index using ClickZetta syntax
        # Note: ClickZetta doesn't support IF NOT EXISTS with vector indexes
        # Use unique index name to avoid conflicts (extract only table name part)
        self.table_name.split(".")[-1]  # Get only the table name part
        create_table_sql = f"""
        CREATE TABLE {self.table_name} (
            {self.id_column} String,
            {self.content_column} String,
            {self.metadata_column} String,
            {self.embedding_column} vector({column_type}, {self.vector_dimension})
        )
        """

        # Check if table already exists first
        try:
            check_results, _ = self.engine.execute_query(
                f"SHOW TABLES LIKE '{self.table_name}'"
            )
            if check_results:
                logger.info(f"Vector table '{self.table_name}' already exists")
                return
        except Exception as e:
            logger.warning(
                f"Could not check if table exists: {e}, proceeding with creation"
            )

        try:
            logger.info(f"Creating vector table with SQL: {create_table_sql}")
            results, _ = self.engine.execute_query(create_table_sql)
            logger.info(f"Vector table '{self.table_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create vector table: {e}")
            # Check if it's because the table already exists
            if "AlreadyExist" in str(e) or "already exists" in str(e).lower():
                logger.info(
                    f"Vector table '{self.table_name}' already exists (from error)"
                )
                return
            raise RuntimeError(
                f"Cannot create vector table '{self.table_name}': {e}"
            ) from e

    def _format_vector(self, vector: list[float]) -> str:
        """Format vector for ClickZetta insertion.

        ClickZetta supports vector() constructor or CAST from array/string.
        Using vector() constructor is the most direct approach.
        """
        # Ensure all values are properly formatted as floats
        formatted_values = []
        for val in vector:
            if isinstance(val, (int, float)):
                formatted_values.append(str(float(val)))
            else:
                # Handle potential None or other invalid values
                formatted_values.append("0.0")

        return f"vector({','.join(formatted_values)})"

    def _format_vector_as_cast(self, vector: list[float]) -> str:
        """Alternative vector formatting using CAST from array.

        This is a fallback method in case direct vector() constructor fails.
        ClickZetta supports: CAST(array_col as vector(float, dimension))
        """
        formatted_values = []
        for val in vector:
            if isinstance(val, (int, float)):
                formatted_values.append(str(float(val)))
            else:
                formatted_values.append("0.0")

        array_str = f"[{','.join(formatted_values)}]"
        return (
            f"CAST('{array_str}' as vector({self.vector_element_type}, {len(vector)}))"
        )

    def _parse_metadata(self, metadata: dict[str, Any] | None) -> str:
        """Parse metadata to JSON string."""
        if metadata is None:
            return "{}"
        return json.dumps(metadata, ensure_ascii=False)

    def _unparse_metadata(self, metadata_str: str) -> dict[str, Any]:
        """Parse metadata from JSON string."""
        try:
            return json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            return {}

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vectorstore
            metadatas: Optional list of metadatas associated with the texts
            ids: Optional list of ids to associate with the texts
            **kwargs: Additional arguments

        Returns:
            List of IDs for the added texts
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(texts_list)

        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(texts_list)

        # Prepare batch insert
        values = []
        for i, (text, embedding, metadata) in enumerate(
            zip(texts_list, embeddings, metadatas)
        ):
            vector_str = self._format_vector(embedding)
            metadata_str = self._parse_metadata(metadata)

            escaped_text = text.replace("'", "''")
            escaped_metadata = metadata_str.replace("'", "''")
            values.append(
                f"('{ids[i]}', '{escaped_text}', '{escaped_metadata}', {vector_str})"
            )

        # Batch insert
        insert_sql = f"""
        INSERT INTO {self.table_name}
        ({self.id_column}, {self.content_column}, {self.metadata_column}, {self.embedding_column})
        VALUES {', '.join(values)}
        """

        try:
            self.engine.execute_query(insert_sql)
            logger.info(f"Added {len(texts_list)} texts to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query with scores.

        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of (Document, score) tuples
        """
        # Get query embedding
        query_embedding = self._embeddings.embed_query(query)
        query_vector = self._format_vector(query_embedding)

        # Build WHERE clause for metadata filtering
        where_clause = ""
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(
                        f"JSONExtractString({self.metadata_column}, '{key}') = '{value}'"
                    )
                elif isinstance(value, (int, float)):
                    conditions.append(
                        f"JSONExtractFloat({self.metadata_column}, '{key}') = {value}"
                    )
                elif isinstance(value, bool):
                    conditions.append(
                        f"JSONExtractBool({self.metadata_column}, '{key}') = {value}"
                    )

            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

        # Get the correct distance function for ClickZetta
        distance_function_map = {
            "cosine": "COSINE_DISTANCE",
            "euclidean": "L2_DISTANCE",
            "l2": "L2_DISTANCE",
            "manhattan": "L1_DISTANCE",
        }
        distance_func = distance_function_map.get(
            self.distance_metric.lower(), "COSINE_DISTANCE"
        )

        # Vector similarity search query using ClickZetta syntax
        search_sql = f"""
        SELECT
            {self.id_column},
            {self.content_column},
            {self.metadata_column},
            {distance_func}({self.embedding_column}, {query_vector}) AS distance
        FROM {self.table_name}
        {where_clause}
        ORDER BY distance ASC
        LIMIT {k}
        """

        try:
            results, _ = self.engine.execute_query(search_sql)

            # Convert results to Documents with scores
            docs_with_scores = []
            for row in results:
                content = row[self.content_column]
                metadata = self._unparse_metadata(row[self.metadata_column])
                distance = row["distance"]

                # Convert distance to similarity score
                # For ClickZetta: lower distance = higher similarity
                # For cosine distance, range is [0,2], we convert to similarity [0,1]
                if self.distance_metric.lower() == "cosine":
                    score = max(
                        0, 1.0 - distance / 2.0
                    )  # Normalize cosine distance to similarity
                else:
                    # For L2/Euclidean distance, use reciprocal for similarity
                    score = 1.0 / (1.0 + distance)

                doc = Document(page_content=content, metadata=metadata)
                docs_with_scores.append((doc, score))

            logger.debug(f"Found {len(docs_with_scores)} similar documents")
            return docs_with_scores

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of Documents most similar to the query
        """
        docs_with_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to
            k: Number of Documents to return
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of Documents most similar to the query vector
        """
        query_vector = self._format_vector(embedding)

        # Build WHERE clause for metadata filtering
        where_clause = ""
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(
                        f"JSONExtractString({self.metadata_column}, '{key}') = '{value}'"
                    )
                elif isinstance(value, (int, float)):
                    conditions.append(
                        f"JSONExtractFloat({self.metadata_column}, '{key}') = {value}"
                    )

            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

        # Get distance function for ClickZetta
        distance_function_map = {
            "cosine": "COSINE_DISTANCE",
            "euclidean": "L2_DISTANCE",
            "l2": "L2_DISTANCE",
            "manhattan": "L1_DISTANCE",
        }
        distance_func = distance_function_map.get(
            self.distance_metric.lower(), "COSINE_DISTANCE"
        )

        # Vector similarity search query using ClickZetta syntax
        search_sql = f"""
        SELECT
            {self.content_column},
            {self.metadata_column}
        FROM {self.table_name}
        {where_clause}
        ORDER BY {distance_func}({self.embedding_column}, {query_vector}) ASC
        LIMIT {k}
        """

        try:
            results, _ = self.engine.execute_query(search_sql)

            # Convert results to Documents
            docs = []
            for row in results:
                content = row[self.content_column]
                metadata = self._unparse_metadata(row[self.metadata_column])
                docs.append(Document(page_content=content, metadata=metadata))

            return docs

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            raise

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector IDs.

        Args:
            ids: List of IDs to delete
            **kwargs: Additional arguments

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
            logger.info(f"Deleted {len(ids)} vectors from store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to
            k: Number of Documents to return
            fetch_k: Number of Documents to fetch before filtering to k
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity
            filter: Optional metadata filter
            **kwargs: Additional arguments

        Returns:
            List of Documents selected by maximal marginal relevance
        """
        # Get more candidates than needed
        docs_with_scores = self.similarity_search_with_score(
            query, fetch_k, filter, **kwargs
        )

        if len(docs_with_scores) <= k:
            return [doc for doc, _ in docs_with_scores]

        # Get embeddings for all candidate documents
        doc_texts = [doc.page_content for doc, _ in docs_with_scores]
        doc_embeddings = self._embeddings.embed_documents(doc_texts)
        query_embedding = self._embeddings.embed_query(query)

        # Implement MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(docs_with_scores)))

        # Select first document (most similar to query)
        selected_indices.append(0)
        remaining_indices.remove(0)

        # Select remaining documents using MMR
        for _ in range(k - 1):
            if not remaining_indices:
                break

            best_score = float("-inf")
            best_idx = None

            for idx in remaining_indices:
                # Similarity to query
                query_sim = np.dot(query_embedding, doc_embeddings[idx]) / (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(doc_embeddings[idx])
                )

                # Maximum similarity to already selected documents
                max_sim = 0
                for selected_idx in selected_indices:
                    sim = np.dot(doc_embeddings[idx], doc_embeddings[selected_idx]) / (
                        np.linalg.norm(doc_embeddings[idx])
                        * np.linalg.norm(doc_embeddings[selected_idx])
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Return selected documents
        return [docs_with_scores[idx][0] for idx in selected_indices]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        engine: ClickZettaEngine,
        table_name: str = "langchain_vectors",
        vector_element_type: str = "float",
        **kwargs: Any,
    ) -> ClickZettaVectorStore:
        """Create a ClickZettaVectorStore from a list of texts.

        Args:
            texts: List of texts to add to the vectorstore
            embeddings: Embedding model to use
            engine: ClickZetta database engine
            metadatas: Optional list of metadatas
            table_name: Name of the table to store vectors
            vector_element_type: Element type for vectors (float, int, tinyint)
            **kwargs: Additional arguments

        Returns:
            ClickZettaVectorStore instance
        """
        vector_store = cls(
            engine=engine,
            embeddings=embedding,
            table_name=table_name,
            vector_element_type=vector_element_type,
            **kwargs,
        )
        vector_store.add_texts(texts, metadatas)
        return vector_store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings,
        *,
        engine: ClickZettaEngine,
        table_name: str = "langchain_vectors",
        vector_element_type: str = "float",
        **kwargs: Any,
    ) -> ClickZettaVectorStore:
        """Create a ClickZettaVectorStore from a list of Documents.

        Args:
            documents: List of Documents to add to the vectorstore
            embeddings: Embedding model to use
            engine: ClickZetta database engine
            table_name: Name of the table to store vectors
            vector_element_type: Element type for vectors (float, int, tinyint)
            **kwargs: Additional arguments

        Returns:
            ClickZettaVectorStore instance
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            engine=engine,
            table_name=table_name,
            vector_element_type=vector_element_type,
            **kwargs,
        )
