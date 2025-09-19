"""ClickZetta hybrid store that supports both vector and full-text search in a single table."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaHybridStore(VectorStore):
    """ClickZetta hybrid store with both vector and full-text search capabilities.

    This store creates a single table with both vector embeddings and inverted index
    for true hybrid search capabilities within ClickZetta.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        embeddings: Embeddings,
        table_name: str = "langchain_hybrid_documents",
        vector_dimension: int | None = None,
        vector_element_type: str = "float",
        distance_metric: str = "cosine",
        text_analyzer: str = "unicode",
        text_mode: str = "smart",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta hybrid store.

        Args:
            engine: ClickZetta database engine
            embeddings: Embeddings model for vector generation
            table_name: Name of the hybrid table
            vector_dimension: Dimension of vector embeddings
            vector_element_type: Type of vector elements (float, int, tinyint)
            distance_metric: Distance metric for vector search (cosine, euclidean, l2)
            text_analyzer: Text analyzer for full-text search (unicode, chinese, english, keyword)
            text_mode: Text analysis mode (smart, max_word)
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
        self.vector_dimension = vector_dimension
        self.vector_element_type = vector_element_type
        self.distance_metric = distance_metric
        self.text_analyzer = text_analyzer
        self.text_mode = text_mode

        # Auto-detect vector dimension if not provided
        if self.vector_dimension is None:
            try:
                test_embedding = self._embeddings.embed_query("test")
                self.vector_dimension = len(test_embedding)
                logger.info(f"Detected vector dimension: {self.vector_dimension}")
            except Exception as e:
                logger.warning(f"Could not detect vector dimension: {e}")
                self.vector_dimension = 1536  # Default

        # Create hybrid table
        self._create_hybrid_table()

    @property
    def embeddings(self) -> Embeddings:
        """Access the query embedding object."""
        return self._embeddings

    def _create_hybrid_table(self) -> None:
        """Create hybrid table with both vector and inverted indexes."""
        # Map distance metric to ClickZetta function name
        distance_function_map = {
            "cosine": "cosine_distance",
            "euclidean": "l2_distance",
            "l2": "l2_distance",
            "manhattan": "l1_distance",
        }
        distance_function = distance_function_map.get(
            self.distance_metric.lower(), "cosine_distance"
        )

        # Map vector element types
        element_type_map = {
            "float": ("float", "f32"),
            "int": ("int", "f32"),
            "tinyint": ("tinyint", "i8"),
        }
        column_type, scalar_type = element_type_map.get(
            self.vector_element_type.lower(), ("float", "f32")
        )

        # Create hybrid table with both vector and inverted indexes
        # Note: Create table first, then add indexes separately due to ClickZetta syntax limitations
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id String,
            content String,
            metadata String,
            embedding vector({column_type}, {self.vector_dimension})
        )
        """

        try:
            self.engine.execute_query(create_table_sql)
            logger.info(f"Hybrid table '{self.table_name}' created successfully")

            # Try to create indexes separately (optional - failures are logged but not fatal)
            self._create_indexes_if_possible(distance_function, scalar_type)

        except Exception as e:
            logger.error(f"Failed to create hybrid table: {e}")
            raise

    def _create_indexes_if_possible(
        self, distance_function: str, scalar_type: str
    ) -> None:
        """Try to create indexes separately - failures are non-fatal."""
        # Try to create full-text index using correct ClickZetta syntax
        # Generate unique index name based on table name to avoid global conflicts
        import hashlib

        table_hash = hashlib.md5(self.table_name.encode()).hexdigest()[:8]
        fulltext_index_name = f"content_fts_{table_hash}"

        # Always try to create index, let ClickZetta handle duplicates
        try:
            fulltext_index_sql = f"""
            CREATE INVERTED INDEX {fulltext_index_name} ON TABLE {self.table_name}(content)
            PROPERTIES('analyzer'='{self.text_analyzer}')
            """
            self.engine.execute_query(fulltext_index_sql)
            logger.info(
                f"Full-text index '{fulltext_index_name}' created for table '{self.table_name}'"
            )

            # Build index for existing data (only if creation succeeded)
            try:
                build_fulltext_sql = (
                    f"BUILD INDEX {fulltext_index_name} ON {self.table_name}"
                )
                self.engine.execute_query(build_fulltext_sql)
                logger.info(
                    f"Full-text index '{fulltext_index_name}' built for existing data"
                )
            except Exception as build_error:
                logger.warning(f"Failed to build inverted index: {build_error}")

        except Exception as e:
            # Check if it's actually an "already exists" error
            if "AlreadyExist" in str(e) or "already exist" in str(e).lower():
                logger.info(f"Full-text index '{fulltext_index_name}' already exists")
                # For existing indexes, skip BUILD to avoid orphaned index issues
            else:
                logger.warning(f"Could not create full-text index: {e}")

        # Try to create vector index using correct ClickZetta syntax
        # Generate unique index name based on table name to avoid global conflicts
        vector_index_name = f"embedding_idx_{table_hash}"

        # Always try to create index, let ClickZetta handle duplicates
        try:
            vector_index_sql = f"""
            CREATE VECTOR INDEX {vector_index_name} ON TABLE {self.table_name}(embedding)
            PROPERTIES(
                "scalar.type" = "f32",
                "distance.function" = "{self.distance_metric}_distance"
            )
            """
            self.engine.execute_query(vector_index_sql)
            logger.info(
                f"Vector index '{vector_index_name}' created for table '{self.table_name}'"
            )

            # Build index for existing data (only if creation succeeded)
            try:
                build_vector_sql = (
                    f"BUILD INDEX {vector_index_name} ON {self.table_name}"
                )
                self.engine.execute_query(build_vector_sql)
                logger.info(
                    f"Vector index '{vector_index_name}' built for existing data"
                )
            except Exception as build_error:
                logger.warning(f"Failed to build vector index: {build_error}")

        except Exception as e:
            # Check if it's actually an "already exists" error
            if "AlreadyExist" in str(e) or "already exist" in str(e).lower():
                logger.info(f"Vector index '{vector_index_name}' already exists")
                # For existing indexes, skip BUILD to avoid orphaned index issues
            else:
                logger.warning(f"Could not create vector index: {e}")

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

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the hybrid store.

        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings
        embeddings = self._embeddings.embed_documents(texts)

        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Prepare batch insert
        values = []
        for i, (text, metadata, embedding) in enumerate(
            zip(texts, metadatas, embeddings)
        ):
            content = text.replace("'", "''")
            metadata_str = json.dumps(metadata, ensure_ascii=False).replace("'", "''")

            # Format embedding as array string
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            values.append(
                f"('{ids[i]}', '{content}', '{metadata_str}', {embedding_str})"
            )

        # Batch insert
        insert_sql = f"""
        INSERT INTO {self.table_name}
        (id, content, metadata, embedding)
        VALUES {', '.join(values)}
        """

        try:
            self.engine.execute_query(insert_sql)
            logger.info(f"Added {len(texts)} documents to hybrid store")

            # Note: Index building is handled during index creation, not after each document addition

            return ids
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query using vector search."""
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and similarity scores using vector search."""
        # Generate query embedding
        query_embedding = self._embeddings.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Map distance metric to ClickZetta function
        distance_function_map = {
            "cosine": "COSINE_DISTANCE",
            "euclidean": "L2_DISTANCE",
            "l2": "L2_DISTANCE",
            "manhattan": "L1_DISTANCE",
        }
        distance_function = distance_function_map.get(
            self.distance_metric.lower(), "COSINE_DISTANCE"
        )

        # Vector similarity search query
        search_sql = f"""
        SELECT
            id,
            content,
            metadata,
            {distance_function}(embedding, {embedding_str}) AS distance
        FROM {self.table_name}
        ORDER BY distance ASC
        LIMIT {k}
        """

        try:
            results, _ = self.engine.execute_query(search_sql)

            documents = []
            for row in results:
                content = row["content"]
                metadata = self._parse_metadata(row["metadata"])
                metadata["document_id"] = row["id"]
                metadata["vector_distance"] = row["distance"]

                doc = Document(page_content=content, metadata=metadata)
                documents.append(
                    (doc, 1.0 - row["distance"])
                )  # Convert distance to similarity

            return documents

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def fulltext_search(
        self,
        query: str,
        k: int = 4,
        search_type: str = "phrase",
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query using full-text search.

        Args:
            query: Search query
            k: Number of results to return
            search_type: Type of search (phrase, all, any)
            **kwargs: Additional arguments

        Returns:
            List of relevant documents
        """
        # Build full-text search condition
        escaped_query = query.replace("'", "''")

        if search_type == "phrase":
            search_condition = f"match_phrase(content, '{escaped_query}', map('analyzer', '{self.text_analyzer}'))"
        elif search_type == "all":
            search_condition = f"match_all(content, '{escaped_query}', map('analyzer', '{self.text_analyzer}'))"
        elif search_type == "any":
            search_condition = f"match_any(content, '{escaped_query}', map('analyzer', '{self.text_analyzer}'))"
        else:
            search_condition = f"match_phrase(content, '{escaped_query}', map('analyzer', '{self.text_analyzer}'))"

        # Full-text search query
        search_sql = f"""
        SELECT
            id,
            content,
            metadata,
            1.0 AS relevance_score
        FROM {self.table_name}
        WHERE {search_condition}
        LIMIT {k}
        """

        try:
            results, _ = self.engine.execute_query(search_sql)

            documents = []
            for row in results:
                content = row["content"]
                metadata = self._parse_metadata(row["metadata"])
                metadata["document_id"] = row["id"]
                metadata["fulltext_score"] = row["relevance_score"]

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        fulltext_search_type: str = "phrase",
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs using hybrid search combining vector and full-text search.

        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for combining scores (0.0 = only full-text, 1.0 = only vector)
            fulltext_search_type: Type of full-text search (phrase, all, any)
            **kwargs: Additional arguments

        Returns:
            List of relevant documents with hybrid scores
        """
        # Get more results from each method for better fusion
        retrieval_k = max(k * 2, 10)

        # Vector search
        vector_docs_with_scores = self.similarity_search_with_score(
            query, k=retrieval_k
        )

        # Full-text search
        fulltext_docs = self.fulltext_search(
            query, k=retrieval_k, search_type=fulltext_search_type
        )

        # Create score mappings with explicit float conversion
        vector_scores = {}
        for doc, score in vector_docs_with_scores:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            vector_scores[doc_id] = float(score)

        fulltext_scores = {}
        for doc in fulltext_docs:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            fulltext_scores[doc_id] = float(doc.metadata.get("fulltext_score", 1.0))

        # Combine all documents
        all_docs = {}
        for doc, _ in vector_docs_with_scores:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            all_docs[doc_id] = doc

        for doc in fulltext_docs:
            doc_id = doc.metadata.get("document_id", str(hash(doc.page_content)))
            all_docs[doc_id] = doc

        # Calculate hybrid scores with float conversion
        hybrid_scores = []
        for doc_id, doc in all_docs.items():
            vector_score = float(vector_scores.get(doc_id, 0.0))
            fulltext_score = float(fulltext_scores.get(doc_id, 0.0))

            # Normalize scores if needed
            if vector_scores:
                max_vector = (
                    float(max(vector_scores.values()))
                    if vector_scores.values()
                    else 1.0
                )
                if max_vector > 0:
                    vector_score = vector_score / max_vector

            if fulltext_scores:
                max_fulltext = (
                    float(max(fulltext_scores.values()))
                    if fulltext_scores.values()
                    else 1.0
                )
                if max_fulltext > 0:
                    fulltext_score = fulltext_score / max_fulltext

            # Calculate hybrid score
            hybrid_score = (1 - alpha) * fulltext_score + alpha * vector_score

            # Add scores to metadata
            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["vector_score"] = vector_score
            doc.metadata["fulltext_score"] = fulltext_score

            hybrid_scores.append((doc, hybrid_score))

        # Sort by hybrid score and return top k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in hybrid_scores[:k]]

        logger.debug(f"Hybrid search returned {len(final_docs)} documents")
        return final_docs

    def _parse_metadata(self, metadata_str: str) -> dict[str, Any]:
        """Parse metadata from JSON string."""
        try:
            return json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            return {}

    def delete(self, ids: list[str], **kwargs: Any) -> bool:
        """Delete documents by IDs."""
        if not ids:
            return False

        ids_str = "', '".join(ids)
        delete_sql = f"DELETE FROM {self.table_name} WHERE id IN ('{ids_str}')"

        try:
            self.engine.execute_query(delete_sql)
            logger.info(f"Deleted {len(ids)} documents from hybrid store")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        engine: ClickZettaEngine | None = None,
        table_name: str = "langchain_hybrid_documents",
        **kwargs: Any,
    ) -> ClickZettaHybridStore:
        """Create hybrid store from texts.

        Args:
            texts: List of texts to add
            embedding: Embeddings model
            metadatas: Optional list of metadata dicts
            engine: ClickZetta database engine
            table_name: Name of the hybrid table
            **kwargs: Additional arguments

        Returns:
            ClickZettaHybridStore instance
        """
        if engine is None:
            raise ValueError("ClickZetta engine must be provided")

        # Create store instance (this will create the table)
        store = cls(
            engine=engine, embeddings=embedding, table_name=table_name, **kwargs
        )

        # Add texts after table is created
        if texts:
            store.add_texts(texts, metadatas)

        return store


class ClickZettaUnifiedRetriever(BaseRetriever):
    """Unified retriever interface for ClickZetta hybrid store.

    This retriever works with ClickZettaHybridStore to provide access to
    single-table hybrid search (vector + full-text in same table).
    """

    hybrid_store: ClickZettaHybridStore = Field(exclude=True)
    k: int = 4
    alpha: float = 0.5
    search_type: str = "hybrid"  # "hybrid", "vector", "fulltext"
    fulltext_search_type: str = "phrase"

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self,
        hybrid_store: ClickZettaHybridStore,
        k: int = 4,
        alpha: float = 0.5,
        search_type: str = "hybrid",
        fulltext_search_type: str = "phrase",
        **kwargs: Any,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            hybrid_store: ClickZetta hybrid store
            k: Number of documents to retrieve
            alpha: Weight for hybrid search
            search_type: Type of search (hybrid, vector, fulltext)
            fulltext_search_type: Type of full-text search
            **kwargs: Additional arguments
        """
        super().__init__(
            hybrid_store=hybrid_store,
            k=k,
            alpha=alpha,
            search_type=search_type,
            fulltext_search_type=fulltext_search_type,
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Get relevant documents using specified search type."""
        if self.search_type == "vector":
            return self.hybrid_store.similarity_search(query, k=self.k)
        elif self.search_type == "fulltext":
            return self.hybrid_store.fulltext_search(
                query, k=self.k, search_type=self.fulltext_search_type
            )
        else:  # hybrid
            return self.hybrid_store.hybrid_search(
                query,
                k=self.k,
                alpha=self.alpha,
                fulltext_search_type=self.fulltext_search_type,
            )
