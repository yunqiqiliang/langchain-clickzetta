"""Tests for ClickZetta vector store."""

import json
from unittest.mock import Mock

from langchain_core.documents import Document

from langchain_clickzetta.vectorstores import ClickZettaVectorStore


class TestClickZettaVectorStore:
    """Test ClickZetta vector store functionality."""

    def test_init(self, mock_engine, mock_embeddings):
        """Test vector store initialization."""
        # Mock the table creation
        mock_engine.execute_query.return_value = ([], [])

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings, table_name="test_vectors"
        )

        assert vector_store.engine == mock_engine
        assert vector_store.embeddings == mock_embeddings
        assert vector_store.table_name == "test_vectors"
        assert vector_store.embedding_column == "embedding"
        assert vector_store.content_column == "content"

        # Verify table creation was called
        mock_engine.execute_query.assert_called()

    def test_format_vector(self, mock_engine, mock_embeddings):
        """Test vector formatting."""
        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        vector = [0.1, 0.2, 0.3]
        formatted = vector_store._format_vector(vector)

        # ClickZetta uses vector() constructor
        assert formatted == "vector(0.1,0.2,0.3)"

    def test_format_vector_with_element_types(self, mock_engine, mock_embeddings):
        """Test vector formatting with different element types."""
        # Test with float type (default)
        vector_store_float = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings, vector_element_type="float"
        )

        vector = [0.1, 0.2, 0.3]
        formatted = vector_store_float._format_vector(vector)
        assert formatted == "vector(0.1,0.2,0.3)"

        # Test CAST formatting
        cast_formatted = vector_store_float._format_vector_as_cast(vector)
        assert cast_formatted == "CAST('[0.1,0.2,0.3]' as vector(float, 3))"

        # Test with int type
        vector_store_int = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings, vector_element_type="int"
        )
        cast_formatted_int = vector_store_int._format_vector_as_cast(vector)
        assert cast_formatted_int == "CAST('[0.1,0.2,0.3]' as vector(int, 3))"

    def test_parse_metadata(self, mock_engine, mock_embeddings):
        """Test metadata parsing."""
        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        # Test with valid JSON
        metadata = {"key": "value", "number": 42}
        parsed = vector_store._parse_metadata(metadata)
        assert json.loads(parsed) == metadata

        # Test with None
        parsed = vector_store._parse_metadata(None)
        assert parsed == "{}"

    def test_unparse_metadata(self, mock_engine, mock_embeddings):
        """Test metadata unparsing."""
        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        # Test with valid JSON string
        metadata_str = '{"key": "value", "number": 42}'
        unparsed = vector_store._unparse_metadata(metadata_str)
        assert unparsed == {"key": "value", "number": 42}

        # Test with empty string
        unparsed = vector_store._unparse_metadata("")
        assert unparsed == {}

        # Test with invalid JSON
        unparsed = vector_store._unparse_metadata("invalid json")
        assert unparsed == {}

    def test_add_texts(self, mock_engine, mock_embeddings):
        """Test adding texts to vector store."""
        mock_engine.execute_query.return_value = ([], [])
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        texts = ["test text 1", "test text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]

        result_ids = vector_store.add_texts(texts, metadatas, ids)

        assert result_ids == ids
        # Verify execute_query was called for insert
        assert mock_engine.execute_query.call_count >= 1

    def test_add_texts_without_ids(self, mock_engine, mock_embeddings):
        """Test adding texts without providing IDs."""
        mock_engine.execute_query.return_value = ([], [])
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2]]

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        texts = ["test text"]
        result_ids = vector_store.add_texts(texts)

        assert len(result_ids) == 1
        assert isinstance(result_ids[0], str)  # UUID string

    def test_similarity_search_with_score(self, mock_engine, mock_embeddings):
        """Test similarity search with scores."""
        # Mock query execution
        mock_engine.execute_query.return_value = (
            [
                {
                    "id": "doc1",
                    "content": "test document 1",
                    "metadata": '{"source": "test1"}',
                    "distance": 0.1,
                },
                {
                    "id": "doc2",
                    "content": "test document 2",
                    "metadata": '{"source": "test2"}',
                    "distance": 0.2,
                },
            ],
            ["id", "content", "metadata", "distance"],
        )

        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        results = vector_store.similarity_search_with_score("test query", k=2)

        assert len(results) == 2
        doc1, score1 = results[0]
        assert isinstance(doc1, Document)
        assert doc1.page_content == "test document 1"
        assert doc1.metadata["source"] == "test1"
        assert score1 == 0.9  # 1 - 0.1 for cosine distance

    def test_similarity_search(self, mock_engine, mock_embeddings):
        """Test basic similarity search."""
        # Mock the similarity_search_with_score method
        mock_docs_with_scores = [
            (Document(page_content="doc1", metadata={}), 0.9),
            (Document(page_content="doc2", metadata={}), 0.8),
        ]

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        # Mock the method
        vector_store.similarity_search_with_score = Mock(
            return_value=mock_docs_with_scores
        )

        results = vector_store.similarity_search("test query", k=2)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_by_vector(self, mock_engine, mock_embeddings):
        """Test similarity search by vector."""
        mock_engine.execute_query.return_value = (
            [{"content": "test document", "metadata": '{"source": "test"}'}],
            ["content", "metadata"],
        )

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        embedding = [0.1, 0.2, 0.3]
        results = vector_store.similarity_search_by_vector(embedding, k=1)

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content == "test document"

    def test_delete(self, mock_engine, mock_embeddings):
        """Test deleting vectors."""
        mock_engine.execute_query.return_value = ([], [])

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        result = vector_store.delete(["id1", "id2"])

        assert result is True
        # Verify delete query was executed
        mock_engine.execute_query.assert_called()

    def test_from_texts(self, mock_engine, mock_embeddings):
        """Test creating vector store from texts."""
        mock_engine.execute_query.return_value = ([], [])
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        texts = ["text 1", "text 2"]
        metadatas = [{"source": "1"}, {"source": "2"}]

        vector_store = ClickZettaVectorStore.from_texts(
            texts=texts,
            embeddings=mock_embeddings,
            engine=mock_engine,
            metadatas=metadatas,
            table_name="test_table",
        )

        assert isinstance(vector_store, ClickZettaVectorStore)
        assert vector_store.table_name == "test_table"

    def test_from_documents(self, mock_engine, mock_embeddings, sample_documents):
        """Test creating vector store from documents."""
        mock_engine.execute_query.return_value = ([], [])
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2]] * len(
            sample_documents
        )

        vector_store = ClickZettaVectorStore.from_documents(
            documents=sample_documents,
            embeddings=mock_embeddings,
            engine=mock_engine,
            table_name="test_table",
        )

        assert isinstance(vector_store, ClickZettaVectorStore)
        assert vector_store.table_name == "test_table"

    def test_similarity_search_with_filter(self, mock_engine, mock_embeddings):
        """Test similarity search with metadata filter."""
        mock_engine.execute_query.return_value = (
            [
                {
                    "id": "doc1",
                    "content": "filtered document",
                    "metadata": '{"category": "tech"}',
                    "distance": 0.1,
                }
            ],
            ["id", "content", "metadata", "distance"],
        )

        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        vector_store = ClickZettaVectorStore(
            engine=mock_engine, embeddings=mock_embeddings
        )

        filter_dict = {"category": "tech"}
        results = vector_store.similarity_search_with_score(
            "test query", k=1, filter=filter_dict
        )

        assert len(results) == 1
        # Verify that the SQL query includes the filter
        calls = mock_engine.execute_query.call_args_list
        sql_query = calls[-1][0][0]  # Get the last SQL query
        # ClickZetta uses JSON extraction functions differently
        assert "JSONExtractString" in sql_query or "JSON_EXTRACT" in sql_query
        assert "category" in sql_query
