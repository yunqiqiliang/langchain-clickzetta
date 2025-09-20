"""Test configuration and fixtures."""

import os
from unittest.mock import Mock

import pytest

from langchain_clickzetta.engine import ClickZettaEngine


@pytest.fixture
def mock_engine():
    """Create a mock ClickZetta engine for testing."""
    engine = Mock(spec=ClickZettaEngine)

    # Mock basic methods
    engine.execute_query.return_value = ([], [])
    engine.get_table_info.return_value = "Mock table info"
    engine.close.return_value = None

    # Mock connection config for vectorstore tests
    engine.connection_config = {
        "workspace": "test-workspace",
        "schema": "test-schema",
        "service": "test-service",
        "instance": "test-instance",
        "username": "test-user",
        "password": "test-password",
        "vcluster": "test-vcluster",
    }

    return engine


@pytest.fixture
def test_connection_config():
    """Provide test connection configuration.

    ClickZetta requires exactly 7 connection parameters.
    """
    return {
        "service": "test-service",
        "instance": "test-instance",
        "workspace": "test-workspace",
        "schema": "test-schema",
        "username": "test-user",
        "password": "test-password",
        "vcluster": "test-vcluster",  # Required parameter
    }


@pytest.fixture
def real_engine(test_connection_config):
    """Create a real ClickZetta engine if credentials are available."""
    # Only create real engine if all 7 required environment variables are set
    required_vars = [
        "CLICKZETTA_SERVICE",
        "CLICKZETTA_INSTANCE",
        "CLICKZETTA_WORKSPACE",
        "CLICKZETTA_SCHEMA",
        "CLICKZETTA_USERNAME",
        "CLICKZETTA_PASSWORD",
        "CLICKZETTA_VCLUSTER",
    ]

    if all(os.getenv(var) for var in required_vars):
        config = {
            "service": os.getenv("CLICKZETTA_SERVICE"),
            "instance": os.getenv("CLICKZETTA_INSTANCE"),
            "workspace": os.getenv("CLICKZETTA_WORKSPACE"),
            "schema": os.getenv("CLICKZETTA_SCHEMA"),
            "username": os.getenv("CLICKZETTA_USERNAME"),
            "password": os.getenv("CLICKZETTA_PASSWORD"),
            "vcluster": os.getenv("CLICKZETTA_VCLUSTER"),
        }
        return ClickZettaEngine(**config)
    else:
        pytest.skip(
            "Real ClickZetta credentials not available - all 7 parameters required"
        )


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 1536  # Mock embedding vector
    embeddings.embed_documents.return_value = [
        [0.1] * 1536,
        [0.2] * 1536,
    ]  # Mock document embeddings
    return embeddings


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={"category": "tech", "source": "test1"},
        ),
        Document(
            page_content="Another document discussing artificial intelligence.",
            metadata={"category": "tech", "source": "test2"},
        ),
        Document(
            page_content="A document about cooking and recipes.",
            metadata={"category": "food", "source": "test3"},
        ),
    ]
