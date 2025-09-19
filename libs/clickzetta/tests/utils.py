"""Test utilities for LangChain ClickZetta integration."""

import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from langchain_clickzetta import ClickZettaEngine


def load_uat_connection() -> dict[str, Any]:
    """Load UAT connection configuration from connections.json.

    Returns:
        Dictionary containing UAT connection parameters

    Raises:
        pytest.skip: If configuration is not found
    """
    config_path = Path.home() / ".clickzetta" / "connections.json"

    if not config_path.exists():
        pytest.skip("ClickZetta connections.json not found")

    with open(config_path, encoding="utf-8") as f:
        config_data = json.load(f)

    # Find UAT connection
    uat_conn = None
    for conn in config_data.get("connections", []):
        if conn.get("name") == "uat":
            uat_conn = conn
            break

    if not uat_conn:
        pytest.skip("UAT connection not found in connections.json")

    return uat_conn


def load_dashscope_config() -> Optional[dict[str, Any]]:
    """Load DashScope configuration from connections.json.

    Returns:
        Dictionary containing DashScope configuration, or None if not found
    """
    config_path = Path.home() / ".clickzetta" / "connections.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return (
            config_data.get("system_config", {})
            .get("embedding", {})
            .get("dashscope", {})
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def create_test_engine() -> ClickZettaEngine:
    """Create a ClickZetta engine for testing.

    Returns:
        ClickZettaEngine instance configured for UAT
    """
    uat_config = load_uat_connection()

    return ClickZettaEngine(
        service=uat_config["service"],
        instance=uat_config["instance"],
        workspace=uat_config["workspace"],
        schema=uat_config["schema"],
        username=uat_config["username"],
        password=uat_config["password"],
        vcluster=uat_config["vcluster"],
        query_timeout=300,
    )


def generate_test_table_name(prefix: str = "test") -> str:
    """Generate a unique test table name.

    Args:
        prefix: Prefix for the table name

    Returns:
        Unique table name with timestamp
    """
    return f"{prefix}_{int(time.time())}"


def cleanup_test_table(engine: ClickZettaEngine, table_name: str) -> None:
    """Clean up a test table.

    Args:
        engine: ClickZetta engine
        table_name: Name of table to clean up
    """
    try:
        engine.execute_query(f"DROP TABLE IF EXISTS {table_name}")
    except Exception:
        # Ignore cleanup errors
        pass


class MockEmbeddings:
    """Mock embeddings for testing without external dependencies."""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for query."""
        import hashlib

        hash_val = hashlib.md5(text.encode()).hexdigest()
        # Generate deterministic mock embedding
        values = []
        for i in range(0, min(32, len(hash_val)), 2):
            values.append(float(int(hash_val[i : i + 2], 16)) / 255.0)

        # Pad or truncate to desired dimension
        while len(values) < self.dimension:
            values.extend(values[: min(len(values), self.dimension - len(values))])

        return values[: self.dimension]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        return [self.embed_query(text) for text in texts]


class MockLLM:
    """Mock LLM for testing without external dependencies."""

    def generate(self, messages):
        """Generate mock LLM response."""

        class MockGeneration:
            def __init__(self, text):
                self.text = text

        class MockResult:
            def __init__(self, generation):
                self.generations = [[generation]]

        # Return a simple SQL query for testing
        return MockResult(
            MockGeneration(
                "SQLQuery: SELECT COUNT(*) as count FROM information_schema.tables;"
            )
        )


def get_real_dashscope_embeddings():
    """Get real DashScope embeddings if available, otherwise return mock."""
    config = load_dashscope_config()
    if config and config.get("api_key"):
        try:
            from langchain_community.embeddings import DashScopeEmbeddings

            return DashScopeEmbeddings(
                dashscope_api_key=config["api_key"],
                model=config.get("model", "text-embedding-v4"),
            )
        except ImportError:
            pass

    return MockEmbeddings()


def get_real_dashscope_llm():
    """Get real DashScope LLM if available, otherwise return mock."""
    config = load_dashscope_config()
    if config and config.get("api_key"):
        try:
            from langchain_community.llms import Tongyi

            return Tongyi(
                dashscope_api_key=config["api_key"],
                model_name="qwen-turbo",
                max_tokens=200,
            )
        except ImportError:
            pass

    return MockLLM()
