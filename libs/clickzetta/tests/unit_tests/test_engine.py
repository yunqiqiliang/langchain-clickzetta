"""Tests for ClickZetta engine."""

from unittest.mock import Mock, patch

from langchain_clickzetta.engine import ClickZettaEngine


class TestClickZettaEngine:
    """Test ClickZetta engine functionality."""

    def test_init(self, test_connection_config):
        """Test engine initialization."""
        engine = ClickZettaEngine(**test_connection_config)

        assert engine.connection_config["service"] == "test-service"
        assert engine.connection_config["instance"] == "test-instance"
        assert engine.connection_config["workspace"] == "test-workspace"
        assert engine.connection_config["schema"] == "test-schema"
        assert engine.connection_timeout == 30
        assert engine.query_timeout == 300

    def test_init_with_custom_timeouts(self, test_connection_config):
        """Test engine initialization with custom timeouts."""
        engine = ClickZettaEngine(
            connection_timeout=60, query_timeout=600, **test_connection_config
        )

        assert engine.connection_timeout == 60
        assert engine.query_timeout == 600

    @patch("langchain_clickzetta.engine.Session")
    def test_get_session(self, mock_session_class, test_connection_config):
        """Test session creation."""
        mock_session = Mock()
        mock_session_class.builder.return_value.configs.return_value.build.return_value = (
            mock_session
        )

        engine = ClickZettaEngine(**test_connection_config)
        session = engine.get_session()

        assert session == mock_session
        mock_session_class.builder.assert_called_once()

    @patch("langchain_clickzetta.engine.create_engine")
    def test_get_sqlalchemy_engine(self, mock_create_engine, test_connection_config):
        """Test SQLAlchemy engine creation."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        engine = ClickZettaEngine(**test_connection_config)
        sqlalchemy_engine = engine.get_sqlalchemy_engine()

        assert sqlalchemy_engine == mock_engine
        mock_create_engine.assert_called_once()

        # Verify connection URL format
        call_args = mock_create_engine.call_args
        url = call_args[0][0]
        assert "clickzetta://" in url
        assert "test-service" in url
        assert "test-workspace" in url

    @patch("langchain_clickzetta.engine.Session")
    def test_execute_query(self, mock_session_class, test_connection_config):
        """Test query execution."""
        # Mock session and result
        mock_session = Mock()
        mock_result = Mock()
        mock_result.to_pandas.return_value.to_dict.return_value = [
            {"id": 1, "name": "test"},
            {"id": 2, "name": "test2"},
        ]
        mock_result.to_pandas.return_value.columns.tolist.return_value = ["id", "name"]

        mock_session.sql.return_value = mock_result
        mock_session_class.builder.return_value.configs.return_value.build.return_value = (
            mock_session
        )

        engine = ClickZettaEngine(**test_connection_config)
        results, columns = engine.execute_query("SELECT * FROM test_table")

        assert len(results) == 2
        assert columns == ["id", "name"]
        mock_session.sql.assert_called_once_with(
            "SELECT * FROM test_table", hints=engine.hints
        )

    @patch("langchain_clickzetta.engine.Session")
    def test_execute_query_with_parameters(
        self, mock_session_class, test_connection_config
    ):
        """Test query execution with parameters."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.to_pandas.return_value.to_dict.return_value = []
        mock_result.to_pandas.return_value.columns.tolist.return_value = []

        mock_session.sql.return_value = mock_result
        mock_session_class.builder.return_value.configs.return_value.build.return_value = (
            mock_session
        )

        engine = ClickZettaEngine(**test_connection_config)
        parameters = {"param1": "value1"}
        engine.execute_query(
            "SELECT * FROM test_table WHERE col = %(param1)s", parameters
        )

        mock_session.sql.assert_called_once_with(
            "SELECT * FROM test_table WHERE col = %(param1)s",
            hints=engine.hints,
            param1="value1",
        )

    def test_get_table_info_mock(self, mock_engine):
        """Test table info retrieval with mock."""
        mock_engine.execute_query.return_value = (
            [
                {
                    "table_name": "test_table",
                    "column_name": "id",
                    "data_type": "INTEGER",
                    "is_nullable": "NO",
                    "column_default": None,
                },
                {
                    "table_name": "test_table",
                    "column_name": "name",
                    "data_type": "VARCHAR",
                    "is_nullable": "YES",
                    "column_default": None,
                },
            ],
            ["table_name", "column_name", "data_type", "is_nullable", "column_default"],
        )

        # Replace the engine's method
        engine = ClickZettaEngine(
            service="test",
            instance="test",
            workspace="test",
            schema="test",
            username="test",
            password="test",
        )
        engine.execute_query = mock_engine.execute_query

        table_info = engine.get_table_info(["test_table"])

        assert "Table: test_table" in table_info
        assert "id INTEGER NOT NULL" in table_info
        assert "name VARCHAR NULL" in table_info

    def test_context_manager(self, test_connection_config):
        """Test context manager functionality."""
        with patch("langchain_clickzetta.engine.Session"):
            with ClickZettaEngine(**test_connection_config) as engine:
                assert isinstance(engine, ClickZettaEngine)

    @patch("langchain_clickzetta.engine.Session")
    def test_close(self, mock_session_class, test_connection_config):
        """Test closing connections."""
        mock_session = Mock()
        mock_session_class.builder.return_value.configs.return_value.build.return_value = (
            mock_session
        )

        engine = ClickZettaEngine(**test_connection_config)
        engine.get_session()  # Create session
        engine.close()

        mock_session.close.assert_called_once()
        assert engine._session is None
