"""ClickZetta database engine for LangChain integration."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote_plus

from clickzetta.zettapark.session import Session
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class ClickZettaEngine:
    """ClickZetta database engine for LangChain integration.

    This class provides a unified interface for connecting to ClickZetta
    and executing SQL queries, supporting both direct Session usage and
    SQLAlchemy engine compatibility.
    """

    def __init__(
        self,
        service: str,
        instance: str,
        workspace: str,
        schema: str,
        username: str,
        password: str,
        vcluster: str,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta engine.

        ClickZetta requires exactly 7 connection parameters:
        service, instance, workspace, schema, username, password, vcluster

        Args:
            service: ClickZetta service name
            instance: ClickZetta instance name
            workspace: ClickZetta workspace name
            schema: ClickZetta schema name
            username: Username for authentication
            password: Password for authentication
            vcluster: Virtual cluster name (required)
            connection_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
            **kwargs: Additional connection parameters
        """
        self.connection_config = {
            "service": service,
            "instance": instance,
            "workspace": workspace,
            "schema": schema,
            "username": username,
            "password": password,
            "vcluster": vcluster,
        }

        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self.additional_params = kwargs

        # Default hints for optimal performance
        self.hints = {
            "sdk.job.timeout": query_timeout,
            "query_tag": "Query from LangChain",
            "cz.storage.parquet.vector.index.read.memory.cache": "true",
            "cz.storage.parquet.vector.index.read.local.cache": "false",
            "cz.sql.table.scan.push.down.filter": "true",
            "cz.sql.table.scan.enable.ensure.filter": "true",
            "cz.storage.always.prefetch.internal": "true",
            "cz.optimizer.generate.columns.always.valid": "true",
            "cz.sql.index.prewhere.enabled": "true",
        }
        self.hints.update(kwargs.get("hints", {}))

        self._session: Session | None = None
        self._sqlalchemy_engine: Engine | None = None

    def get_session(self) -> Session:
        """Get or create ClickZetta session."""
        if self._session is None:
            # Use clickzetta.connect() method directly
            import clickzetta

            self._session = clickzetta.connect(**self.connection_config)
            logger.info(
                f"Created ClickZetta session for workspace: {self.connection_config['workspace']}"
            )
        return self._session

    def get_sqlalchemy_engine(self) -> Engine:
        """Get or create SQLAlchemy engine for ClickZetta.

        This creates a SQLAlchemy engine that can work with ClickZetta
        through the connector protocol.
        """
        if self._sqlalchemy_engine is None:
            # Build connection URL for ClickZetta SQLAlchemy connector
            password_encoded = quote_plus(self.connection_config["password"])
            username_encoded = quote_plus(self.connection_config["username"])

            url_parts = [
                f"clickzetta://{username_encoded}:{password_encoded}",
                f"@{self.connection_config['service']}",
                f"/{self.connection_config['workspace']}",
                f"?instance={self.connection_config['instance']}",
                f"&schema={self.connection_config['schema']}",
            ]

            # vcluster is required for ClickZetta
            url_parts.append(f"&vcluster={self.connection_config['vcluster']}")

            connection_url = "".join(url_parts)

            self._sqlalchemy_engine = create_engine(
                connection_url,
                connect_args={
                    "timeout": self.connection_timeout,
                    "hints": self.hints,
                },
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
            )
            logger.info("Created SQLAlchemy engine for ClickZetta")

        return self._sqlalchemy_engine

    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Execute a SQL query and return results.

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Tuple of (results as list of dicts, column names)
        """
        connection = self.get_session()

        try:
            # Use cursor to execute query
            cursor = connection.cursor()
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)

            # Get results and column names
            results = cursor.fetchall()
            columns = (
                [desc[0] for desc in cursor.description] if cursor.description else []
            )

            # Convert result to list of dictionaries
            if results and columns:
                # Convert rows to dictionaries using column names
                records = []
                for row in results:
                    if isinstance(row, (tuple, list)):
                        # Convert tuple/list to dict using column names
                        record = dict(zip(columns, row))
                    else:
                        # Row is already a dict-like object
                        record = dict(row)
                    records.append(record)
            else:
                records = []

            logger.debug(f"Query executed successfully, returned {len(records)} rows")
            return records, columns

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_sql_with_engine(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute SQL using SQLAlchemy engine.

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Query results as list of dictionaries
        """
        engine = self.get_sqlalchemy_engine()

        with engine.connect() as conn:
            if parameters:
                result = conn.execute(text(query), parameters)
            else:
                result = conn.execute(text(query))

            # Convert to list of dictionaries
            columns = result.keys()
            records = [dict(zip(columns, row)) for row in result.fetchall()]

            logger.debug(f"SQLAlchemy query executed, returned {len(records)} rows")
            return records

    def get_table_names(self, schema: str | None = None) -> list[str]:
        """Get all table names in the specified or current schema.

        Note: This method is deprecated in LangChain but still required for compatibility.
        Use get_usable_table_names() instead.

        Args:
            schema: Optional schema name. If not provided, uses the current schema.

        Returns:
            List of table names
        """
        return list(self.get_usable_table_names(schema=schema))

    def get_usable_table_names(self, schema: str | None = None) -> list[str]:
        """Get available table names in the specified or current schema.

        This is the preferred method for getting table names in LangChain.

        Args:
            schema: Optional schema name. If not provided, uses the current schema.

        Returns:
            List of available table names
        """
        # Use provided schema or default to connection schema
        target_schema = schema or self.connection_config["schema"]

        try:
            # Try to use SHOW TABLES first (most compatible)
            # Some databases support SHOW TABLES FROM schema syntax
            if schema:
                query = f"SHOW TABLES FROM {target_schema}"
            else:
                query = "SHOW TABLES"
            results, _ = self.execute_query(query)

            # Extract table names from result
            table_names = []
            for row in results:
                # SHOW TABLES usually returns a single column with table name
                if isinstance(row, dict):
                    # Find the table name field (could be 'table_name', 'Tables_in_*', etc.)
                    table_name = None
                    for key, value in row.items():
                        if 'table' in key.lower() or len(row) == 1:
                            table_name = value
                            break
                    if table_name:
                        table_names.append(table_name)

            if table_names:
                return sorted(table_names)

        except Exception as e:
            logger.warning(f"SHOW TABLES failed: {e}, trying alternative method")

        try:
            # Fallback: try information_schema
            query = f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = '{target_schema}'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
            results, _ = self.execute_query(query)
            return [row["table_name"] for row in results]

        except Exception as e:
            logger.warning(f"information_schema query failed: {e}")

        # Last resort: return empty list
        logger.warning("Could not retrieve table names, returning empty list")
        return []

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        This method is required for LangChain SQL compatibility.

        Args:
            command: SQL command to execute
            fetch: Fetch mode ("all", "one", or "cursor")

        Returns:
            String representation of query results
        """
        try:
            results, columns = self.execute_query(command)

            if not results:
                return ""

            # Format results as string table
            if not columns:
                return str(results)

            # Create a simple table format
            lines = []

            # Header
            lines.append(" | ".join(columns))
            lines.append("-" * len(lines[0]))

            # Data rows
            for row in results:
                if isinstance(row, dict):
                    row_values = [str(row.get(col, "")) for col in columns]
                else:
                    row_values = [str(val) for val in row]
                lines.append(" | ".join(row_values))

            return "\n".join(lines)

        except Exception as e:
            # Return error message as string (LangChain pattern)
            return f"Error executing query: {str(e)}"

    @property
    def dialect(self) -> str:
        """Get database dialect string.

        Returns:
            String representation of database dialect
        """
        return "clickzetta"

    @property
    def table_info(self) -> str:
        """Get information about all tables in the database.

        Returns:
            String containing information about all tables
        """
        table_names = self.get_usable_table_names()
        return self.get_table_info(table_names=table_names)

    def get_table_info(self, table_names: list[str] | None = None, schema: str | None = None) -> str:
        """Get information about tables in the specified or current schema.

        Args:
            table_names: Optional list of specific table names to describe
            schema: Optional schema name. If not provided, uses the current schema.

        Returns:
            String containing table information
        """
        # Use provided schema or default to connection schema
        target_schema = schema or self.connection_config["schema"]

        # ClickZetta uses SHOW COLUMNS or system tables for metadata
        # Try SHOW COLUMNS approach first, fallback to system tables if needed
        if table_names:
            # For specific tables, use SHOW COLUMNS for each table
            results = []
            columns = [
                "table_name",
                "column_name",
                "data_type",
                "is_nullable",
                "column_default",
            ]

            for table_name in table_names:
                try:
                    # Include schema in the SHOW COLUMNS query if different from connection schema
                    if schema and schema != self.connection_config["schema"]:
                        show_sql = f"SHOW COLUMNS FROM {target_schema}.{table_name}"
                    else:
                        show_sql = f"SHOW COLUMNS FROM {table_name}"
                    table_results, _ = self.execute_query(show_sql)

                    # Transform SHOW COLUMNS results to match expected format
                    for row in table_results:
                        # Check if data is already in the expected format (for testing)
                        if "table_name" in row and "column_name" in row and "is_nullable" in row:
                            # Data is already in the correct format, use as-is
                            results.append(row)
                        elif "column_name" in row and "data_type" in row:
                            # ClickZetta SHOW COLUMNS format: schema_name, table_name, column_name, data_type, comment
                            results.append(
                                {
                                    "table_name": row.get("table_name", table_name),
                                    "column_name": row.get("column_name", ""),
                                    "data_type": row.get("data_type", ""),
                                    "is_nullable": "YES",  # ClickZetta doesn't return nullable info via SHOW COLUMNS
                                    "column_default": row.get("comment", ""),  # Use comment as default since no default field
                                }
                            )
                        else:
                            # MySQL/Standard SHOW COLUMNS format: Field, Type, Null, Key, Default, Extra
                            results.append(
                                {
                                    "table_name": table_name,
                                    "column_name": row.get(
                                        "Field", row.get("column_name", "")
                                    ),
                                    "data_type": row.get("Type", row.get("data_type", "")),
                                    "is_nullable": (
                                        "YES" if row.get("Null", "YES") == "YES" else "NO"
                                    ),
                                    "column_default": row.get(
                                        "Default", row.get("column_default")
                                    ),
                                }
                            )
                except Exception:
                    # If SHOW COLUMNS fails, skip this table
                    logger.warning(f"Could not get column info for table {table_name}")
                    continue
        else:
            # For all tables, try to use system information if available
            query = f"""
            SELECT
                table_name,
                column_name,
                data_type,
                'YES' as is_nullable,
                '' as column_default
            FROM information_schema.columns
            WHERE table_schema = '{target_schema}'
            ORDER BY table_name, column_name
            """
            try:
                results, columns = self.execute_query(query)
            except Exception:
                # Fallback: return basic info
                logger.warning(
                    "Could not access information_schema, returning basic table info"
                )
                results = []

        # Format table information
        table_info = []
        current_table = None
        columns = []

        for row in results:
            table_name = row["table_name"]
            if current_table != table_name:
                if current_table is not None:
                    table_info.append(f"Table: {current_table}")
                    table_info.extend([f"  {col}" for col in columns])
                    table_info.append("")
                current_table = table_name
                columns = []

            nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
            default = (
                f" DEFAULT {row['column_default']}" if row["column_default"] else ""
            )
            columns.append(
                f"{row['column_name']} {row['data_type']} {nullable}{default}"
            )

        # Add the last table
        if current_table is not None:
            table_info.append(f"Table: {current_table}")
            table_info.extend([f"  {col}" for col in columns])

        return "\n".join(table_info)

    def close(self) -> None:
        """Close the database connections."""
        if self._session:
            try:
                self._session.close()
                logger.info("ClickZetta session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self._session = None

        if self._sqlalchemy_engine:
            try:
                self._sqlalchemy_engine.dispose()
                logger.info("SQLAlchemy engine disposed")
            except Exception as e:
                logger.warning(f"Error disposing engine: {e}")
            finally:
                self._sqlalchemy_engine = None

    def __enter__(self) -> ClickZettaEngine:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
