"""ClickZetta-based storage implementations for LangChain."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.stores import BaseStore

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaStore(BaseStore):
    """ClickZetta-based implementation of LangChain BaseStore.

    Uses ClickZetta tables to provide persistent key-value storage
    for LangChain applications.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        table_name: str = "langchain_store",
        key_column: str = "store_key",
        value_column: str = "store_value",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta store.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the table to store key-value pairs
            key_column: Name of the key column
            value_column: Name of the value column
            **kwargs: Additional arguments
        """
        self.engine = engine
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

        self.key_column = key_column
        self.value_column = value_column

        # Initialize table if it doesn't exist
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create the storage table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.key_column} String,
            {self.value_column} String,
            created_at Timestamp DEFAULT CURRENT_TIMESTAMP,
            updated_at Timestamp DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            self.engine.execute_query(create_table_sql)
            logger.info(f"Storage table '{self.table_name}' created or verified")
        except Exception as e:
            logger.error(f"Failed to create storage table: {e}")
            raise

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get values for multiple keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            List of values (as bytes) or None if key not found
        """
        if not keys:
            return []

        # Build SQL query with IN clause
        keys_str = "', '".join(keys)
        select_sql = f"""
        SELECT {self.key_column}, {self.value_column}
        FROM {self.table_name}
        WHERE {self.key_column} IN ('{keys_str}')
        """

        try:
            results, _ = self.engine.execute_query(select_sql)

            # Create a mapping from key to value
            key_value_map = {}
            for row in results:
                key = row[self.key_column]
                value = row[self.value_column]
                # Convert string value back to bytes
                key_value_map[key] = value.encode("utf-8") if value else None

            # Return values in the same order as requested keys
            return [key_value_map.get(key) for key in keys]

        except Exception as e:
            logger.error(f"Failed to get values: {e}")
            return [None] * len(keys)

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set values for multiple keys.

        Args:
            key_value_pairs: List of (key, value) tuples
        """
        if not key_value_pairs:
            return

        # Use MERGE INTO for ClickZetta UPSERT operations
        for key, value in key_value_pairs:
            value_str = value.decode("utf-8") if value else ""
            # Escape single quotes
            escaped_key = key.replace("'", "''")
            escaped_value = value_str.replace("'", "''")

            merge_sql = f"""
            MERGE INTO {self.table_name} AS target
            USING (SELECT '{escaped_key}' AS key, '{escaped_value}' AS value, CURRENT_TIMESTAMP AS ts) AS source
            ON target.{self.key_column} = source.key
            WHEN MATCHED THEN UPDATE SET
                {self.value_column} = source.value,
                updated_at = source.ts
            WHEN NOT MATCHED THEN INSERT
                ({self.key_column}, {self.value_column}, created_at, updated_at)
                VALUES (source.key, source.value, source.ts, source.ts)
            """

            try:
                self.engine.execute_query(merge_sql)
            except Exception as e:
                logger.error(f"Failed to set key-value pair {key}: {e}")
                raise

        logger.debug(f"Set {len(key_value_pairs)} key-value pairs")

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete multiple keys.

        Args:
            keys: List of keys to delete
        """
        if not keys:
            return

        keys_str = "', '".join(keys)
        delete_sql = f"""
        DELETE FROM {self.table_name}
        WHERE {self.key_column} IN ('{keys_str}')
        """

        try:
            self.engine.execute_query(delete_sql)
            logger.debug(f"Deleted {len(keys)} keys")
        except Exception as e:
            logger.error(f"Failed to delete keys: {e}")
            raise

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Yield keys that match the given prefix.

        Args:
            prefix: Key prefix to match

        Yields:
            Keys that start with the prefix
        """
        if prefix:
            # Use LIKE pattern for prefix matching
            escaped_prefix = prefix.replace("'", "''")
            where_clause = f"WHERE {self.key_column} LIKE '{escaped_prefix}%'"
        else:
            where_clause = ""

        select_sql = f"""
        SELECT {self.key_column}
        FROM {self.table_name}
        {where_clause}
        ORDER BY {self.key_column}
        """

        try:
            results, _ = self.engine.execute_query(select_sql)
            for row in results:
                yield row[self.key_column]
        except Exception as e:
            logger.error(f"Failed to yield keys: {e}")
            return


class ClickZettaDocumentStore(ClickZettaStore):
    """ClickZetta table-based document store for LangChain Documents.

    Uses ClickZetta tables to provide structured document storage
    with metadata support and SQL queryability.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        table_name: str = "langchain_documents_store",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta document store.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the table to store documents
            **kwargs: Additional arguments
        """
        super().__init__(
            engine=engine,
            table_name=table_name,
            key_column="doc_id",
            value_column="doc_content",
            **kwargs,
        )

    def _create_table_if_not_exists(self) -> None:
        """Create the document storage table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.key_column} String,
            {self.value_column} String,
            metadata String,
            created_at Timestamp DEFAULT CURRENT_TIMESTAMP,
            updated_at Timestamp DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            self.engine.execute_query(create_table_sql)
            logger.info(
                f"Document storage table '{self.table_name}' created or verified"
            )
        except Exception as e:
            logger.error(f"Failed to create document storage table: {e}")
            raise

    def store_document(self, doc_id: str, content: str, metadata: dict = None) -> None:
        """Store a document with metadata.

        Args:
            doc_id: Unique document identifier
            content: Document content
            metadata: Document metadata dictionary
        """
        metadata_str = json.dumps(metadata or {}, ensure_ascii=False)

        # Escape single quotes
        escaped_id = doc_id.replace("'", "''")
        escaped_content = content.replace("'", "''")
        escaped_metadata = metadata_str.replace("'", "''")

        merge_sql = f"""
        MERGE INTO {self.table_name} AS target
        USING (SELECT '{escaped_id}' AS id, '{escaped_content}' AS content, '{escaped_metadata}' AS meta, CURRENT_TIMESTAMP AS ts) AS source
        ON target.{self.key_column} = source.id
        WHEN MATCHED THEN UPDATE SET
            {self.value_column} = source.content,
            metadata = source.meta,
            updated_at = source.ts
        WHEN NOT MATCHED THEN INSERT
            ({self.key_column}, {self.value_column}, metadata, created_at, updated_at)
            VALUES (source.id, source.content, source.meta, source.ts, source.ts)
        """

        try:
            self.engine.execute_query(merge_sql)
            logger.debug(f"Stored document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to store document {doc_id}: {e}")
            raise

    def get_document(self, doc_id: str) -> tuple[str, dict] | None:
        """Get a document with metadata.

        Args:
            doc_id: Document identifier

        Returns:
            Tuple of (content, metadata) or None if not found
        """
        escaped_id = doc_id.replace("'", "''")
        select_sql = f"""
        SELECT {self.value_column}, metadata
        FROM {self.table_name}
        WHERE {self.key_column} = '{escaped_id}'
        """

        try:
            results, _ = self.engine.execute_query(select_sql)
            if results:
                row = results[0]
                content = row[self.value_column]
                metadata_str = row.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    metadata = {}
                return content, metadata
            return None
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None


class ClickZettaFileStore(BaseStore):
    """ClickZetta Volume-based file storage for binary data.

    Uses ClickZetta Volume to provide native file storage
    capabilities for embeddings, models, and other binary data.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        volume_type: str = "user",
        volume_name: str | None = None,
        subdirectory: str = "langchain_files",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta Volume file store.

        Args:
            engine: ClickZetta database engine
            volume_type: Type of volume ('user', 'table', 'named')
            volume_name: Name of table (for table volume) or volume (for named volume)
            subdirectory: Subdirectory within volume for file storage
            **kwargs: Additional arguments
        """
        from langchain_clickzetta.volume_store import ClickZettaVolumeStore

        self.volume_store = ClickZettaVolumeStore(
            engine=engine,
            volume_type=volume_type,
            volume_name=volume_name,
            subdirectory=subdirectory,
            **kwargs,
        )

    def store_file(
        self,
        file_path: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> None:
        """Store a file with metadata.

        Args:
            file_path: File path/identifier
            content: File content as bytes
            mime_type: MIME type of the file
        """
        # Create metadata
        file_metadata = {"mime_type": mime_type, "file_size": len(content)}

        # Store metadata separately with a special key
        metadata_key = f"_metadata_{file_path}"
        metadata_json = json.dumps(file_metadata, ensure_ascii=False)

        # Store both file content and metadata
        self.volume_store.mset(
            [(file_path, content), (metadata_key, metadata_json.encode("utf-8"))]
        )

    def get_file(self, file_path: str) -> tuple[bytes, str] | None:
        """Get a file with metadata.

        Args:
            file_path: File path/identifier

        Returns:
            Tuple of (content, mime_type) or None if not found
        """
        # Get both file and metadata
        metadata_key = f"_metadata_{file_path}"
        results = self.volume_store.mget([file_path, metadata_key])

        content = results[0]
        metadata_bytes = results[1]

        if content:
            mime_type = "application/octet-stream"  # default
            if metadata_bytes:
                try:
                    metadata_json = metadata_bytes.decode("utf-8")
                    metadata = json.loads(metadata_json)
                    mime_type = metadata.get("mime_type", mime_type)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Use default mime_type

            return content, mime_type
        return None

    def list_files(self, prefix: str = "") -> list[tuple[str, int, str]]:
        """List files with metadata.

        Args:
            prefix: Path prefix to filter files

        Returns:
            List of (file_path, file_size, mime_type) tuples
        """
        files = []
        for key in self.volume_store.yield_keys(prefix=prefix):
            # Skip metadata keys
            if key.startswith("_metadata_"):
                continue

            # Get file info
            result = self.get_file(key)
            if result:
                content, mime_type = result
                files.append((key, len(content), mime_type))

        return files

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get values for multiple keys."""
        return self.volume_store.mget(keys)

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set values for multiple keys."""
        self.volume_store.mset(key_value_pairs)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete multiple keys."""
        # Also delete metadata for these keys
        all_keys_to_delete = []
        for key in keys:
            all_keys_to_delete.append(key)
            all_keys_to_delete.append(f"_metadata_{key}")

        self.volume_store.mdelete(all_keys_to_delete)

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Yield keys that match the given prefix."""
        for key in self.volume_store.yield_keys(prefix=prefix):
            # Skip metadata keys
            if not key.startswith("_metadata_"):
                yield key
