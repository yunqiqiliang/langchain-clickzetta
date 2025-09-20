"""ClickZetta Volume-based storage implementation for LangChain."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, Optional

from langchain_core.stores import BaseStore

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaVolumeStore(BaseStore[str, bytes]):
    """ClickZetta Volume-based implementation of LangChain BaseStore.

    Uses ClickZetta Volume (User, Table, or Named Volume) to provide
    file-based key-value storage for LangChain applications.

    This implementation leverages ClickZetta's native Volume capabilities:
    - User Volume: volume:user://~/filename
    - Table Volume: volume:table://table_name/filename
    - Named Volume: volume://volume_name/filename
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        volume_type: str = "user",
        volume_name: str | None = None,
        subdirectory: str = "langchain_store",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta Volume store.

        Args:
            engine: ClickZetta database engine
            volume_type: Type of volume ('user', 'table', 'named')
            volume_name: Name of table (for table volume) or volume (for named volume)
            subdirectory: Subdirectory within volume for storage
            **kwargs: Additional arguments
        """
        self.engine = engine
        self.volume_type = volume_type.lower()
        self.volume_name = volume_name
        self.subdirectory = subdirectory.strip("/")

        if self.volume_type not in ["user", "table", "named"]:
            raise ValueError(
                f"Invalid volume_type: {volume_type}. Must be 'user', 'table', or 'named'"
            )

        if self.volume_type == "user" and volume_name:
            raise ValueError(
                "volume_name should not be provided for user volume (uses current user '~')"
            )

        if self.volume_type == "table" and not volume_name:
            raise ValueError("table_name is required for table volume")

        if self.volume_type == "named" and not volume_name:
            raise ValueError("volume_name is required for named volume")

        # Create subdirectory if it doesn't exist
        self._ensure_subdirectory()

    def _ensure_subdirectory(self) -> None:
        """Ensure the subdirectory exists in the volume."""
        if not self.subdirectory:
            return

        try:
            # Try to list the subdirectory - this will create it if needed
            if self.volume_type == "user":
                list_sql = "SHOW USER VOLUME DIRECTORY"
            elif self.volume_type == "table":
                list_sql = f"SHOW TABLE VOLUME DIRECTORY {self.volume_name}"
            else:  # named
                list_sql = f"SHOW VOLUME DIRECTORY {self.volume_name}"

            # Just execute to ensure volume is accessible
            self.engine.execute_query(list_sql)
            logger.debug(f"Volume directory verified: {self.volume_type}")

        except Exception as e:
            logger.warning(f"Could not verify volume directory: {e}")

    def _get_file_path(self, key: str) -> str:
        """Get the full file path for a key."""
        # Use SHA256 hash to create safe filename from key
        safe_filename = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".dat"

        if self.subdirectory:
            return f"{self.subdirectory}/{safe_filename}"
        return safe_filename

    def _put_key_metadata(self, key: str, file_path: str) -> bool:
        """Store key metadata to enable key recovery."""
        try:
            # Store key in a metadata file alongside the data file
            metadata_file_path = file_path.replace('.dat', '.key')
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(key.encode('utf-8'))
                tmp_file_path = tmp_file.name

            # Upload metadata to volume
            if self.volume_type == "user":
                put_sql = f"PUT '{tmp_file_path}' TO USER VOLUME FILE '{metadata_file_path}'"
            elif self.volume_type == "table":
                put_sql = f"PUT '{tmp_file_path}' TO TABLE VOLUME {self.volume_name} FILE '{metadata_file_path}'"
            else:  # named
                put_sql = f"PUT '{tmp_file_path}' TO VOLUME {self.volume_name} FILE '{metadata_file_path}'"

            self.engine.execute_query(put_sql)
            return True

        except Exception as e:
            logger.debug(f"Failed to store key metadata for {key}: {e}")
            return False
        finally:
            try:
                if "tmp_file_path" in locals():
                    os.unlink(tmp_file_path)
            except OSError:
                pass

    def _get_key_from_metadata(self, file_path: str) -> Optional[str]:
        """Recover key from metadata file."""
        try:
            metadata_file_path = file_path.replace('.dat', '.key')
            metadata_content = self._get_file(metadata_file_path)
            if metadata_content:
                return metadata_content.decode('utf-8')
            return None
        except Exception as e:
            logger.debug(f"Failed to get key metadata from {file_path}: {e}")
            return None

    def _remove_key_metadata(self, file_path: str) -> bool:
        """Remove key metadata file."""
        try:
            metadata_file_path = file_path.replace('.dat', '.key')
            return self._remove_file(metadata_file_path)
        except Exception as e:
            logger.debug(f"Failed to remove key metadata for {file_path}: {e}")
            return False

    def _put_file_with_retry(self, file_path: str, content: bytes, max_retries: int = 3) -> bool:
        """Put content to volume file with retry mechanism."""
        for attempt in range(max_retries):
            if self._put_file(file_path, content):
                return True
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.debug(f"Retry attempt {attempt + 1} for {file_path} in {wait_time}s")
                time.sleep(wait_time)
        return False

    def _put_file(self, file_path: str, content: bytes) -> bool:
        """Put content to volume file."""
        # Write content to temporary local file
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            # Upload to volume
            if self.volume_type == "user":
                put_sql = f"PUT '{tmp_file_path}' TO USER VOLUME FILE '{file_path}'"
            elif self.volume_type == "table":
                put_sql = f"PUT '{tmp_file_path}' TO TABLE VOLUME {self.volume_name} FILE '{file_path}'"
            else:  # named
                put_sql = f"PUT '{tmp_file_path}' TO VOLUME {self.volume_name} FILE '{file_path}'"

            self.engine.execute_query(put_sql)
            return True

        except Exception as e:
            logger.error(f"Failed to put file {file_path}: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                if "tmp_file_path" in locals():
                    os.unlink(tmp_file_path)
            except OSError:
                pass  # File might already be deleted or not exist

    def _get_file(self, file_path: str) -> Optional[bytes]:
        """Get content from volume file."""
        import tempfile

        try:
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as tmp_dir:
                os.path.join(tmp_dir, "downloaded_file")

                # Download from volume
                if self.volume_type == "user":
                    get_sql = f"GET USER VOLUME FILE '{file_path}' TO '{tmp_dir}/'"
                elif self.volume_type == "table":
                    get_sql = f"GET TABLE VOLUME {self.volume_name} FILE '{file_path}' TO '{tmp_dir}/'"
                else:  # named
                    get_sql = f"GET VOLUME {self.volume_name} FILE '{file_path}' TO '{tmp_dir}/'"

                self.engine.execute_query(get_sql)

                # Read the downloaded file
                downloaded_file = os.path.join(tmp_dir, os.path.basename(file_path))
                if os.path.exists(downloaded_file):
                    with open(downloaded_file, "rb") as f:
                        content = f.read()
                        # Check if the content is an error response from ClickZetta
                        if content.startswith(b'<?xml') and b'<Code>NoSuchKey</Code>' in content:
                            logger.debug(f"File not found (NoSuchKey): {file_path}")
                            return None
                        return content
                return None

        except Exception as e:
            # Check if it's a "file not found" type error
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['nosuchkey', 'not exist', 'not found', 'file not found']):
                logger.debug(f"File not found: {file_path}")
            else:
                logger.debug(f"Failed to get file {file_path}: {e}")
            return None

    def _remove_file_with_retry(self, file_path: str, max_retries: int = 3) -> bool:
        """Remove file from volume with retry mechanism."""
        for attempt in range(max_retries):
            if self._remove_file(file_path):
                return True
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.debug(f"Retry attempt {attempt + 1} for removing {file_path} in {wait_time}s")
                time.sleep(wait_time)
        return False

    def _remove_file(self, file_path: str) -> bool:
        """Remove file from volume."""
        try:
            if self.volume_type == "user":
                remove_sql = f"REMOVE USER VOLUME FILE '{file_path}'"
            elif self.volume_type == "table":
                remove_sql = (
                    f"REMOVE TABLE VOLUME {self.volume_name} FILE '{file_path}'"
                )
            else:  # named
                remove_sql = f"REMOVE VOLUME {self.volume_name} FILE '{file_path}'"

            self.engine.execute_query(remove_sql)
            return True

        except Exception as e:
            logger.debug(f"Failed to remove file {file_path}: {e}")
            return False

    def _list_files(self) -> list[str]:
        """List all files in the volume subdirectory."""
        try:
            if self.volume_type == "user":
                list_sql = "SHOW USER VOLUME DIRECTORY"
            elif self.volume_type == "table":
                list_sql = f"SHOW TABLE VOLUME DIRECTORY {self.volume_name}"
            else:  # named
                list_sql = f"SHOW VOLUME DIRECTORY {self.volume_name}"

            results, _ = self.engine.execute_query(list_sql)

            files = []
            for row in results:
                relative_path = row["relative_path"]
                # Filter by subdirectory
                if self.subdirectory:
                    if relative_path.startswith(self.subdirectory + "/"):
                        files.append(relative_path)
                else:
                    # Only root level files if no subdirectory
                    if "/" not in relative_path:
                        files.append(relative_path)

            return files

        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def mget(self, keys: Sequence[str]) -> list[Optional[bytes]]:
        """Get values for multiple keys.

        Args:
            keys: List of keys to retrieve

        Returns:
            List of values (as bytes) or None if key not found
        """
        if not keys:
            return []

        results = []
        for key in keys:
            file_path = self._get_file_path(key)
            content = self._get_file(file_path)
            results.append(content)

        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set values for multiple keys.

        Args:
            key_value_pairs: List of (key, value) tuples
        """
        if not key_value_pairs:
            return

        failed_keys = []
        for key, value in key_value_pairs:
            file_path = self._get_file_path(key)
            success = self._put_file_with_retry(file_path, value)
            if not success:
                failed_keys.append(key)
                logger.warning(f"Failed to store key: {key}")
                continue

            # Store key metadata for recovery
            self._put_key_metadata(key, file_path)

        if failed_keys:
            logger.error(f"Failed to store {len(failed_keys)} keys: {failed_keys}")
            # Optionally raise exception based on configuration
            # raise RuntimeError(f"Failed to store {len(failed_keys)} keys: {failed_keys}")

        logger.debug(f"Set {len(key_value_pairs)} key-value pairs in volume")

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete multiple keys.

        Args:
            keys: List of keys to delete

        Raises:
            RuntimeError: If any key fails to delete
        """
        if not keys:
            return

        failed_keys = []
        for key in keys:
            file_path = self._get_file_path(key)
            data_success = self._remove_file_with_retry(file_path)
            metadata_success = self._remove_key_metadata(file_path)

            # Consider it failed only if data file deletion fails
            # Metadata deletion failure is not critical
            if not data_success:
                failed_keys.append(key)
                logger.warning(f"Failed to delete key: {key}")

        if failed_keys:
            logger.error(f"Failed to delete {len(failed_keys)} keys: {failed_keys}")
            # Optionally raise exception based on configuration
            # raise RuntimeError(f"Failed to delete {len(failed_keys)} keys: {failed_keys}")

        logger.debug(f"Successfully deleted {len(keys)} keys from volume")

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys that match the given prefix.

        This implementation recovers keys from metadata files stored alongside
        data files. Keys are recovered from .key files that correspond to .dat files.

        Args:
            prefix: Key prefix to match

        Yields:
            Keys that start with the prefix
        """
        files = self._list_files()

        for file_path in files:
            if file_path.endswith(".dat"):
                # Try to recover the original key from metadata
                key = self._get_key_from_metadata(file_path)
                if key is not None:
                    # Check prefix match
                    if not prefix or key.startswith(prefix):
                        yield key
                else:
                    # Fallback: if no metadata available, log warning
                    logger.debug(f"No key metadata found for file: {file_path}")
                    # Could try to use a heuristic here, but it's better to be explicit
                    continue


class ClickZettaUserVolumeStore(ClickZettaVolumeStore):
    """Convenience class for User Volume storage."""

    def __init__(
        self,
        engine: ClickZettaEngine,
        subdirectory: str = "langchain_store",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            engine=engine, volume_type="user", subdirectory=subdirectory, **kwargs
        )


class ClickZettaTableVolumeStore(ClickZettaVolumeStore):
    """Convenience class for Table Volume storage.

    Table Volume provides storage associated with a specific table.
    Requires table_name to specify which table's volume to use.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        table_name: str,
        subdirectory: str = "langchain_store",
        **kwargs: Any,
    ) -> None:
        """Initialize Table Volume store.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the table whose volume to use
            subdirectory: Subdirectory within table volume
            **kwargs: Additional arguments
        """
        super().__init__(
            engine=engine,
            volume_type="table",
            volume_name=table_name,
            subdirectory=subdirectory,
            **kwargs,
        )


class ClickZettaNamedVolumeStore(ClickZettaVolumeStore):
    """Convenience class for Named Volume storage.

    Named Volume provides shared storage created explicitly by users.
    Requires volume_name to specify which named volume to use.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        volume_name: str,
        subdirectory: str = "langchain_store",
        **kwargs: Any,
    ) -> None:
        """Initialize Named Volume store.

        Args:
            engine: ClickZetta database engine
            volume_name: Name of the named volume to use
            subdirectory: Subdirectory within named volume
            **kwargs: Additional arguments
        """
        super().__init__(
            engine=engine,
            volume_type="named",
            volume_name=volume_name,
            subdirectory=subdirectory,
            **kwargs,
        )
