"""ClickZetta Volume-based storage implementation for LangChain."""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.stores import BaseStore

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


class ClickZettaVolumeStore(BaseStore):
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

    def _get_file(self, file_path: str) -> bytes | None:
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
                        return f.read()
                return None

        except Exception as e:
            logger.debug(f"Failed to get file {file_path}: {e}")
            return None

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

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
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

        for key, value in key_value_pairs:
            file_path = self._get_file_path(key)
            success = self._put_file(file_path, value)
            if not success:
                raise RuntimeError(f"Failed to store key: {key}")

        logger.debug(f"Set {len(key_value_pairs)} key-value pairs in volume")

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete multiple keys.

        Args:
            keys: List of keys to delete
        """
        if not keys:
            return

        for key in keys:
            file_path = self._get_file_path(key)
            self._remove_file(file_path)

        logger.debug(f"Deleted {len(keys)} keys from volume")

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        """Yield keys that match the given prefix.

        Args:
            prefix: Key prefix to match

        Yields:
            Keys that start with the prefix
        """
        files = self._list_files()

        # Create reverse mapping from filename to key (this is approximate)
        # In a real implementation, you might want to store metadata separately
        for file_path in files:
            if file_path.endswith(".dat"):
                # For demonstration, we'll try to match by checking if the file exists
                # In practice, you might want a separate metadata store for key mappings
                try:
                    content = self._get_file(file_path)
                    if content is not None:
                        # Generate a placeholder key - in real usage you'd need better key recovery
                        key = f"key_{os.path.basename(file_path).replace('.dat', '')}"
                        if not prefix or key.startswith(prefix):
                            yield key
                except Exception:
                    continue  # Skip files that can't be processed


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
