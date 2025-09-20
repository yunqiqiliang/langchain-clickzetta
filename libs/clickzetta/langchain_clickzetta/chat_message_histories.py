"""ClickZetta chat message history implementation for LangChain."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_clickzetta.engine import ClickZettaEngine

logger = logging.getLogger(__name__)


def _clean_json_string(json_str: str) -> str:
    """Clean JSON string by removing problematic control characters.

    Args:
        json_str: JSON string that might contain control characters

    Returns:
        Cleaned JSON string safe for SQL storage and JSON parsing
    """
    # Remove harmful control characters but preserve \t (0x09), \n (0x0A), \r (0x0D)
    # Remove: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F-0x9F
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', json_str)
    return cleaned


class ClickZettaChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by ClickZetta database.

    This class stores and retrieves chat message history from ClickZetta,
    enabling persistent conversation memory across sessions.
    """

    def __init__(
        self,
        engine: ClickZettaEngine,
        session_id: str,
        table_name: str = "langchain_chat_history",
        session_id_column: str = "session_id",
        message_column: str = "message",
        timestamp_column: str = "timestamp",
        **kwargs: Any,
    ) -> None:
        """Initialize ClickZetta chat message history.

        Args:
            engine: ClickZetta database engine
            session_id: Unique identifier for the chat session
            table_name: Name of the table to store chat history
            session_id_column: Name of the session ID column
            message_column: Name of the message column
            timestamp_column: Name of the timestamp column
            **kwargs: Additional arguments
        """
        self.engine = engine
        self.session_id = session_id
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
        self.session_id_column = session_id_column
        self.message_column = message_column
        self.timestamp_column = timestamp_column

        # Initialize table if it doesn't exist
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create the chat history table if it doesn't exist."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {self.session_id_column} String,
            {self.message_column} String,
            {self.timestamp_column} Timestamp DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            self.engine.execute_query(create_table_sql)
            logger.info(f"Chat history table '{self.table_name}' created or verified")
        except Exception as e:
            logger.error(f"Failed to create chat history table: {e}")
            raise

    @property
    def buffer(self) -> str:
        """Get the conversation buffer as a formatted string.

        This property formats the conversation history as a string,
        compatible with LangChain memory systems.

        Returns:
            Formatted string representation of the conversation
        """
        messages = self.messages
        if not messages:
            return ""

        # Format messages as "Human: ... AI: ..." string
        buffer_lines = []
        for message in messages:
            if hasattr(message, 'type'):
                if message.type == 'human':
                    prefix = "Human"
                elif message.type == 'ai':
                    prefix = "AI"
                else:
                    prefix = message.type.title()
            else:
                # Fallback for messages without type
                prefix = message.__class__.__name__.replace('Message', '')

            content = message.content if hasattr(message, 'content') else str(message)
            buffer_lines.append(f"{prefix}: {content}")

        return "\n".join(buffer_lines)

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save context from a conversation to the chat history.

        This method is compatible with LangChain memory systems that expect
        save_context functionality.

        Args:
            inputs: Dictionary containing input data (typically has "input" key)
            outputs: Dictionary containing output data (typically has "output" key)
        """
        from langchain_core.messages import AIMessage, HumanMessage

        # Extract input message
        if "input" in inputs:
            input_content = inputs["input"]
            human_message = HumanMessage(content=input_content)
            self.add_message(human_message)

        # Extract output message
        if "output" in outputs:
            output_content = outputs["output"]
            ai_message = AIMessage(content=output_content)
            self.add_message(ai_message)

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return memory variables for LangChain memory compatibility.

        This method is compatible with BaseMemory interface.

        Args:
            inputs: Dictionary of input values (not used in this implementation)

        Returns:
            Dictionary containing memory variables with chat history
        """
        # inputs parameter is not used in this implementation but kept for interface compatibility
        _ = inputs
        return {
            "history": self.buffer,
            "chat_history": self.buffer,
            "messages": self.messages
        }

    @property
    def memory_variables(self) -> list[str]:
        """Return list of memory variable keys.

        Returns:
            List of memory variable keys that this memory provides
        """
        return ["history", "chat_history", "messages"]

    def add_user_message(self, message: str) -> None:
        """Add a user message to the chat history.

        Args:
            message: The user message content
        """
        from langchain_core.messages import HumanMessage
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the chat history.

        Args:
            message: The AI message content
        """
        from langchain_core.messages import AIMessage
        self.add_message(AIMessage(content=message))

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve all messages for the current session."""
        select_sql = f"""
        SELECT {self.message_column}
        FROM {self.table_name}
        WHERE {self.session_id_column} = '{self.session_id}'
        ORDER BY {self.timestamp_column} ASC
        """

        try:
            results, _ = self.engine.execute_query(select_sql)

            messages = []
            for row in results:
                message_data = json.loads(row[self.message_column])
                message_list = messages_from_dict([message_data])
                if message_list:
                    messages.append(message_list[0])

            logger.debug(
                f"Retrieved {len(messages)} messages for session {self.session_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """Add a new message to the chat history.

        Args:
            message: The message to add to the chat history
        """
        message_dict = message_to_dict(message)
        # Ensure proper JSON serialization with control character handling
        message_json = json.dumps(
            message_dict, ensure_ascii=True, separators=(",", ":")
        )
        # Clean control characters and escape single quotes for SQL
        cleaned_json = _clean_json_string(message_json)
        escaped_message_json = cleaned_json.replace("'", "''")

        insert_sql = f"""
        INSERT INTO {self.table_name}
        ({self.session_id_column}, {self.message_column})
        VALUES ('{self.session_id}', '{escaped_message_json}')
        """

        try:
            self.engine.execute_query(insert_sql)
            logger.debug(f"Added message to session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages to the chat history.

        Args:
            messages: Sequence of messages to add to the chat history
        """
        if not messages:
            return

        # Prepare batch insert
        values = []
        for message in messages:
            message_dict = message_to_dict(message)
            # Ensure proper JSON serialization with control character handling
            message_json = json.dumps(
                message_dict, ensure_ascii=True, separators=(",", ":")
            )
            # Clean control characters and escape single quotes for SQL
            cleaned_json = _clean_json_string(message_json)
            escaped_message_json = cleaned_json.replace("'", "''")
            values.append(f"('{self.session_id}', '{escaped_message_json}')")

        insert_sql = f"""
        INSERT INTO {self.table_name}
        ({self.session_id_column}, {self.message_column})
        VALUES {', '.join(values)}
        """

        try:
            self.engine.execute_query(insert_sql)
            logger.debug(f"Added {len(messages)} messages to session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to add messages: {e}")
            raise

    def clear(self) -> None:
        """Clear all messages for the current session."""
        delete_sql = f"""
        DELETE FROM {self.table_name}
        WHERE {self.session_id_column} = '{self.session_id}'
        """

        try:
            self.engine.execute_query(delete_sql)
            logger.info(f"Cleared chat history for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to clear chat history: {e}")
            raise

    def get_messages_by_count(self, count: int) -> list[BaseMessage]:
        """Retrieve the last N messages for the current session.

        Args:
            count: Number of recent messages to retrieve

        Returns:
            List of the most recent messages
        """
        select_sql = f"""
        SELECT {self.message_column}
        FROM {self.table_name}
        WHERE {self.session_id_column} = '{self.session_id}'
        ORDER BY {self.timestamp_column} DESC
        LIMIT {count}
        """

        try:
            results, _ = self.engine.execute_query(select_sql)

            messages = []
            # Reverse the results to get chronological order
            for row in reversed(results):
                try:
                    message_json = row[self.message_column]
                    # Handle potential control characters and encoding issues
                    if isinstance(message_json, bytes):
                        message_json = message_json.decode("utf-8", errors="replace")

                    # Clean control characters using our helper function
                    message_json = _clean_json_string(message_json)

                    message_data = json.loads(message_json)
                    message_list = messages_from_dict([message_data])
                    if message_list:
                        messages.append(message_list[0])
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to parse message JSON: {e}")
                    continue

            logger.debug(
                f"Retrieved {len(messages)} recent messages for session {self.session_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve recent messages: {e}")
            return []

    def get_messages_by_time_range(
        self, start_time: str | None = None, end_time: str | None = None
    ) -> list[BaseMessage]:
        """Retrieve messages within a specific time range.

        Args:
            start_time: Start time in ISO format (e.g., '2023-01-01 00:00:00')
            end_time: End time in ISO format (e.g., '2023-01-02 00:00:00')

        Returns:
            List of messages within the time range
        """
        where_conditions = [f"{self.session_id_column} = '{self.session_id}'"]

        if start_time:
            where_conditions.append(f"{self.timestamp_column} >= '{start_time}'")
        if end_time:
            where_conditions.append(f"{self.timestamp_column} <= '{end_time}'")

        where_clause = " AND ".join(where_conditions)

        select_sql = f"""
        SELECT {self.message_column}
        FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY {self.timestamp_column} ASC
        """

        try:
            results, _ = self.engine.execute_query(select_sql)

            messages = []
            for row in results:
                message_data = json.loads(row[self.message_column])
                message_list = messages_from_dict([message_data])
                if message_list:
                    messages.append(message_list[0])

            logger.debug(
                f"Retrieved {len(messages)} messages in time range for session {self.session_id}"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve messages by time range: {e}")
            return []

    def get_session_count(self) -> int:
        """Get the total number of messages in the current session.

        Returns:
            Total number of messages in the session
        """
        count_sql = f"""
        SELECT COUNT(*) as message_count
        FROM {self.table_name}
        WHERE {self.session_id_column} = '{self.session_id}'
        """

        try:
            results, _ = self.engine.execute_query(count_sql)
            count = results[0]["message_count"] if results else 0
            return count
        except Exception as e:
            logger.error(f"Failed to get session count: {e}")
            return 0

    def delete_session(self) -> None:
        """Delete the entire session and all its messages.

        This is equivalent to clear() but more explicit about the operation.
        """
        self.clear()

    @classmethod
    def get_all_sessions(
        cls,
        engine: ClickZettaEngine,
        table_name: str = "langchain_chat_history",
        session_id_column: str = "session_id",
    ) -> list[str]:
        """Get all session IDs from the chat history table.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the chat history table
            session_id_column: Name of the session ID column

        Returns:
            List of all session IDs
        """
        select_sql = f"""
        SELECT DISTINCT {session_id_column}
        FROM {table_name}
        ORDER BY {session_id_column}
        """

        try:
            results, _ = engine.execute_query(select_sql)
            sessions = [row[session_id_column] for row in results]
            return sessions
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []

    @classmethod
    def delete_all_sessions(
        cls, engine: ClickZettaEngine, table_name: str = "langchain_chat_history"
    ) -> None:
        """Delete all chat history from the table.

        Args:
            engine: ClickZetta database engine
            table_name: Name of the chat history table
        """
        truncate_sql = f"TRUNCATE TABLE {table_name}"

        try:
            engine.execute_query(truncate_sql)
            logger.info(f"Truncated chat history table {table_name}")
        except Exception as e:
            logger.error(f"Failed to truncate chat history table: {e}")
            raise
