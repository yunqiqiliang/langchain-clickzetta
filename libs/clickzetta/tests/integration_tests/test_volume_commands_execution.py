"""Test to verify Volume commands are actually executed by monitoring SQL execution."""

import os
import sys

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    "/Users/liangmo/Documents/GitHub/langchain-clickzetta/libs/clickzetta/langchain_clickzetta",
)

# Import modules directly to avoid dependency issues
from engine import ClickZettaEngine  # noqa: E402
from stores import ClickZettaFileStore  # noqa: E402


class SQLMonitoringEngine:
    """Wrapper around ClickZettaEngine to monitor SQL execution."""

    def __init__(self, engine):
        self.engine = engine
        self.executed_sqls = []

    def execute_query(self, sql, *args, **kwargs):
        self.executed_sqls.append(sql)
        print(f"  SQL EXECUTED: {sql}")
        return self.engine.execute_query(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def test_volume_commands_execution():
    """Test that Volume PUT/REMOVE commands are actually executed."""
    print("=== Volume Commands Execution Verification ===\n")

    # Initialize ClickZetta engine
    base_engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster"),
    )

    # Wrap with monitoring
    engine = SQLMonitoringEngine(base_engine)

    try:
        # Initialize FileStore
        file_store = ClickZettaFileStore(
            engine=engine,
            volume_type="user",
            subdirectory="command_test"
        )
        print("✓ ClickZettaFileStore initialized with SQL monitoring")

        print("\n--- PHASE 1: Monitor file storage commands ---")

        # Store a file and monitor SQL
        test_file = "command_test/monitor.txt"
        test_content = b"Content for SQL monitoring"

        engine.executed_sqls.clear()
        file_store.store_file(test_file, test_content, "text/plain")

        put_commands = [sql for sql in engine.executed_sqls if sql.startswith("PUT")]
        print(f"\n✓ PUT commands executed: {len(put_commands)}")
        for cmd in put_commands:
            print(f"  {cmd}")

        assert len(put_commands) >= 2, "Expected at least 2 PUT commands (file + metadata)"

        print("\n--- PHASE 2: Monitor file deletion commands ---")

        # Delete the file and monitor SQL
        engine.executed_sqls.clear()
        file_store.mdelete([test_file])

        remove_commands = [sql for sql in engine.executed_sqls if sql.startswith("REMOVE")]
        print(f"\n✓ REMOVE commands executed: {len(remove_commands)}")
        for cmd in remove_commands:
            print(f"  {cmd}")

        assert len(remove_commands) >= 2, "Expected at least 2 REMOVE commands (file + metadata)"

        print("\n--- PHASE 3: Verify command syntax ---")

        # Verify that the correct User Volume syntax is used
        user_volume_removes = [cmd for cmd in remove_commands if "REMOVE USER VOLUME FILE" in cmd]
        print(f"\n✓ User Volume REMOVE commands: {len(user_volume_removes)}")

        assert len(user_volume_removes) >= 2, "Expected User Volume REMOVE commands for both file and metadata"

        # Verify file paths are correct
        for cmd in user_volume_removes:
            assert "command_test/" in cmd, f"Command should contain subdirectory: {cmd}"
            print(f"✓ Command contains correct path: {cmd}")

        print("\n--- PHASE 4: Verify commands actually work ---")

        # Verify the file is actually gone by trying to retrieve it
        result = file_store.get_file(test_file)
        assert result is None, "File should be gone after REMOVE command"
        print("✓ File confirmed deleted by REMOVE command")

        # Verify via Volume Store directly
        volume_result = file_store.volume_store.mget([test_file])
        assert volume_result[0] is None, "File should be gone from Volume Store"
        print("✓ File confirmed deleted from Volume Store")

        print("\n=== RESULT: Volume commands execution VERIFIED ===")
        print("✓ PUT commands are executed for file storage")
        print("✓ REMOVE commands are executed for file deletion")
        print("✓ Correct User Volume syntax is used")
        print("✓ Commands actually delete files from ClickZetta Volume")
        print("✓ No evidence of command interception or pseudo-deletion")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            base_engine.close()
            print("\n✓ Connection closed")
        except:
            pass


if __name__ == "__main__":
    test_volume_commands_execution()
