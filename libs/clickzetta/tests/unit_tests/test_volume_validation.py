"""Test Volume Store parameter validation."""

import os
import sys

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_clickzetta.engine import ClickZettaEngine
from langchain_clickzetta.volume_store import (
    ClickZettaNamedVolumeStore,
    ClickZettaTableVolumeStore,
    ClickZettaUserVolumeStore,
    ClickZettaVolumeStore,
)


def test_volume_parameter_validation():
    """Test that Volume Store parameters are validated correctly."""
    print("=== Testing Volume Store Parameter Validation ===\n")

    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster"),
    )

    # Test 1: User Volume should NOT accept volume_name
    print("1. Testing User Volume (should not accept volume_name)")
    try:
        # This should work - no volume_name provided
        _ = ClickZettaUserVolumeStore(engine=engine)
        print("✓ ClickZettaUserVolumeStore() - OK")

        # This should fail - volume_name provided
        try:
            _ = ClickZettaVolumeStore(
                engine=engine, volume_type="user", volume_name="some_name"
            )
            print("✗ User volume accepted volume_name (should have failed)")
        except ValueError as e:
            print(f"✓ User volume correctly rejected volume_name: {e}")
    except Exception as e:
        print(f"✗ User Volume Test Failed: {e}")

    print()

    # Test 2: Table Volume REQUIRES table_name
    print("2. Testing Table Volume (requires table_name)")
    try:
        # This should work - table_name provided
        _ = ClickZettaTableVolumeStore(engine=engine, table_name="test_table")
        print("✓ ClickZettaTableVolumeStore(table_name='test_table') - OK")

        # This should fail - no table_name provided
        try:
            _ = ClickZettaVolumeStore(engine=engine, volume_type="table")
            print("✗ Table volume accepted missing table_name (should have failed)")
        except ValueError as e:
            print(f"✓ Table volume correctly required table_name: {e}")
    except Exception as e:
        print(f"✗ Table Volume Test Failed: {e}")

    print()

    # Test 3: Named Volume REQUIRES volume_name
    print("3. Testing Named Volume (requires volume_name)")
    try:
        # This should work - volume_name provided
        _ = ClickZettaNamedVolumeStore(engine=engine, volume_name="my_volume")
        print("✓ ClickZettaNamedVolumeStore(volume_name='my_volume') - OK")

        # This should fail - no volume_name provided
        try:
            _ = ClickZettaVolumeStore(engine=engine, volume_type="named")
            print("✗ Named volume accepted missing volume_name (should have failed)")
        except ValueError as e:
            print(f"✓ Named volume correctly required volume_name: {e}")
    except Exception as e:
        print(f"✗ Named Volume Test Failed: {e}")

    print()

    # Test 4: Invalid volume type
    print("4. Testing Invalid Volume Type")
    try:
        _ = ClickZettaVolumeStore(engine=engine, volume_type="invalid")
        print("✗ Invalid volume type accepted (should have failed)")
    except ValueError as e:
        print(f"✓ Invalid volume type correctly rejected: {e}")

    print("\n=== Volume Parameter Validation Tests Complete ===")


if __name__ == "__main__":
    test_volume_parameter_validation()
