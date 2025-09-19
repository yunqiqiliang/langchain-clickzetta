"""Test LangChain BaseStore interface compatibility."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.stores import BaseStore

from langchain_clickzetta import (
    ClickZettaDocumentStore,
    ClickZettaFileStore,
    ClickZettaStore,
    ClickZettaUserVolumeStore,
)


def test_langchain_basestore_interface():
    """Test that all our stores implement LangChain BaseStore interface correctly."""
    print("=== Testing LangChain BaseStore Interface Compatibility ===\n")

    engine_params = {
        "service": os.getenv("CLICKZETTA_SERVICE", "test-service"),
        "instance": os.getenv("CLICKZETTA_INSTANCE", "test-instance"),
        "workspace": os.getenv("CLICKZETTA_WORKSPACE", "test-workspace"),
        "schema": os.getenv("CLICKZETTA_SCHEMA", "test-schema"),
        "username": os.getenv("CLICKZETTA_USERNAME", "test-username"),
        "password": os.getenv("CLICKZETTA_PASSWORD", "test-password"),
        "vcluster": os.getenv("CLICKZETTA_VCLUSTER", "test-vcluster"),
    }

    # Required BaseStore methods
    required_methods = ["mget", "mset", "mdelete", "yield_keys"]

    stores_to_test = [
        ("ClickZettaStore", ClickZettaStore, {"table_name": "test_table"}),
        ("ClickZettaDocumentStore", ClickZettaDocumentStore, {"volume_type": "user"}),
        ("ClickZettaFileStore", ClickZettaFileStore, {"volume_type": "user"}),
        ("ClickZettaUserVolumeStore", ClickZettaUserVolumeStore, {}),
    ]

    for store_name, store_class, _extra_params in stores_to_test:
        print(f"Testing {store_name}:")

        # Test 1: Check if it's a subclass of BaseStore
        if issubclass(store_class, BaseStore):
            print(f"  ✓ {store_name} is a subclass of BaseStore")
        else:
            print(f"  ✗ {store_name} is NOT a subclass of BaseStore")
            continue

        # Test 2: Check if it has all required methods
        missing_methods = []
        for method in required_methods:
            if not hasattr(store_class, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"  ✗ {store_name} missing methods: {missing_methods}")
        else:
            print(f"  ✓ {store_name} has all required methods: {required_methods}")

        # Test 3: Check method signatures (basic check)
        try:
            from inspect import signature

            # Check mget signature
            mget_sig = signature(store_class.mget)
            if len(mget_sig.parameters) >= 2:  # self + keys parameter
                print(f"  ✓ {store_name}.mget has correct signature")
            else:
                print(f"  ✗ {store_name}.mget has incorrect signature")

            # Check mset signature
            mset_sig = signature(store_class.mset)
            if len(mset_sig.parameters) >= 2:  # self + key_value_pairs parameter
                print(f"  ✓ {store_name}.mset has correct signature")
            else:
                print(f"  ✗ {store_name}.mset has incorrect signature")

        except Exception as e:
            print(f"  ⚠ Could not check method signatures for {store_name}: {e}")

        print()

    print("=== Interface Compatibility Test Complete ===")


def test_basestore_method_compatibility():
    """Test that our stores can be used as BaseStore instances."""
    print("\n=== Testing BaseStore Method Usage ===\n")

    try:
        from langchain_clickzetta.engine import ClickZettaEngine

        engine = ClickZettaEngine(
            **{
                "service": os.getenv("CLICKZETTA_SERVICE", "test"),
                "instance": os.getenv("CLICKZETTA_INSTANCE", "test"),
                "workspace": os.getenv("CLICKZETTA_WORKSPACE", "test"),
                "schema": os.getenv("CLICKZETTA_SCHEMA", "test"),
                "username": os.getenv("CLICKZETTA_USERNAME", "test"),
                "password": os.getenv("CLICKZETTA_PASSWORD", "test"),
                "vcluster": os.getenv("CLICKZETTA_VCLUSTER", "test"),
            }
        )

        # Test with ClickZettaStore (table-based)
        print("1. Testing ClickZettaStore as BaseStore:")
        store: BaseStore = ClickZettaStore(engine=engine, table_name="test_compat")

        # Test BaseStore methods
        test_data = [("test_key", b"test_value")]

        # This should work since our store implements BaseStore
        store.mset(test_data)
        print("  ✓ mset() works")

        values = store.mget(["test_key"])
        print(f"  ✓ mget() works: {values}")

        keys = list(store.yield_keys(prefix="test"))
        print(f"  ✓ yield_keys() works: {keys}")

        store.mdelete(["test_key"])
        print("  ✓ mdelete() works")

    except Exception as e:
        print(f"  ✗ BaseStore method compatibility test failed: {e}")

    print("\n=== Method Usage Test Complete ===")


if __name__ == "__main__":
    test_langchain_basestore_interface()
    test_basestore_method_compatibility()
