"""Setup demo data for SQL Chain examples."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_clickzetta import ClickZettaEngine


def setup_demo_data():
    """Create demo tables and insert sample data for SQL Chain examples."""
    # Initialize ClickZetta engine
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster")
    )

    print("=== Setting up Demo Data for SQL Chain ===\n")

    workspace = engine.connection_config['workspace']
    schema = engine.connection_config['schema']

    # Create demo_customers table
    customers_table = f"{workspace}.{schema}.demo_customers"

    try:
        # Drop table if exists
        drop_sql = f"DROP TABLE IF EXISTS {customers_table}"
        engine.execute_query(drop_sql)

        # Create customers table
        create_customers_sql = f"""
        CREATE TABLE {customers_table} (
            customer_id INT,
            first_name String,
            last_name String,
            email String,
            phone String,
            registration_date Date,
            country String,
            city String,
            total_orders INT,
            total_spent DECIMAL(10,2)
        )
        """
        engine.execute_query(create_customers_sql)
        print(f"✓ Created table: {customers_table}")

        # Insert sample data using DATE() function
        insert_sql = f"""
        INSERT INTO {customers_table} VALUES
        (1, 'Alice', 'Johnson', 'alice.johnson@email.com', '+1-555-0101', DATE('2023-01-15'), 'USA', 'New York', 12, 2450.00),
        (2, 'Bob', 'Smith', 'bob.smith@email.com', '+1-555-0102', DATE('2023-02-20'), 'USA', 'Los Angeles', 8, 1890.50),
        (3, 'Charlie', 'Brown', 'charlie.brown@email.com', '+1-555-0103', DATE('2023-03-10'), 'Canada', 'Toronto', 15, 3200.75),
        (4, 'Diana', 'Wilson', 'diana.wilson@email.com', '+44-555-0104', DATE('2023-04-05'), 'UK', 'London', 6, 1100.25),
        (5, 'Eve', 'Davis', 'eve.davis@email.com', '+1-555-0105', DATE('2023-05-12'), 'USA', 'Chicago', 20, 4500.00),
        (6, 'Frank', 'Miller', 'frank.miller@email.com', '+49-555-0106', DATE('2023-06-08'), 'Germany', 'Berlin', 9, 2100.80),
        (7, 'Grace', 'Taylor', 'grace.taylor@email.com', '+33-555-0107', DATE('2023-07-22'), 'France', 'Paris', 11, 2750.60),
        (8, 'Henry', 'Anderson', 'henry.anderson@email.com', '+61-555-0108', DATE('2023-08-14'), 'Australia', 'Sydney', 14, 3450.90),
        (9, 'Iris', 'Thomas', 'iris.thomas@email.com', '+81-555-0109', DATE('2023-09-30'), 'Japan', 'Tokyo', 7, 1650.40),
        (10, 'Jack', 'Moore', 'jack.moore@email.com', '+1-555-0110', DATE('2023-10-18'), 'USA', 'San Francisco', 13, 3100.25)
        """
        engine.execute_query(insert_sql)
        print(f"✓ Inserted sample data into {customers_table}")

    except Exception as e:
        print(f"✗ Error setting up customers table: {e}")

    # Create demo_orders table
    orders_table = f"{workspace}.{schema}.demo_orders"

    try:
        # Drop table if exists
        drop_sql = f"DROP TABLE IF EXISTS {orders_table}"
        engine.execute_query(drop_sql)

        # Create orders table
        create_orders_sql = f"""
        CREATE TABLE {orders_table} (
            order_id INT,
            customer_id INT,
            order_date Date,
            product_name String,
            quantity INT,
            unit_price DECIMAL(10,2),
            total_amount DECIMAL(10,2),
            status String
        )
        """
        engine.execute_query(create_orders_sql)
        print(f"✓ Created table: {orders_table}")

        # Insert sample data using DATE() function
        insert_sql = f"""
        INSERT INTO {orders_table} VALUES
        (101, 1, DATE('2024-01-10'), 'Laptop Pro 16', 1, 2499.99, 2499.99, 'Delivered'),
        (102, 1, DATE('2024-02-15'), 'Wireless Mouse', 2, 79.99, 159.98, 'Delivered'),
        (103, 2, DATE('2024-01-20'), 'Smartphone X', 1, 899.99, 899.99, 'Delivered'),
        (104, 2, DATE('2024-03-05'), 'Tablet Air', 1, 599.99, 599.99, 'Shipped'),
        (105, 3, DATE('2024-01-25'), 'Gaming Chair', 1, 399.99, 399.99, 'Delivered'),
        (106, 3, DATE('2024-02-28'), 'Mechanical Keyboard', 1, 149.99, 149.99, 'Delivered'),
        (107, 4, DATE('2024-03-12'), 'Monitor 4K', 1, 449.99, 449.99, 'Processing'),
        (108, 5, DATE('2024-02-08'), 'Desktop Computer', 1, 1299.99, 1299.99, 'Delivered'),
        (109, 5, DATE('2024-03-18'), 'External SSD', 2, 199.99, 399.98, 'Shipped'),
        (110, 6, DATE('2024-01-30'), 'Headphones Pro', 1, 299.99, 299.99, 'Delivered')
        """
        engine.execute_query(insert_sql)
        print(f"✓ Inserted sample data into {orders_table}")

    except Exception as e:
        print(f"✗ Error setting up orders table: {e}")

    # Create demo_products table
    products_table = f"{workspace}.{schema}.demo_products"

    try:
        # Drop table if exists
        drop_sql = f"DROP TABLE IF EXISTS {products_table}"
        engine.execute_query(drop_sql)

        # Create products table
        create_products_sql = f"""
        CREATE TABLE {products_table} (
            product_id INT,
            product_name String,
            category String,
            brand String,
            price DECIMAL(10,2),
            stock_quantity INT,
            rating DECIMAL(3,2),
            release_date Date
        )
        """
        engine.execute_query(create_products_sql)
        print(f"✓ Created table: {products_table}")

        # Insert sample data using DATE() function
        insert_sql = f"""
        INSERT INTO {products_table} VALUES
        (1, 'Laptop Pro 16', 'Electronics', 'TechBrand', 2499.99, 25, 4.8, DATE('2023-09-15')),
        (2, 'Smartphone X', 'Electronics', 'PhoneCorp', 899.99, 50, 4.6, DATE('2023-10-20')),
        (3, 'Wireless Mouse', 'Accessories', 'TechBrand', 79.99, 100, 4.4, DATE('2023-08-10')),
        (4, 'Gaming Chair', 'Furniture', 'ComfortSeats', 399.99, 15, 4.7, DATE('2023-07-05')),
        (5, 'Monitor 4K', 'Electronics', 'DisplayTech', 449.99, 30, 4.5, DATE('2023-11-12')),
        (6, 'Mechanical Keyboard', 'Accessories', 'KeyMaster', 149.99, 75, 4.3, DATE('2023-06-18')),
        (7, 'Tablet Air', 'Electronics', 'TabletInc', 599.99, 40, 4.2, DATE('2023-12-01')),
        (8, 'Desktop Computer', 'Electronics', 'PCBuilder', 1299.99, 20, 4.9, DATE('2023-05-22')),
        (9, 'Headphones Pro', 'Electronics', 'AudioMax', 299.99, 60, 4.6, DATE('2023-08-30')),
        (10, 'External SSD', 'Storage', 'DataDrive', 199.99, 80, 4.7, DATE('2023-09-08'))
        """
        engine.execute_query(insert_sql)
        print(f"✓ Inserted sample data into {products_table}")

    except Exception as e:
        print(f"✗ Error setting up products table: {e}")

    # Show summary
    print("\n=== Demo Data Setup Complete ===")
    try:
        # Show table counts
        for table_name, display_name in [
            (customers_table, "demo_customers"),
            (orders_table, "demo_orders"),
            (products_table, "demo_products")
        ]:
            result, _ = engine.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            count = result[0]['count'] if result else 0
            print(f"  {display_name}: {count} records")
    except Exception as e:
        print(f"Error getting table counts: {e}")

    # Close connection
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")


if __name__ == "__main__":
    setup_demo_data()
