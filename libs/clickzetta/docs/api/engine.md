# ClickZettaEngine API 参考

`ClickZettaEngine` 是 LangChain ClickZetta 集成的核心类，负责管理数据库连接和查询执行。

## 类定义

```python
class ClickZettaEngine:
    """ClickZetta数据库引擎，提供连接管理和查询执行功能。"""
```

## 构造函数

### `__init__()`

```python
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
    hints: dict[str, Any] | None = None,
    **kwargs: Any
) -> None:
```

创建 ClickZettaEngine 实例。

#### 参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `service` | `str` | ✅ | ClickZetta 服务地址 |
| `instance` | `str` | ✅ | ClickZetta 实例名称 |
| `workspace` | `str` | ✅ | 工作空间名称 |
| `schema` | `str` | ✅ | 模式名称 |
| `username` | `str` | ✅ | 用户名 |
| `password` | `str` | ✅ | 密码 |
| `vcluster` | `str` | ✅ | 虚拟集群名称（必需参数） |
| `connection_timeout` | `int` | ❌ | 连接超时时间（秒），默认30 |
| `query_timeout` | `int` | ❌ | 查询超时时间（秒），默认300 |
| `hints` | `dict` | ❌ | 查询提示参数 |

#### 示例

```python
# 基本用法
engine = ClickZettaEngine(
    service="clickzetta.example.com",
    instance="my-instance",
    workspace="my-workspace",
    schema="my-schema",
    username="user",
    password="password",
    vcluster="default"
)

# 带超时和提示的高级配置
engine = ClickZettaEngine(
    service="clickzetta.example.com",
    instance="my-instance",
    workspace="my-workspace",
    schema="my-schema",
    username="user",
    password="password",
    vcluster="default",
    connection_timeout=60,
    query_timeout=600,
    hints={
        "sdk.job.timeout": 1200,
        "query_tag": "langchain_app"
    }
)
```

## 核心方法

### `execute_query()`

```python
def execute_query(
    self,
    query: str,
    parameters: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
```

执行SQL查询并返回结果。

#### 参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `query` | `str` | ✅ | 要执行的SQL查询 |
| `parameters` | `dict` | ❌ | 查询参数（用于参数化查询） |

#### 返回值

返回元组 `(results, columns)`：
- `results`: `list[dict[str, Any]]` - 查询结果，每行一个字典
- `columns`: `list[str]` - 列名列表

#### 示例

```python
# 简单查询
results, columns = engine.execute_query("SELECT COUNT(*) as total FROM users")
print(f"用户总数: {results[0]['total']}")

# 参数化查询
results, columns = engine.execute_query(
    "SELECT * FROM users WHERE age > ? AND city = ?",
    parameters={"age": 25, "city": "北京"}
)

# 复杂查询
sql = """
SELECT
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC
"""
results, columns = engine.execute_query(sql)

for row in results:
    print(f"部门: {row['department']}, 人数: {row['employee_count']}, 平均薪资: {row['avg_salary']}")
```

### `get_table_names()`

```python
def get_table_names(self, schema: str | None = None) -> list[str]:
```

获取指定模式下的所有表名。

#### 参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `schema` | `str` | ❌ | 模式名称，默认使用实例配置的模式 |

#### 返回值

返回表名列表。

#### 示例

```python
# 获取默认模式下的所有表
tables = engine.get_table_names()
print(f"数据库中的表: {tables}")

# 获取指定模式下的表
tables = engine.get_table_names(schema="public")
print(f"public模式下的表: {tables}")
```

### `get_table_info()`

```python
def get_table_info(
    self,
    table_names: list[str] | None = None,
    schema: str | None = None
) -> str:
```

获取表结构信息，格式化为SQL DDL形式。

> **注意**: ClickZetta的SHOW COLUMNS返回格式为：`schema_name, table_name, column_name, data_type, comment`，与MySQL标准格式不同。本方法已经处理了这种差异，提供统一的接口。

#### 参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `table_names` | `list[str]` | ❌ | 表名列表，默认获取所有表 |
| `schema` | `str` | ❌ | 模式名称，默认使用实例配置的模式 |

#### 返回值

返回格式化的表结构信息字符串。

#### 示例

```python
# 获取所有表的结构信息
table_info = engine.get_table_info()
print(table_info)

# 获取指定表的结构信息
table_info = engine.get_table_info(table_names=["users", "orders"])
print(table_info)

# 获取指定模式下指定表的信息
table_info = engine.get_table_info(
    table_names=["products"],
    schema="inventory"
)
```

### `run()`

```python
def run(
    self,
    command: str,
    fetch: str = "all"
) -> str | Sequence[dict[str, Any]]:
```

执行SQL命令并返回格式化结果。

#### 参数

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `command` | `str` | ✅ | 要执行的SQL命令 |
| `fetch` | `str` | ❌ | 获取模式："all"或"one"，默认"all" |

#### 返回值

根据 `fetch` 参数返回字符串（格式化结果）或字典序列。

#### 示例

```python
# 获取格式化的结果字符串
result = engine.run("SELECT * FROM users LIMIT 5")
print(result)

# 获取单行结果
result = engine.run("SELECT COUNT(*) as total FROM users", fetch="one")
print(result)
```

## 属性

### `connection_config`

```python
@property
def connection_config(self) -> dict[str, str]:
```

获取当前连接配置信息。

#### 示例

```python
config = engine.connection_config
print(f"当前工作空间: {config['workspace']}")
print(f"当前模式: {config['schema']}")
```

### `dialect`

```python
@property
def dialect(self) -> str:
```

获取数据库方言标识，总是返回 "clickzetta"。

#### 示例

```python
print(f"数据库方言: {engine.dialect}")  # 输出: clickzetta
```

## 连接管理

### 连接池

ClickZettaEngine 内部使用连接池来管理数据库连接，提供以下特性：

- **自动重连**: 连接断开时自动重建
- **连接复用**: 避免频繁创建/销毁连接
- **超时管理**: 支持连接和查询超时
- **线程安全**: 支持多线程环境使用

### 最佳实践

```python
# 1. 使用单例模式共享引擎实例
class DatabaseManager:
    _engine = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = ClickZettaEngine(
                # 连接参数...
            )
        return cls._engine

# 2. 在应用生命周期中复用引擎
engine = DatabaseManager.get_engine()

# 3. 使用上下文管理器（如果需要）
class EngineContext:
    def __enter__(self):
        self.engine = ClickZettaEngine(...)
        return self.engine

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 清理资源
        pass

# 4. 配置合适的超时时间
engine = ClickZettaEngine(
    # 连接参数...
    connection_timeout=60,      # 连接超时
    query_timeout=1800,         # 长查询超时30分钟
    hints={
        "sdk.job.timeout": 3600  # 作业超时1小时
    }
)
```

## 错误处理

### 常见异常

```python
from langchain_clickzetta.engine import ClickZettaEngine

try:
    engine = ClickZettaEngine(...)
    results, columns = engine.execute_query("SELECT * FROM invalid_table")
except ConnectionError as e:
    print(f"连接错误: {e}")
except TimeoutError as e:
    print(f"超时错误: {e}")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 错误类型

| 异常类型 | 描述 | 处理建议 |
|----------|------|----------|
| `ConnectionError` | 数据库连接失败 | 检查网络和连接参数 |
| `TimeoutError` | 查询或连接超时 | 增加超时时间或优化查询 |
| `ValueError` | 参数验证失败 | 检查传入参数的格式和值 |
| `RuntimeError` | 查询执行错误 | 检查SQL语法和权限 |

## 性能优化

### 查询优化

```python
# 1. 使用查询提示
engine = ClickZettaEngine(
    # 连接参数...
    hints={
        "sdk.job.timeout": 3600,
        "query_tag": "analytics_app",
        "max_rows": 1000000
    }
)

# 2. 批量操作
# 避免在循环中执行单个查询
# 好的做法：
batch_sql = "INSERT INTO table VALUES " + ",".join(["(?, ?, ?)" for _ in data])
engine.execute_query(batch_sql, parameters=flat_data)

# 不好的做法：
# for item in data:
#     engine.execute_query("INSERT INTO table VALUES (?, ?, ?)", item)

# 3. 使用适当的LIMIT
results, _ = engine.execute_query("SELECT * FROM large_table LIMIT 1000")
```

### 连接优化

```python
# 复用引擎实例，避免频繁创建
engine = ClickZettaEngine(...)

# 使用长连接进行多次查询
for query in queries:
    results, _ = engine.execute_query(query)
    process_results(results)
```

## 完整示例

```python
from langchain_clickzetta import ClickZettaEngine
import os

def main():
    # 从环境变量获取配置
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE"),
        instance=os.getenv("CLICKZETTA_INSTANCE"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE"),
        schema=os.getenv("CLICKZETTA_SCHEMA"),
        username=os.getenv("CLICKZETTA_USERNAME"),
        password=os.getenv("CLICKZETTA_PASSWORD"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER"),
        connection_timeout=60,
        query_timeout=1800
    )

    try:
        # 测试连接
        results, _ = engine.execute_query("SELECT CURRENT_TIMESTAMP as now")
        print(f"连接成功，当前时间: {results[0]['now']}")

        # 获取表信息
        tables = engine.get_table_names()
        print(f"可用表: {tables}")

        # 执行业务查询
        if "users" in tables:
            results, _ = engine.execute_query("""
                SELECT
                    COUNT(*) as total_users,
                    COUNT(DISTINCT city) as cities_count
                FROM users
            """)
            print(f"用户统计: {results[0]}")

    except Exception as e:
        print(f"操作失败: {e}")

if __name__ == "__main__":
    main()
```