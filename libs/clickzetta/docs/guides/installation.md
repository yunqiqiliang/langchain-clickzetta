# 安装与配置

本指南将帮助您快速安装和配置 LangChain ClickZetta 集成。

## 📦 安装

### 使用 pip 安装（推荐）

```bash
pip install langchain-clickzetta
```

### 开发安装

如果您想从源码安装或参与开发：

```bash
git clone https://github.com/yunqiqiliang/langchain-clickzetta.git
cd langchain-clickzetta/libs/clickzetta
pip install -e ".[dev]"
```

## 🔧 依赖要求

### Python 版本
- Python 3.9 或更高版本

### 核心依赖
安装包时会自动安装以下依赖：

```
langchain-core>=0.1.0
clickzetta-connector-python>=0.8.92
clickzetta-zettapark-python>=0.1.3
sqlalchemy>=2.0.0
numpy>=1.20.0
pydantic>=2.0.0
typing-extensions>=4.0.0
```

> **版本信息**: 当前版本 0.1.13 已修复所有LangChain兼容性问题，包括vcluster参数支持和SHOW COLUMNS格式处理。

### 可选依赖

#### 中文AI优化（推荐）
```bash
pip install dashscope  # 阿里云灵积平台
```

#### 开发工具
```bash
pip install langchain-clickzetta[dev]
```

包含：pytest、ruff、black、mypy等开发工具

## 🏗️ ClickZetta 环境配置

### 获取 ClickZetta 访问权限

1. **注册云器科技账号**
   - 访问 [云器科技官网](https://www.yunqi.tech/)
   - 注册并申请 ClickZetta 试用

2. **获取连接信息**
   ClickZetta需要以下7个必需连接参数：
   - `service` - 服务地址
   - `instance` - 实例名称
   - `workspace` - 工作空间
   - `schema` - 模式名称
   - `username` - 用户名
   - `password` - 密码
   - `vcluster` - 虚拟集群名称（必需参数）

### 环境变量配置

创建 `.env` 文件或设置环境变量：

```bash
# ClickZetta 连接配置
export CLICKZETTA_SERVICE="your-service"
export CLICKZETTA_INSTANCE="your-instance"
export CLICKZETTA_WORKSPACE="your-workspace"
export CLICKZETTA_SCHEMA="your-schema"
export CLICKZETTA_USERNAME="your-username"
export CLICKZETTA_PASSWORD="your-password"
export CLICKZETTA_VCLUSTER="your-vcluster"

# 可选：灵积DashScope配置（推荐用于中文AI）
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 连接配置文件

您也可以使用配置文件，创建 `~/.clickzetta/connections.json`：

```json
{
  "default": {
    "service": "your-service",
    "instance": "your-instance",
    "workspace": "your-workspace",
    "schema": "your-schema",
    "username": "your-username",
    "password": "your-password",
    "vcluster": "your-vcluster"
  },
  "uat": {
    "service": "uat-service",
    "instance": "uat-instance",
    "workspace": "test",
    "schema": "test_schema",
    "username": "test-user",
    "password": "test-password",
    "vcluster": "test-cluster"
  }
}
```

## 🧪 验证安装

### 基本导入测试

```python
# 测试基本导入
try:
    from langchain_clickzetta import ClickZettaEngine
    print("✅ LangChain ClickZetta 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
```

### 连接测试

```python
from langchain_clickzetta import ClickZettaEngine

# 创建引擎实例
engine = ClickZettaEngine(
    service="your-service",
    instance="your-instance",
    workspace="your-workspace",
    schema="your-schema",
    username="your-username",
    password="your-password",
    vcluster="your-vcluster"
)

# 测试连接
try:
    results, columns = engine.execute_query("SELECT 1 as test")
    print("✅ ClickZetta 连接成功")
    print(f"测试结果: {results}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
```

### 完整功能测试

```python
from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaVectorStore,
    ClickZettaStore
)
from langchain_community.embeddings import DashScopeEmbeddings

# 初始化组件
engine = ClickZettaEngine(
    # ... 你的连接参数
)

# 测试向量存储
try:
    embeddings = DashScopeEmbeddings(
        dashscope_api_key="your-api-key",
        model="text-embedding-v4"
    )

    vector_store = ClickZettaVectorStore(
        engine=engine,
        embedding=embeddings,
        table_name="test_vectors"
    )
    print("✅ 向量存储初始化成功")
except Exception as e:
    print(f"⚠️  向量存储初始化失败: {e}")

# 测试键值存储
try:
    store = ClickZettaStore(engine=engine, table_name="test_store")
    print("✅ 键值存储初始化成功")
except Exception as e:
    print(f"❌ 键值存储初始化失败: {e}")
```

## ⚠️ 常见问题

### 连接问题

**问题**: `连接超时`
```
解决方案:
1. 检查网络连接
2. 确认ClickZetta服务地址正确
3. 增加connection_timeout参数
```

**问题**: `认证失败`
```
解决方案:
1. 确认用户名密码正确
2. 检查用户权限
3. 确认vcluster参数正确
```

### 依赖问题

**问题**: `ModuleNotFoundError: No module named 'clickzetta'`
```
解决方案:
pip install clickzetta-connector-python
```

**问题**: `版本冲突`
```
解决方案:
pip install --upgrade langchain-clickzetta
```

### 权限问题

**问题**: `权限不足，无法创建表`
```
解决方案:
1. 联系管理员授予CREATE TABLE权限
2. 使用现有表名
3. 确认workspace和schema权限
```

### LangChain兼容性问题

**问题**: `'is_nullable' KeyError`
```
解决方案:
这已在v0.1.13中修复。请升级到最新版本：
pip install --upgrade langchain-clickzetta
```

**问题**: `missing vcluster parameter`
```
解决方案:
确保提供所有7个必需参数，包括vcluster:
engine = ClickZettaEngine(
    service="...",
    instance="...",
    workspace="...",
    schema="...",
    username="...",
    password="...",
    vcluster="..."  # 必需参数
)
```

## 🚀 下一步

安装完成后，建议阅读：

1. [5分钟上手指南](quickstart.md) - 快速体验核心功能
2. [配置最佳实践](configuration.md) - 生产环境优化
3. [基础示例](../examples/basic.md) - 实际使用示例

## 💡 提示

- 建议在生产环境使用连接池
- 定期更新到最新版本
- 使用环境变量管理敏感信息
- 开启日志记录以便调试