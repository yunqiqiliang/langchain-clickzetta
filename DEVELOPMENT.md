# 开发指南

## 项目结构

```
langchain-clickzetta/
├── langchain_clickzetta/     # 主要源代码
│   ├── __init__.py          # 包初始化
│   ├── engine.py            # ClickZetta数据库引擎
│   ├── vectorstores.py      # 向量存储实现
│   ├── sql_chain.py         # SQL查询链
│   ├── chat_message_histories.py  # 聊天历史
│   └── retrievers.py        # 检索器实现
├── tests/                   # 测试代码
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
├── scripts/                # 开发脚本
│   ├── debug/              # 调试脚本
│   └── integration/        # 集成测试脚本
├── examples/               # 使用示例
├── docs/                   # 文档
├── pyproject.toml          # 项目配置
├── README.md               # 项目说明
└── DEVELOPMENT.md          # 开发指南
```

## 开发环境设置

### 1. 克隆仓库
```bash
git clone https://github.com/your-repo/langchain-clickzetta.git
cd langchain-clickzetta
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或者 venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -e ".[dev]"
```

### 4. 配置ClickZetta连接
创建 `~/.clickzetta/connections.json` 文件:
```json
{
  "system_config": {
    "embedding": {
      "dashscope": {
        "api_key": "your-dashscope-api-key",
        "model": "text-embedding-v4",
        "dimensions": 1024
      }
    }
  },
  "connections": [
    {
      "name": "uat",
      "service": "your-service",
      "username": "your-username",
      "password": "your-password",
      "instance": "your-instance",
      "workspace": "your-workspace",
      "schema": "your-schema",
      "vcluster": "your-vcluster"
    }
  ]
}
```

## 运行测试

### 单元测试
```bash
pytest tests/unit/ -v
```

### 集成测试 (需要真实ClickZetta连接)
```bash
pytest tests/integration/ -v
```

### 完整真实性测试
```bash
# 基础集成测试
python scripts/integration/test_real_integration.py

# DashScope服务集成测试
python scripts/integration/test_real_dashscope_integration.py
```

## 代码质量

### 格式化代码
```bash
black langchain_clickzetta tests
```

### 类型检查
```bash
mypy langchain_clickzetta
```

### 代码检查
```bash
ruff check .
```

## 调试

开发过程中可以使用 `scripts/debug/` 目录下的调试脚本：

```bash
# 调试向量存储
python scripts/debug/debug_vectorstore.py

# 调试表创建
python scripts/debug/debug_table_creation.py
```

## 贡献代码

1. 创建功能分支: `git checkout -b feature/amazing-feature`
2. 编写代码和测试
3. 确保所有测试通过: `pytest`
4. 提交更改: `git commit -m 'Add amazing feature'`
5. 推送分支: `git push origin feature/amazing-feature`
6. 创建Pull Request

## 发布流程

1. 更新版本号在 `pyproject.toml`
2. 运行完整测试套件
3. 创建git标签: `git tag v0.1.0`
4. 推送标签: `git push origin v0.1.0`
5. 构建和发布: `python -m build && twine upload dist/*`