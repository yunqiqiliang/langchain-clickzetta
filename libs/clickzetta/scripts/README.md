# Scripts Directory

这个目录包含了开发和测试过程中使用的脚本工具。

## 目录结构

### `debug/` - 调试脚本
开发过程中用于调试和验证的脚本：

- `debug_vectorstore.py` - 向量存储调试脚本
- `debug_table_creation.py` - 表创建调试脚本
- `debug_show_tables.py` - 表查询语法调试
- `debug_exact_syntax.py` - 精确语法测试
- `debug_manual_create.py` - 手动创建测试
- `debug_doc_syntax.py` - 文档语法验证

### `integration/` - 集成测试脚本
完整的端到端集成测试：

- `test_real_integration.py` - 基础真实性集成测试
- `test_real_dashscope_integration.py` - DashScope服务集成测试

## 使用方法

### 运行调试脚本
```bash
# 激活虚拟环境
source venv/bin/activate

# 运行特定调试脚本
python scripts/debug/debug_vectorstore.py
```

### 运行集成测试
```bash
# 激活虚拟环境
source venv/bin/activate

# 运行完整的DashScope集成测试
python scripts/integration/test_real_dashscope_integration.py
```

## 注意事项

- 所有脚本都需要先配置好 `~/.clickzetta/connections.json`
- DashScope相关测试需要有效的API密钥
- 调试脚本主要用于开发阶段，生产环境请使用正式的测试套件