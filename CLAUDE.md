# CLAUDE 开发笔记

## ClickZetta 产品文档位置

/Users/liangmo/Documents/GitHub/mcp-clickzetta-server/docs/clickzetta_product_doc

## Python虚拟环境

/Users/liangmo/Documents/GitHub/langchain-clickzetta/libs/clickzetta/.venv

## 何时需要查看产品文档

1. **SQL语法相关** - 必须查看ClickZetta SQL参考文档
   - 向量表创建语法 (vector index creation)
   - 距离函数名称 (L2_DISTANCE, COSINE_DISTANCE)
   - 全文搜索函数 (match_phrase, match_query)
   - 数据类型定义 (vector(float, dimension))

2. **向量搜索功能** - 查看向量搜索特性文档
   - 向量数据类型支持
   - 索引类型 (HNSW)
   - 距离计算方法
   - 向量维度限制

3. **连接参数** - 查看连接器文档
   - 连接ClickZetta必须7个参数: service, instance, workspace, schema, username, password, vcluster
   - 所有参数都是必需的，不能省略

## 重要注意事项

### SQL语法规范
- ClickZetta有特定的SQL语法，不能直接使用标准SQL
- 向量表创建不支持 `IF NOT EXISTS`
- 向量索引语法: `index idx_name (column_name) using vector properties (...)`
- 距离函数: `L2_DISTANCE()`, `COSINE_DISTANCE()`

### 向量数据类型
- 支持的元素类型: float, int, tinyint
- scalar.type 映射: f32 (float), i32 (int), i8 (tinyint)
- 向量维度必须在创建表时指定

### 连接配置
- 配置文件位置: `~/.clickzetta/connections.json`
- UAT环境连接名称: "uat"
- DashScope配置在 system_config.embedding.dashscope

### 开发规范
- 运行测试前确保: `source venv/bin/activate`
- 格式化代码: `black langchain_clickzetta tests`
- 类型检查: `mypy langchain_clickzetta`
- 代码检查: `ruff check .`

## 调试脚本使用

### 向量存储调试
```bash
python scripts/debug/debug_vectorstore.py
```

### 表创建调试
```bash
python scripts/debug/debug_table_creation.py
```

### 完整集成测试
```bash
python scripts/integration/test_real_dashscope_integration.py
```

## 常见错误和解决方案

1. **向量表创建失败**
   - 检查是否使用了 `IF NOT EXISTS` (不支持)
   - 检查向量维度是否正确
   - 检查距离函数名称是否正确

2. **连接失败**
   - 确保所有7个连接参数都提供
   - 检查vcluster参数是否遗漏

3. **SQL语法错误**
   - 查阅ClickZetta官方SQL语法文档
   - 不能直接使用MySQL/PostgreSQL语法

4. **向量搜索错误**
   - 检查向量维度是否匹配
   - 检查向量元素类型是否正确

5. **全文搜索没有结果**
   - 确保在添加文档后执行 `BUILD INDEX index_name ON table_name`
   - 检查分词器设置是否正确 (unicode/chinese/english/keyword)
   - 使用 `TOKENIZE()` 函数测试分词结果
   - 确保搜索词和文档内容的分词结果有交集

## 倒排索引实现确认

✅ **倒排索引已完整实现**：
- 建表时自动创建倒排索引：`INDEX content_fts (content) USING INVERTED PROPERTIES ('analyzer' = 'unicode')`
- 支持中文分词：unicode分词器能正确处理中英文混合文本
- 需要构建索引：添加文档后需要执行 `BUILD INDEX` 对存量数据建索引
- 搜索函数：支持 `match_phrase`, `match_all`, `match_any` 等全文搜索函数
- 测试验证：已通过真实ClickZetta环境测试确认功能正常

## 真正的混合检索实现

✅ **已实现同一表支持向量和倒排索引**：
- 新增 `ClickZettaHybridStore` 类，单表同时创建向量索引和倒排索引
- 表结构包含：`id`, `content`, `metadata`, `embedding vector(float,1024)`
- 同时创建两种索引：
  ```sql
  INDEX content_fts (content) USING INVERTED PROPERTIES ('analyzer' = 'unicode')
  INDEX embedding_idx (embedding) USING VECTOR PROPERTIES ('distance.function' = 'cosine_distance')
  ```
- 支持三种搜索模式：
  - 纯向量搜索：`similarity_search()`
  - 纯全文搜索：`fulltext_search()`
  - 混合搜索：`hybrid_search()` (可调权重)
- 完全兼容LangChain VectorStore标准接口
- 优于原有的跨表混合搜索，实现真正的单表混合检索

## 最后更新

最后更新时间: 2025-09-19
更新内容: 创建CLAUDE开发笔记，记录ClickZetta开发注意事项