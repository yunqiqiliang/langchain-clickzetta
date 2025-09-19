# ClickZetta 存储服务改进总结

## 问题修正

### 1. UPSERT 操作优化：从 DELETE+INSERT 改为 MERGE INTO

**问题**: 之前使用的 DELETE+INSERT 模式不够高效，且不是原子操作。

**解决方案**: 根据 ClickZetta 产品文档，使用标准的 `MERGE INTO` 语法：

```sql
-- 修正前 (DELETE+INSERT)
DELETE FROM table WHERE key = 'key1';
INSERT INTO table (key, value, updated_at) VALUES ('key1', 'value1', CURRENT_TIMESTAMP);

-- 修正后 (MERGE INTO)
MERGE INTO table AS target
USING (SELECT 'key1' AS key, 'value1' AS value, CURRENT_TIMESTAMP AS ts) AS source
ON target.key = source.key
WHEN MATCHED THEN UPDATE SET
    value = source.value,
    updated_at = source.ts
WHEN NOT MATCHED THEN INSERT
    (key, value, created_at, updated_at)
    VALUES (source.key, source.value, source.ts, source.ts)
```

**优势**:
- ✅ 原子性操作，避免并发问题
- ✅ 更高效的性能
- ✅ 符合 ClickZetta 标准 SQL 规范
- ✅ 单条 SQL 完成 UPSERT 操作

### 2. Volume 存储类型名称规范化

**问题**: Volume Store 的参数验证和命名规则不够明确。

**解决方案**: 根据 ClickZetta Volume 文档，明确三种类型的命名规则：

#### User Volume
```python
# ✅ 正确用法 - 不需要指定 volume_name
user_store = ClickZettaUserVolumeStore(engine=engine)

# ❌ 错误用法 - User Volume 不应指定 volume_name
user_store = ClickZettaVolumeStore(engine=engine, volume_type="user", volume_name="xxx")
# 抛出: ValueError: volume_name should not be provided for user volume (uses current user '~')
```

#### Table Volume
```python
# ✅ 正确用法 - 必须指定 table_name
table_store = ClickZettaTableVolumeStore(engine=engine, table_name="my_table")

# ❌ 错误用法 - Table Volume 必须指定 table_name
table_store = ClickZettaVolumeStore(engine=engine, volume_type="table")
# 抛出: ValueError: table_name is required for table volume
```

#### Named Volume
```python
# ✅ 正确用法 - 必须指定 volume_name
named_store = ClickZettaNamedVolumeStore(engine=engine, volume_name="shared_volume")

# ❌ 错误用法 - Named Volume 必须指定 volume_name
named_store = ClickZettaVolumeStore(engine=engine, volume_type="named")
# 抛出: ValueError: volume_name is required for named volume
```

## 技术实现细节

### MERGE INTO 语法结构

所有存储服务现在使用统一的 MERGE INTO 模式：

1. **ClickZettaStore** (键值存储)
2. **ClickZettaDocumentStore** (文档存储)
3. **ClickZettaFileStore** (文件存储)

### Volume 协议规范

根据 ClickZetta 文档，Volume 地址格式：

| Volume 类型 | 地址格式 | SQL 操作示例 |
|------------|----------|-------------|
| User Volume | `volume:user://~/filename` | `PUT 'file' TO USER VOLUME FILE 'filename'` |
| Table Volume | `volume:table://table_name/filename` | `PUT 'file' TO TABLE VOLUME table_name FILE 'filename'` |
| Named Volume | `volume://volume_name/filename` | `PUT 'file' TO VOLUME volume_name FILE 'filename'` |

## 验证测试

创建了完整的参数验证测试 (`test_volume_validation.py`)，确保：

- ✅ User Volume 不接受 volume_name 参数
- ✅ Table Volume 必须提供 table_name 参数
- ✅ Named Volume 必须提供 volume_name 参数
- ✅ 无效的 volume_type 被正确拒绝

## 性能和可靠性提升

### MERGE INTO 的优势

1. **原子性**: 单条 SQL 完成完整的 UPSERT 操作
2. **性能**: 避免了两次数据库往返（DELETE + INSERT）
3. **并发安全**: 减少了竞态条件的风险
4. **标准化**: 使用 ClickZetta 原生支持的标准 SQL 语法

### Volume 类型安全

1. **编译时检查**: 参数验证在对象创建时进行
2. **清晰的错误信息**: 明确指出参数要求
3. **类型安全**: 不同 Volume 类型有专门的便捷类
4. **文档一致性**: 与 ClickZetta 产品文档完全一致

## 向后兼容性

所有改进都保持了 API 的向后兼容性：

- ✅ 现有的 `mget()`, `mset()`, `mdelete()`, `yield_keys()` 方法签名不变
- ✅ 现有的构造函数参数保持兼容
- ✅ 只是内部实现从 DELETE+INSERT 优化为 MERGE INTO
- ✅ 增加了更严格的参数验证，提高了代码质量

## 测试验证

所有存储服务都通过了完整的测试：

```bash
# 运行完整的存储服务演示
python examples/storage_example.py

# 运行 Volume 参数验证测试
python test_volume_validation.py
```

两个测试都成功通过，证明了改进的正确性和稳定性。