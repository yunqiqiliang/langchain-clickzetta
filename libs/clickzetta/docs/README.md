# LangChain ClickZetta 开发者手册

欢迎使用 LangChain ClickZetta 集成！本文档为开发者提供全面的产品使用指南。

## 📚 文档导航

### 🚀 快速开始
- [安装与配置](guides/installation.md) - 环境搭建和基本配置
- [5分钟上手指南](guides/quickstart.md) - 快速体验核心功能
- [配置最佳实践](guides/configuration.md) - 生产环境配置建议

### 📖 功能指南
- [SQL查询与自然语言处理](guides/sql-queries.md) - AI驱动的SQL查询
- [向量存储与相似性搜索](guides/vector-storage.md) - 高性能向量搜索
- [全文搜索与索引](guides/fulltext-search.md) - 企业级文本检索
- [混合搜索](guides/hybrid-search.md) - 向量+全文的统一搜索
- [存储服务](guides/storage-services.md) - 企业级数据存储解决方案
- [聊天历史管理](guides/chat-history.md) - 对话记忆与会话管理

### 🎯 实用教程
- [构建RAG应用](tutorials/rag-application.md) - 端到端RAG系统构建
- [中文AI应用开发](tutorials/chinese-ai-app.md) - 针对中文场景的优化
- [企业级部署](tutorials/enterprise-deployment.md) - 生产环境部署指南
- [性能优化](tutorials/performance-optimization.md) - 系统性能调优
- [错误处理与调试](tutorials/debugging.md) - 常见问题排查

### 📋 API参考
- [核心引擎 API](api/engine.md) - ClickZettaEngine 详细说明
- [SQL链 API](api/sql-chain.md) - ClickZettaSQLChain 接口文档
- [向量存储 API](api/vector-store.md) - ClickZettaVectorStore 完整API
- [存储服务 API](api/storage.md) - 所有存储类的详细接口
- [检索器 API](api/retrievers.md) - 各种检索器的使用方法

### 💡 示例代码
- [基础示例](examples/basic.md) - 简单功能演示
- [高级用例](examples/advanced.md) - 复杂场景实现
- [集成示例](examples/integrations.md) - 与其他系统的集成
- [最佳实践](examples/best-practices.md) - 推荐的使用模式

## 🏗️ 架构概览

LangChain ClickZetta 集成提供以下核心组件：

### 核心组件
1. **引擎** (`langchain_clickzetta.engine`) - 数据库连接和查询执行
2. **向量存储** (`langchain_clickzetta.vectorstores`) - 向量相似性搜索
3. **存储服务** (`langchain_clickzetta.stores`) - 持久化键值存储
4. **SQL链** (`langchain_clickzetta.sql_chain`) - 自然语言转SQL
5. **检索器** (`langchain_clickzetta.retrievers`) - 文档检索系统

### 存储服务架构

#### 基于表的存储
- **ClickZettaStore** - 使用SQL表的键值存储
- **ClickZettaDocumentStore** - 带元数据的文档存储（继承自ClickZettaStore）

#### 基于Volume的存储
- **ClickZettaFileStore** - 使用ClickZetta Volume的二进制文件存储
- **ClickZettaUserVolumeStore** - 用户专属volume存储
- **ClickZettaTableVolumeStore** - 表专属volume存储
- **ClickZettaNamedVolumeStore** - 命名volume存储

## 🔧 开发者资源

### 环境要求
- Python 3.9+
- ClickZetta 数据库实例
- 推荐：灵积DashScope API密钥（用于中文优化）

### 相关链接
- [PyPI 包](https://pypi.org/project/langchain-clickzetta/) - 最新版本安装
- [GitHub 仓库](https://github.com/yunqiqiliang/langchain-clickzetta) - 源码和问题反馈
- [ClickZetta 官网](https://www.yunqi.tech/) - ClickZetta 产品信息
- [LangChain 文档](https://python.langchain.com/) - LangChain 框架文档

## 📋 关键特性

### ✅ LangChain兼容性
- 完整的 `BaseStore` 接口实现
- 支持同步和异步方法
- 标准的LangChain使用模式
- 类型安全操作

### ✅ ClickZetta原生特性
- MERGE INTO操作实现原子化UPSERT
- Volume存储支持二进制数据
- SQL可查询的文档存储
- 完善的错误处理和日志记录

### ✅ 性能优化
- 多键批量操作
- 使用MERGE INTO的原子事务
- 高效的基于前缀的键过滤
- 连接池和连接复用

## 📞 技术支持

### 社区支持
- [GitHub Issues](https://github.com/yunqiqiliang/langchain-clickzetta/issues) - 问题反馈和功能请求
- [GitHub Discussions](https://github.com/yunqiqiliang/langchain-clickzetta/discussions) - 技术讨论和经验分享

### 企业支持
如需企业级技术支持，请联系云器科技团队。

---

> 💡 **提示**：建议按照文档顺序阅读，从安装配置开始，逐步掌握各项功能。所有示例代码都经过测试验证，可以直接在您的环境中运行。