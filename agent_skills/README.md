# Leek Agent Skills

这些 Skills 帮助 AI 编程助手理解 leek-core 的架构和接口，提高量化交易策略开发效率。

## 包含的 Skills

| Skill | 用途 |
|-------|------|
| `leek-strategy` | 开发交易策略（Strategy、CTAStrategy） |
| `leek-indicators` | 使用和开发技术指标（50+ 内置指标） |
| `leek-data-source` | 配置和开发数据源（WebSocket、数据库） |
| `leek-executor` | 开发交易执行器（REST/WebSocket） |
| `leek-backtest` | 使用回测系统 |
| `leek-risk-position` | 风险和仓位管理 |
| `leek-factors` | 开发机器学习因子（DualModeFactor） |
| `leek-ml-training` | 训练机器学习模型（TrainingEngine、标签、训练器） |
| `leek-strategy-evaluation` | 评估策略回测结果，判断策略是否可用 |

## 安装方式

### Cursor

```bash
# 安装单个 skill
cp -r leek-strategy ~/.cursor/skills/

# 安装所有 skills
cp -r leek-* ~/.cursor/skills/
```

### Windsurf

```bash
# 安装单个 skill
cp -r leek-strategy ~/.windsurf/skills/

# 安装所有 skills
cp -r leek-* ~/.windsurf/skills/
```

### GitHub Copilot / Claude / 其他工具

将对应 skill 目录下的 `SKILL.md` 内容添加到你的 AI 工具的：
- System Prompt
- Custom Instructions
- Knowledge Base
- 项目 README 或文档

## Skill 结构说明

每个 skill 目录包含：

```
leek-xxx/
├── SKILL.md          # 主文件，包含核心概念和使用方法
└── reference/        # （可选）详细参考文档
    └── xx-xxx.md
```

- `SKILL.md` 保持精简（<500行），包含关键接口和示例
- `reference/` 目录包含完整文档，供深入学习时参考

## 使用建议

1. **按需安装**：根据开发任务选择需要的 skill，无需全部安装
2. **配合源码**：skill 提供快速参考，复杂实现请参考 `src/leek_core/` 源码
3. **参考测试**：`tests/` 目录包含大量使用示例

## 相关文档

- 完整文档：`leek-core/docs/`
- 源代码：`leek-core/src/leek_core/`
- 测试用例：`leek-core/src/leek_core/tests/`
