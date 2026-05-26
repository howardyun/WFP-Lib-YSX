---
name: reproduce-repair
description: 使用 reproduce.yaml 和 tools/run_reproduce.py 对当前科研项目进行可信复现和自动勘误。
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

你正在执行一个科研项目的可信复现任务。

目标不是简单地让脚本运行通过，而是在尽可能保留原始方法逻辑的前提下，完成可追踪、可验证、可报告的复现实验。

## 必须遵循的流程

1. 先阅读 `CLAUDE.md`。
2. 阅读 `README.md`。
3. 阅读 `setup.py`。
4. 阅读 `reproduce.yaml`。
5. 检查目标 shell 脚本，例如 `scripts/AWF.sh`。
6. 通过以下统一入口运行目标命令：

```bash
python tools/run_reproduce.py --config reproduce.yaml --only <command_name>
```

7. 如果运行失败，必须检查：

   - `logs/latest_run_summary.json`
   - 最新的 stderr 日志
   - 最新的 stdout 日志

8. 在修改任何文件之前，必须先对错误进行分类。

9. 按以下优先级考虑修复方案：

   - 依赖安装问题
   - Python 版本兼容问题
   - CUDA 或 PyTorch 兼容问题
   - 数据集缺失问题
   - checkpoint 或模型权重缺失问题
   - 相对路径或工作目录问题
   - shell 脚本参数问题
   - API 废弃或版本变化问题
   - 源码实现错误

10. 如果必须修改代码，只能做最小、局部、可回退的修改。

11. 修改后必须重新运行完全相同的失败命令。

12. 每次修复尝试都必须追加记录到 `logs/repair_history.md`。

## 修复记录格式

每次修复尝试都按如下格式记录：

```markdown
## Attempt N

- 运行命令：
- 错误类别：
- 错误证据：
- 根本原因：
- 已检查文件：
- 已修改文件：
- 补丁摘要：
- 重跑命令：
- 重跑结果：
- 剩余问题：
```

## 停止规则

对于同一个命令，如果连续 3 次修复仍然失败，应停止继续修改，并总结当前无法解决的阻塞点。

## 严格限制

- 不得修改核心算法逻辑，除非有充分证据证明其存在实现错误。
- 不得删除模型组件。
- 不得跳过数据加载。
- 不得跳过训练或评估。
- 不得伪造输出文件。
- 不得在没有日志和输出验证的情况下报告成功。