#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("缺少依赖：PyYAML。请先安装：pip install pyyaml")
    sys.exit(2)


def load_config(path: Path) -> dict:
    """读取 reproduce.yaml 配置文件。"""
    print(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_command(name: str, command: str, timeout: int, log_dir: Path) -> dict:
    """运行单条复现实验命令，并保存标准输出、标准错误和运行状态。"""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace("/", "_").replace(" ", "_").replace("-", "_")

    stdout_log = log_dir / f"{timestamp}_{safe_name}_stdout.log"
    stderr_log = log_dir / f"{timestamp}_{safe_name}_stderr.log"

    print(f"[运行任务] {name}")
    print(f"[执行命令] {command}")
    print(f"[标准输出日志] {stdout_log}")
    print(f"[标准错误日志] {stderr_log}")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")

        return {
            "name": name,
            "command": command,
            "timeout": timeout,
            "returncode": proc.returncode,
            "status": "passed" if proc.returncode == 0 else "failed",
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    except subprocess.TimeoutExpired as e:
        stdout_log.write_text(e.stdout or "", encoding="utf-8", errors="replace")
        stderr_log.write_text(e.stderr or "", encoding="utf-8", errors="replace")

        return {
            "name": name,
            "command": command,
            "timeout": timeout,
            "returncode": -1,
            "status": "timeout",
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="统一运行复现实验脚本，并保存日志。")
    parser.add_argument("--config", default="reproduce.yaml", help="复现实验配置文件路径")
    parser.add_argument("--only", default=None, help="只运行指定名称的命令")
    args = parser.parse_args()

    root = Path.cwd()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    config = load_config(Path(args.config))
    commands = config.get("commands", [])

    if args.only:
        commands = [cmd for cmd in commands if cmd.get("name") == args.only]
        if not commands:
            print(f"未找到名为 {args.only} 的命令。")
            return 2

    results = []

    for cmd in commands:
        result = run_command(
            name=cmd["name"],
            command=cmd["command"],
            timeout=int(cmd.get("timeout", 600)),
            log_dir=log_dir,
        )
        results.append(result)

        if result["status"] != "passed":
            break

    summary_path = log_dir / "latest_run_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[运行摘要] {summary_path}")

    failed = [r for r in results if r["status"] != "passed"]
    if failed:
        print("[最终结果] 运行失败")
        return 1

    print("[最终结果] 运行成功")
    return 0


if __name__ == "__main__":
    sys.exit(main())