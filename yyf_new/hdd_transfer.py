#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hdd_transfer.py

用于 Orin 上把 /projects 下指定前缀的数据目录批量传输到 8T 机械硬盘。
推荐用法：

1) 自动进入 tmux，传输 /projects 下 park* 和 supermarket* 到 /data/hdd8t/0521：
   python3 hdd_transfer.py --dest /data/hdd8t/0521 --patterns park supermarket --tmux

2) 只预览，不传输：
   python3 hdd_transfer.py --dest /data/hdd8t/0521 --patterns park supermarket --dry-run

3) 传输更多前缀：
   python3 hdd_transfer.py --dest /data/hdd8t/0521 --patterns park supermarket CBICR 0518 --tmux

说明：
- 脚本只复制，不删除 /projects 源数据。
- 默认会先列出匹配目录，并要求输入 yes 确认。
- 传输使用 rsync：可中断、可续传、显示总体进度。
"""

import argparse
import fnmatch
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print("\n$ " + shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def inside_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def start_or_attach_tmux(session_name: str, original_args: List[str]) -> None:
    if inside_tmux():
        return

    if not command_exists("tmux"):
        print("错误：系统没有安装 tmux。请先执行：sudo apt install -y tmux", file=sys.stderr)
        sys.exit(2)

    # 去掉 --tmux，避免进入 tmux 后无限递归启动 tmux
    args_without_tmux = [a for a in original_args if a != "--tmux"]

    script_path = str(Path(__file__).resolve())
    cmd_inside_tmux = shlex.join([sys.executable, script_path] + args_without_tmux)

    # 如果会话已经存在，就直接 attach
    has_session = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

    if has_session:
        print(f"tmux 会话已存在，正在进入：{session_name}")
        os.execvp("tmux", ["tmux", "attach", "-t", session_name])

    print(f"即将进入 tmux 会话：{session_name}")
    print("传输过程中可按 Ctrl+B，然后按 D 退出 tmux，任务会继续在 Orin 上运行。")
    print("脚本结束或报错后，tmux 不会自动关闭，会留在 bash 里方便你查看原因。")

    wrapped_cmd = (
        "set +e; "
        + cmd_inside_tmux
        + "; status=$?; "
        + "echo; "
        + "echo '========================================'; "
        + "echo \"hdd_transfer.py 已结束，退出码: $status\"; "
        + "echo '如果退出码不是 0，请查看上面的报错信息。'; "
        + "echo '你现在仍在 tmux 里，可继续输入命令。'; "
        + "echo '退出 tmux 但让会话继续：Ctrl+B 然后 D'; "
        + "echo '彻底结束这个 tmux：输入 exit'; "
        + "echo '========================================'; "
        + "exec bash"
    )

    os.execvp("tmux", ["tmux", "new", "-s", session_name, "bash", "-lc", wrapped_cmd])


def normalize_prefix(pattern: str) -> str:
    # 用户输入 park -> park*
    # 用户输入 park_* -> 保持 park_*
    if any(ch in pattern for ch in ["*", "?", "["]):
        return pattern
    return pattern + "*"


def find_source_dirs(source_root: Path, patterns: Iterable[str]) -> List[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"源目录不存在：{source_root}")

    normalized = [normalize_prefix(p) for p in patterns]
    result = []

    for item in source_root.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if any(fnmatch.fnmatch(name, pat) for pat in normalized):
            result.append(item)

    return sorted(set(result), key=lambda p: p.name)


def ensure_hdd_mounted(dest: Path, mountpoint: Path, automount_unit: str) -> None:
    # 启动 automount。即使已经启动，也不会有问题。
    if command_exists("systemctl"):
        subprocess.run(
            ["sudo", "systemctl", "start", automount_unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # 访问挂载点，触发 systemd automount
    mountpoint.mkdir(parents=True, exist_ok=True)
    try:
        list(mountpoint.iterdir())
    except Exception:
        pass

    # 检查挂载点是否真的挂载
    proc = subprocess.run(["findmnt", str(mountpoint)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"错误：{mountpoint} 当前没有挂载。")
        print("请确认 8T 硬盘已经插电、USB 已接入 Orin，并执行：")
        print(f"  lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT,MODEL")
        print(f"  sudo systemctl start {automount_unit}")
        print(f"  ls {mountpoint}")
        sys.exit(3)

    # 确保目标目录在挂载点下面，避免不小心写到系统盘
    try:
        dest.resolve().relative_to(mountpoint.resolve())
    except ValueError:
        print(f"错误：目标目录 {dest} 不在挂载点 {mountpoint} 下面。")
        print("为避免误写系统盘，脚本已停止。")
        sys.exit(4)


def count_files(paths: List[Path]) -> int:
    total = 0
    for root in paths:
        for _, _, files in os.walk(root):
            total += len(files)
    return total


def human_du(path_list: List[Path]) -> None:
    if not path_list:
        return
    run(["du", "-sch"] + [str(p) for p in path_list] + ["|", "tail", "-1"], check=False)


def run_shell(command: str, check: bool = True) -> subprocess.CompletedProcess:
    print("\n$ " + command, flush=True)
    return subprocess.run(command, shell=True, check=check)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 /projects 下指定前缀的数据目录批量传输到 8T 机械硬盘。"
    )
    parser.add_argument(
        "--source-root",
        default="/projects",
        help="源数据根目录，默认：/projects",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="目标目录，例如：/data/hdd8t/0521",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        required=True,
        help="要匹配的目录名前缀，例如：park supermarket。输入 park 会匹配 park*。",
    )
    parser.add_argument(
        "--mountpoint",
        default="/data/hdd8t",
        help="8T 硬盘挂载点，默认：/data/hdd8t",
    )
    parser.add_argument(
        "--automount-unit",
        default="data-hdd8t.automount",
        help="systemd automount unit 名称，默认：data-hdd8t.automount",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="tmux 会话名，默认根据目标目录自动生成。",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="自动在 tmux 里运行。推荐长时间传输时使用。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览 rsync，不实际复制。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过 yes 确认，直接执行。",
    )

    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    source_root = Path(args.source_root).resolve()
    mountpoint = Path(args.mountpoint).resolve()

    session_name = args.session
    if not session_name:
        safe_dest = dest.name.replace("/", "_") or "transfer"
        session_name = f"transfer_{safe_dest}"

    if args.tmux:
        start_or_attach_tmux(session_name, sys.argv[1:])

    print("========== 传输配置 ==========")
    print(f"源数据根目录: {source_root}")
    print(f"目标目录:     {dest}")
    print(f"匹配规则:     {', '.join(normalize_prefix(p) for p in args.patterns)}")
    print(f"挂载点:       {mountpoint}")
    print("=============================")

    ensure_hdd_mounted(dest, mountpoint, args.automount_unit)

    source_dirs = find_source_dirs(source_root, args.patterns)

    if not source_dirs:
        print("没有找到匹配的源目录。请检查 --patterns 是否正确。")
        sys.exit(0)

    print("\n将要传输以下目录：")
    for p in source_dirs:
        print(f"  {p}")

    print("\n源目录文件数量统计中...")
    src_count = count_files(source_dirs)
    print(f"源目录文件总数: {src_count}")

    print("\n源目录总大小：")
    run_shell("du -sch " + " ".join(shlex.quote(str(p)) for p in source_dirs) + " | tail -1", check=False)

    if args.dry_run:
        print("\n当前是 dry-run 模式，只预览 rsync，不实际复制。")
    else:
        dest.mkdir(parents=True, exist_ok=True)

    if not args.yes:
        print("\n确认要开始传输吗？")
        print("请输入 yes 后回车继续；其他输入会取消。")
        answer = input("> ").strip()
        if answer != "yes":
            print("已取消。")
            sys.exit(0)

    print("\n========== 开始传输 ==========")
    for src in source_dirs:
        print("\n======================================")
        print(f"正在处理: {src}")
        print(f"目标位置: {dest}/")
        print("======================================")

        rsync_cmd = [
            "rsync",
            "-avh",
            "--info=progress2",
            "--partial",
            "--append-verify",
        ]

        if args.dry_run:
            rsync_cmd.append("--dry-run")

        # 这里 src 后面不加 /，用于保留 park_xxx / supermarket_xxx 目录本身
        rsync_cmd += [str(src), str(dest) + "/"]
        run(rsync_cmd)

    print("\n========== 传输阶段结束 ==========")
    print("目标目录大小：")
    run_shell("du -sh " + shlex.quote(str(dest)), check=False)

    print("\n目标目录文件数量：")
    run_shell("find " + shlex.quote(str(dest)) + " -type f | wc -l", check=False)

    print("\n建议你再手动确认一次：")
    print(f"  du -sch {source_root}/park* {source_root}/supermarket* | tail -1")
    print(f"  du -sh {dest}")
    print(f"  find {dest} -type f | wc -l")

    print("\n确认无误后，如需安全拔盘，请执行：")
    print("  cd ~")
    print("  sync")
    print(f"  sudo umount {mountpoint}")
    print(f"  sudo systemctl stop {args.automount_unit}")
    print(f"  findmnt {mountpoint}")
    print("  lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT,MODEL")
    print("\n注意：脚本不会自动删除 /projects 源数据。确认备份无误后再手动删除。")


if __name__ == "__main__":
    main()
