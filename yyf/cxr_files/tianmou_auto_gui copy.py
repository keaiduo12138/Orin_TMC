#!/usr/bin/env python3
import csv
import os
import queue
import re
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class UdpCameraController:
    def __init__(self, ip: str, port: int, log_fn):
        self.ip = ip
        self.port = port
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.log = log_fn

    def update_addr(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.addr = (ip, port)
        self.log(f"[UDP] 目标地址更新为 {self.addr}")

    def send(self, msg: str):
        try:
            self.sock.sendto(msg.encode("utf-8"), self.addr)
            self.log(f"[UDP] -> {self.addr}: {msg}")
        except Exception as exc:
            self.log(f"[ERR] UDP 发送失败: {exc}")

    def set_save_addr(self, path: str):
        self.send("set_addr:" + path)

    def start_camera(self):
        self.send("start_camera")

    def start_record(self):
        self.send("start_record")

    def stop_record(self):
        self.send("stop_record")

    def stop_camera(self):
        self.send("stop_camera")


class TianmouAutoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("天眸自动测试")
        self.geometry("550x380")
        self.minsize(500, 320)

        self.log_queue = queue.Queue()
        self.main_proc = None
        self.workflow_running = False

        repo_root = "/home/nvidia/Desktop/cxr_multi_sensor"
        self.main_exec_var = tk.StringVar(value=os.path.join(repo_root, "yyf", "build", "main"))
        self.main_cwd_var = tk.StringVar(value=os.path.join(repo_root, "yyf", "build"))
        self.ts_exec_var = tk.StringVar(value=os.path.join(repo_root, "test_timestamp", "read_timestamp"))
        self.ts_save_dir_var = tk.StringVar(value="/projects/cxr_data/temporal_files")
        self.gpio_script_var = tk.StringVar(value=os.path.join(repo_root, "yyf", "power_1v8_ctrl.py"))

        self.save_path_var = tk.StringVar()
        self.start_delay_var = tk.StringVar(value="1.0")

        self._generate_save_path()

        self.ctrl = UdpCameraController("127.0.0.1", 8889, self._log)
        self._build_ui()
        self._update_status("空闲")
        self.after(100, self._drain_log_queue)

    def _generate_save_path(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        base_dir = "/projects/cxr_data"
        full_path = os.path.join(base_dir, ts)
        self.save_path_var.set(full_path)

    def _build_ui(self):
        root = ttk.Frame(self, padding=20)
        root.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(root, text="天眸自动测试系统", font=("TkDefaultFont", 14, "bold"))
        title_label.pack(pady=(0, 15))

        path_frame = ttk.Frame(root)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(path_frame, text="保存路径:").pack(side=tk.LEFT)
        ttk.Entry(path_frame, textvariable=self.save_path_var, width=35).pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="刷新", command=self._on_refresh_path, width=6).pack(side=tk.LEFT)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=(0, 10))
        self.start_btn = ttk.Button(btn_frame, text="开始", command=self.on_start, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(btn_frame, text="结束", command=self.on_stop, width=12, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.ts_btn = ttk.Button(btn_frame, text="读取时间戳", command=self.on_read_timestamp, width=12)
        self.ts_btn.pack(side=tk.LEFT, padx=5)

        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, text="状态: ").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="空闲", foreground="green", font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(root, text="日志")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _on_refresh_path(self):
        self._generate_save_path()
        self._log("[SYS] 已刷新保存路径")

    def _log(self, message: str):
        self.log_queue.put(message)

    def _drain_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}\n"
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")
        self.after(100, self._drain_log_queue)

    def _update_status(self, status: str, color: str = "green"):
        self.status_label.configure(text=status, foreground=color)

    def _read_stream(self, proc: subprocess.Popen, tag: str):
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                self._log(f"[{tag}] {line.rstrip()}")
        except Exception as exc:
            self._log(f"[ERR] 日志读取 {tag} 失败: {exc}")

    def _start_subprocess(self, cmd_list, cwd, tag):
        proc = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        threading.Thread(target=self._read_stream, args=(proc, tag), daemon=True).start()
        return proc

    def _run_gpio(self, action: str):
        gpio_script = self.gpio_script_var.get().strip()
        if not os.path.isfile(gpio_script):
            self._log(f"[ERR] GPIO 脚本不存在: {gpio_script}")
            return False

        self._log(f"[GPIO] 执行 {action} (Pin 15)...")
        try:
            result = subprocess.run(
                ["sudo", "python3", gpio_script, action],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self._log(f"[GPIO] {action} 成功")
                return True
            else:
                self._log(f"[ERR] GPIO {action} 失败: {result.stderr}")
                return False
        except Exception as exc:
            self._log(f"[ERR] GPIO 执行异常: {exc}")
            return False

    def on_start(self):
        if self.workflow_running:
            messagebox.showwarning("警告", "自动化流程已在运行中")
            return
        self._generate_save_path()
        threading.Thread(target=self._start_workflow, daemon=True).start()

    def on_stop(self):
        if not self.workflow_running:
            messagebox.showinfo("提示", "没有运行中的自动化流程")
            return
        threading.Thread(target=self._stop_workflow, daemon=True).start()

    def on_read_timestamp(self):
        # 读取已保存的最新 CSV 文件
        ts_dir = self.ts_save_dir_var.get().strip()
        if not os.path.isdir(ts_dir):
            self._log(f"[ERR] 时间戳保存目录不存在: {ts_dir}")
            return

        csv_files = [f for f in os.listdir(ts_dir) if f.startswith("tianmou_timestamp_") and f.endswith(".csv")]
        if not csv_files:
            self._log(f"[ERR] 未找到已保存的时间戳文件")
            return

        csv_files.sort(reverse=True)
        latest_file = os.path.join(ts_dir, csv_files[0])

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = reader.__next__()

            status = row.get("status", "unknown")
            ts_us = row.get("timestamp_us", "0")
            exit_code = row.get("exit_code", "-1")

            self._log(f"[TS] 读取文件: {csv_files[0]}")
            self._log(f"[TS] Status: {status}")
            self._log(f"[TS] Timestamp (us): {ts_us}")
            self._log(f"[TS] Exit code: {exit_code}")
        except Exception as exc:
            self._log(f"[ERR] 读取时间戳文件失败: {exc}")

    def _start_workflow(self):
        try:
            main_exec = self.main_exec_var.get().strip()
            main_cwd = self.main_cwd_var.get().strip()
            start_delay = float(self.start_delay_var.get().strip())
            tianmou_save = self.save_path_var.get().strip()

            if not os.path.isfile(main_exec):
                raise FileNotFoundError(f"主程序不存在: {main_exec}")
            if not os.path.isdir(main_cwd):
                raise FileNotFoundError(f"工作目录不存在: {main_cwd}")

            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self._update_status("启动中...", "orange")

            self._log("[SYS] ===== 开始自动化流程 =====")
            self._log(f"[SYS] 保存路径: {tianmou_save}")

            # 0. 预防性清理：工作流开始前先拉低 GPIO（处理上次异常退出的残留状态）
            self._log("[SYS] 0/5 预防性拉低 GPIO...")
            self._run_gpio("off")
            time.sleep(0.5)

            # 0.1 预防性清理：删除可能残留的 .rosbag 索引文件
            bag_base_dir = os.path.dirname(tianmou_save.rstrip('/'))
            if os.path.isdir(bag_base_dir):
                removed_count = 0
                for f in os.listdir(bag_base_dir):
                    if f.endswith(".rosbag"):
                        try:
                            os.remove(os.path.join(bag_base_dir, f))
                            removed_count += 1
                        except Exception:
                            pass
                if removed_count > 0:
                    self._log(f"[SYS] 已清理 {removed_count} 个残留 .rosbag 文件")
                else:
                    self._log("[SYS] 无需清理 .rosbag 文件")
            else:
                self._log("[SYS] 数据目录尚不存在，跳过 .rosbag 清理")

            # 1. 启动天眸相机 (run天眸)
            self._log("[SYS] 1/5 启动天眸相机 (run天眸)...")
            self.ctrl.start_camera()
            self._log("[SYS] 等待天眸相机启动 (7秒)...")
            time.sleep(7.0)

            # 2. 启动主程序 (main.C)
            self._log("[SYS] 2/5 启动主程序 (main.C)...")
            self.main_proc = self._start_subprocess([main_exec], cwd=main_cwd, tag="main")
            # 🔥🔥🔥【改进点 A】建议在这里增加一个长延迟
            self._log("[SYS] 等待 RealSense 硬件标定与流稳定 (增加到3.5秒)...")
            time.sleep(3.5) # RealSense 开启 Auto-Exposure 和初始化需要时间，给够它 3 秒

            # 3. 设置保存路径并开始录制
            self._log("[SYS] 3/5 设置保存路径并开始录制...")
            self.ctrl.set_save_addr(tianmou_save)
            time.sleep(0.5)
            self.ctrl.start_record()
            time.sleep(2.0)

            # 4. 拉高 GPIO (Pin 15)
            self._log("[SYS] 4/5 拉高 GPIO (Pin 15)...")
            self._run_gpio("on")
            # time.sleep(1.0)
            self.workflow_running = True
            self._update_status("运行中", "blue")
            self._log("[SYS] ===== 自动化流程已启动 =====")

        except Exception as exc:
            self._log(f"[ERR] 启动失败: {exc}")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._update_status("启动失败", "red")

    def _stop_workflow(self):
        self._log("[SYS] ===== 开始停止自动化流程 =====")
        self._update_status("停止中...", "orange")

        # 1. 终止主程序
        self._log("[SYS] 1/5 终止主程序...")
        if self.main_proc:
            self._terminate_proc(self.main_proc, "main")
            self.main_proc = None
        time.sleep(1.0)

        # 2. 停止天眸录制
        self._log("[SYS] 2/5 停止天眸录制...")
        self.ctrl.stop_record()
        time.sleep(1.0)

        # 3. 关闭天眸相机 (Stop天眸)
        self._log("[SYS] 3/5 关闭天眸相机 (Stop天眸)...")
        self.ctrl.stop_camera()
        time.sleep(1.0)

        # 4. 读取时间戳
        self._log("[SYS] 4/5 读取时间戳...")
        self._read_and_save_timestamp()
        time.sleep(1.0)

        # 5. 拉低 GPIO
        self._log("[SYS] 5/5 拉低 GPIO (Pin 15)...")
        self._run_gpio("off")

        self.workflow_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._update_status("已停止", "black")
        self._log("[SYS] ===== 自动化流程已停止 =====")

    def _terminate_proc(self, proc, tag: str, force: bool = False):
        try:
            sig = signal.SIGTERM if force else signal.SIGINT
            os.killpg(os.getpgid(proc.pid), sig)
            self._log(f"[SYS] 已发送 {sig.name} 到 {tag}")
            proc.wait(timeout=8)
            self._log(f"[SYS] {tag} 已退出 (code={proc.returncode})")
        except subprocess.TimeoutExpired:
            self._log(f"[WARN] {tag} 未响应，准备强制终止...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception as exc:
                self._log(f"[ERR] SIGKILL 失败: {exc}")
        except Exception as exc:
            self._log(f"[ERR] 终止 {tag} 失败: {exc}")

    def _read_and_save_timestamp(self):
        ts_exec = self.ts_exec_var.get().strip()
        ts_dir = self.ts_save_dir_var.get().strip()
        if not os.path.isdir(ts_dir):
            os.makedirs(ts_dir, exist_ok=True)

        if not ts_exec or not os.path.isfile(ts_exec):
            self._log(f"[WARN] 时间戳程序不存在: {ts_exec}")
            return

        self._log(f"[SYS] 读取时间戳...")
        try:
            result = subprocess.run(["sudo", ts_exec], capture_output=True, text=True, timeout=10)
            output = (result.stdout or "") + (result.stderr or "")
            for line in output.splitlines():
                self._log(f"[TS] {line}")

            match = re.search(r"Timestamp in microseconds:\s*(\d+)", output)
            ts_us = int(match.group(1)) if match else 0
            if ts_us > 0:
                readable_time = datetime.fromtimestamp(ts_us / 1_000_000).strftime("%Y-%m-%d %H:%M:%S")
                self._log(f"[TS] 可读时间: {readable_time}")

            out_name = datetime.now().strftime("tianmou_timestamp_%Y%m%d_%H%M%S.csv")
            out_path = os.path.join(ts_dir, out_name)

            def write_csv(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["status", "timestamp_us", "exit_code"])
                    writer.writerow(["ok" if ts_us > 0 else "failed", ts_us, result.returncode])

            try:
                write_csv(out_path)
                self._log(f"[SYS] 时间戳已保存: {out_path}")
            except OSError as e:
                if e.errno == 28:
                    self._log(f"[ERR] 磁盘空间不足，无法保存到 {ts_dir}")
                    self._log(f"[TS] 已读取的时间戳(请手动记录): {ts_us} us")
                    fallback = os.path.join("/tmp", out_name)
                    try:
                        write_csv(fallback)
                        self._log(f"[SYS] 已临时保存到: {fallback}")
                    except Exception as e2:
                        self._log(f"[ERR] 临时保存也失败: {e2}")
                else:
                    raise
        except Exception as exc:
            self._log(f"[ERR] 时间戳读取/保存失败: {exc}")


if __name__ == "__main__":
    app = TianmouAutoApp()
    app.mainloop()
