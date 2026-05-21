#!/usr/bin/env python3
import csv
import os
import queue
import re
import shlex
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
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.log = log_fn

    def update_addr(self, ip: str, port: int):
        self.addr = (ip, port)
        self.log(f"[SYS] UDP target updated to {self.addr}")

    def send(self, msg: str):
        try:
            self.sock.sendto(msg.encode("utf-8"), self.addr)
            self.log(f"[UDP] -> {self.addr}: {msg}")
        except Exception as exc:
            self.log(f"[ERR] UDP send failed: {exc}")

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


class SyncWorkflowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Sensor Sync Workflow")
        self.geometry("980x720")
        self.minsize(900, 620)

        self.log_queue = queue.Queue()
        self.main_proc = None
        self.gpio_proc = None
        self.workflow_running = False

        repo_root = "/home/nvidia/Desktop/cxr_multi_sensor"
        self.main_exec_var = tk.StringVar(value=os.path.join(repo_root, "build", "main"))
        self.main_cwd_var = tk.StringVar(value=os.path.join(repo_root, "build"))
        self.ts_exec_var = tk.StringVar(value=os.path.join(repo_root, "test_timestamp", "read_timestamp"))
        self.ts_save_dir_var = tk.StringVar(value=os.path.join(repo_root, "temporal_files"))

        self.udp_ip_var = tk.StringVar(value="10.42.0.1")
        self.udp_port_var = tk.StringVar(value="8889")
        self.tianmou_save_var = tk.StringVar(value="/home/nvidia/UAV1016/1")
        self.start_delay_var = tk.StringVar(value="1.0")

        self.use_gpio_var = tk.BooleanVar(value=True)
        self.gpio_cmd_var = tk.StringVar(value=f"python3 {os.path.join(repo_root, 'gpio_30hz_agx.py')}")

        self.status_var = tk.StringVar(value="Idle")
        self.log_text = None

        self.ctrl = UdpCameraController(self.udp_ip_var.get(), int(self.udp_port_var.get()), self._log)
        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        proc_frame = ttk.LabelFrame(root, text="Process Paths")
        proc_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_labeled_entry(proc_frame, "build/main", self.main_exec_var, 0, browse_file=True)
        self._add_labeled_entry(proc_frame, "main cwd", self.main_cwd_var, 1, browse_dir=True)
        self._add_labeled_entry(proc_frame, "read_timestamp", self.ts_exec_var, 2, browse_file=True)
        self._add_labeled_entry(proc_frame, "timestamp save dir", self.ts_save_dir_var, 3, browse_dir=True)

        udp_frame = ttk.LabelFrame(root, text="Tianmou UDP")
        udp_frame.pack(fill=tk.X, pady=(0, 8))
        self._add_labeled_entry(udp_frame, "UDP IP", self.udp_ip_var, 0)
        self._add_labeled_entry(udp_frame, "UDP Port", self.udp_port_var, 1)
        self._add_labeled_entry(udp_frame, "Tianmou save path", self.tianmou_save_var, 2, browse_dir=True)
        self._add_labeled_entry(udp_frame, "Start delay (sec)", self.start_delay_var, 3)

        gpio_frame = ttk.LabelFrame(root, text="GPIO Trigger")
        gpio_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(gpio_frame, text="Enable GPIO trigger process", variable=self.use_gpio_var).grid(
            row=0, column=0, padx=8, pady=8, sticky="w"
        )
        ttk.Label(gpio_frame, text="GPIO command").grid(row=1, column=0, padx=8, pady=(0, 8), sticky="w")
        ttk.Entry(gpio_frame, textvariable=self.gpio_cmd_var).grid(row=1, column=1, padx=8, pady=(0, 8), sticky="we")
        gpio_frame.grid_columnconfigure(1, weight=1)

        ctrl_frame = ttk.LabelFrame(root, text="Workflow Control")
        ctrl_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(ctrl_frame, text="Apply UDP Address", command=self.on_apply_udp).grid(row=0, column=0, padx=8, pady=8, sticky="we")
        ttk.Button(ctrl_frame, text="Start Workflow", command=self.on_start_workflow).grid(row=0, column=1, padx=8, pady=8, sticky="we")
        ttk.Button(ctrl_frame, text="Stop Workflow", command=self.on_stop_workflow).grid(row=0, column=2, padx=8, pady=8, sticky="we")
        ttk.Button(ctrl_frame, text="Emergency Stop All", command=self.on_emergency_stop).grid(row=0, column=3, padx=8, pady=8, sticky="we")
        for col in range(4):
            ctrl_frame.grid_columnconfigure(col, weight=1)

        log_frame = ttk.LabelFrame(root, text="Logs")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=18, wrap="word", state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        status = ttk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X)

    def _add_labeled_entry(self, parent, label, var, row, browse_file=False, browse_dir=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=8, pady=6, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, padx=8, pady=6, sticky="we")
        if browse_file:
            ttk.Button(parent, text="...", width=4, command=lambda: self._pick_file(var)).grid(
                row=row, column=2, padx=4, pady=6
            )
        if browse_dir:
            ttk.Button(parent, text="...", width=4, command=lambda: self._pick_dir(var)).grid(
                row=row, column=3, padx=4, pady=6
            )
        parent.grid_columnconfigure(1, weight=1)

    def _pick_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _pick_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

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

    def _read_stream(self, proc: subprocess.Popen, tag: str):
        try:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                self._log(f"[{tag}] {line.rstrip()}")
        except Exception as exc:
            self._log(f"[ERR] log reader {tag} failed: {exc}")

    def _parse_udp(self):
        ip = self.udp_ip_var.get().strip()
        port_s = self.udp_port_var.get().strip()
        try:
            port = int(port_s)
            if port <= 0 or port >= 65536:
                raise ValueError
            socket.inet_aton(ip)
            return ip, port
        except Exception:
            raise ValueError("Invalid UDP IP/port")

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

    def on_apply_udp(self):
        try:
            ip, port = self._parse_udp()
            self.ctrl.update_addr(ip, port)
            self.status_var.set(f"UDP target: {ip}:{port}")
        except Exception as exc:
            messagebox.showerror("UDP Error", str(exc))

    def on_start_workflow(self):
        if self.workflow_running:
            messagebox.showwarning("Workflow", "Workflow is already running.")
            return
        threading.Thread(target=self._start_sequence, daemon=True).start()

    def _start_sequence(self):
        try:
            main_exec = self.main_exec_var.get().strip()
            main_cwd = self.main_cwd_var.get().strip()
            ts_exec = self.ts_exec_var.get().strip()
            gpio_cmd = self.gpio_cmd_var.get().strip()
            start_delay = float(self.start_delay_var.get().strip())
            tianmou_save = self.tianmou_save_var.get().strip()

            if not os.path.isfile(main_exec):
                raise FileNotFoundError(f"main not found: {main_exec}")
            if not os.path.isdir(main_cwd):
                raise FileNotFoundError(f"main cwd not found: {main_cwd}")
            if not os.path.isfile(ts_exec):
                self._log(f"[WARN] read_timestamp not found now: {ts_exec}")

            self.status_var.set("Starting build/main...")
            self.main_proc = self._start_subprocess([main_exec], cwd=main_cwd, tag="main")
            self._log("[SYS] build/main started")
            time.sleep(0.8)

            ip, port = self._parse_udp()
            self.ctrl.update_addr(ip, port)
            if tianmou_save:
                self.ctrl.set_save_addr(tianmou_save)
            self.ctrl.start_camera()
            time.sleep(max(start_delay, 0.0))
            self.ctrl.start_record()
            self._log("[SYS] Tianmou start sequence sent")

            if self.use_gpio_var.get() and gpio_cmd:
                gpio_cmd_list = shlex.split(gpio_cmd)
                self.gpio_proc = self._start_subprocess(gpio_cmd_list, cwd=os.path.dirname(gpio_cmd_list[0]) if os.path.isabs(gpio_cmd_list[0]) else None, tag="gpio")
                self._log(f"[SYS] GPIO trigger started: {gpio_cmd}")

            self.workflow_running = True
            self.status_var.set("Workflow running")
        except Exception as exc:
            self._log(f"[ERR] start workflow failed: {exc}")
            self.status_var.set("Start failed")

    def on_stop_workflow(self):
        if not self.workflow_running and self.main_proc is None and self.gpio_proc is None:
            messagebox.showinfo("Workflow", "No running workflow found.")
            return
        threading.Thread(target=self._stop_sequence, daemon=True).start()

    def _stop_sequence(self):
        self.status_var.set("Stopping workflow...")
        try:
            self.ctrl.stop_record()
            time.sleep(0.2)
            self.ctrl.stop_camera()
            self._log("[SYS] Tianmou stop sequence sent")
        except Exception as exc:
            self._log(f"[WARN] stop Tianmou failed: {exc}")

        self._terminate_proc(self.gpio_proc, "gpio")
        self.gpio_proc = None

        self._read_and_save_timestamp()

        self._terminate_proc(self.main_proc, "main")
        self.main_proc = None

        self.workflow_running = False
        self.status_var.set("Workflow stopped")
        self._log("[SYS] Workflow stopped")

    def on_emergency_stop(self):
        self._log("[SYS] Emergency stop triggered")
        try:
            self.ctrl.stop_record()
            self.ctrl.stop_camera()
        except Exception as exc:
            self._log(f"[WARN] emergency UDP stop failed: {exc}")
        self._terminate_proc(self.gpio_proc, "gpio", force=True)
        self._terminate_proc(self.main_proc, "main", force=True)
        self.gpio_proc = None
        self.main_proc = None
        self.workflow_running = False
        self.status_var.set("Emergency stopped")

    def _terminate_proc(self, proc: subprocess.Popen, tag: str, force=False):
        if proc is None:
            return
        if proc.poll() is not None:
            self._log(f"[SYS] {tag} already exited (code={proc.returncode})")
            return
        try:
            sig = signal.SIGTERM if force else signal.SIGINT
            os.killpg(proc.pid, sig)
            self._log(f"[SYS] sent {sig.name} to {tag} process group")
            proc.wait(timeout=8)
            self._log(f"[SYS] {tag} exited (code={proc.returncode})")
        except subprocess.TimeoutExpired:
            self._log(f"[WARN] {tag} did not exit, sending SIGKILL")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception as exc:
                self._log(f"[ERR] SIGKILL {tag} failed: {exc}")
        except Exception as exc:
            self._log(f"[ERR] terminate {tag} failed: {exc}")

    def _read_and_save_timestamp(self):
        ts_exec = self.ts_exec_var.get().strip()
        ts_dir = self.ts_save_dir_var.get().strip()
        if not ts_exec or not os.path.isfile(ts_exec):
            self._log(f"[WARN] read_timestamp executable not found: {ts_exec}")
            return

        os.makedirs(ts_dir, exist_ok=True)
        self._log(f"[SYS] Running timestamp reader: {ts_exec}")

        try:
            result = subprocess.run([ts_exec], capture_output=True, text=True, timeout=10)
            output = (result.stdout or "") + (result.stderr or "")
            for line in output.splitlines():
                self._log(f"[timestamp] {line}")

            match = re.search(r"Timestamp in microseconds:\s*(\d+)", output)
            ts_us = int(match.group(1)) if match else 0

            out_name = datetime.now().strftime("sync_stop_timestamp_%Y%m%d_%H%M%S.csv")
            out_path = os.path.join(ts_dir, out_name)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["status", "timestamp_us", "read_timestamp_exit_code"])
                writer.writerow(["ok" if ts_us > 0 else "parse_failed", ts_us, result.returncode])
            self._log(f"[SYS] Timestamp saved: {out_path}")
        except Exception as exc:
            self._log(f"[ERR] timestamp read/save failed: {exc}")


if __name__ == "__main__":
    app = SyncWorkflowApp()
    app.mainloop()
