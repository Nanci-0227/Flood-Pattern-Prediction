"""洪涝风险预警系统的本地桌面启动界面。"""
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = ROOT / "result"


class FloodWarningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("洪涝风险预测与公安应急处置系统")
        self.geometry("880x620")
        self.minsize(760, 540)
        self.process = None
        self.log_queue = queue.Queue()

        self.video_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="unet")
        self.rain_var = tk.StringVar(value="25")
        self.output_var = tk.StringVar(value=str(DEFAULT_RESULT_DIR / "output.mp4"))
        self.event_var = tk.StringVar(value=str(DEFAULT_RESULT_DIR / "risk_events.csv"))
        self.status_var = tk.StringVar(value="请选择监控视频后开始分析。")
        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")

        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="洪涝风险预测与公安应急处置", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text="视频积水识别 · 趋势预测 · 四级预警 · 应急处置建议",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(3, 16))

        form = ttk.LabelFrame(container, text="分析参数", padding=12)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self._path_row(form, 0, "监控视频", self.video_var, self._choose_video)
        self._path_row(form, 1, "输出视频", self.output_var, self._choose_output, save=True)
        self._path_row(form, 2, "事件记录", self.event_var, self._choose_event, save=True)

        ttk.Label(form, text="分割方式").grid(row=3, column=0, sticky="w", pady=7)
        backend_box = ttk.Combobox(form, textvariable=self.backend_var, values=("unet", "hsv"), state="readonly", width=15)
        backend_box.grid(row=3, column=1, sticky="w", pady=7)
        ttk.Label(form, text="unet 优先使用训练权重；hsv 为传统兜底方法", style="Hint.TLabel").grid(row=3, column=2, sticky="w", padx=10)

        ttk.Label(form, text="当前降雨 (mm/h)").grid(row=4, column=0, sticky="w", pady=7)
        ttk.Entry(form, textvariable=self.rain_var, width=18).grid(row=4, column=1, sticky="w", pady=7)
        ttk.Label(form, text="例如：10 小雨、25 中雨、50 大雨", style="Hint.TLabel").grid(row=4, column=2, sticky="w", padx=10)

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=14)
        self.run_button = ttk.Button(actions, text="开始风险分析", command=self._start_analysis)
        self.run_button.pack(side="left")
        self.open_video_button = ttk.Button(actions, text="打开结果视频", command=lambda: self._open_file(self.output_var.get()), state="disabled")
        self.open_video_button.pack(side="left", padx=8)
        self.open_event_button = ttk.Button(actions, text="打开事件记录", command=lambda: self._open_file(self.event_var.get()), state="disabled")
        self.open_event_button.pack(side="left")

        ttk.Label(container, textvariable=self.status_var, style="Hint.TLabel").pack(anchor="w")
        log_frame = ttk.LabelFrame(container, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    @staticmethod
    def _path_row(parent, row, label, variable, command, save=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=7)
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=2, sticky="w", padx=10, pady=7)

    def _choose_video(self):
        path = filedialog.askopenfilename(
            title="选择监控视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")],
        )
        if path:
            self.video_var.set(path)

    def _choose_output(self):
        path = filedialog.asksaveasfilename(
            title="保存分析视频", defaultextension=".mp4", filetypes=[("MP4 视频", "*.mp4")]
        )
        if path:
            self.output_var.set(path)

    def _choose_event(self):
        path = filedialog.asksaveasfilename(
            title="保存风险事件记录", defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")]
        )
        if path:
            self.event_var.set(path)

    def _start_analysis(self):
        if self.process is not None:
            return
        video = Path(self.video_var.get())
        if not video.is_file():
            messagebox.showerror("无法启动", "请选择一个存在的监控视频文件。")
            return
        try:
            rain = float(self.rain_var.get())
            if rain < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "降雨强度必须是大于或等于 0 的数字。")
            return

        output = Path(self.output_var.get())
        events = Path(self.event_var.get())
        output.parent.mkdir(parents=True, exist_ok=True)
        events.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "infer_video.py", "--video", str(video),
            "--backend", self.backend_var.get(), "--rain", str(rain),
            "--output", str(output), "--event_log", str(events),
        ]
        self._append_log("启动分析：" + " ".join(command) + "\n")
        self.status_var.set("正在分析视频，请勿关闭窗口…")
        self.run_button.configure(state="disabled")
        self.open_video_button.configure(state="disabled")
        self.open_event_button.configure(state="disabled")
        thread = threading.Thread(target=self._run_command, args=(command,), daemon=True)
        thread.start()

    def _run_command(self, command):
        try:
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for line in self.process.stdout:
                self.log_queue.put(("log", line))
            self.log_queue.put(("done", self.process.wait()))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            kind, value = self.log_queue.get()
            if kind == "log":
                self._append_log(value)
            elif kind == "done":
                self.process = None
                self.run_button.configure(state="normal")
                if value == 0:
                    self.status_var.set("分析完成。可打开结果视频和风险事件记录。")
                    self.open_video_button.configure(state="normal")
                    self.open_event_button.configure(state="normal")
                else:
                    self.status_var.set(f"分析结束，返回代码：{value}。请查看运行日志。")
            else:
                self.process = None
                self.run_button.configure(state="normal")
                self.status_var.set("启动失败，请查看运行日志。")
                self._append_log(f"错误：{value}\n")
        self.after(100, self._drain_log_queue)

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _open_file(path):
        target = Path(path)
        if not target.exists():
            messagebox.showwarning("文件尚未生成", f"尚未找到：\n{target}")
            return
        os.startfile(str(target))


if __name__ == "__main__":
    FloodWarningApp().mainloop()
