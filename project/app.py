"""洪涝风险预测系统的本地分析与训练界面。"""
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
DEVICE_OPTIONS = {
    "自动选择（推荐）": "auto",
    "仅使用 CPU": "cpu",
    "使用 NVIDIA 显卡（CUDA）": "cuda",
}
TRAIN_OPTIONS = {
    "积水区域分割（U-Net）": "seg",
    "积水趋势预测（LSTM）": "lstm",
    "风险等级融合（MLP）": "fusion",
}


class FloodWarningApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("洪涝风险预测与公安应急处置系统")
        self.geometry("960x730")
        self.minsize(820, 600)
        self.process = None
        self.active_task = None
        self.log_queue = queue.Queue()

        self.video_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="unet")
        self.rain_var = tk.StringVar(value="25")
        self.output_var = tk.StringVar(value=str(DEFAULT_RESULT_DIR / "output.mp4"))
        self.event_var = tk.StringVar(value=str(DEFAULT_RESULT_DIR / "risk_events.csv"))
        self.train_type_var = tk.StringVar(value="积水区域分割（U-Net）")
        self.device_var = tk.StringVar(value="自动选择（推荐）")
        self.image_dir_var = tk.StringVar(value=str(ROOT / "data" / "floodnet" / "images"))
        self.mask_dir_var = tk.StringVar(value=str(ROOT / "data" / "floodnet" / "masks"))
        self.series_csv_var = tk.StringVar(value=str(ROOT / "data" / "series" / "ratio.csv"))
        self.fusion_csv_var = tk.StringVar(value=str(ROOT / "data" / "series" / "fusion_features.csv"))
        self.epochs_var = tk.StringVar(value="30")
        self.batch_size_var = tk.StringVar(value="8")
        self.lr_var = tk.StringVar(value="0.0001")
        self.val_split_var = tk.StringVar(value="0.2")
        self.status_var = tk.StringVar(value="请选择分析视频，或在“模型训练”页配置训练任务。")
        self.cuda_message = self._cuda_message()
        self._build_ui()
        self.after(100, self._drain_log_queue)

    @staticmethod
    def _cuda_message():
        try:
            import torch
            if torch.cuda.is_available():
                return f"已检测到 CUDA：{torch.cuda.get_device_name(0)}"
            return "当前环境未启用 CUDA；选择 CUDA 会在训练前提示。"
        except Exception:
            return "无法检测 CUDA；请确认 PyTorch 已正确安装。"

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        container = ttk.Frame(self, padding=18)
        container.pack(fill="both", expand=True)
        ttk.Label(container, text="洪涝风险预测与公安应急处置", style="Title.TLabel").pack(anchor="w")
        ttk.Label(container, text="视频积水识别 · 趋势预测 · 四级预警 · 本地模型训练", style="Hint.TLabel").pack(anchor="w", pady=(3, 12))

        notebook = ttk.Notebook(container)
        notebook.pack(fill="x")
        analysis_tab = ttk.Frame(notebook, padding=12)
        training_tab = ttk.Frame(notebook, padding=12)
        notebook.add(analysis_tab, text="视频风险分析")
        notebook.add(training_tab, text="模型训练")
        self._build_analysis_tab(analysis_tab)
        self._build_training_tab(training_tab)

        actions = ttk.Frame(container)
        actions.pack(fill="x", pady=12)
        self.run_button = ttk.Button(actions, text="开始风险分析", command=self._start_analysis)
        self.run_button.pack(side="left")
        self.train_button = ttk.Button(actions, text="开始模型训练", command=self._start_training)
        self.train_button.pack(side="left", padx=8)
        self.open_video_button = ttk.Button(actions, text="打开结果视频", command=lambda: self._open_file(self.output_var.get()), state="disabled")
        self.open_video_button.pack(side="left", padx=8)
        self.open_event_button = ttk.Button(actions, text="打开事件记录", command=lambda: self._open_file(self.event_var.get()), state="disabled")
        self.open_event_button.pack(side="left")

        ttk.Label(container, textvariable=self.status_var, style="Hint.TLabel").pack(anchor="w")
        log_frame = ttk.LabelFrame(container, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=14, wrap="word", state="disabled", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_analysis_tab(self, parent):
        form = ttk.LabelFrame(parent, text="分析参数", padding=12)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self._path_row(form, 0, "监控视频", self.video_var, self._choose_video)
        self._path_row(form, 1, "输出视频", self.output_var, self._choose_output)
        self._path_row(form, 2, "事件记录", self.event_var, self._choose_event)
        ttk.Label(form, text="分割方式").grid(row=3, column=0, sticky="w", pady=7)
        ttk.Combobox(form, textvariable=self.backend_var, values=("unet", "hsv"), state="readonly", width=18).grid(row=3, column=1, sticky="w", pady=7)
        ttk.Label(form, text="unet 使用训练权重；hsv 是传统兜底方法", style="Hint.TLabel").grid(row=3, column=2, sticky="w", padx=10)
        ttk.Label(form, text="当前降雨 (mm/h)").grid(row=4, column=0, sticky="w", pady=7)
        ttk.Entry(form, textvariable=self.rain_var, width=18).grid(row=4, column=1, sticky="w", pady=7)
        ttk.Label(form, text="例如：10 小雨、25 中雨、50 大雨", style="Hint.TLabel").grid(row=4, column=2, sticky="w", padx=10)

    def _build_training_tab(self, parent):
        form = ttk.LabelFrame(parent, text="训练参数", padding=12)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="训练任务").grid(row=0, column=0, sticky="w", pady=7)
        ttk.Combobox(form, textvariable=self.train_type_var, values=tuple(TRAIN_OPTIONS), state="readonly", width=28).grid(row=0, column=1, sticky="w", pady=7)
        ttk.Label(form, text="选择任务后按对应数据路径训练", style="Hint.TLabel").grid(row=0, column=2, sticky="w", padx=10)
        ttk.Label(form, text="训练设备").grid(row=1, column=0, sticky="w", pady=7)
        ttk.Combobox(form, textvariable=self.device_var, values=tuple(DEVICE_OPTIONS), state="readonly", width=28).grid(row=1, column=1, sticky="w", pady=7)
        ttk.Label(form, text=self.cuda_message, style="Hint.TLabel", wraplength=350).grid(row=1, column=2, sticky="w", padx=10)
        self._path_row(form, 2, "分割图像目录", self.image_dir_var, lambda: self._choose_dir(self.image_dir_var))
        self._path_row(form, 3, "分割掩码目录", self.mask_dir_var, lambda: self._choose_dir(self.mask_dir_var))
        self._path_row(form, 4, "LSTM 时序 CSV", self.series_csv_var, lambda: self._choose_csv(self.series_csv_var))
        self._path_row(form, 5, "融合特征 CSV", self.fusion_csv_var, lambda: self._choose_csv(self.fusion_csv_var))
        ttk.Label(form, text="训练轮数").grid(row=6, column=0, sticky="w", pady=7)
        ttk.Entry(form, textvariable=self.epochs_var, width=12).grid(row=6, column=1, sticky="w", pady=7)
        ttk.Label(form, text="首次测试建议设置为 1–3 轮", style="Hint.TLabel").grid(row=6, column=2, sticky="w", padx=10)
        ttk.Label(form, text="批大小").grid(row=7, column=0, sticky="w", pady=7)
        ttk.Entry(form, textvariable=self.batch_size_var, width=12).grid(row=7, column=1, sticky="w", pady=7)
        ttk.Label(form, text="显存不足时请调小，例如 2 或 4", style="Hint.TLabel").grid(row=7, column=2, sticky="w", padx=10)
        ttk.Label(form, text="学习率 / 验证比例").grid(row=8, column=0, sticky="w", pady=7)
        row = ttk.Frame(form)
        row.grid(row=8, column=1, sticky="w", pady=7)
        ttk.Entry(row, textvariable=self.lr_var, width=12).pack(side="left")
        ttk.Label(row, text="  /  ").pack(side="left")
        ttk.Entry(row, textvariable=self.val_split_var, width=12).pack(side="left")

    @staticmethod
    def _path_row(parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=7)
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=2, sticky="w", padx=10, pady=7)

    def _choose_video(self):
        path = filedialog.askopenfilename(title="选择监控视频", filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if path:
            self.video_var.set(path)

    def _choose_output(self):
        path = filedialog.asksaveasfilename(title="保存分析视频", defaultextension=".mp4", filetypes=[("MP4 视频", "*.mp4")])
        if path:
            self.output_var.set(path)

    def _choose_event(self):
        path = filedialog.asksaveasfilename(title="保存风险事件记录", defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")])
        if path:
            self.event_var.set(path)

    @staticmethod
    def _choose_dir(variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    @staticmethod
    def _choose_csv(variable):
        path = filedialog.askopenfilename(title="选择 CSV 数据文件", filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")])
        if path:
            variable.set(path)

    def _start_analysis(self):
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
        output, events = Path(self.output_var.get()), Path(self.event_var.get())
        output.parent.mkdir(parents=True, exist_ok=True)
        events.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "infer_video.py", "--video", str(video), "--backend", self.backend_var.get(), "--rain", str(rain), "--output", str(output), "--event_log", str(events)]
        self._start_command(command, "analysis", "正在分析视频，请勿关闭窗口…")

    def _start_training(self):
        task = TRAIN_OPTIONS[self.train_type_var.get()]
        try:
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            lr, val_split = float(self.lr_var.get()), float(self.val_split_var.get())
            if epochs < 1 or batch_size < 1 or lr <= 0 or not 0 < val_split < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "请填写有效的轮数、批大小、学习率和 0–1 之间的验证比例。")
            return
        device = DEVICE_OPTIONS[self.device_var.get()]
        if device == "cuda" and "未启用" in self.cuda_message:
            messagebox.showwarning("CUDA 尚不可用", "当前安装的是 CPU 版 PyTorch。请先安装 CUDA 版 PyTorch，或选择 CPU/自动模式。")
            return
        command = [sys.executable, f"train_{task}.py", "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr), "--val_split", str(val_split), "--device", device]
        if task == "seg":
            image_dir, mask_dir = Path(self.image_dir_var.get()), Path(self.mask_dir_var.get())
            if not image_dir.is_dir() or not mask_dir.is_dir():
                messagebox.showerror("数据目录无效", "请选择存在的分割图像目录和掩码目录。")
                return
            command += ["--image_dir", str(image_dir), "--mask_dir", str(mask_dir)]
        else:
            csv_path = Path(self.series_csv_var.get() if task == "lstm" else self.fusion_csv_var.get())
            if not csv_path.is_file():
                messagebox.showerror("数据文件无效", "请选择存在的 CSV 数据文件。")
                return
            command += ["--csv", str(csv_path)]
        self._start_command(command, "training", "正在训练模型，请勿关闭窗口…")

    def _start_command(self, command, task, status):
        if self.process is not None:
            messagebox.showinfo("任务进行中", "当前已有任务在运行，请等待其完成。")
            return
        self._append_log("启动命令：" + " ".join(command) + "\n")
        self.active_task = task
        self.status_var.set(status)
        self.run_button.configure(state="disabled")
        self.train_button.configure(state="disabled")
        self.open_video_button.configure(state="disabled")
        self.open_event_button.configure(state="disabled")
        threading.Thread(target=self._run_command, args=(command,), daemon=True).start()

    def _run_command(self, command):
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
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
                completed_task = self.active_task
                self.process, self.active_task = None, None
                self.run_button.configure(state="normal")
                self.train_button.configure(state="normal")
                if value == 0:
                    self.status_var.set("任务完成。")
                    if completed_task == "analysis":
                        self.open_video_button.configure(state="normal")
                        self.open_event_button.configure(state="normal")
                else:
                    self.status_var.set(f"任务结束，返回代码：{value}。请查看运行日志。")
            else:
                self.process, self.active_task = None, None
                self.run_button.configure(state="normal")
                self.train_button.configure(state="normal")
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
