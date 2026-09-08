"""本机 Web 控制台：视频分析、模型训练与运行日志。"""
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
job = {"process": None, "kind": None, "logs": [], "returncode": None}
job_lock = threading.Lock()


def cuda_info():
    try:
        import torch
        available = torch.cuda.is_available()
        return {
            "available": available,
            "device": torch.cuda.get_device_name(0) if available else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    except Exception as exc:
        return {"available": False, "device": None, "torch": None, "cuda": None, "error": str(exc)}


def append_log(line):
    with job_lock:
        job["logs"].append(line.rstrip())
        job["logs"] = job["logs"][-400:]


def run_command(command, kind):
    try:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        with job_lock:
            job.update(process=process, kind=kind, returncode=None)
        for line in process.stdout:
            append_log(line)
        code = process.wait()
        append_log(f"任务结束，返回代码：{code}")
        with job_lock:
            job["returncode"] = code
            job["process"] = None
    except Exception as exc:
        append_log(f"启动失败：{exc}")
        with job_lock:
            job["returncode"] = -1
            job["process"] = None


def start(command, kind):
    with job_lock:
        if job["process"] is not None:
            return False, "已有任务正在运行。"
        job.update(logs=["启动命令：" + " ".join(command)], returncode=None, kind=kind)
        # 预占位置，防止浏览器连续点击启动多个任务。
        job["process"] = "starting"
    threading.Thread(target=run_command, args=(command, kind), daemon=True).start()
    return True, "任务已启动。"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with job_lock:
        running = job["process"] is not None
        payload = {"running": running, "kind": job["kind"], "logs": job["logs"], "returncode": job["returncode"]}
    payload["cuda"] = cuda_info()
    return jsonify(payload)


@app.post("/api/analyze")
def analyze():
    data = request.get_json(force=True)
    video = Path(data.get("video", ""))
    if not video.is_file():
        return jsonify(error="请选择存在的本地视频路径。"), 400
    try:
        rain = float(data.get("rain", 25))
        if rain < 0:
            raise ValueError
    except ValueError:
        return jsonify(error="降雨强度必须为非负数字。"), 400
    output = Path(data.get("output") or ROOT / "result" / "output.mp4")
    event_log = Path(data.get("event_log") or ROOT / "result" / "risk_events.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    event_log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "infer_video.py", "--video", str(video), "--backend", data.get("backend", "unet"), "--rain", str(rain), "--output", str(output), "--event_log", str(event_log)]
    ok, message = start(command, "视频风险分析")
    return jsonify(message=message), 202 if ok else 409


@app.post("/api/train")
def train():
    data = request.get_json(force=True)
    task = data.get("task")
    if task not in {"seg", "lstm", "fusion"}:
        return jsonify(error="未知训练任务。"), 400
    device = data.get("device", "auto")
    if device == "cuda" and not cuda_info()["available"]:
        return jsonify(error="当前 PyTorch 未启用 CUDA，请选择 CPU/自动或完成 CUDA 版安装。"), 400
    try:
        epochs, batch = int(data.get("epochs", 30)), int(data.get("batch_size", 8))
        lr, val_split = float(data.get("lr", 0.0001)), float(data.get("val_split", 0.2))
        if epochs < 1 or batch < 1 or lr <= 0 or not 0 < val_split < 1:
            raise ValueError
    except ValueError:
        return jsonify(error="训练参数无效。"), 400
    command = [sys.executable, f"train_{task}.py", "--epochs", str(epochs), "--batch_size", str(batch), "--lr", str(lr), "--val_split", str(val_split), "--device", device]
    if task == "seg":
        images, masks = Path(data.get("image_dir", "")), Path(data.get("mask_dir", ""))
        if not images.is_dir() or not masks.is_dir():
            return jsonify(error="请选择存在的图像目录和掩码目录。"), 400
        command += ["--image_dir", str(images), "--mask_dir", str(masks)]
    else:
        csv_path = Path(data.get("csv", ""))
        if not csv_path.is_file():
            return jsonify(error="请选择存在的 CSV 文件。"), 400
        command += ["--csv", str(csv_path)]
    ok, message = start(command, "模型训练")
    return jsonify(message=message), 202 if ok else 409


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
