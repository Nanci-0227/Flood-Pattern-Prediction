# Flood Pattern Prediction

课程项目：**基于人工智能/深度学习的洪涝风险预测与公安应急处置研究**。

系统从道路监控视频中分割积水区域，预测积水面积变化趋势，并融合降雨信息生成四级风险预警及处置建议。

## 项目入口

实际代码位于 [`project/`](project/README.md)。进入该目录后可安装依赖并运行训练或推理：

```bash
cd project
pip install -r requirements.txt
python infer_video.py --video data/video/demo.mp4 --backend unet --rain 30
```

## 快捷启动界面

在 Windows 中双击根目录的 `start_app.bat`，即可打开桌面控制台。通过界面选择监控视频、填写降雨强度、选择 `unet` 或 `hsv` 分割方式，再点击“开始风险分析”。分析完成后可直接打开叠加预警的视频和风险事件 CSV。

若首次启动失败，请在 `project/` 目录安装依赖：

```bash
pip install -r requirements.txt
```

## 复现实验要求

- 分割数据应提供道路图像与同名二值积水掩码。
- 时序数据至少包含 `ratio` 列；风险融合数据需包含 `ratio`、`change_rate`、`pred_ratio`、`rain`，`drain_rate` 缺失时按 0 处理。
- 应按摄像头点位、降雨事件或视频片段划分训练/验证/测试集，避免相邻帧泄漏到不同集合。

## 评估输出

- 分割训练输出 Dice、IoU、Precision、Recall。
- LSTM 预测按时间顺序留出未来数据，并输出验证集 MSE、MAE、RMSE。
- 风险融合训练将 CSV 尾部时间段作为验证集，输出 Accuracy、Macro-F1 及各风险等级召回率；橙色、红色等级的漏报应重点分析。

模型权重、原始视频与实验输出默认不提交到仓库；请在实验记录中注明数据来源、划分方式、随机种子和评估指标。
