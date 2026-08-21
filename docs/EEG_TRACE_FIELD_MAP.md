# EEG Trace Field Map

| 字段 | 来源函数 | 含义 |
|---|---|---|
| Sample Rate | `train_rest10_stream()` / `predict_rest10_stream()` | 当前流式输入采样率 |
| Input channels | `train_rest10_stream()` / `predict_rest10_stream()` | 后端输入矩阵的通道数量 |
| Algorithm channels | `REQUIRED_CHANNELS` | 当前预处理保留的 8 个通道顺序 |
| Action ROI channels | `ACTION_CHANNELS` | 当前 Stage1 action score 使用的 6 个通道 |
| ADC range | `EEGAlgorithmBridge.feed()` | 当前抽样帧的 16 通道 ADC count 范围 |
| uV range | `EEGAlgorithmBridge.feed()` | 同一帧 ADC 换算后的微伏范围 |
| Preprocess | `preprocess_raw_trial()` | 50Hz notch + 8~30Hz bandpass 的真实参数 |
| RMS | `preprocess_raw_trial()` | 原始窗口 RMS 范围到预处理输出 RMS 范围 |
| Quality | `evaluate_prediction_window_quality()` / `evaluate_rest_trial_quality()` | 当前窗口/Rest trial 的质量检测结果摘要 |
| Power | `compute_trial_channel_powers()` | 当前窗口 8~30Hz、0.5~2.5s 的通道功率 |
| Baseline | `profile.channel_baselines` | 当前用户 Rest calibration 得到的个人基线 |
| ERD | `compute_channel_erd()` | `(Power - Baseline) / Baseline * 100%` |
| Weight | `LEFT_ACTION_WEIGHTS` / `RIGHT_ACTION_WEIGHTS` | C=1.0、CP=0.9、FC=0.8 |
| Contribution | `compute_action_score_details()` | `ERD * Weight`，只用于解释当前 action score |
| Left Score | `compute_action_score_details()` | 左 ROI 动态归一化加权 evidence |
| Right Score | `compute_action_score_details()` | 右 ROI 动态归一化加权 evidence |
| Action Score | `compute_action_score_details()` | `max(left_score, right_score)` |
| Threshold | `profile.action_threshold` 或 `profile.action_thresholds_without_channel` | 当前窗口实际使用的 T_all 或 T_no_xxx |
| Margin | `predict_rest10_stream()` | `action_score - action_threshold` |
| Code | `predict_rest10_stream()` | 当前未经过 Bridge display latch 改写前的 prediction code |
| Backend Trial Count | `DataModelStrategy.count` -> `EEGAlgorithmBridge.set_backend_trial_count()` -> `predict_rest10_stream()` | 后端业务 Trial 编号；缺失时 Trace 显示 N/A，不参与算法 |
| Trial Window Index | `predict_rest10_stream()` | 当前 backend Trial 内第几个完整 Prediction Window；backend count 改变时从 1 重新开始 |
| Global Window Index | `Rest10StreamSession.prediction_trial_count` | 原有全局 Prediction Window 序号；保留 `[EEG_TRACE][WINDOW#n]` 搜索标签 |

Trace 主日志保存到：

```text
logs/eeg_trace_observation.log
```

`EEG_TRACE_DETAIL=True` 时，DETAIL 日志会包含完整数组、逐通道 RMS、底层 POWER/ERD/ROI 明细等排障信息。
