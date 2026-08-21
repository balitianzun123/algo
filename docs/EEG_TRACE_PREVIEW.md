# EEG Trace Preview

> SAMPLE FORMAT / 演示数据。以下数值只展示 Trace 阅读格式，不是真实 EEG 采集结果。

MAIN Trace 现在只保留四类：`SESSION`、`ADC`、`CALIBRATION`、`WINDOW#n`。Power、Baseline、ERD、Weight、ROI Score、Threshold、Margin 不再分散在多个块里，而是集中到一张 Prediction Window 表。

## SESSION

一次 calibration 或 prediction session 的基本配置只打印一次；sample rate、mapping、window、step 改变，或者 session reset 后才会重新打印。

```text
[EEG_TRACE][SESSION]
Case        : P001
Mode        : prediction
Sample Rate : 2000.0 Hz
Input       : 16 channels
Algorithm   : FC3 FC4 C3 C4 CP3 CPz CP4 Cz
Action ROI  : FC3 FC4 C3 C4 CP3 CP4
Window      : 4.0 s
Step        : 1.0 s
Status      : READY
```

## ADC

逐帧入口日志已压缩成一行，并按 `sample_rate` 节流，默认约每 1 秒输出一次。

```text
[EEG_TRACE][ADC]
16ch | ADC 8388590~8388630 -> uV -0.40~+0.49 | offset=8388608 | scale=0.0223517 | status=OK
```

`EEG_TRACE_DETAIL=True` 时才会额外出现完整 `raw_counts=[...]` 和 `samples_uv=[...]`。

## CAL Progress

Rest calibration 收集阶段只保留每个 trial 的一行结果。

```text
[EEG_TRACE][CAL#01]
ACCEPT | quality=6/6 | bad=none | accepted=1/10

[EEG_TRACE][CAL#02]
REJECT | quality=5/6 | bad=CP3 | accepted=1/10
```

## CAL Summary

Calibration 完成后输出一张最终汇总。

```text
[EEG_TRACE][CALIBRATION]
==================================================
REST CALIBRATION COMPLETE
==================================================
Accepted : 10
Rejected : 2
Attempts : 12

LOO Action Scores
R01=8.12
R02=10.34
R03=7.91
R04=11.20
R05=9.75
R06=12.80
R07=6.88
R08=10.02
R09=13.45
R10=8.66

T_all = 12.48

Single-channel Thresholds
T_no_FC3 = 12.10
T_no_FC4 = 12.75
T_no_C3  = 11.96
T_no_C4  = 12.42
T_no_CP3 = 12.20
T_no_CP4 = 12.60

Personal Baseline
FC3=40.50
C3 =43.70
CP3=44.00

FC4=40.90
C4 =42.20
CP4=44.10
==================================================
```

## Prediction Window

一个真实 prediction window 只输出一张窗口表。Power、Baseline、ERD、权重贡献、ROI Score、Threshold 和 Margin 都集中在这里。

```text
[EEG_TRACE][WINDOW#18]
============================================================
TRIAL #5 | TRIAL WINDOW #3 | GLOBAL WINDOW #18 | 17.0s -> 21.0s
============================================================
Data       : 8ch x 8000 | 2000.0 Hz | 4.0 s
Preprocess : 50.0Hz notch + 8.0~30.0Hz bandpass | OK
RMS        : 16.80~20.10 uV -> 6.50~8.20 uV
Quality    : 6/6 | bad=none
Mode       : all_channels

Channel   Power      Baseline        ERD     W   Contribution
----------------------------------------------------------------
C3           28.40        43.70     -35.01%   1.0         -35.01
CP3          35.10        44.00     -20.23%   0.9         -18.21
FC3          32.60        40.50     -19.51%   0.8         -15.61

C4           39.80        42.20      -5.69%   1.0          -5.69
CP4          42.50        44.10      -3.63%   0.9          -3.27
FC4          41.20        40.90      +0.73%   0.8          +0.58
----------------------------------------------------------------

Left ROI  : -(-68.83) / 2.70 = 25.49
Right ROI : -( -8.38) / 2.70 = 3.10

Action Score : max(25.49, 3.10) = 25.49
Threshold    : T_all = 12.80
Margin       : +12.69
Code         : 1
============================================================
```

单坏通道会自然显示 `EXCLUDED` 和动态 denominator，例如 C3 被排除：

```text
Quality    : 5/6 | bad=C3
Mode       : single_channel_excluded

C3       EXCLUDED
CP3          35.10        44.00     -20.23%   0.9         -18.21
FC3          32.60        40.50     -19.51%   0.8         -15.61

Left ROI  : -(-33.82) / 1.70 = 19.89
Threshold : T_no_C3 = 11.96
```

`main ROI` 只解释 action evidence 的来源，不代表左手/右手 MI 分类。
