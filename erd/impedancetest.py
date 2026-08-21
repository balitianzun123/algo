
import impedance

import impedance
import numpy as np

# 构造 4通道，每通道512个采样点 的模拟EEG数据 (n_channels, n_samples)
eeg_data = np.random.randn(16, 512)
# 采样率250Hz
fs = 250

# 调用阻抗&通道评估接口
res = impedance.eeg_signal_impedance(data=eeg_data, fs=fs)

# 打印结果
print(f"result = {res}")
# 遍历每个通道结果
for ch_info in res["channel_eval"]:
    print(f"通道{ch_info['channel']}({ch_info['name']}) 状态：{ch_info['status']} 编码：{ch_info['status_code']}")

# import impedance
# # 查看阻抗计算函数详情
# print("==== eeg_signal_impedance 参数详情 ====")
help(impedance.eeg_signal_impedance)
#
# # 顺便查看信号质量函数
# print("\n==== eeg_signal_quality 参数详情 ====")
# help(impedance.eeg_signal_quality)