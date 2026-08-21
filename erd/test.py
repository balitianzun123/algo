# async def start_eeg_detection(case_no: str, sampling_rate: int, type: int, bus_type: int, age: int, gender: int):
#     global _is_running, _device_connection_status, _all_electrode_status, _current_cue_label, _calib_result
#     if _is_running:
#         logger.info("电极状态解析任务已在运行")
#         return
#
#     # 从上下文取出队列和停止事件
#     raw_q = TASK_CONTEXT.get("raw_q")
#     stop_evt = TASK_CONTEXT.get("stop_event")
#     if not raw_q or not stop_evt:
#         logger.info("❌ 未初始化任务队列，请先启动采集任务")
#         return
#
#     init_electrode_status()
#     _device_connection_status.update({
#         "is_connected": True,
#         "connection_error": "",
#         "stream_interrupted": False,
#         "last_connect_time": time.strftime("%Y-%m-%d %H:%M:%S")
#     })
#     _is_running = True
#     logger.info("✅ 电极状态解析任务启动，读取 TASK_CONTEXT 内的 raw_q")
#     loop = asyncio.get_running_loop()
#
#     try:
#         # 校准(type=0)：清空累积缓冲，起始标签用传入的 bus_type
#         # 在线(type=1)：清在线会话状态
#         if type == 0:
#             reset_calibration()
#             _current_cue_label = int(bus_type)
#         else:
#             reset_session()
#
#         # 双停止条件：外部标记 + 线程停止事件
#         while _is_running and not stop_evt.is_set():
#             try:
#                 filtered_samples = await loop.run_in_executor(
#                     None,
#                     lambda: raw_q.get(timeout=0.1)
#                 )
#             except Empty:
#                 current_time = time.strftime("%H:%M:%S")
#                 _device_connection_status["stream_interrupted"] = True
#                 with STATUS_LOCK:
#                     for label in STANDARD_MAPPING.keys():
#                         _all_electrode_status[label]["status"] = STATUS_STREAM_INTERRUPTED
#                         _all_electrode_status[label]["timestamp"] = current_time
#                 print(f"[{current_time}] ⚠ raw_q 无数据，数据流中断")
#                 continue
#
#             _device_connection_status["stream_interrupted"] = False
#             current_time = time.strftime("%H:%M:%S")
#
#             # ============ 算法：校准只累积，在线才逐帧识别 ============
#             action_code = 0
#             if type == 0:
#                 # 校准：把本小段 + “当前提示标签”逐点存起来，先不建模
#                 _calib_chunks.append(np.asarray(filtered_samples, dtype=np.float64))
#                 _calib_labels.append(np.full(len(filtered_samples), _current_cue_label, dtype=np.int64))
#             else:
#                 # 在线：每 chunk 识别一次，bus_type 传 None
#                 code, success, err = sim_demo(case_no, age, gender, sampling_rate,
#                                               1, None, filtered_samples, STANDARD_MAPPING)
#                 if success and code == 1:
#                     action_code = code
#                 elif not success:
#                     logger.info(f"[在线识别] 失败：{err}")
#
#             # ============ 逐帧原始 24bit ADC 电极状态（前端显示，两种模式都更新）============
#             with STATUS_LOCK:
#                 for frame_data in filtered_samples:
#                     for label in STANDARD_MAPPING.keys():
#                         raw_adc = frame_data[STANDARD_MAPPING[label]]   # ← 用真实通道号，不是顺序下标
#
#                         if abs(raw_adc) > SATURATION_RAW_THRESHOLD:
#                             stat = STATUS_DETECT_ERROR
#                         else:
#                             stat = _get_electrode_status(raw_adc)
#
#                         _all_electrode_status[label] = {
#                             "electrode": label,
#                             "ch_num": STANDARD_MAPPING[label],
#                             "status": stat,
#                             "recognition_result": action_code,
#                             "raw_adc": round(raw_adc, 2),
#                             "uv_value": 0.0,
#                             "timestamp": current_time,
#                             "ts": time.time()
#                         }
#
#         # ============ 循环结束（录完）：校准在这里“一次性建模” ============
#         if type == 0:
#             if not _calib_chunks:
#                 _calib_result = (0, False, "没有累积到任何校准数据")
#                 logger.info("[校准] ❌ 没有累积到任何数据")
#             else:
#                 filtered_all = np.concatenate(_calib_chunks, axis=0)     # (总采样, 全通道)
#                 bus_all      = np.concatenate(_calib_labels, axis=0)     # (总采样,) 逐点对齐
#                 uniq = {int(k): int(v) for k, v in zip(*np.unique(bus_all, return_counts=True))}
#                 logger.info(f"[校准] 总采样={len(bus_all)} 各类点数={uniq}")
#                 code, success, err = sim_demo(case_no, age, gender, sampling_rate,
#                                               0, bus_all, filtered_all, STANDARD_MAPPING)
#                 _calib_result = (code, success, err)
#                 logger.info(f"[校准建模] code={code} success={success} err={err!r}")
#             reset_calibration()
#
#     except Exception as e:
#         err_time = time.strftime("%Y-%m-%d %H:%M:%S")
#         err_msg = str(e)
#         _device_connection_status.update({
#             "is_connected": False,
#             "connection_error": err_msg,
#             "last_disconnect_time": err_time
#         })
#         current_time = time.strftime("%H:%M:%S")
#         with STATUS_LOCK:
#             for label in STANDARD_MAPPING.keys():
#                 _all_electrode_status[label]["status"] = STATUS_DETECT_ERROR
#                 _all_electrode_status[label]["timestamp"] = current_time
#         logger.info(f"\n❌ 电极解析异常：{err_msg}")
#     finally:
#         _is_running = False
#         stop_time = time.strftime("%Y-%m-%d %H:%M:%S")
#         _device_connection_status.update({
#             "is_connected": False,
#             "stream_interrupted": True,
#             "last_disconnect_time": stop_time,
#             "connection_error": "任务停止"
#         })
#         logger.info("===== 电极状态解析任务安全退出 =====")