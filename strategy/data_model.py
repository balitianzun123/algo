# models/data_model.py
import inspect
import json
import logging
import threading
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List, Optional, Dict, Any

from dao.bci_experiment_dao import BciExperimentDao
from db.ApplicationContextBean import Container
from erd.algorithm_bridge import EEGAlgorithmBridge
from erd.algorithm_core.reporting import (
    format_calibration_final_summary_report,
    format_calibration_loo_report,
    format_prediction_success_window_summary,
    format_window_identity_for_report,
)
from schemes.bci_experiment_schema import BciExperimentCreateRequest
from utils.img_util import ImgUtil
from .base import BaseModelStrategy

logger = logging.getLogger(__name__)

class DataModelStrategy(BaseModelStrategy):
    """纯数学模型策略"""
    def __init__(self):
        # 不在这里获取，延迟到使用时
        self.model_name = "DataModel"
        self.version = "1.0.0"

        #当前正在统计的是哪个 Trial
        self.current_prediction_trial_count = 0

        # Bridge 在服务生命周期内只创建一次，持续缓存不会丢失。
        self.algorithm = EEGAlgorithmBridge(
            case_no="patient_001", age=30, gender=1, sample_rate=2000,
            required_rest_trials=10, prediction_step_seconds=1.0,
            profile_dir=r"algo/erd/profiles",
        )
        logger.debug(
            f"algorithm bridge loaded from {inspect.getfile(EEGAlgorithmBridge)}; "
            "dc_offset_baseline=disabled"
        )
        self._lock = threading.Lock()

        # 仅保护JSONL追加写入，
        # 防止并发算法调用造成记录交错
        self._trace_lock = threading.Lock()
        #校准开关
        self.calibration_started = False
        #预测开关
        self.predict_started = False

        self.latest_algorithm_result: Optional[Dict[str, Any]] = None
        # 当前校准阶段已经接受的有效 trial 数，仅用于向后端暴露校准进度
        self._calibration_accepted_trials: int = 0
        # 本轮预测开始的时间。
        self.prediction_started_at: Optional[float] = None
        # T2：第一个4秒预测窗口准备完成的时间。
        self.first_prediction_window_ready_at: Optional[float] = None
        self.first_prediction_window_ready_perf: Optional[float] = None

        # T3：第一次预测计算完成的时间。
        self.first_prediction_result_at: Optional[float] = None

        # T4：第一次 MI 判定成功(code=1)的时间。
        self.imagery_success_at: Optional[float] = None
        self.imagery_success_perf: Optional[float] = None
        # 本轮预测报告是否已经生成，防止每帧重复生成。
        self.prediction_report_generated = False

        # 报告保存目录。
        self.prediction_report_dir = Path(
            r"algo/erd/profiles"
        )
        self.prediction_report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.algorithm_trace_dir = Path(
            r"algo/erd/profiles"
        )

        self.algorithm_trace_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.algorithm_trace_session_id = (datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

        self._calibration_profile_ready: bool = False
        #定义是第几次训练
        self.count = 0


    def calibration(self,channels_16: List[int]) -> None:
        if not self.calibration_started:
            calibration_res = self.start_algorithm_calibration()
            logger.info(f"{self.algorithm.case_no} 进入校准模式: {calibration_res}")

            if calibration_res.get("success"):
                self.calibration_started = True
                self.predict_started = False

        self._call_algorithm(channels_16)
        self.predict_started = False

    def prediction(self, channels_16: List[int]) -> int:
        if not self.predict_started:
            prediction_res = self.start_algorithm_prediction()
            logger.info(f"{self.algorithm.case_no} 进入预测模式: {prediction_res}")

            if prediction_res.get("success"):
                use_res = self.start_using_prediction_result()
                logger.info(f"后端指令开始使用预测结果: {use_res}")

                if use_res.get("success"):
                    self.predict_started = True

                    # T1：正式开始本轮预测。
                    self.prediction_started_at = time.time()

                    # 清空上一轮时间。
                    self.first_prediction_window_ready_at = None
                    self.first_prediction_window_ready_perf = None
                    self.first_prediction_result_at = None
                    self.imagery_success_at = None
                    self.imagery_success_perf = None

                    # 允许本轮重新生成报告。
                    self.prediction_report_generated = False

        if self.predict_started and self.prediction_started_at is None:
            self.prediction_started_at = time.time()

        # ==================================================
        # 后端 count 改变 = 新 Prediction Trial
        # ==================================================
        if ( self.predict_started and self.count != self.current_prediction_trial_count ):
            # 先结算上一 Trial
            if self.current_prediction_trial_count != 0:

                # 上一个 Trial 没出现过成功的 1
                if not self.prediction_report_generated:

                    self._generate_failed_prediction_report(self.current_prediction_trial_count)

            # 正式切换到新 Trial
            self.current_prediction_trial_count = self.count

            # 新 Trial 重新统计
            self.prediction_started_at = time.time()
            self.first_prediction_window_ready_at = None
            self.first_prediction_window_ready_perf = None
            self.first_prediction_result_at = None
            self.imagery_success_at = None
            self.imagery_success_perf = None
            self.prediction_report_generated = False

            # 防止新 Trial 读取上一 Trial 的缓存结果
            self.latest_algorithm_result = None

        self._call_algorithm(channels_16)
        self.calibration_started = False

        code = 0
        result = self.latest_algorithm_result

        if (result is not None and result.get("success")):
            if result.get("status") == "predicted":
                window_ready_epoch = result.get("prediction_window_ready_epoch")
                window_ready_perf = result.get("prediction_window_ready_perf")
                result_epoch = result.get("prediction_result_epoch")

                # T2只记录第一个完整预测窗口。
                if (self.first_prediction_window_ready_at is None and window_ready_epoch is not None):
                    self.first_prediction_window_ready_at = float(window_ready_epoch)

                if (self.first_prediction_window_ready_perf is None and window_ready_perf is not None):
                    self.first_prediction_window_ready_perf = float( window_ready_perf )

                # T3只记录第一次预测计算完成时间。
                if (self.first_prediction_result_at is None and result_epoch is not None):
                    self.first_prediction_result_at = float( result_epoch )

                code = int(result.get("display_code") or 0)

                # T4：第一次稳定识别到想象成功。
                # 当前 count Trial 第一次实际 display_code == 1 时，记录成功。
                if (self.current_prediction_trial_count != 0 and code == 1 and not self.prediction_report_generated ):
                    self.imagery_success_at = time.time()
                    self.imagery_success_perf = time.perf_counter()

                    self._generate_prediction_report(result)

        return code

    def _generate_failed_prediction_report(
            self,
            trial_count: int,
    ) -> None:

        report_path = (self.prediction_report_dir / (f"{self.algorithm.case_no}_"f"{self.algorithm_trace_session_id}_"f"prediction_report.txt"))

        # 如果这是整个报告的第一个 Trial，
        # 而且它一直没成功，此时 TXT 还不存在。
        # 先创建报告头和校准静息功率基线摘要。
        if not report_path.exists():
            profile_data: dict[str, Any] = {}

            result = self.latest_algorithm_result or {}
            profile_path_value = result.get("profile_path")

            if profile_path_value:
                try:
                    profile_data = json.loads(Path(profile_path_value).read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning( f"读取校准功率基线摘要失败：{exc}")

            loo_scores_text = format_calibration_loo_report(profile_data)
            calibration_final_summary_text = format_calibration_final_summary_report(profile_data)

            report_header = (
                "脑电运动想象预测报告\n"
                "====================\n"
                "\n"
                "校准静息功率基线与MI阈值配置\n"
                "------------------------------\n"
                f"{loo_scores_text}"
                "\n"
                f"{calibration_final_summary_text}"
                "\n"
            )

            try:
                report_path.write_text(report_header,encoding="utf-8",)

            except Exception as exc:
                logger.error( f"预测报告文件头保存失败：{exc}")


        def format_time(value):
            if value is None:
                return "未记录"

            return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        result = self.latest_algorithm_result or {}

        failed_content = (
            "================================\n"
            f"Prediction Trial: {trial_count}\n"
            "================================\n"
            f"后端 count：{trial_count}\n"
            "\n"
            f"Trial开始时间：{format_time(self.prediction_started_at)}\n"
            f"首个预测窗口：{format_time(self.first_prediction_window_ready_at)}\n"
            "首次想象成功：未成功\n"
            "\n"
            "响应时间（T4-T2）：未成功\n"
            "\n"
            "首次成功 prediction_index：未成功\n"
            "首次成功动作得分：未成功\n"
            f"动作阈值：{result.get('action_threshold')}\n"
            "\n"
        )

        try:
            with report_path.open("a",
                    encoding="utf-8",) as file:
                file.write(failed_content)

        except Exception as exc:
            logger.error(
                f"未成功 Trial 报告保存失败：{exc}"
            )

    def _generate_prediction_report(self, prediction_result: Dict[str, Any],) -> None:
        """首次稳定识别成功后，生成TXT报告。"""

        window_ready_epoch = prediction_result.get("prediction_window_ready_epoch")
        window_ready_perf = prediction_result.get("prediction_window_ready_perf")
        result_epoch = prediction_result.get("prediction_result_epoch")

        if self.first_prediction_window_ready_at is None and window_ready_epoch is not None:
            self.first_prediction_window_ready_at = float(window_ready_epoch)
        if self.first_prediction_window_ready_perf is None and window_ready_perf is not None:
            self.first_prediction_window_ready_perf = float(window_ready_perf)
        if self.first_prediction_result_at is None and result_epoch is not None:
            self.first_prediction_result_at = float(result_epoch)
        if self.prediction_started_at is None:
            self.prediction_started_at = self.first_prediction_window_ready_at or self.imagery_success_at

        if self.first_prediction_window_ready_perf is None:
            logger.error("报告生成失败：未记录第一个预测窗口准备时间")
            return

        if self.imagery_success_perf is None:
            logger.error("报告生成失败：未记录想象成功时间")
            return

        if self.imagery_success_at is None:
            logger.error("报告生成失败：未记录想象成功的实际时间")
            return

        response_time = round(
            max(
                0.0,
                self.imagery_success_perf
                - self.first_prediction_window_ready_perf,
                ),
            3,
        )

        def format_time(value: Optional[float]) -> str:
            if value is None:
                return "未记录"

            return datetime.fromtimestamp(value).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]

        # timestamp = datetime.fromtimestamp(
        #     self.imagery_success_at
        # ).strftime("%Y%m%d_%H%M%S_%f")

        report_path = (
                self.prediction_report_dir
                / f"{self.algorithm.case_no}_{self.algorithm_trace_session_id}_prediction_report.txt"
        )

        # 读取本次预测实际使用的个人 Profile，
        # 取出校准阶段 N 个有效 Rest Trial 的逐通道功率基线摘要。
        profile_data: dict[str, Any] = {}
        profile_path_value = prediction_result.get("profile_path")
        if profile_path_value:
            try:
                profile_data = json.loads(Path(profile_path_value).read_text( encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"读取校准功率基线摘要失败：{exc}")

        loo_scores_text = format_calibration_loo_report(profile_data)
        calibration_final_summary_text = format_calibration_final_summary_report(profile_data)

        report_header = (
            "脑电运动想象预测报告\n"
            "====================\n"
            "\n"
            "校准静息功率基线与MI阈值配置\n"
            "------------------------------\n"
            f"{loo_scores_text}"
            "\n"
            f"{calibration_final_summary_text}"
            "\n"
        )

        window_identity = format_window_identity_for_report(prediction_result)
        success_window_summary_text = format_prediction_success_window_summary(prediction_result)

        trial_content = (
            "========================================\n"
            f"Prediction Trial #{self.count}\n"
            "========================================\n"
            f"后端 Trial       : {window_identity.get('backend_trial_count', self.count)}\n"
            f"Trial Window     : {window_identity.get('trial_prediction_index', 'N/A')}\n"
            f"Global Window    : {window_identity.get('prediction_index', prediction_result.get('prediction_index'))}\n"
            "\n"
            "结果             : 成功\n"
            f"最终 display_code: {prediction_result.get('display_code')}\n"
            "\n"
            "时间\n"
            "----------------------------------------\n"
            f"T1 Trial开始       : {format_time(self.prediction_started_at)}\n"
            f"T2 首个窗口就绪    : {format_time(self.first_prediction_window_ready_at)}\n"
            f"T3 首次计算完成    : {format_time(self.first_prediction_result_at)}\n"
            f"T4 首次MI判定成功   : {format_time(self.imagery_success_at)}\n"
            f"响应时间 T4-T2     : {response_time:.3f} s\n"
            "\n"
            f"{success_window_summary_text}"
            "\n"
            "========================================\n"
        )

        # 在Service层构建 CreateRequest 对象
        req = BciExperimentCreateRequest(
            subject_code=self.algorithm.case_no,
            sample_rate=self.algorithm.sample_rate,
            t1_enter_predict=format_time(self.prediction_started_at),
            t2_window_ready=format_time(self.first_prediction_window_ready_at),
            t3_calc_finish=format_time(self.first_prediction_result_at),
            t4_imagination_finish=format_time(self.imagery_success_at),
            response_sec=self.round_to_6(response_time),
            predict_result=int(prediction_result.get('display_code') or 0),
            action_score=self.round_to_6(prediction_result.get('action_score')),
            action_threshold=self.round_to_6(prediction_result.get('action_threshold')),
            loo_scores_text=loo_scores_text,
        )
        # 传给DAO
        logger.info(f"== {self.algorithm.case_no}运动想象成功，采样率{self.algorithm.sample_rate}Hz 报告已生成 响应时间={response_time}====")
        if self.algorithm.case_no is not None:
            bci_experiment_dao = Container.get_dao(BciExperimentDao)
            bci_experiment_dao.create(req)

        try:
            # 文件第一次出现，只写一次标题和校准静息功率基线摘要
            if not report_path.exists():
                report_path.write_text(
                    report_header,
                    encoding="utf-8",
                )

            # 每个Trial第一次成功，往后追加一段
            with report_path.open("a", encoding="utf-8") as file:
                file.write(trial_content)
        except Exception as exc:
            logger.error(f"预测报告保存失败：{exc}")
            return

        self.prediction_report_generated = True

        # logger.info(
        #     f"运动想象成功，报告已生成：{report_path}，"
        #     f"响应时间={response_time}秒"
        # )

    def round_to_6(self,v):
        """将数值四舍五入到6位小数"""
        if v is None:
            return None
        return Decimal(str(v)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def get_channels_status(self) -> dict[str, Any]:
        """获取8个通道状态"""
        return self.algorithm.get_live_signal_status()

    def update_channels_status(self, channels_16: List[int]) -> dict[str, Any]:
        """Feed one EEG frame into the live channel-quality path only."""
        try:
            return self.algorithm.update_live_signal_quality(channels_16)
        except Exception as exc:
            logger.error(f"EEG live signal quality update error: {exc}")
            try:
                status = dict(self.algorithm.get_live_signal_status())
            except Exception:
                return {
                    "status": "error",
                    "ready": False,
                    "message": str(exc),
                    "channels": [],
                }
            status["status"] = "error"
            status["ready"] = False
            status["message"] = str(exc)
            return status

    def get_algorithm_topmap(self) -> str:
        result = self.algorithm.get_session_bandpower_topomap_image()
        if result["image_path"] is not None:
            return ImgUtil.image_to_base64(result["image_path"])
        return ""

    def get_calibration_accepted_trials(self) -> int:
        """获取当前校准阶段已经接受的有效 trial 数。"""
        with self._lock:
            return int(self._calibration_accepted_trials)

    def get_calibration_profile_ready(self) -> bool:
        """获取个人校准 Profile 是否已经生成完成。"""
        with self._lock:
            return bool(self._calibration_profile_ready)

    def _call_algorithm(self, channels_16: List[int]) -> None:
        """每收到一帧 16 通道 EEG 数据，就调用算法桥接层处理一次。"""

        try:
            # feed() 是算法唯一的数据入口；内部会处理基线、训练、预测、通道质量和脑地图累计。
            set_backend_trial_count = getattr(
                self.algorithm,
                "set_backend_trial_count",
                None,
            )
            if callable(set_backend_trial_count):
                try:
                    set_backend_trial_count(
                        self.count if self.predict_started else None
                    )
                except Exception as exc:
                    logger.debug("failed to update EEG trace trial count: %s", exc)

            result = self.algorithm.feed(channels_16)

        except Exception as exc:
            logger.error(f"EEG algorithm bridge error: {exc}")
            return
        # None 表示当前还没攒够一个算法处理块，或者算法暂时没有新状态需要返回。
        if result is None:
            return
        # 保存最近一次算法结果，get_latest_data() 会把它返回给前端/后端查询。
        with self._lock:
            self.latest_algorithm_result = result
            # 如果本次算法结果携带 accepted_trials，
            # 就同步保存校准有效 trial 数。
            if "accepted_trials" in result:
                self._calibration_accepted_trials = int(result["accepted_trials"])
        status = result.get("status")

        self._append_algorithm_trace(result)

        if status == "waiting_for_mode_command":
            logger.info(
                "waiting_for_mode_command; 等待后端调用 start_algorithm_calibration() 或 start_algorithm_prediction()"
            )
        elif status == "baseline_skipping":
            logger.info(
                f"baseline_skipping baseline_skip_samples={result.get('baseline_skip_samples')} "
                f"required_baseline_skip_samples={result.get('required_baseline_skip_samples')}"
            )
        elif status == "baseline_calibrating":
            logger.info(
                f"baseline_calibrating baseline_samples={result.get('baseline_samples')} "
                f"required_baseline_samples={result.get('required_baseline_samples')}"
            )
        elif status == "baseline_ready":
            logger.info("baseline_ready; starting algorithm buffering")
        elif status == "calibrating":
            logger.info(
                f"calibrating collected_trials={result.get('collected_trials')} "
                f"required_trials={result.get('required_trials')}"
            )
        elif status == "profile_created":
            self._calibration_profile_ready = True
            logger.info(
                f"profile_created profile_path={result.get('profile_path')}; "
                f"waiting for backend start_prediction command"
            )

        elif status == "profile_ready_waiting_for_prediction":
            logger.debug(
                f"profile_ready_waiting_for_prediction profile_path={result.get('profile_path')}"
            )
        elif status == "collecting":
            logger.info(f"collecting buffer_samples={result.get('buffer_samples')} / required_samples={result.get('required_samples')}")
        elif status == "predicted":
            realtime_code = result.get("realtime_code", result.get("code"))
            realtime_label = result.get("realtime_label", result.get("label"))
            display_code = result.get("display_code", 0)
            display_label = result.get("display_label", "rest")
            result_enabled = result.get("prediction_result_enabled", False)
            success_latched = result.get("success_latched", False)

            logger.debug(
                f"predicted realtime_code={realtime_code} realtime_label={realtime_label} "
                f"display_code={display_code} display_label={display_label} "
                f"result_enabled={result_enabled} success_latched={success_latched} "
                f"action_score={result.get('action_score')} action_threshold={result.get('action_threshold')} "
                f"profile_source={result.get('profile_source')}"
            )
        else:
            logger.warning(f"algorithm status={status} message={result.get('message')}")

    def start_using_prediction_result(self) -> dict:
        """前端点击“开始预测/使用结果”时调用；当前 display_code 直接等于窗口 code。"""
        result = self.algorithm.start_using_prediction_result()
        logger.info(f"开始使用预测结果: {result}")
        with self._lock:
            self.latest_algorithm_result = result
        return result

    def reset_prediction_result_latch(self) -> dict:
        """兼容旧后端调用；当前算法没有成功锁存。"""
        result = self.algorithm.reset_prediction_result_latch()
        logger.info(f"重置预测结果锁存: {result}")
        with self._lock:
            self.latest_algorithm_result = result
        return result

    def start_algorithm_calibration(self) -> dict:
        """后端点击“开始校准/训练”时调用；只切算法状态，后续数据仍由 _call_algorithm() 持续喂入。"""
        result = self.algorithm.start_calibration()
        logger.info(f"开始算法静息校准: {result}")
        with self._lock:
            self.latest_algorithm_result = result
            # 新一轮校准，从 0/10 重新开始
            if result.get("success"):
                self._calibration_accepted_trials = 0
                self._calibration_profile_ready = False
                self.count = 0
                self.current_prediction_trial_count = 0
                # 新一次校准 = 新的一次实验报告
                self.algorithm_trace_session_id = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
        return result

    def start_algorithm_prediction(self) -> dict:
        """后端点击“进入预测模式”时调用；profile 准备好后，后续 feed() 才会真正预测。"""
        result = self.algorithm.start_prediction()
        logger.info(f"进入算法预测模式: {result}")
        with self._lock:
            self.latest_algorithm_result = result
        return result

    def update_sample_rate(self,case_no:str,rate_hz: int) -> None:
        self.algorithm = EEGAlgorithmBridge(
            case_no=case_no + "_" + str(int(time.time())),
            age=30,
            gender=1,
            sample_rate=rate_hz,
            required_rest_trials=10,
            prediction_step_seconds=1.0,
            profile_dir=r"algo/erd/profiles",
        )
        # self.algorithm.sample_rate = rate_hz
        # self.algorithm.case_no = case_no
        self.latest_algorithm_result = None
        logger.info(
            f"算法已按采样率重建：病历号={self.algorithm.case_no}；"
            f"算法已按采样率重建：采样率={self.algorithm.sample_rate}Hz；"
            "DC基线=disabled"
        )
        return None

    def reset_eeg_params(self, count: int) -> None:
        self.count = int(count)
        return None

    @staticmethod
    def _algorithm_trace_json_default(value: Any,) -> Any:
        """把算法结果中的少量非标准JSON类型安全转换。"""

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        # numpy array等
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        # numpy scalar等
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        raise TypeError(
            f"Object of type "
            f"{type(value).__name__} "
            f"is not JSON serializable"
        )

    @staticmethod
    def _algorithm_trace_json_default(
            value: Any,
    ) -> Any:
        """把算法结果中的少量非标准JSON类型安全转换。"""

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        # numpy array等
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        # numpy scalar等
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        raise TypeError(
            f"Object of type "
            f"{type(value).__name__} "
            f"is not JSON serializable"
        )

    def _algorithm_trace_phase(self,result: Dict[str, Any],) -> Optional[str]:
        """判断当前结果是否属于需要持久化的算法诊断记录。"""

        status = str(
            result.get("status") or ""
        )

        if status in {
            "rest_trial_accepted",
            "rest_trial_rejected",
            "profile_created",
        }:
            return "calibration"

        if status == "predicted":
            return "prediction"

        return None

    def _append_algorithm_trace(self, result: Dict[str, Any],) -> None:
        """把完整算法结果追加写入Calibration或Prediction JSONL。"""

        phase = self._algorithm_trace_phase(result)

        if phase is None:
            return

        # case_no可能进入文件名，因此做最基本的安全处理。
        case_no = str(self.algorithm.case_no)

        safe_case_no = "".join(char
            if (char.isalnum() or char in ("-", "_")
            ) else "_" for char in case_no
        )

        if not safe_case_no:
            safe_case_no = "patient"

        trace_path = (
                self.algorithm_trace_dir
                / (
                    f"{safe_case_no}_"
                    f"{self.algorithm_trace_session_id}_"
                    f"{phase}_trace.jsonl"
                )
        )

        # 不修改Bridge返回的原result。
        # 只复制一份用于持久化。
        record = dict(result)

        record["_trace"] = {
            "phase": phase,
            "session_id": (self.algorithm_trace_session_id),
            "recorded_at_epoch": float(time.time()),
            "recorded_at_iso": (datetime.now().astimezone().isoformat(timespec="milliseconds")),
        }

        try:
            serialized_record = json.dumps(record,ensure_ascii=False,default=self._algorithm_trace_json_default,)
            with self._trace_lock:
                with trace_path.open("a",encoding="utf-8",) as file:
                        file.write(serialized_record + "\n"
                    )

        except Exception as exc:
            # 日志保存失败绝对不能影响EEG算法和后端。
            logger.error("algorithm trace save failed: "f"{exc}")
