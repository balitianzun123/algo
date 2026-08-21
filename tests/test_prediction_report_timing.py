from __future__ import annotations

import sys
import threading
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from algo.strategy.data_model import DataModelStrategy


class DummyAlgorithm:
    case_no = "report_timing_case"
    sample_rate = 250

    def feed(self, channels_16):
        return {
            "success": True,
            "status": "predicted",
            "display_code": 1,
            "prediction_window_ready_epoch": 1786000000.0,
            "prediction_window_ready_perf": 10.0,
            "prediction_result_epoch": 1786000001.0,
            "action_score": 29.895784478264726,
            "action_threshold": 16.83143912434964,
        }


def test_prediction_report_backfills_missing_timing_fields():
    report_dir = Path(__file__).resolve().parent
    pattern = "report_timing_case_prediction_report_*.txt"
    for old_file in report_dir.glob(pattern):
        old_file.unlink(missing_ok=True)

    strategy = DataModelStrategy.__new__(DataModelStrategy)
    strategy.algorithm = DummyAlgorithm()
    strategy._lock = threading.Lock()
    strategy.predict_started = True
    strategy.latest_algorithm_result = None
    strategy.prediction_started_at = None
    strategy.first_prediction_window_ready_at = None
    strategy.first_prediction_window_ready_perf = None
    strategy.first_prediction_result_at = None
    strategy.imagery_success_at = None
    strategy.imagery_success_perf = None
    strategy.prediction_report_generated = False
    strategy.prediction_report_dir = report_dir

    try:
        assert strategy.prediction([0] * 16) == 1
        assert strategy.prediction_started_at is not None
        reports = list(report_dir.glob(pattern))
        assert len(reports) == 1
        content = reports[0].read_text(encoding="utf-8")
        assert "未知" not in content
        assert "None 秒" not in content
        assert "响应时间" in content
        assert "动作得分" in content
        assert "动作阈值" in content
    finally:
        for report_file in report_dir.glob(pattern):
            report_file.unlink(missing_ok=True)
