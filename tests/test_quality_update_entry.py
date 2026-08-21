from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from algo.erd import algorithm_bridge as bridge_module
from algo.erd.algorithm_bridge import EEGAlgorithmBridge, LIVE_CHANNEL_NAMES
from algo.erd.algorithm_core import constants as core_constants
from algo.erd.algorithm_core import trace as eeg_trace
from algo.erd.enter import DEVICE_STANDARD_MAPPING


SAMPLE_RATE = 200


def _runtime_config() -> dict[str, object]:
    return {
        "required_rest_trials": 10,
        "prediction_step_seconds": 1.0,
        "live_quality_window_seconds": 0.02,
        "live_quality_step_seconds": 0.02,
        "live_quality_filter_warmup_seconds": 0.0,
        "prediction_smoothing_enabled": True,
        "prediction_smoothing_history_size": 3,
        "prediction_smoothing_min_votes": 2,
        "prediction_smoothing_action_margin": 0.0,
        "prediction_smoothing_hold_seconds": 0.0,
    }


def _adc_frame_from_uv(values_uv: np.ndarray) -> np.ndarray:
    return bridge_module.RAW_ADC_OFFSET + np.asarray(values_uv, dtype=np.float64) / bridge_module.RAW_ADC_SCALE


def _signal_uv(sample_index: int) -> np.ndarray:
    values = np.zeros(bridge_module.DEVICE_CHANNEL_COUNT, dtype=np.float64)
    t = float(sample_index + 1) / float(SAMPLE_RATE)
    for offset, protocol_index in enumerate(DEVICE_STANDARD_MAPPING.values()):
        values[protocol_index] = 8.0 * np.sin(2.0 * np.pi * 10.0 * t + 0.17 * offset)
    return values


def test_quality_update_entry_starts_without_dc_baseline(monkeypatch):
    impedance_calls = []

    def fake_impedance(data, fs, channel_names):
        impedance_calls.append(
            {
                "data": np.asarray(data),
                "fs": float(fs),
                "channel_names": tuple(channel_names),
            }
        )
        return {
            "channel_eval": [
                {"name": name, "channel": index, "status": "good", "status_code": 0}
                for index, name in enumerate(channel_names)
            ]
        }

    monkeypatch.setattr(
        bridge_module,
        "sim_demo",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quality-only entry must not call sim_demo")),
    )
    monkeypatch.setattr(
        bridge_module,
        "_load_impedance_module",
        lambda: SimpleNamespace(
            eeg_signal_impedance=fake_impedance,
            eeg_signal_quality=fake_impedance,
        ),
    )

    bridge = EEGAlgorithmBridge(
        case_no="quality_only",
        sample_rate=SAMPLE_RATE,
        runtime_config=_runtime_config(),
    )

    assert not hasattr(bridge, "baseline_values")
    first = bridge.update_live_signal_quality(_adc_frame_from_uv(_signal_uv(0)))
    assert first["status"] in {"collecting", "ready"}
    assert first["ready"] is False
    assert "baseline_ready" not in first

    result = first
    for sample_index in range(1, bridge.live_quality_sample_count + 2):
        result = bridge.update_live_signal_quality(_adc_frame_from_uv(_signal_uv(sample_index)))

    assert result["status"] == "ready"
    assert result["ready"] is True
    assert len(impedance_calls) >= 1
    assert impedance_calls[0]["data"].shape == (len(LIVE_CHANNEL_NAMES), bridge.live_quality_sample_count)
    assert impedance_calls[0]["channel_names"] == tuple(LIVE_CHANNEL_NAMES)
    assert bridge._sample_buffer == []


def test_feed_adc_trace_has_no_dc_baseline_and_sim_demo_receives_raw_uv(caplog, monkeypatch):
    monkeypatch.setattr(core_constants, "EEG_TRACE_ENABLED", True)
    monkeypatch.setattr(core_constants, "EEG_TRACE_DETAIL", False)
    caplog.set_level(logging.INFO, logger=eeg_trace.__name__)
    captured_blocks = []

    def fake_sim_demo(*args, **kwargs):
        block = np.asarray(args[6], dtype=np.float64)
        captured_blocks.append(block)
        return {"success": True, "status": "collecting", "code": 0}

    monkeypatch.setattr(bridge_module, "sim_demo", fake_sim_demo)

    bridge = EEGAlgorithmBridge(
        case_no="adc_no_dc",
        sample_rate=SAMPLE_RATE,
        runtime_config=_runtime_config(),
    )
    bridge._trace_adc_stride_samples = 1

    first_uv = np.linspace(-12.0, 15.0, bridge_module.DEVICE_CHANNEL_COUNT)
    for sample_index in range(SAMPLE_RATE):
        bridge.feed(_adc_frame_from_uv(first_uv + sample_index * 0.001))

    assert len(captured_blocks) == 1
    first_block_uv = (captured_blocks[0][0] - bridge_module.RAW_ADC_OFFSET) * bridge_module.RAW_ADC_SCALE
    np.testing.assert_allclose(first_block_uv, first_uv, rtol=0.0, atol=1e-8)

    trace_text = "\n".join(record.message for record in caplog.records)
    assert "[EEG_TRACE][ADC]" in trace_text
    assert "dc_baseline" not in trace_text
    assert "[EEG_TRACE][DC_BASELINE]" not in trace_text
    assert "uV -12.00~+15.00" in trace_text


def test_prediction_smoothing_is_deprecated_noop_and_display_code_matches_code():
    bridge = EEGAlgorithmBridge(
        case_no="direct_codes",
        sample_rate=SAMPLE_RATE,
        runtime_config=_runtime_config(),
    )
    bridge._prediction_result_enabled = True

    outputs = []
    for code in (1, 0, 1):
        label = "action" if code == 1 else "rest"
        result = {
            "success": True,
            "status": "predicted",
            "classification_available": True,
            "code": code,
            "label": label,
            "stage1_result": label,
            "final_result": label,
            "action_score": 20.0 if code == 1 else 19.0,
            "action_threshold": 20.0,
        }
        smoothed = bridge._apply_prediction_smoothing(result)
        displayed = bridge._apply_prediction_result_latch(smoothed)
        outputs.append(displayed)

    assert [item["code"] for item in outputs] == [1, 0, 1]
    assert [item["realtime_code"] for item in outputs] == [1, 0, 1]
    assert [item["display_code"] for item in outputs] == [1, 0, 1]
    assert all("prediction_smoothing" not in item for item in outputs)
    assert all(item["success_latched"] is False for item in outputs)


def test_strategy_quality_update_entry_forwards_channels_without_query_signature_change(monkeypatch):
    pytest.importorskip("dao.bci_experiment_dao", reason="backend DAO package is not available in this test environment")
    from algo.strategy import data_model, model_context

    class DummyBridge:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def update_live_signal_quality(self, channels_16):
            self.calls.append(list(channels_16))
            return {"status": "updated", "channels": []}

        def get_live_signal_status(self):
            return {"status": "queried", "channels": []}

        def feed(self, channels_16):
            raise AssertionError("quality update must not call feed")

        def start_calibration(self):
            return {"success": True}

        def start_prediction(self):
            return {"success": True}

        def start_using_prediction_result(self):
            return {"success": True}

        def get_session_bandpower_topomap_image(self):
            return {"image_path": None}

    monkeypatch.setattr(data_model, "EEGAlgorithmBridge", DummyBridge)

    strategy = data_model.DataModelStrategy()
    assert strategy.update_channels_status([1] * bridge_module.DEVICE_CHANNEL_COUNT) == {
        "status": "updated",
        "channels": [],
    }
    assert strategy.get_channels_status() == {"status": "queried", "channels": []}
    assert strategy.algorithm.calls == [[1] * bridge_module.DEVICE_CHANNEL_COUNT]

    context = model_context.ModelStrategyContext()
    assert context.update_channels_status(0, [2] * bridge_module.DEVICE_CHANNEL_COUNT) == {
        "status": "updated",
        "channels": [],
    }
    assert context.get_channels_status(0) == {"status": "queried", "channels": []}
    assert context._strategies[0].algorithm.calls == [[2] * bridge_module.DEVICE_CHANNEL_COUNT]
