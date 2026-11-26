import asyncio
import sys
from types import SimpleNamespace

from vcb.config import STTConfig
from vcb.stt import parakeet_stt


def test_load_logs_before_parakeet_warmup(monkeypatch, caplog):
    class FakeModel:
        def recognize(self, *_args, **_kwargs):
            assert any("Preparing speech recognition on cuda:1" in record.message
                       for record in caplog.records)

    model = FakeModel()
    fake_asr = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: model,
    )
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        set_default_logger_severity=lambda _level: None,
    )
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(parakeet_stt, "register_cuda_dlls", lambda: None)
    caplog.set_level("INFO", logger="vcb.stt")

    engine = parakeet_stt.ParakeetSTT(STTConfig(device="cuda", device_index=1))
    asyncio.run(engine.load())

    assert engine._model is model
    assert any("parakeet-tdt-0.6b-v3 loaded on cuda:1" in record.message
               for record in caplog.records)
