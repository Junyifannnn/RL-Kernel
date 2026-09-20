# SPDX-License-Identifier: Apache-2.0
"""Reject plausible-looking profiles that bypass the configured Attention."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "benchmarks/profile_rocm_rollout_decode.py"
spec = importlib.util.spec_from_file_location("rollout_profile_under_test", path)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize("impl,record,error", [
    ("ROCmAttentionImpl", {"call_count": 36}, "bypassed"),
    ("RlKernelAttentionImpl", None, "no Attention execution"),
    ("RlKernelAttentionImpl", {"call_count": 36, "backend": "native"}, "requested Triton"),
])
def test_profile_rejects_wrong_or_unobserved_attention(monkeypatch, impl, record, error):
    import rl_engine.integrations.state as state
    module = type("Attention", (), {"impl": type(impl, (), {})()})()
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: [module])))
    monkeypatch.setattr(state, "get_active_integration", lambda _: SimpleNamespace(
        readback=lambda: {"operators": {"attention": record}}))
    with pytest.raises(RuntimeError, match=error):
        profile.verify_attention_route(worker, "R/R", "triton")


def test_profile_preserves_execution_evidence(monkeypatch):
    import rl_engine.integrations.state as state
    record = {"call_count": 36, "paged_kernel": "rlkernel.rocm.triton_chunked_flash_attention.v1"}
    module = type("Attention", (), {"impl": type("RlKernelAttentionImpl", (), {})()})()
    worker = SimpleNamespace(rank=2, model_runner=SimpleNamespace(
        get_model=lambda: SimpleNamespace(modules=lambda: [module] * 36)))
    monkeypatch.setattr(state, "get_active_integration", lambda _: SimpleNamespace(
        readback=lambda: {"operators": {"attention": record}}))
    assert profile.verify_attention_route(worker, "R/R", "triton") == {
        "rank": 2, "attention_layers": 36, "attention": record,
    }
