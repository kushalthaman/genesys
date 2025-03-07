### for sample testing kernel bench/checking setup works locally

import pytest
from genesys.verifiers.kernelbench_modal_verifier import KernelBenchModalVerifier, EvalFunc, app

class FakeResponse:
    def __init__(self, llm_response, ref_arch_src, metadata=None):
        self.llm_response = llm_response
        self.verification_info = {"ref_arch_src": ref_arch_src}
        self.metadata = metadata or {"verbose": False, "gpu": "L40S", "gpu_arch": ["Ada"]}

class DummyContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyEvalFuncSuccess:
    class DummyMethod:
        def remote(self, *args, **kwargs):
            return {
                "compiled": True,
                "correct": True,
                "avg_runtime": 7.0,
                "runtime_stats": {},
                "error": ""
            }
    @property
    def eval_single_sample_modal(self):
        return self.DummyMethod()

class DummyEvalFuncFailure:
    class DummyMethod:
        def remote(self, *args, **kwargs):
            return {
                "compiled": False,
                "correct": False,
                "avg_runtime": None,
                "runtime_stats": {},
                "error": "compilation error"
            }
    @property
    def eval_single_sample_modal(self):
        return self.DummyMethod()

def dummy_run():
    return DummyContext()

def create_fake_response():
    return FakeResponse(
        llm_response="candidate kernel code",
        ref_arch_src="reference kernel code",
        metadata={"verbose": False, "gpu": "L40S", "gpu_arch": ["Ada"]}
    )

def test_kernelbench_modal_verifier_success(monkeypatch):
    monkeypatch.setattr(app, "run", lambda: dummy_run())
    monkeypatch.setattr(EvalFunc, "with_options", lambda **kwargs: lambda: DummyEvalFuncSuccess())
    
    verifier = KernelBenchModalVerifier()
    fake_response = create_fake_response()
    result = verifier.verify(fake_response)
    
    assert result["score"] == 1.0
    assert result["verification_result_info"]["compiled"] is True
    assert result["verification_result_info"]["correct"] is True

def test_kernelbench_modal_verifier_failure(monkeypatch):
    monkeypatch.setattr(app, "run", lambda: dummy_run())
    monkeypatch.setattr(EvalFunc, "with_options", lambda **kwargs: lambda: DummyEvalFuncFailure())
    
    verifier = KernelBenchModalVerifier()
    fake_response = create_fake_response()
    result = verifier.verify(fake_response)
    
    assert result["score"] == 0.0
    assert result["verification_result_info"]["compiled"] is False
    assert result["verification_result_info"]["correct"] is False
