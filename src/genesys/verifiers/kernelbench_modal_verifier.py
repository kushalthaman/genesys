from typing import Dict
from genesys.schemas import Response
from genesys.verifiers.base_verifier import BaseVerifier
import modal
from genesys.local_kernel_eval import eval_kernel_against_ref, set_gpu_arch

# modal image
image = modal.Image.debian_slim().pip_install(
    "pydantic",
    "torch",
    "h2",
    "grpclib",
    "datasets",
    "certifi",
    "google-cloud-storage",
    "grpcio",
    "rich",
    "huggingface_hub"
).add_local_python_source("genesys")
app = modal.App("genesys_kernelbench_modal_verifier", image=image)

@app.cls()   
class EvalFunc:
    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src: str, custom_cuda: str, verbose: bool, gpu_arch: list) -> dict:
        set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src,
            custom_cuda,
            verbose=verbose,
            measure_performance=True,
            num_correct_trials=5,
            num_perf_trials=100,
        )

class KernelBenchModalVerifier(BaseVerifier):
    """
    A Genesys verifier that uses Modal to evaluate generated GPU kernels in real time.
    
    Expects the Response object to have:
      - llm_response: the LLM-generated kernel source.
      - verification_info: a dict containing 'ref_arch_src' (the reference kernel source).
      - metadata: may include GPU settings, e.g., "gpu" (default "L40S") and "gpu_arch" (default ["Ada"]).
    
    Calls a Modal sandbox that compiles, executes, and tests the kernel, returning a verification dict.
    Currently score is set to 1.0 if the kernel both compiles and produces correct results; otherwise 0.0. 
    We can add functionality for incorporating average runtime/detailed runtime stats, or design a reward function
    as needed. 
    """
    max_parallel = 1
    timeout = 300   # sec

    def verify(self, result: Response) -> Dict:
        candidate_kernel = result.llm_response
        ref_arch_src = result.verification_info.get("ref_arch_src")
        if not ref_arch_src:
            raise ValueError("Missing 'ref_arch_src' in verification_info for KernelBenchModalVerifier.")
        
        verbose = result.metadata.get("verbose", False)
        gpu_raw = result.metadata.get("gpu", "NVIDIA L40S")
        gpu = gpu_raw.replace("NVIDIA ", "")
        gpu_arch = result.metadata.get("gpu_arch", ["Ada"])
        
        with app.run():
            kernel_exec_result = EvalFunc.with_options(gpu=gpu)().eval_single_sample_modal.remote(
                ref_arch_src, candidate_kernel, verbose, gpu_arch
            )
        
        score = 1.0 if kernel_exec_result.get("compiled") and kernel_exec_result.get("correct") else 0.0
        
        return {
            "score": score,
            "verification_result_info": kernel_exec_result,
        }

### sample testing block 
if __name__ == "__main__":
    import json
    from datasets import load_dataset

    try:
        dataset = load_dataset("ScalingIntelligence/kernelbench-samples", split="test", streaming=True)
        sample = next(iter(dataset))
    except Exception as e:
        print("Error loading dataset:", e)
        exit(1)
    
    candidate_kernel = sample.get("kernel") or sample.get("kernel_code") or sample.get("result")
    if not candidate_kernel:
        raise ValueError("No kernel code found in sample. Available keys: " + str(list(sample.keys())))
    
    hardware = sample.get("hardware", "NVIDIA L40S") 
    # we can replace this with anything from https://modal.com/docs/reference/modal.gpu 
    gpu = hardware.replace("NVIDIA ", "")
    
    FakeResponse = type("FakeResponse", (), {})()
    FakeResponse.llm_response = candidate_kernel
    FakeResponse.verification_info = {"ref_arch_src": candidate_kernel}
    FakeResponse.metadata = {
        "gpu": gpu,
        "gpu_arch": ["Ada"],
        "verbose": True,
    }
    
    verifier = KernelBenchModalVerifier()
    result = verifier.verify(FakeResponse)

    print("Final Evaluation Result:")
    print(json.dumps(result, indent=2))
