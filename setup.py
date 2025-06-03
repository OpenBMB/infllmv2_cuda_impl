import os, glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

# 支持多种常见的GPU架构
# 70: V100, 75: T4/RTX, 80: A100, 86: RTX 3090/RTX A6000, 89: RTX 4090, 90: H100
supported_archs = ["70", "75", "80", "86", "87", "89", "90"]
arch_flags = []
for arch in supported_archs:
    arch_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

setup(
    name='infllm_v2',
    version='0.0.0',
    author_email="acha131441373@gmail.com",
    description="infllm_v2 cuda implementation",
    packages=find_packages(),
    setup_requires=[
        "pybind11",
    ],
    ext_modules=[
        CUDAExtension(
            name='infllm_v2.C',
            sources = [
                "csrc/entry.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    [
                        "-O3", "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ] + arch_flags
                ),
            },
        )
    ],
    cmdclass={
        'build_ext': NinjaBuildExtension
    }
) 
