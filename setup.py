import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# FORCE_BUILD: Force a fresh build locally
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("INFLLM_V2_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("INFLLM_V2_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("INFLLM_V2_FORCE_CXX11_ABI", "FALSE") == "TRUE"

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )

def append_nvcc_threads(nvcc_extra_args):
    # Increase thread count based on available CPU cores
    import multiprocessing
    num_threads = min(multiprocessing.cpu_count(), 8)  # Use up to 8 threads
    return nvcc_extra_args + ["--threads", str(num_threads)]

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

cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_cuda_home_none("infllm_v2")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "InfLLM V2 is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    
    # 支持多种常见的GPU架构
    # 70: V100, 75: T4/RTX, 80: A100, 86: RTX 3090/RTX A6000, 89: RTX 4090, 90: H100
    supported_archs = ["80", "86"]
    for arch in supported_archs:
        cc_flag.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    
    # Flash Attention CUDA源文件列表 - 只编译 hdim128, bf16 版本
    flash_attn_sources = [
                # "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim96_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim96_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim160_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim160_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim192_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim192_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim256_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim96_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim96_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim160_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim160_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim192_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim192_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim256_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_bwd_hdim256_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
                "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_causal_sm80.cu",
                # "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
    ]
    
    # 过滤掉不存在的文件
    existing_flash_attn_sources = []
    for source in flash_attn_sources:
        if os.path.exists(source):
            existing_flash_attn_sources.append(source)
    
    ext_modules.append(
        CUDAExtension(
            name="infllm_v2.C",
            sources=[
                "csrc/entry.cu",
                "csrc/flash_attn/flash_api.cpp",
            ] + existing_flash_attn_sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        "-DFLASHATTENTION_DISABLE_DROPOUT",
                        "-DFLASHATTENTION_DISABLE_ALIBI",
                        "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        "-DFLASHATTENTION_DISABLE_LOCAL",
                    ]
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "flash_attn",
                Path(this_dir) / "csrc" / "flash_attn" / "src", 
                Path(this_dir) / "csrc" / "cutlass" / "include",
                # Path(this_dir) / "3rd" / "cutlass" / "include",
            ],
        )
    )

setup(
    name='infllm_v2',
    version='0.0.0',
    author_email="acha131441373@gmail.com",
    description="infllm_v2 cuda implementation with flash attention and cutlass",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension} if ext_modules else {},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "psutil",
    ],
) 
