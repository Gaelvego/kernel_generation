"""
Benchmark prompts for Triton kernel generation evaluation.
We focus on three canonical benchmarks that exercise memory, compute, and reductions.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Benchmark:
    name: str
    user_prompt: str
    description: str


BENCHMARKS: List[Benchmark] = [
    Benchmark(
        name="vector_add",
        description="Element-wise addition of two vectors. Tests basic memory ops and arithmetic.",
        user_prompt=(
            "Write a Triton kernel that performs element-wise addition of two 1-D tensors (vectors). "
            "The kernel should load blocks of data from two input pointers, add them, and store the result "
            "to an output pointer. Use a BLOCK_SIZE compile-time constant. Handle boundary conditions where "
            "the vector length is not a multiple of BLOCK_SIZE."
        ),
    ),
    Benchmark(
        name="fused_softmax",
        description="Row-wise online softmax. Tests reductions, exponentials, and broadcasting.",
        user_prompt=(
            "Write a Triton kernel that computes fused softmax over the rows of a 2-D matrix. "
            "For each row, compute the max value, subtract it from every element, exponentiate, sum the "
            "exponentials, and divide each element by the sum. Use an online algorithm to keep numerical "
            "stability. Use BLOCK_SIZE for the row dimension and handle partial blocks at the end of each row."
        ),
    ),
    Benchmark(
        name="matmul",
        description="Tiled matrix multiplication. Tests shared memory, dot product, and loop tiling.",
        user_prompt=(
            "Write a Triton kernel that performs matrix multiplication C = A @ B. Use a blocked (tiled) algorithm "
            "with BLOCK_SIZE_M, BLOCK_SIZE_N, and BLOCK_SIZE_K as compile-time constants. Load tiles of A and B "
            "into shared memory, compute the dot product using tl.dot, and accumulate into an output tile. "
            "Handle matrices whose dimensions are not perfect multiples of the block sizes."
        ),
    ),
]
