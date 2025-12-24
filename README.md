# High-Performance GEMM Engine (C++ / AVX2 / OpenMP)

A high-performance General Matrix Multiplication (GEMM) kernel engineered from scratch in C++. This project demonstrates the evolution of a compute-intensive kernel from a naive $O(N^3)$ implementation to a hardware-saturated, multithreaded engine achieving **283x speedup**.

Achieved **~89 GFLOPS** on an AMD Zen 3 (Ryzen 5 5625U), reaching **~41% of the hardware's Theoretical Peak Performance** by exploiting memory hierarchy, SIMD vectorization, and register-level blocking.

##  Performance Benchmarks (N=1024)

| Version | Description | Time (s) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **v1. Naive** | Standard Triple Loop | 6.82 s | 0.31 | **1.0x** |
| **v2. Reordered** | Memory-Aware (i-k-j) | 0.1152 s | 18.63 | **~9x** |
| **v3. OpenMP** | Multithreaded + Tiling | 0.0276 s | 77.83 | **~75x** |
| **v4. Ultimate** | **AVX2 + Register Blocking** | **0.024 s** | **89.38** | **283x** |

<img width="1600" height="1018" alt="ryzen result" src="https://github.com/user-attachments/assets/fb7aa631-ea1b-41a8-8909-f8e509ab545b" />

##  The Optimization Journey of C = A . B

This project is structured as a series of incremental optimizations, identifying specific hardware bottlenecks at each stage and resolving them using Systems Engineering principles.

### Phase 1: The Memory Wall (Spatial Locality)
* **Bottleneck:** The Naive implementation (`i-j-k` loop order) accesses Matrix B in column-major order. Since C++ stores arrays in row-major order, this caused a **Cache Miss** on almost every access, fetching entire cache lines just to use a single double.
* **Solution:** Reordered loops to `i-k-j`. This accesses Matrix B sequentially (Stride-1), allowing the CPU's hardware prefetcher to load data into L1 cache efficiently.
* **Result:** **59x Speedup** purely from memory access pattern changes.

### Phase 2: The Parallelism Wall (Multithreading)
* **Bottleneck:** A single core cannot saturate the memory bandwidth or compute potential of modern CPUs.
* **Solution:** Implemented **OpenMP** threading with `collapse(2)` to parallelize the outer loops.
* **Engineering Challenge:** Identified a race condition in the initial `i-k-j` parallelization where multiple threads fought for the same `C[i][j]` accumulator. Resolved this by parallelizing the *output* coordinates (`i` and `j`), giving each thread a disjoint block of the Result Matrix to compute.
* **Result:** **~247x Speedup** (Scaling linearly with core count).

### Phase 3: The Arithmetic Intensity Wall (The "Ultimate" Kernel)
* **Bottleneck:** Even with cache blocking (`BS=32/40`), the CPU spent more cycles loading data from L1 Cache into Registers than actually performing math. This is the classic "Von Neumann Bottleneck" at the L1 level.
* **Solution:** **Register Blocking (K-Unrolling)** with AVX2 Intrinsics.
    * **SIMD:** Used `_mm256_fmadd_pd` (Fused Multiply-Add) to process 4 doubles per instruction.
    * **Register Blocking:** Unrolled the K-loop by 4. Instead of `Load C -> Math -> Store C`, the kernel loads a vector of C into a **YMM Register** and keeps it there while accumulating results from 4 different K-steps.
    * **Impact:** Reduced L1 Cache load/store traffic by **75%**, increasing Arithmetic Intensity and ensuring the FPUs (Floating Point Units) are fed constantly.
* **Result:** **283x Speedup**, achieving 89.38 GFLOPS.

##  Technical Implementation Details

### Cache Blocking & Alignment
The kernel uses a Block Size (`BS`) of 40.
* **Why 40?** A cache line is 64 bytes (8 doubles). A block width of 40 doubles consumes exactly 5 cache lines ($40/8 = 5$), ensuring clean memory alignment and minimizing "split loads" across cache boundaries.
* **L2 Residence:** The blocking strategy ensures the working set fits comfortably within the 1.25MB L2 cache of the Tiger Lake architecture.

### AVX2 Intrinsics
The core computational kernel allows the CPU to perform 16 floating-point operations per cycle per core (assuming FMA throughput).

```cpp
// Example of the Inner Micro-Kernel
// We hold 'vec_C' in a register to minimize memory access
for (; r < k_limit - 3; r += 4) {
    // Broadcast A values
    __m256d vec_A0 = _mm256_set1_pd(A_row[r]);
    // ...
    
    // Perform Fused Multiply-Add on 4 streams at once
    vec_C = _mm256_fmadd_pd(vec_A0, vec_B0, vec_C);
    // ...
}
```

##  Build & Run
### Requirements:

* GCC or Clang with OpenMP support.

* x86-64 CPU with AVX2 support (Haswell or newer).

### Compilation:


```
# -O3: Maximum optimization
# -march=native: Enable AVX2/FMA instructions specific to your CPU
# -fopenmp: Enable multithreading
g++ Source.cpp -o gemm_engine -O3 -march=native -fopenmp
```
### Execution:
```
./gemm_engine
```
###  Roofline Analysis
## Hardware: Ryzen 5 5625U (6 cores, 12 threads).

## Theoretical Peak: ~220.8 GFLOPS (Base Clock) / ~412.8 GFLOPS (Turbo).

## Achieved: 89.38 GFLOPS.

## Efficiency: The kernel achieves ~41% of the theoretical hardware limit. The remaining gap is attributed to the thermal throttling characteristic of mobile SKUs running dense AVX2 workloads and the lack of Assembly-level prefetching.

### Author
Aditya Kumar

### References

Based on principles from "Computer Systems: A Programmer's Perspective" (Bryant & O'Hallaron).
README.md formatted by Gemini.
