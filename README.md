# Introduction to GPU gospel

Programming a GPU kernel is like programming a website by specifying each pixel individually. Doing it by pixels makes sense because in the end, the website is rendered into pixels.

However, this approach is highly inefficient, so a compiler acts as an intermediary. It takes your manually specified pixels and converts them into functionally equivalent HTML. Later, the HTML is still rendered into pixels.

You care a lot about performance and performance depends on the generated HTML. You program pixels and there is no obvious way to see how they relate to the HTML. Your code certainly does not contain any HTML.

Technically, there are ways to inspect the generated HTML, but they are complicated. Additionally, for some reason, some pixels are more efficient than others. Sometimes it is faster to draw multiple pixels of the same color, sometimes it is not. Sometimes it is faster to draw pixels in strange orders and you do not really know why.

So, welcome to the **GPU Gospel**, a list of rules, concepts, and commandments for programming GPU kernels. Most of these principles are universal, but some numbers and details are specific to the NVIDIA A100-PCIE-40GB.

# The gospel

1. A GPU consists of SMs and memory.
1. A GPU does 2 things. SMs compute, memory stores and serves data. Both can happen simultaneously.
1. Always have computations to perform, always have data to transfer, but avoid unnecessary computations and transfers.
1. Understand how many computations (peak FLOPS) your GPU can perform per 1 byte of transfered data (peak memory bandwidth). This is called the ops/byte ratio.
1. Understand how many computations per 1 byte of transfered data your algorithm requires. This is called arithmetic intensity.
1. GPU computation is distributed over a grid, which consists of multiple thread blocks. Each thread block consists of multiple threads. 32 threads together form a warp.
1. A kernel is a function executed by all threads in the grid.
1. An SM is the hardware that executes computations.
1. Each thread block is assigned to one SM.
1. An SM takes a thread block and, at each step, selects a few warps to execute.
1. Threads in the same warp
   - Execute the same function at the same time but on different data (SIMD model).
   - Acualy on NVIDIA has SIMT, there threads execute in lockstep. If they diverge (contidionals) execution is sequential. 
   - Can efficiently communicate using registers and collaborate via primitives like `__shfl_down_sync`.
1. Threads in the same block
   - Can be synchronized with `__syncthreads`.
   - Share the SM’s resources: registers, L1 memory, and execution cycles.
1. There are multiple levels of memory.
   - Global (DRAM): Large but slow memory (typically 1–80GB). ~300 cycles to access.
   - L2 cache (SMEM): Fast, smaller memory shared across all threads (typically 4MB–96MB). ~30-100 cycles to access.
   - L1 cache: Very fast memory directly on SM. Configurable into shared memory and cache. ~30 cycles to access.
   - Registers: Extremely fast memory located directly in the SM. 1-cycle access time. Typically 65k registers per SM.  
1. Each memory type has a preferred access pattern.
1. Global memory is accessed in transactions of 32B, 64B, or 128B. 128B is 1 float per thread in a warp: 4B*32 = 128B.
   - When threads in a warp access shared memory, their needs are handled with the minimal set of transactions (coalesced).
   - If all threads fit into a continuous 128B block, a single transaction is used (efficient).
   - If threads access data more than 128B apart, 32 memory transactions are required (inefficient).
1. Shared memory is divided into 32 banks. Each consecutive 4B chunk belongs to a different bank.
   - One bank can serve 4B to one thread at a time.
   - If multiple threads access different addresses in the same bank, memory access is serialized, increasing latency.
   - If multiple threads access the same address within a bank, access is broadcasted (efficient).
1. To improve coalescing, thread can access multiple elements at the same time, and load 4floats = 16B = 128b.

# Languag
- **SM**: streaming multiprocessor
- **DRAM**: dynamic random access memory
- **Kernel**: function that is to be executed by every thread
- **SIMD**: Single Instruction/Multiple Data

# What was this?
 cheat sheet of fundamental "axioms" for GPU programming. This does not explain much—it simply lists key statements that the author found important and (hopefully) true.

This was created while I was learning the basics of kernel programming. It is naive, incomplete, and possibly incorrect. Use at your own risk.

# Resources
- https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
- https://maharshi.bearblog.dev/optimizing-sgemv-cuda/
- https://siboehm.com/articles/22/CUDA-MMM

# Contributions
Feel free to raise issues/PRs if there is something you would like to add. Below are some open questions that should be addressed in future versions of the Gospel.

# Opened question
- How does vectorized memory access relate to bank conflicts?
- How much slower is broadcasting compared to standard memory access?
- Double-check memory access speeds.
- When exactly does vectorization help?
- Double check ops/flops difference. 
