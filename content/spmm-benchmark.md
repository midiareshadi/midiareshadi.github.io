# Why Outer-Product Beats Vendor Sparse Libraries by up to 198x for LLM Attention Sparsity

I benchmarked three sparse SpMSpM algorithms — inner-product,
                    outer-product, and Gustavson's algorithm — against vendor-optimized
                    libraries, namely cuSPARSE on NVIDIA T4 and rocSPARSE on AMD MI300x,
                    using LLM-realistic top-k attention sparsity. The results surprised me:
                    all three Triton implementations outperformed the vendor libraries in
                    every test case, with the outer-product approach achieving up to a
                    **198× speedup over rocSPARSE**.

This post explains what I benchmarked, what I found, and — most
                    importantly — why outer-product wins.

## Background

Modern LLMs exhibit natural sparsity in their attention score matrices.
                    After softmax, most attention weights are near-zero — only a small
                    fraction of token pairs carry significant attention mass. This is the
                    "heavy hitter" phenomenon observed by <a href="https://arxiv.org/abs/2306.14048" target="_blank">H2O</a>, <a href="https://arxiv.org/abs/2404.14469" target="_blank">SnapKV</a>, and <a href="https://arxiv.org/abs/2309.17453" target="_blank">StreamingLLM</a>.
                    Exploiting this sparsity requires efficient sparse matrix multiplication
                    (SpMSpM) kernels.

The <a href="https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf" target="_blank">Zhang et al.</a> analyse inner-product, outer-product,
                    and Gustavson's algorithms for SpMSpM on hardware accelerators. My work extends this to
                    GPU hardware using Triton and adds a direct comparison against
                    vendor-optimised sparse libraries.

## Experimental Setup

### Hardware

| Hardware | Architecture | Memory | Peak BW |
| --- | --- | --- | --- |
| NVIDIA T4 | Turing, Tensor Cores | 16GB GDDR6 | 300 GB/s |
| AMD MI300x | CDNA3, Matrix Cores | 192GB HBM3 | 5,300 GB/s |

### Workloads

6 attention score matrix shapes from LLaMA 1-7B — Q×K<sup>T</sup>
                    and scores×V at sequence lengths 512, 1024, and 2048. Sparsity
                    levels: 70%, 90%, 95% (top-k per row, LLM-realistic).

### Algorithms

| Algorithm | Strategy | Format |
| --- | --- | --- |
| Vendor | cuSPARSE / rocSPARSE | CSR |
| Inner-product | C[i,j] = dot(A[i,:], B[:,j]) | A: CSR, B: dense |
| Outer-product | C += Σ A[:,k] × B[k,:] | dense tiling + tl.dot |
| Gustavson | C[i,:] += A[i,k] * B[k,:] | A: CSR, B: dense |

## Results

### Algorithm comparison

<div class="finding-box">
<p>Outer-product wins <strong>36/36 cases</strong> on both T4 and MI300x.
                    All three Triton algorithms beat vendor on <strong>54/54 cases</strong>.
                    Maximum speedup: <strong>198x over rocSPARSE</strong> on MI300x at 70% sparsity.</p>
</div>

FIGURE_SPARSE_WINNER_HEATMAP

FIGURE_FIG2_SPEEDUP_VS_VENDOR

### Sparsity sensitivity

Higher sparsity amplifies the advantage. At 70% sparsity,
                    outer-product achieves a mean 73x speedup over vendor. At 95%,
                    this narrows to 15x — still very significant.

| Sparsity | T4 mean speedup | MI300x mean speedup |
| --- | --- | --- |
| 70% | 73.5x | 70.2x |
| 90% | 24.2x | 32.1x |
| 95% | 15.3x | 24.9x |

### GPU reverses the CPU algorithm ranking

The GAMMA paper found Gustavson outperforms inner-product on tensor accelerators.
                    On GPU, the two are essentially identical — within 3% of each other
                    on both T4 and MI300x. Outer-product dominates both on GPU.

## Why Outer-Product Wins

The performance gap is explained by memory bandwidth utilization.
                    I measured achieved memory bandwidth for each algorithm:

FIGURE_FIG6_BANDWIDTH_UTILIZATION

<table>
<tr><th>Algorithm</th><th>T4 (300 GB/s peak)</th><th>MI300x (5300 GB/s peak)</th></tr>
<tr><td>outer_product</td><td class="highlight">52.5 GB/s (17.5%)</td><td class="highlight">272 GB/s (5.1%)</td></tr>
<tr><td>inner_product</td><td>16.0 GB/s (5.3%)</td><td>40.6 GB/s (0.8%)</td></tr>
<tr><td>gustavson</td><td>16.6 GB/s (5.5%)</td><td>39.6 GB/s (0.7%)</td></tr>
<tr><td>vendor</td><td>1.4 GB/s (0.5%)</td><td>3.6 GB/s (0.1%)</td></tr>
</table>

The reason is algorithmic. Outer-product processes the computation as:

```
C += sum_k outer(A[:,k], B[k,:])  # rank-1 updates via tl.dot()
```

This maps directly to `tl.dot()` in Triton, which compiles
                    to Tensor Core (NVIDIA) or Matrix Core (AMD) operations. The memory
                    access pattern is regular and coalesced — tiles of A and B are loaded
                    sequentially, accumulated into the output.

Inner-product and Gustavson both require gather operations — loading
                    specific rows of B indexed by sparse column indices of A. This produces
                    scattered memory loads that cannot leverage Tensor Cores efficiently,
                    resulting in 3x lower bandwidth utilization.

Vendor libraries (cuSPARSE/rocSPARSE) use CSR format and process only
                    non-zeros — but cannot utilize Tensor Cores for sparse operations,
                    achieving less than 1% of peak bandwidth.

<div class="finding-box">
<p><strong>Key insight:</strong> For top-k attention sparsity, outer-product's
                    regular tiling pattern enables Tensor Core utilization — achieving 3-7x
                    higher memory bandwidth than gather-based algorithms and 35-140x higher
                    than vendor CSR kernels.</p>
</div>

### MI300x: compute-bound, not memory-bound

On MI300x, outer-product achieves 272 GB/s against a 5,300 GB/s peak
                    — only 5.1% utilization. This doesn't mean the kernel is inefficient.
                    It means the kernel is *compute-bound* (Tensor Core limited),
                    not memory-bound. MI300x's HBM3 bandwidth is so large that it feeds
                    the Tensor Cores faster than they can consume data — the bottleneck
                    shifts from memory to compute. This is why MI300x achieves even larger
                    speedups than T4 for outer-product.

## Dense Dataflow: a Separate Finding

I also benchmarked Weight Stationary (WS), Input Stationary (IS),
                    and Output Stationary (OS) dataflow strategies for dense matmul across
                    13 LLaMA 1-7B shapes.

FIGURE_DENSE_WINNER_HEATMAP

<div class="finding-box">
<p>Dataflow choice is hardware-dependent. IS wins on MI300x for all shapes —
                    HBM3 bandwidth absorbs dataflow differences. WS wins on T4 for 11/13 shapes —
                    bandwidth constraints make weight reuse critical. OS is consistently worst
                    (3.3x penalty on T4, 1.26x on MI300x).</p>
</div>

## Code

All kernels, benchmark scripts, results, and analysis are available at:
                    <a href="https://github.com/midiareshadi/dataflow-bench" target="_blank">github.com/midiareshadi/dataflow-bench</a>

```
git clone https://github.com/midiareshadi/dataflow-bench.git
cd dataflow-bench
pip install torch triton pandas numpy matplotlib

# Dense dataflow benchmark
python3 dense/benchmark_t4.py

# Sparse SpMSpM benchmark
python3 sparse/benchmark_t4.py

# Bandwidth utilization analysis
python3 sparse/analysis/roofline.py
```

## References

- Gamma: Leveraging Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication:
                        <a href="https://people.csail.mit.edu/sanchez/papers/2021.gamma.asplos.pdf" target="_blank">Zhang et al., ASPLOS 2021</a>
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models:
                        <a href="https://arxiv.org/abs/2306.14048" target="_blank">Zhang et al., 2023</a>
- SnapKV: LLM Knows What You are Looking for Before Generation:
                        <a href="https://arxiv.org/abs/2404.14469" target="_blank">Li et al., 2024</a>
- StreamingLLM — Efficient streaming with attention sinks:
                        <a href="https://arxiv.org/abs/2309.17453" target="_blank">Xiao et al., 2023</a>
- Triton:
                        <a href="https://github.com/openai/triton" target="_blank">github.com/openai/triton</a>
