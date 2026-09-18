# Results

Eval subset: held-out steps [503, 1855) of 1855 (15% of trajectories). Catalog size 6372. Backbone Qwen/Qwen2.5-1.5B-Instruct (float16) on cuda.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 1352 | 22.3% | 83.3% | 0.0% | - | - | 22.7% (n=980) |
| most-frequent-candidate | 1352 | 27.5% | 88.8% | 0.0% | - | - | 0.0% (n=980) |
| baseline-generation-sft | 1352 | 63.4% | 89.5% | 0.9% | 263 | 267 | 64.0% (n=980) |
