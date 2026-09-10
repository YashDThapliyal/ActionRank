# Results

Eval subset: first 8 of 1855 held-out step examples (15% of trajectories). Catalog size 6372. Backbone Qwen/Qwen2.5-1.5B-Instruct (float16) on mps.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 8 | 12.5% | 62.5% | 0.0% | - | - | 0.0% (n=6) |
| most-frequent-candidate | 8 | 25.0% | 100.0% | 0.0% | - | - | 0.0% (n=6) |
| actionrank-span | 8 | 25.0% | 87.5% | 0.0% | 226 | 217 | 16.7% (n=6) |
