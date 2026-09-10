# Results

Eval subset: first 500 of 1855 held-out step examples (15% of trajectories). Catalog size 6372. Backbone Qwen/Qwen2.5-1.5B-Instruct (float16) on mps.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 500 | 20.8% | 85.0% | 0.0% | - | - | 19.1% (n=362) |
| most-frequent-candidate | 500 | 27.6% | 88.0% | 0.0% | - | - | 0.0% (n=362) |
| actionrank-tier1 | 500 | 50.0% | 90.0% | 0.0% | 249 | 228 | 35.6% (n=362) |
| actionrank-span | 500 | 62.0% | 96.2% | 0.0% | 264 | 239 | 53.0% (n=362) |
| actionrank-tier2 | 500 | 53.0% | 90.6% | 0.0% | 284 | 250 | 40.3% (n=362) |
