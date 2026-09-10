# Results

Eval subset: first 500 of 1855 held-out step examples (15% of trajectories). Catalog size 6372. Backbone Qwen/Qwen2.5-1.5B-Instruct (float16) on mps.

| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |
|---|---:|---:|---:|---:|---:|---:|---:|
| random-candidate | 500 | 20.8% | 85.0% | 0.0% | - | - | 19.1% (n=362) |
| most-frequent-candidate | 500 | 27.6% | 88.0% | 0.0% | - | - | 0.0% (n=362) |
| actionrank-tier1 | 500 | 49.4% | 91.4% | 0.0% | 235 | 219 | 37.3% (n=362) |
| actionrank-span | 500 | 57.8% | 97.2% | 0.0% | 231 | 217 | 45.9% (n=362) |
| baseline-generation | 500 | 31.8% | 53.4% | 1.8% | 726 | 653 | 43.9% (n=362) |
| baseline-generation-sft | 500 | 66.8% | 91.4% | 1.4% | 639 | 592 | 60.2% (n=362) |
| actionrank-tier2 | 500 | 50.4% | 91.8% | 0.0% | 251 | 230 | 38.7% (n=362) |
