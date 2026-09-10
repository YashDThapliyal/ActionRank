# Prediction analysis (first 500 held-out steps)

| system | predicts Finish | Finish recall (n) | top-1 non-Finish | top-1 seen tools (n) | top-1 unseen tools (n) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0% | 0.0% (138) | 43.9% | 46.5% (241) | 38.8% (121) |
| span | 43.8% | 89.1% (138) | 45.9% | 52.3% (241) | 33.1% (121) |
| tier1 | 38.8% | 81.2% (138) | 37.3% | 47.3% (241) | 17.4% (121) |
