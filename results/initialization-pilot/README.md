# initialization study

These results use a split grouped by name spelling. All scores below are train/dev; the test partition was not evaluated.

| Run | Context | Activation | Init | Steps | Parameters | Train NLL | Dev NLL | Seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| unscaled-bs3-tanh-s0 | 3 | tanh | unscaled | 10000 | 11897 | 2.410075 | 2.421453 | 4.38 |
| unscaled-bs3-blend-s0 | 3 | blend | unscaled | 10000 | 12097 | 2.349277 | 2.353500 | 8.88 |
| unscaled-bs4-tanh-s0 | 4 | tanh | unscaled | 10000 | 13897 | 2.395267 | 2.410748 | 8.44 |
| unscaled-bs4-blend-s0 | 4 | blend | unscaled | 10000 | 14097 | 2.326090 | 2.325818 | 15.58 |
| scaled-bs3-tanh-s0 | 3 | tanh | scaled | 10000 | 11897 | 2.201305 | 2.214066 | 8.19 |
| scaled-bs3-blend-s0 | 3 | blend | scaled | 10000 | 12097 | 2.210748 | 2.217225 | 14.99 |
| scaled-bs4-tanh-s0 | 4 | tanh | scaled | 10000 | 13897 | 2.170050 | 2.187547 | 8.57 |
| scaled-bs4-blend-s0 | 4 | blend | scaled | 10000 | 14097 | 2.176493 | 2.192065 | 15.59 |
| unscaled-bs3-tanh-s1 | 3 | tanh | unscaled | 10000 | 11897 | 2.383415 | 2.390979 | 8.39 |
| unscaled-bs3-blend-s1 | 3 | blend | unscaled | 10000 | 12097 | 2.343870 | 2.345479 | 15.21 |
| unscaled-bs4-tanh-s1 | 4 | tanh | unscaled | 10000 | 13897 | 2.392688 | 2.401631 | 8.62 |
| unscaled-bs4-blend-s1 | 4 | blend | unscaled | 10000 | 14097 | 2.328557 | 2.331146 | 15.42 |
| scaled-bs3-tanh-s1 | 3 | tanh | scaled | 10000 | 11897 | 2.204105 | 2.216773 | 8.64 |
| scaled-bs3-blend-s1 | 3 | blend | scaled | 10000 | 12097 | 2.213517 | 2.225603 | 15.39 |
| scaled-bs4-tanh-s1 | 4 | tanh | scaled | 10000 | 13897 | 2.173879 | 2.189042 | 8.51 |
| scaled-bs4-blend-s1 | 4 | blend | scaled | 10000 | 14097 | 2.181929 | 2.195482 | 15.48 |
| unscaled-bs3-tanh-s2 | 3 | tanh | unscaled | 10000 | 11897 | 2.417618 | 2.435275 | 8.20 |
| unscaled-bs3-blend-s2 | 3 | blend | unscaled | 10000 | 12097 | 2.336686 | 2.341601 | 15.45 |
| unscaled-bs4-tanh-s2 | 4 | tanh | unscaled | 10000 | 13897 | 2.382123 | 2.392179 | 8.58 |
| unscaled-bs4-blend-s2 | 4 | blend | unscaled | 10000 | 14097 | 2.317032 | 2.320243 | 15.65 |
| scaled-bs3-tanh-s2 | 3 | tanh | scaled | 10000 | 11897 | 2.194225 | 2.206411 | 8.57 |
| scaled-bs3-blend-s2 | 3 | blend | scaled | 10000 | 12097 | 2.204056 | 2.212937 | 15.24 |
| scaled-bs4-tanh-s2 | 4 | tanh | scaled | 10000 | 13897 | 2.167705 | 2.185778 | 8.54 |
| scaled-bs4-blend-s2 | 4 | blend | scaled | 10000 | 14097 | 2.175529 | 2.189589 | 15.76 |

## Matched activation comparisons

Differences are blend minus tanh. Negative values favour blend. The standard error quantifies seed variation at this fixed split and protocol; it is not a significance filter for exploratory selection.

- Context 3, unscaled, 10000 updates: n = 3, mean difference = -0.069043, SE = 0.013917.
- Context 4, unscaled, 10000 updates: n = 3, mean difference = -0.075784, SE = 0.004592.
- Context 3, scaled, 10000 updates: n = 3, mean difference = 0.006172, SE = 0.001647.
- Context 4, scaled, 10000 updates: n = 3, mean difference = 0.004923, SE = 0.000785.

## First observed dev-loss targets

Hits are observed at scheduled evaluations, after the stated number of updates. They are not interpolated or claims of sustained attainment. Elapsed time includes initial diagnostics and dev evaluations but excludes setup and saving.

| Run | Target dev NLL | First observed step | Seconds |
| --- | ---: | ---: | ---: |
| unscaled-bs3-tanh-s0 | 2.4 | Not reached | — |
| unscaled-bs3-tanh-s0 | 2.2 | Not reached | — |
| unscaled-bs3-tanh-s0 | 2.15 | Not reached | — |
| unscaled-bs3-tanh-s0 | 2.13 | Not reached | — |
| unscaled-bs3-blend-s0 | 2.4 | 6000 | 4.58 |
| unscaled-bs3-blend-s0 | 2.2 | Not reached | — |
| unscaled-bs3-blend-s0 | 2.15 | Not reached | — |
| unscaled-bs3-blend-s0 | 2.13 | Not reached | — |
| unscaled-bs4-tanh-s0 | 2.4 | Not reached | — |
| unscaled-bs4-tanh-s0 | 2.2 | Not reached | — |
| unscaled-bs4-tanh-s0 | 2.15 | Not reached | — |
| unscaled-bs4-tanh-s0 | 2.13 | Not reached | — |
| unscaled-bs4-blend-s0 | 2.4 | 6000 | 7.07 |
| unscaled-bs4-blend-s0 | 2.2 | Not reached | — |
| unscaled-bs4-blend-s0 | 2.15 | Not reached | — |
| unscaled-bs4-blend-s0 | 2.13 | Not reached | — |
| scaled-bs3-tanh-s0 | 2.4 | 2000 | 1.45 |
| scaled-bs3-tanh-s0 | 2.2 | Not reached | — |
| scaled-bs3-tanh-s0 | 2.15 | Not reached | — |
| scaled-bs3-tanh-s0 | 2.13 | Not reached | — |
| scaled-bs3-blend-s0 | 2.4 | 2000 | 2.62 |
| scaled-bs3-blend-s0 | 2.2 | Not reached | — |
| scaled-bs3-blend-s0 | 2.15 | Not reached | — |
| scaled-bs3-blend-s0 | 2.13 | Not reached | — |
| scaled-bs4-tanh-s0 | 2.4 | 1000 | 0.86 |
| scaled-bs4-tanh-s0 | 2.2 | 6000 | 4.15 |
| scaled-bs4-tanh-s0 | 2.15 | Not reached | — |
| scaled-bs4-tanh-s0 | 2.13 | Not reached | — |
| scaled-bs4-blend-s0 | 2.4 | 1000 | 1.67 |
| scaled-bs4-blend-s0 | 2.2 | 7000 | 8.21 |
| scaled-bs4-blend-s0 | 2.15 | Not reached | — |
| scaled-bs4-blend-s0 | 2.13 | Not reached | — |
| unscaled-bs3-tanh-s1 | 2.4 | 9000 | 5.94 |
| unscaled-bs3-tanh-s1 | 2.2 | Not reached | — |
| unscaled-bs3-tanh-s1 | 2.15 | Not reached | — |
| unscaled-bs3-tanh-s1 | 2.13 | Not reached | — |
| unscaled-bs3-blend-s1 | 2.4 | 6000 | 6.75 |
| unscaled-bs3-blend-s1 | 2.2 | Not reached | — |
| unscaled-bs3-blend-s1 | 2.15 | Not reached | — |
| unscaled-bs3-blend-s1 | 2.13 | Not reached | — |
| unscaled-bs4-tanh-s1 | 2.4 | Not reached | — |
| unscaled-bs4-tanh-s1 | 2.2 | Not reached | — |
| unscaled-bs4-tanh-s1 | 2.15 | Not reached | — |
| unscaled-bs4-tanh-s1 | 2.13 | Not reached | — |
| unscaled-bs4-blend-s1 | 2.4 | 6000 | 6.90 |
| unscaled-bs4-blend-s1 | 2.2 | Not reached | — |
| unscaled-bs4-blend-s1 | 2.15 | Not reached | — |
| unscaled-bs4-blend-s1 | 2.13 | Not reached | — |
| scaled-bs3-tanh-s1 | 2.4 | 1000 | 0.88 |
| scaled-bs3-tanh-s1 | 2.2 | Not reached | — |
| scaled-bs3-tanh-s1 | 2.15 | Not reached | — |
| scaled-bs3-tanh-s1 | 2.13 | Not reached | — |
| scaled-bs3-blend-s1 | 2.4 | 1000 | 1.57 |
| scaled-bs3-blend-s1 | 2.2 | Not reached | — |
| scaled-bs3-blend-s1 | 2.15 | Not reached | — |
| scaled-bs3-blend-s1 | 2.13 | Not reached | — |
| scaled-bs4-tanh-s1 | 2.4 | 1000 | 0.85 |
| scaled-bs4-tanh-s1 | 2.2 | 7000 | 4.60 |
| scaled-bs4-tanh-s1 | 2.15 | Not reached | — |
| scaled-bs4-tanh-s1 | 2.13 | Not reached | — |
| scaled-bs4-blend-s1 | 2.4 | 1000 | 1.61 |
| scaled-bs4-blend-s1 | 2.2 | 9000 | 10.23 |
| scaled-bs4-blend-s1 | 2.15 | Not reached | — |
| scaled-bs4-blend-s1 | 2.13 | Not reached | — |
| unscaled-bs3-tanh-s2 | 2.4 | Not reached | — |
| unscaled-bs3-tanh-s2 | 2.2 | Not reached | — |
| unscaled-bs3-tanh-s2 | 2.15 | Not reached | — |
| unscaled-bs3-tanh-s2 | 2.13 | Not reached | — |
| unscaled-bs3-blend-s2 | 2.4 | 6000 | 6.87 |
| unscaled-bs3-blend-s2 | 2.2 | Not reached | — |
| unscaled-bs3-blend-s2 | 2.15 | Not reached | — |
| unscaled-bs3-blend-s2 | 2.13 | Not reached | — |
| unscaled-bs4-tanh-s2 | 2.4 | 8000 | 5.41 |
| unscaled-bs4-tanh-s2 | 2.2 | Not reached | — |
| unscaled-bs4-tanh-s2 | 2.15 | Not reached | — |
| unscaled-bs4-tanh-s2 | 2.13 | Not reached | — |
| unscaled-bs4-blend-s2 | 2.4 | 6000 | 7.05 |
| unscaled-bs4-blend-s2 | 2.2 | Not reached | — |
| unscaled-bs4-blend-s2 | 2.15 | Not reached | — |
| unscaled-bs4-blend-s2 | 2.13 | Not reached | — |
| scaled-bs3-tanh-s2 | 2.4 | 2000 | 1.45 |
| scaled-bs3-tanh-s2 | 2.2 | Not reached | — |
| scaled-bs3-tanh-s2 | 2.15 | Not reached | — |
| scaled-bs3-tanh-s2 | 2.13 | Not reached | — |
| scaled-bs3-blend-s2 | 2.4 | 2000 | 2.72 |
| scaled-bs3-blend-s2 | 2.2 | Not reached | — |
| scaled-bs3-blend-s2 | 2.15 | Not reached | — |
| scaled-bs3-blend-s2 | 2.13 | Not reached | — |
| scaled-bs4-tanh-s2 | 2.4 | 2000 | 1.51 |
| scaled-bs4-tanh-s2 | 2.2 | 7000 | 4.69 |
| scaled-bs4-tanh-s2 | 2.15 | Not reached | — |
| scaled-bs4-tanh-s2 | 2.13 | Not reached | — |
| scaled-bs4-blend-s2 | 2.4 | 1000 | 1.65 |
| scaled-bs4-blend-s2 | 2.2 | 7000 | 8.30 |
| scaled-bs4-blend-s2 | 2.15 | Not reached | — |
| scaled-bs4-blend-s2 | 2.13 | Not reached | — |
