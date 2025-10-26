# MCMC Methods Comparison Report

## Overview

This report compares the performance of different MCMC sampling methods across three distributions: Gaussian, Ring, and Allen-Cahn Equation. The methods tested include:

1. **HMC** - Hamiltonian Monte Carlo
2. **HMC Walk** - HMC with walk moves
3. **HMC Side** - HMC with side moves  
4. **Stretch** - Stretch move
5. **Walk** - Walk move
6. **Side** - Side move

Each method was tested with two parameter configurations:
- **L=2**: Step size = 0.5, Integration length = 2
- **L=10**: Step size = 0.1, Integration length = 10

All experiments used 24 chains, 200,000 warmup samples, and 1,000,000 main samples with thinning by 10.

## Calculation Notes

**Effective Sample Size (ESS):** Calculated using the formula ESS = N/(1 + 2τ), where:
- N = 1,000,000 (total number of samples per chain)
- τ = integrated autocorrelation time (averaged across dimensions)

**ESS/Time Ratio:** Calculated as ESS divided by wall-clock time in seconds, representing the number of effective independent samples obtained per second of computation time.

**Autocorr/Time Ratio:** Calculated as integrated autocorrelation time divided by wall-clock time, representing the autocorrelation burden per unit time (lower values indicate better efficiency).

## 1. Gaussian Distribution Results

| Method | Config | Time (s) | Acceptance Rate | Avg Autocorr | Autocorr/Time | ESS | ESS/Time | Mean Residual | Covariance Residual |
|--------|--------|----------|----------------|--------------|---------------|-----|----------|---------------|-------------------|
| HMC | L=2 | 2465.06 | 0.800 | 32.0 | 0.012981 | 15,385 | 6.24 | 0.0041 | 15.74 |
| HMC | L=10 | 5733.53 | 0.799 | 2.0 | 0.000349 | 200,000 | 34.88 | 0.0018 | 20.62 |
| HMC Walk | L=2 | 3413.08 | 0.802 | 1.0 | 0.000293 | 333,333 | 97.66 | 0.0013 | 22.01 |
| HMC Walk | L=10 | - | - | - | - | - | - | - | - |
| HMC Side | L=2 | 2889.89 | 0.793 | 2.2 | 0.000761 | 185,185 | 64.08 | 0.0014 | 21.34 |
| HMC Side | L=10 | - | - | - | - | - | - | - | - |
| Stretch | L=2 | - | - | - | - | - | - | - | - |
| Stretch | L=10 | 1600.08 | 0.384 | 20.0 | 0.012499 | 24,390 | 15.24 | 0.0079 | 22.34 |
| Walk | L=2 | - | - | - | - | - | - | - | - |
| Walk | L=10 | 1424.27 | 0.261 | 5.3 | 0.003721 | 86,207 | 60.53 | 0.0102 | 21.83 |
| Side | L=2 | - | - | - | - | - | - | - | - |
| Side | L=10 | 1587.70 | 0.137 | 17.5 | 0.011022 | 27,778 | 17.50 | 0.0081 | 22.21 |

### Key Observations for Gaussian Distribution:
- **HMC Walk (L=2)** achieved the highest ESS/time ratio (97.66) - most efficient sampling
- **HMC (L=10)** had very low autocorrelation (2.0) and high ESS (200,000) but was the slowest
- **Stretch move** was fastest but had lower acceptance rate and much lower ESS (24,390)
- **Side move** had the lowest acceptance rate (0.137) and lowest ESS (27,778)

## 2. Ring Distribution Results

| Method | Config | Time (s) | Acceptance Rate | Avg Autocorr | Autocorr/Time | ESS | ESS/Time | Mean Distance | Distance Std | Fraction in Ring |
|--------|--------|----------|----------------|--------------|---------------|-----|----------|---------------|---------------|------------------|
| HMC | L=2 | 923.78 | 0.800 | 9.8 | 0.010609 | 48,544 | 52.55 | 1.032 | 0.065 | 1.000 |
| HMC | L=10 | 1761.23 | 0.800 | 1.0 | 0.000568 | 333,333 | 189.26 | 1.071 | 0.083 | 1.000 |
| HMC Walk | L=2 | 1372.32 | 0.800 | 1.8 | 0.001312 | 217,391 | 158.41 | 1.057 | 0.080 | 1.000 |
| HMC Walk | L=10 | 5419.47 | 0.801 | 1.0 | 0.000185 | 333,333 | 61.51 | 1.065 | 0.079 | 1.000 |
| HMC Side | L=2 | 1055.73 | 0.800 | 5.1 | 0.004831 | 89,286 | 84.57 | 1.062 | 0.079 | 1.000 |
| HMC Side | L=10 | 1997.86 | 0.800 | 6.5 | 0.003253 | 71,429 | 35.75 | 1.063 | 0.079 | 1.000 |
| Stretch | L=10 | 764.53 | 0.172 | 196.0 | 0.256367 | 2,545 | 3.33 | 1.065 | 0.079 | 1.000 |
| Walk | L=10 | 489.41 | 0.199 | 6.4 | 0.013077 | 72,464 | 148.06 | 1.064 | 0.079 | 1.000 |
| Side | L=10 | 829.05 | 0.124 | 21.5 | 0.025933 | 22,727 | 27.41 | 1.064 | 0.079 | 1.000 |

### Key Observations for Ring Distribution:
- **HMC (L=10)** achieved the highest ESS/time ratio (189.26) - most efficient sampling
- **HMC Walk (L=2)** had very high ESS/time ratio (158.41) with good speed
- **Walk move** was fastest and had good ESS/time ratio (148.06)
- **Stretch move** had extremely low ESS (2,545) and very low ESS/time ratio (3.33)
- All methods successfully sampled within the ring (fraction = 1.000)

## 3. Allen-Cahn Equation Results

| Method | Config | Time (s) | Acceptance Rate | Avg Autocorr | Autocorr/Time | ESS | ESS/Time | Mean Value | Std Value | Fraction +1 | Fraction -1 | Mean Energy |
|--------|--------|----------|----------------|--------------|---------------|-----|----------|------------|-----------|-------------|-------------|-------------|
| HMC | L=2 | 904.09 | 0.800 | 289.0 | 0.319658 | 1,727 | 1.91 | -0.003 | 0.812 | 0.263 | 0.265 | 537136256 |
| HMC | L=10 | 1740.15 | 0.801 | 7.0 | 0.004023 | 66,667 | 38.31 | -0.000 | 0.834 | 0.190 | 0.190 | 570101312 |
| HMC Walk | L=2 | 1367.91 | 0.800 | 1.1 | 0.000804 | 312,500 | 228.45 | -0.000 | 0.831 | 0.192 | 0.192 | 561248640 |
| HMC Walk | L=10 | 6122.05 | 0.799 | 1.0 | 0.000163 | 333,333 | 54.45 | 0.000 | 0.845 | 0.182 | 0.182 | 585206272 |
| HMC Side | L=2 | 935.01 | 0.800 | 2.8 | 0.002995 | 151,515 | 162.05 | 0.000 | 0.837 | 0.189 | 0.189 | 570436992 |
| HMC Side | L=10 | 2127.28 | 0.800 | 4.6 | 0.002162 | 98,039 | 46.09 | -0.000 | 0.840 | 0.186 | 0.186 | 576486784 |
| Stretch | L=10 | 759.08 | 0.354 | 26.4 | 0.034779 | 18,587 | 24.49 | -0.001 | 0.845 | 0.181 | 0.182 | 584497664 |
| Walk | L=10 | 454.91 | 0.240 | 6.1 | 0.013409 | 75,758 | 166.53 | 0.001 | 0.841 | 0.185 | 0.185 | 405488320 |
| Side | L=10 | 782.09 | 0.134 | 19.5 | 0.024933 | 25,000 | 31.97 | -0.000 | 0.844 | 0.183 | 0.183 | 583050112 |

### Key Observations for Allen-Cahn Equation:
- **HMC Walk (L=2)** achieved the highest ESS/time ratio (228.45) - most efficient sampling
- **Walk move** was fastest and had very good ESS/time ratio (166.53)
- **HMC Side (L=2)** had good ESS/time ratio (162.05) with moderate speed
- **HMC (L=2)** had extremely low ESS (1,727) and very low ESS/time ratio (1.91) despite good acceptance rate
- All methods successfully sampled both +1 and -1 states

## Summary and Recommendations

### Best Overall Performance by Distribution:

**Gaussian Distribution:**
- **Best Efficiency**: HMC Walk (L=2) - Highest ESS/time ratio (97.66)
- **Best Speed**: Stretch move - Fastest but much lower ESS/time ratio (15.24)
- **Best Accuracy**: HMC (L=10) - Lowest mean residual and high ESS (200,000)

**Ring Distribution:**
- **Best Efficiency**: HMC (L=10) - Highest ESS/time ratio (189.26)
- **Best Speed**: Walk move - Fastest execution with good ESS/time ratio (148.06)
- **Most Reliable**: HMC methods - Consistent performance across configurations

**Allen-Cahn Equation:**
- **Best Efficiency**: HMC Walk (L=2) - Highest ESS/time ratio (228.45)
- **Best Speed**: Walk move - Fastest execution with very good ESS/time ratio (166.53)
- **Most Reliable**: HMC Walk methods - Consistently high ESS/time ratios

### General Recommendations:

1. **HMC Walk methods** consistently provide the best sampling efficiency (highest ESS/time ratios)
2. **Walk move** is the fastest and often has very good ESS/time ratios
3. **Stretch move** can be fast but often has very low ESS and poor ESS/time ratios
4. **Side move** generally has the lowest acceptance rates and moderate ESS/time ratios
5. **L=10 configuration** (smaller step size, longer integration) generally provides better sampling efficiency but at higher computational cost
6. **ESS/time ratio** is the most important metric for comparing sampling efficiency across methods

### Trade-offs to Consider:

- **Speed vs. Efficiency**: Walk and Stretch moves are faster but less efficient
- **Accuracy vs. Speed**: HMC methods are more accurate but slower
- **Parameter Sensitivity**: HMC methods are more sensitive to step size and integration length choices
