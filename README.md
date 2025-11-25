# `kalimusada`: a Python library for solving the Ma-Chen financial chaotic system

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/kalimusada.svg)](https://pypi.org/project/kalimusada/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1103582949.svg)](https://doi.org/10.5281/zenodo.17710349)

[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)
[![imageio](https://img.shields.io/badge/imageio-%23172B4D.svg?logo=python&logoColor=white)](https://imageio.github.io/)
[![tqdm](https://img.shields.io/badge/tqdm-%23FFC107.svg?logo=tqdm&logoColor=black)](https://tqdm.github.io/)

A Python-based solver for demonstrating sensitivity to initial conditions in economic dynamics.

<p align="center">
  <img src="ma_chen_chaos.gif" alt="Ma-Chen Chaotic Dynamics" width="600">
</p>

## Model

The Ma-Chen system describes financial dynamics through three coupled ordinary differential equations:

$$ \dot{x} = z + (y - a)x, \quad \dot{y} = 1 - by - x^2, \quad \dot{z} = -x - cz $$

where the state variables are:

| Variable | Description | Economic interpretation |
|:--------:|:------------|:------------------------|
| $x(t)$ | Interest rate | Cost of borrowing capital |
| $y(t)$ | Investment demand | Aggregate investment activity |
| $z(t)$ | Price index | General price level |

and the parameters are:

| Parameter | Description | Chaotic value |
|:---------:|:------------|:-------------:|
| $a$ | Savings rate | $0.9$ |
| $b$ | Investment cost coefficient | $0.2$ |
| $c$ | Demand elasticity | $1.2$ |

The solver simulates two trajectories with infinitesimal initial separation $\delta_0 \sim \mathcal{O}(10^{-5})$ to visualize exponential divergence characteristic of deterministic chaos.

## Installation

**From PyPI:**
```bash
pip install kalimusada
```

**From source:**
```bash
git clone https://github.com/sandyherho/kalimusada.git
cd kalimusada
pip install -e .
```

## Quick start

**CLI:**
```bash
kalimusada case1          # run standard chaos scenario
kalimusada --all          # run all test cases
```

**Python API:**
```python
from kalimusada import MaChenSolver, MaChenSystem

system = MaChenSystem(a=0.9, b=0.2, c=1.2)
solver = MaChenSolver()

result = solver.solve(
    system=system,
    init_A=[1.0, 2.0, 0.5],
    init_B=[1.00001, 2.0, 0.5],
    t_span=(0, 250),
    n_points=100000
)

print(f"Max divergence: {result['max_euclidean_distance']:.6f}")
```

## Features

- High-precision ODE integration (LSODA)
- Dual trajectory sensitivity analysis
- Error metrics: Euclidean distance, RMSE, log divergence
- Output formats: CSV, NetCDF, PNG, GIF

## License

MIT © Sandy H. S. Herho

## Citation

```bibtex
@software{herho2025_kalimusada,
  title   = {kalimusada: A Python library for solving the Ma-Chen financial chaotic system},
  author  = {Herho, Sandy H. S.},
  year    = {2025},
  url     = {https://github.com/sandyherho/kalimusada}
}
```
