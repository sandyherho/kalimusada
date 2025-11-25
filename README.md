# kalimusada: Ma-Chen Financial Chaotic System Solver

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`kalimusada` is a Python solver for the Ma-Chen Financial Chaotic System (2001), designed to demonstrate sensitivity to initial conditions (the butterfly effect) in financial dynamics. The package simulates two parallel economic trajectories with infinitesimal initial differences to visualize chaotic divergence.

## Physics

The Ma-Chen system models financial dynamics through three coupled ODEs:
```
dx/dt = z + (y - a) * x    [Interest Rate]
dy/dt = 1 - b * y - x^2    [Investment Demand]
dz/dt = -x - c * z         [Price Index]
```

where:
- x: Interest rate
- y: Investment demand
- z: Price index
- a, b, c: System parameters

For chaotic behavior: a = 0.9, b = 0.2, c = 1.2

## Features

- High-precision ODE integration (LSODA method)
- Dual trajectory simulation for sensitivity analysis
- Comprehensive error metrics (Euclidean distance, RMSE, log divergence)
- Professional visualizations (static plots and animations)
- Multiple output formats (CSV, NetCDF, PNG, GIF)
- Configurable via text files
- Detailed logging with timing information

## Installation

**From source:**
```bash
git clone https://github.com/sandyherho/kalimusada.git
cd kalimusada
pip install -e .
```

## Command Line Usage
```bash
# Run single test case
kalimusada case1

# Run all test cases
kalimusada --all

# Custom configuration
kalimusada --config my_config.txt

# Quiet mode
kalimusada case2 --quiet

# Specify output directory
kalimusada case1 --output-dir results
```

## Python API
```python
from kalimusada import MaChenSolver, MaChenSystem

# Create system
system = MaChenSystem(a=0.9, b=0.2, c=1.2)

# Create solver
solver = MaChenSolver()

# Initial conditions (two economies with tiny difference)
init_A = [1.0, 2.0, 0.5]
init_B = [1.00001, 2.0, 0.5]

# Solve
result = solver.solve(
    system=system,
    init_A=init_A,
    init_B=init_B,
    t_span=(0, 250),
    n_points=100000
)

# Access results
print(f"Max divergence: {result['max_euclidean_distance']:.6f}")
```

## Output Structure
```
outputs/
    csv/          # Solution and error metrics
    netcdf/       # NetCDF4 files with full data
    figs/         # Static PNG plots
    gifs/         # Animated phase space
logs/             # Detailed simulation logs
```

## Author

Sandy H. S. Herho

## License

MIT License - See [LICENSE](LICENSE) file for details

## Citation
```bibtex
@software{herho2025_kalimusada,
  title   = {kalimusada: Ma-Chen Financial Chaotic System Solver},
  author  = {Herho, Sandy H. S.},
  year    = {2025},
  version = {0.0.1},
  url     = {https://github.com/sandyherho/kalimusada}
}
```

## References

Ma, J. H., & Chen, Y. S. (2001). Study for the bifurcation topological structure and the global complicated character of a kind of nonlinear finance system. Applied Mathematics and Mechanics, 22(11), 1240-1251.
