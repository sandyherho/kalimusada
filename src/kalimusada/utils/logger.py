"""Comprehensive simulation logger for Ma-Chen chaotic system simulations."""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class SimulationLogger:
    """Logger for Ma-Chen simulations with detailed diagnostics."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs",
                 verbose: bool = True):
        """
        Initialize simulation logger.
        
        Args:
            scenario_name: Scenario name (for log filename)
            log_dir: Directory for log files
            verbose: Print messages to console
        """
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging."""
        logger = logging.getLogger(f"kalimusada_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log all simulation parameters."""
        self.info("=" * 70)
        self.info("MA-CHEN FINANCIAL CHAOTIC SYSTEM SIMULATION")
        self.info(f"Scenario: {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 70)
        self.info("")
        
        self.info("SYSTEM PARAMETERS:")
        self.info(f"  a (savings): {params.get('a', 0.9)}")
        self.info(f"  b (investment cost): {params.get('b', 0.2)}")
        self.info(f"  c (elasticity): {params.get('c', 1.2)}")
        
        self.info("")
        self.info("INITIAL CONDITIONS - ECONOMY A:")
        self.info(f"  x (interest rate): {params.get('init_A_x', 1.0)}")
        self.info(f"  y (investment): {params.get('init_A_y', 2.0)}")
        self.info(f"  z (price index): {params.get('init_A_z', 0.5)}")
        
        self.info("")
        self.info("INITIAL CONDITIONS - ECONOMY B:")
        self.info(f"  x (interest rate): {params.get('init_B_x', 1.00001)}")
        self.info(f"  y (investment): {params.get('init_B_y', 2.0)}")
        self.info(f"  z (price index): {params.get('init_B_z', 0.5)}")
        
        perturbation = abs(params.get('init_B_x', 1.00001) - params.get('init_A_x', 1.0))
        self.info(f"  Initial perturbation: {perturbation:.2e}")
        
        self.info("")
        self.info("NUMERICAL PARAMETERS:")
        self.info(f"  Time span: [{params.get('t_start', 0)}, {params.get('t_end', 250)}]")
        self.info(f"  Number of points: {params.get('n_points', 100000)}")
        self.info(f"  Method: {params.get('method', 'LSODA')}")
        self.info(f"  Relative tolerance: {params.get('rtol', 1e-10)}")
        self.info(f"  Absolute tolerance: {params.get('atol', 1e-12)}")
        self.info(f"  RMSE window size: {params.get('window_size', 500)}")
        
        self.info("")
        self.info("OUTPUT OPTIONS:")
        self.info(f"  Save CSV: {params.get('save_csv', True)}")
        self.info(f"  Save NetCDF: {params.get('save_netcdf', True)}")
        self.info(f"  Save PNG: {params.get('save_png', True)}")
        self.info(f"  Save GIF: {params.get('save_gif', True)}")
        self.info(f"  Output directory: {params.get('output_dir', 'outputs')}")
        
        self.info("")
        self.info("ANIMATION PARAMETERS:")
        self.info(f"  FPS: {params.get('animation_fps', 30)}")
        self.info(f"  DPI: {params.get('animation_dpi', 150)}")
        self.info(f"  Frame skip: {params.get('animation_skip', 8)}")
        
        self.info("=" * 70)
        self.info("")
    
    def log_timing(self, timing: Dict[str, float]):
        """Log timing breakdown."""
        self.info("=" * 70)
        self.info("TIMING BREAKDOWN:")
        self.info("=" * 70)
        
        # Ordered timing sections
        sections = [
            ('system_init', 'System initialization'),
            ('solver_init', 'Solver initialization'),
            ('initial_conditions', 'Initial conditions setup'),
            ('simulation', 'ODE integration'),
            ('csv_save', 'CSV file saving'),
            ('netcdf_save', 'NetCDF file saving'),
            ('png_save', 'Static plot generation'),
            ('gif_save', 'Animation generation'),
            ('visualization', 'Total visualization')
        ]
        
        for key, desc in sections:
            if key in timing:
                self.info(f"  {desc}: {timing[key]:.3f} s")
        
        # Other timings
        for key, value in sorted(timing.items()):
            if key not in [s[0] for s in sections] and key != 'total':
                self.info(f"  {key}: {value:.3f} s")
        
        self.info(f"  {'-' * 40}")
        total_time = timing.get('total', sum(timing.values()))
        self.info(f"  TOTAL: {total_time:.3f} s")
        
        self.info("=" * 70)
        self.info("")
    
    def log_results(self, results: Dict[str, Any]):
        """Log simulation results."""
        self.info("=" * 70)
        self.info("SIMULATION RESULTS:")
        self.info("=" * 70)
        self.info("")
        
        system = results['system']
        self.info(f"SYSTEM: {system}")
        self.info(f"  Parameter a: {system.a}")
        self.info(f"  Parameter b: {system.b}")
        self.info(f"  Parameter c: {system.c}")
        
        self.info("")
        self.info("NUMERICAL SUMMARY:")
        self.info(f"  Time points: {len(results['time'])}")
        self.info(f"  Time range: [{results['time'][0]:.4f}, {results['time'][-1]:.4f}]")
        dt = results['time'][1] - results['time'][0]
        self.info(f"  Time step (dt): {dt:.6e}")
        
        self.info("")
        self.info("TRAJECTORY A STATISTICS:")
        self.info(f"  x range: [{results['sol_A']['x'].min():.6f}, {results['sol_A']['x'].max():.6f}]")
        self.info(f"  y range: [{results['sol_A']['y'].min():.6f}, {results['sol_A']['y'].max():.6f}]")
        self.info(f"  z range: [{results['sol_A']['z'].min():.6f}, {results['sol_A']['z'].max():.6f}]")
        
        self.info("")
        self.info("TRAJECTORY B STATISTICS:")
        self.info(f"  x range: [{results['sol_B']['x'].min():.6f}, {results['sol_B']['x'].max():.6f}]")
        self.info(f"  y range: [{results['sol_B']['y'].min():.6f}, {results['sol_B']['y'].max():.6f}]")
        self.info(f"  z range: [{results['sol_B']['z'].min():.6f}, {results['sol_B']['z'].max():.6f}]")
        
        self.info("")
        self.info("DIVERGENCE ANALYSIS:")
        self.info(f"  Maximum Euclidean distance: {results['max_euclidean_distance']:.6f}")
        self.info(f"  Mean Euclidean distance: {results['mean_euclidean_distance']:.6f}")
        self.info(f"  Final Euclidean distance: {results['final_euclidean_distance']:.6f}")
        self.info(f"  Maximum relative error: {results['relative_error'].max():.6e}")
        self.info(f"  Mean relative error: {results['relative_error'].mean():.6e}")
        
        # Estimate Lyapunov exponent (crude)
        log_div = results['log_divergence']
        time = results['time']
        valid_mask = ~np.isinf(log_div) & ~np.isnan(log_div)
        if np.sum(valid_mask) > 100:
            # Linear fit to log divergence
            import numpy as np
            t_valid = time[valid_mask]
            ld_valid = log_div[valid_mask]
            # Use first half for fit (before saturation)
            n_fit = len(t_valid) // 4
            if n_fit > 10:
                coeffs = np.polyfit(t_valid[:n_fit], ld_valid[:n_fit], 1)
                lyap_estimate = coeffs[0] * np.log(10)  # Convert from log10 to ln
                self.info(f"  Estimated Lyapunov exponent: {lyap_estimate:.4f} (crude)")
        
        self.info("=" * 70)
        self.info("")
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 70)
        self.info("SIMULATION SUMMARY:")
        self.info("=" * 70)
        self.info("")
        
        if self.errors:
            self.info(f"ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                self.info(f"  {i}. {err}")
        else:
            self.info("ERRORS: None")
        
        self.info("")
        
        if self.warnings:
            self.info(f"WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                self.info(f"  {i}. {warn}")
        else:
            self.info("WARNINGS: None")
        
        self.info("")
        self.info(f"Log file: {self.log_file}")
        self.info("=" * 70)
        self.info(f"Simulation completed: {self.scenario_name}")
        self.info(f"Timestamp: {datetime.now().isoformat()}")
        self.info("=" * 70)


# Import numpy for log_results
import numpy as np
