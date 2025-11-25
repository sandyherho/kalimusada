"""
Ma-Chen Financial Chaotic System Solver.

Implements high-precision ODE integration for dual trajectory simulation
to demonstrate sensitivity to initial conditions.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from .systems import MaChenSystem


class MaChenSolver:
    """
    Solver for the Ma-Chen Financial Chaotic System.
    
    Simulates two parallel trajectories with infinitesimal initial differences
    to demonstrate chaotic sensitivity to initial conditions.
    """
    
    def __init__(self, rtol: float = 1e-10, atol: float = 1e-12,
                 method: str = 'LSODA'):
        """
        Initialize solver.
        
        Args:
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            method: Integration method (LSODA, RK45, DOP853, etc.)
        """
        self.rtol = rtol
        self.atol = atol
        self.method = method
    
    def solve(self, system: MaChenSystem,
              init_A: List[float], init_B: List[float],
              t_span: Tuple[float, float] = (0, 250),
              n_points: int = 100000,
              window_size: int = 500,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Solve the Ma-Chen system for two trajectories.
        
        Args:
            system: MaChenSystem instance
            init_A: Initial conditions for trajectory A [x, y, z]
            init_B: Initial conditions for trajectory B [x, y, z]
            t_span: (t_start, t_end) time interval
            n_points: Number of evaluation points
            window_size: Window size for RMSE calculation
            verbose: Print progress information
        
        Returns:
            Dictionary with solution data and error metrics
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        if verbose:
            print(f"      Time span: [{t_span[0]}, {t_span[1]}]")
            print(f"      Points: {n_points}")
            print(f"      dt: {(t_span[1] - t_span[0]) / n_points:.6e}")
        
        # Solve trajectory A
        if verbose:
            print("      Solving trajectory A...")
        
        sol_A = solve_ivp(
            system,
            t_span,
            init_A,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        if not sol_A.success:
            raise RuntimeError(f"Trajectory A failed: {sol_A.message}")
        
        # Solve trajectory B
        if verbose:
            print("      Solving trajectory B...")
        
        sol_B = solve_ivp(
            system,
            t_span,
            init_B,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol
        )
        
        if not sol_B.success:
            raise RuntimeError(f"Trajectory B failed: {sol_B.message}")
        
        # Compute error metrics
        if verbose:
            print("      Computing error metrics...")
        
        error_metrics = self._compute_error_metrics(
            sol_A, sol_B, window_size, verbose
        )
        
        # Compile results
        result = {
            'time': sol_A.t,
            'sol_A': {
                'x': sol_A.y[0],
                'y': sol_A.y[1],
                'z': sol_A.y[2]
            },
            'sol_B': {
                'x': sol_B.y[0],
                'y': sol_B.y[1],
                'z': sol_B.y[2]
            },
            'init_A': init_A,
            'init_B': init_B,
            'system': system,
            't_span': t_span,
            'n_points': n_points,
            **error_metrics
        }
        
        return result
    
    def _compute_error_metrics(self, sol_A, sol_B, window_size: int,
                               verbose: bool) -> Dict[str, Any]:
        """
        Compute comprehensive error metrics between trajectories.
        
        Args:
            sol_A: Solution for trajectory A
            sol_B: Solution for trajectory B
            window_size: Window size for RMSE calculation
            verbose: Print progress
        
        Returns:
            Dictionary of error metrics
        """
        n_points = len(sol_A.t)
        
        # Component differences
        delta_x = sol_A.y[0] - sol_B.y[0]
        delta_y = sol_A.y[1] - sol_B.y[1]
        delta_z = sol_A.y[2] - sol_B.y[2]
        
        # Euclidean distance
        euclidean_distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        
        # Relative error
        mag_A = np.sqrt(sol_A.y[0]**2 + sol_A.y[1]**2 + sol_A.y[2]**2)
        relative_error = euclidean_distance / (mag_A + 1e-12)
        
        # Log divergence (for Lyapunov-like analysis)
        log_divergence = np.log10(euclidean_distance + 1e-16)
        
        # Windowed RMSE
        if verbose:
            print("      Computing windowed RMSE...")
        
        rmse_windowed = np.zeros(n_points)
        
        for i in tqdm(range(n_points), desc="      RMSE", ncols=70,
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                     disable=not verbose):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n_points, i + window_size // 2)
            rmse_windowed[i] = np.sqrt(np.mean(euclidean_distance[start_idx:end_idx]**2))
        
        return {
            'delta_x': delta_x,
            'delta_y': delta_y,
            'delta_z': delta_z,
            'euclidean_distance': euclidean_distance,
            'relative_error': relative_error,
            'log_divergence': log_divergence,
            'rmse_windowed': rmse_windowed,
            'abs_delta_x': np.abs(delta_x),
            'abs_delta_y': np.abs(delta_y),
            'abs_delta_z': np.abs(delta_z),
            'max_euclidean_distance': np.max(euclidean_distance),
            'mean_euclidean_distance': np.mean(euclidean_distance),
            'final_euclidean_distance': euclidean_distance[-1],
            'window_size': window_size
        }
