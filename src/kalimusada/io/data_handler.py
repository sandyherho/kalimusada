"""Data handler for saving simulation results to CSV and NetCDF."""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DataHandler:
    """Handle saving simulation data to various formats."""
    
    @staticmethod
    def save_solution_csv(filepath: str, result: Dict[str, Any]):
        """
        Save trajectory solution data to CSV.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'Time': result['time'],
            'Economy_A_Interest_x': result['sol_A']['x'],
            'Economy_A_Investment_y': result['sol_A']['y'],
            'Economy_A_Price_z': result['sol_A']['z'],
            'Economy_B_Interest_x': result['sol_B']['x'],
            'Economy_B_Investment_y': result['sol_B']['y'],
            'Economy_B_Price_z': result['sol_B']['z']
        })
        
        df.to_csv(filepath, index=False, float_format='%.12e')
    
    @staticmethod
    def save_error_csv(filepath: str, result: Dict[str, Any]):
        """
        Save error metrics to CSV.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'Time': result['time'],
            'Delta_x': result['delta_x'],
            'Delta_y': result['delta_y'],
            'Delta_z': result['delta_z'],
            'Euclidean_Distance': result['euclidean_distance'],
            'Relative_Error': result['relative_error'],
            'Log10_Divergence': result['log_divergence'],
            'RMSE_Windowed': result['rmse_windowed'],
            'Abs_Delta_x': result['abs_delta_x'],
            'Abs_Delta_y': result['abs_delta_y'],
            'Abs_Delta_z': result['abs_delta_z']
        })
        
        df.to_csv(filepath, index=False, float_format='%.12e')
    
    @staticmethod
    def save_netcdf(filepath: str, result: Dict[str, Any],
                    config: Dict[str, Any]):
        """
        Save complete simulation data to NetCDF format.
        
        Args:
            filepath: Output file path
            result: Simulation result dictionary
            config: Configuration dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            # Dimensions
            n_time = len(result['time'])
            nc.createDimension('time', n_time)
            nc.createDimension('state', 3)
            nc.createDimension('trajectory', 2)
            
            # Time coordinate
            nc_time = nc.createVariable('time', 'f8', ('time',), zlib=True)
            nc_time[:] = result['time']
            nc_time.units = "dimensionless"
            nc_time.long_name = "simulation_time"
            nc_time.standard_name = "time"
            
            # Trajectory A
            nc_Ax = nc.createVariable('economy_A_x', 'f8', ('time',), zlib=True)
            nc_Ax[:] = result['sol_A']['x']
            nc_Ax.units = "dimensionless"
            nc_Ax.long_name = "interest_rate_economy_A"
            
            nc_Ay = nc.createVariable('economy_A_y', 'f8', ('time',), zlib=True)
            nc_Ay[:] = result['sol_A']['y']
            nc_Ay.units = "dimensionless"
            nc_Ay.long_name = "investment_demand_economy_A"
            
            nc_Az = nc.createVariable('economy_A_z', 'f8', ('time',), zlib=True)
            nc_Az[:] = result['sol_A']['z']
            nc_Az.units = "dimensionless"
            nc_Az.long_name = "price_index_economy_A"
            
            # Trajectory B
            nc_Bx = nc.createVariable('economy_B_x', 'f8', ('time',), zlib=True)
            nc_Bx[:] = result['sol_B']['x']
            nc_Bx.units = "dimensionless"
            nc_Bx.long_name = "interest_rate_economy_B"
            
            nc_By = nc.createVariable('economy_B_y', 'f8', ('time',), zlib=True)
            nc_By[:] = result['sol_B']['y']
            nc_By.units = "dimensionless"
            nc_By.long_name = "investment_demand_economy_B"
            
            nc_Bz = nc.createVariable('economy_B_z', 'f8', ('time',), zlib=True)
            nc_Bz[:] = result['sol_B']['z']
            nc_Bz.units = "dimensionless"
            nc_Bz.long_name = "price_index_economy_B"
            
            # Error metrics
            nc_delta_x = nc.createVariable('delta_x', 'f8', ('time',), zlib=True)
            nc_delta_x[:] = result['delta_x']
            nc_delta_x.units = "dimensionless"
            nc_delta_x.long_name = "difference_in_x"
            
            nc_delta_y = nc.createVariable('delta_y', 'f8', ('time',), zlib=True)
            nc_delta_y[:] = result['delta_y']
            nc_delta_y.units = "dimensionless"
            nc_delta_y.long_name = "difference_in_y"
            
            nc_delta_z = nc.createVariable('delta_z', 'f8', ('time',), zlib=True)
            nc_delta_z[:] = result['delta_z']
            nc_delta_z.units = "dimensionless"
            nc_delta_z.long_name = "difference_in_z"
            
            nc_euclid = nc.createVariable('euclidean_distance', 'f8', ('time',), zlib=True)
            nc_euclid[:] = result['euclidean_distance']
            nc_euclid.units = "dimensionless"
            nc_euclid.long_name = "euclidean_distance_between_trajectories"
            
            nc_rel_err = nc.createVariable('relative_error', 'f8', ('time',), zlib=True)
            nc_rel_err[:] = result['relative_error']
            nc_rel_err.units = "dimensionless"
            nc_rel_err.long_name = "relative_error"
            
            nc_log_div = nc.createVariable('log_divergence', 'f8', ('time',), zlib=True)
            nc_log_div[:] = result['log_divergence']
            nc_log_div.units = "dimensionless"
            nc_log_div.long_name = "log10_of_euclidean_distance"
            
            nc_rmse = nc.createVariable('rmse_windowed', 'f8', ('time',), zlib=True)
            nc_rmse[:] = result['rmse_windowed']
            nc_rmse.units = "dimensionless"
            nc_rmse.long_name = "windowed_root_mean_square_error"
            
            # Global attributes
            system = result['system']
            nc.title = f"Ma-Chen Financial Chaotic System Simulation"
            nc.scenario_name = config.get('scenario_name', 'unknown')
            nc.institution = "Ronin Institute"
            nc.source = "kalimusada v0.0.2"
            nc.history = f"Created {datetime.now().isoformat()}"
            nc.Conventions = "CF-1.8"
            
            # System parameters
            nc.parameter_a = float(system.a)
            nc.parameter_b = float(system.b)
            nc.parameter_c = float(system.c)
            
            # Initial conditions
            nc.init_A_x = float(result['init_A'][0])
            nc.init_A_y = float(result['init_A'][1])
            nc.init_A_z = float(result['init_A'][2])
            nc.init_B_x = float(result['init_B'][0])
            nc.init_B_y = float(result['init_B'][1])
            nc.init_B_z = float(result['init_B'][2])
            
            # Numerical parameters
            nc.n_points = int(result['n_points'])
            nc.t_start = float(result['t_span'][0])
            nc.t_end = float(result['t_span'][1])
            nc.rtol = float(config.get('rtol', 1e-10))
            nc.atol = float(config.get('atol', 1e-12))
            nc.method = config.get('method', 'LSODA')
            
            # Summary statistics
            nc.max_euclidean_distance = float(result['max_euclidean_distance'])
            nc.mean_euclidean_distance = float(result['mean_euclidean_distance'])
            nc.final_euclidean_distance = float(result['final_euclidean_distance'])
            nc.window_size = int(result['window_size'])
            
            # References
            nc.references = "Ma & Chen (2001) Applied Mathematics and Mechanics 22(11):1240-1251"
            nc.author = "Sandy H. S. Herho"
            nc.license = "MIT"
