#!/usr/bin/env python
"""
Command Line Interface for kalimusada Ma-Chen Financial Chaotic System Solver.
"""

import argparse
import sys
from pathlib import Path

from .core.solver import MaChenSolver
from .core.systems import MaChenSystem
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 15 + "kalimusada: Ma-Chen Financial Chaotic System")
    print(" " * 25 + "Version 0.0.1")
    print("=" * 70)
    print("\n  Sensitivity Analysis for Chaotic Financial Dynamics")
    print("  Butterfly Effect in Economic Systems")
    print("\n  Author: Sandy H. S. Herho")
    print("  License: MIT")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    if clean.startswith('case_'):
        parts = clean.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            case_num = parts[1]
            rest = '_'.join(parts[2:])
            clean = f"case{case_num}_{rest}"
    
    clean = clean.rstrip('_')
    return clean


def run_scenario(config: dict, output_dir: str = "outputs",
                verbose: bool = True):
    """Run a complete Ma-Chen simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 70}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        # [1/7] Initialize system
        with timer.time_section("system_init"):
            if verbose:
                print("\n[1/7] Initializing Ma-Chen system...")
            
            system = MaChenSystem(
                a=config.get('a', 0.9),
                b=config.get('b', 0.2),
                c=config.get('c', 1.2)
            )
            
            if verbose:
                print(f"      Parameters: a={system.a}, b={system.b}, c={system.c}")
        
        # [2/7] Initialize solver
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[2/7] Initializing solver...")
            
            solver = MaChenSolver(
                rtol=config.get('rtol', 1e-10),
                atol=config.get('atol', 1e-12),
                method=config.get('method', 'LSODA')
            )
            
            if verbose:
                print(f"      Method: {solver.method}")
                print(f"      Tolerance: rtol={solver.rtol}, atol={solver.atol}")
        
        # [3/7] Setup initial conditions
        with timer.time_section("initial_conditions"):
            if verbose:
                print("\n[3/7] Setting up initial conditions...")
            
            init_A = [
                config.get('init_A_x', 1.0),
                config.get('init_A_y', 2.0),
                config.get('init_A_z', 0.5)
            ]
            init_B = [
                config.get('init_B_x', 1.00001),
                config.get('init_B_y', 2.0),
                config.get('init_B_z', 0.5)
            ]
            
            perturbation = abs(init_B[0] - init_A[0])
            
            if verbose:
                print(f"      Economy A: x={init_A[0]}, y={init_A[1]}, z={init_A[2]}")
                print(f"      Economy B: x={init_B[0]}, y={init_B[1]}, z={init_B[2]}")
                print(f"      Initial perturbation: {perturbation:.2e}")
        
        # [4/7] Solve ODEs
        with timer.time_section("simulation"):
            if verbose:
                print("\n[4/7] Solving ODEs...")
            
            t_span = (config.get('t_start', 0.0), config.get('t_end', 250.0))
            n_points = config.get('n_points', 100000)
            
            result = solver.solve(
                system=system,
                init_A=init_A,
                init_B=init_B,
                t_span=t_span,
                n_points=n_points,
                window_size=config.get('window_size', 500),
                verbose=verbose
            )
            
            logger.log_results(result)
            
            if verbose:
                print(f"\n      Max Euclidean distance: {result['max_euclidean_distance']:.6f}")
                print(f"      Final Euclidean distance: {result['final_euclidean_distance']:.6f}")
        
        # [5/7] Save CSV data
        if config.get('save_csv', True):
            with timer.time_section("csv_save"):
                if verbose:
                    print("\n[5/7] Saving CSV data...")
                
                csv_dir = Path(output_dir) / "csv"
                csv_dir.mkdir(parents=True, exist_ok=True)
                
                solution_file = csv_dir / f"{clean_name}_solution.csv"
                error_file = csv_dir / f"{clean_name}_error_metrics.csv"
                
                DataHandler.save_solution_csv(solution_file, result)
                DataHandler.save_error_csv(error_file, result)
                
                if verbose:
                    print(f"      Saved: {solution_file}")
                    print(f"      Saved: {error_file}")
        
        # [6/7] Save NetCDF
        if config.get('save_netcdf', True):
            with timer.time_section("netcdf_save"):
                if verbose:
                    print("\n[6/7] Saving NetCDF data...")
                
                nc_dir = Path(output_dir) / "netcdf"
                nc_dir.mkdir(parents=True, exist_ok=True)
                
                nc_file = nc_dir / f"{clean_name}.nc"
                DataHandler.save_netcdf(nc_file, result, config)
                
                if verbose:
                    print(f"      Saved: {nc_file}")
        
        # [7/7] Generate visualizations
        with timer.time_section("visualization"):
            if verbose:
                print("\n[7/7] Generating visualizations...")
            
            animator = Animator(
                fps=config.get('animation_fps', 30),
                dpi=config.get('animation_dpi', 150)
            )
            
            if config.get('save_png', True):
                with timer.time_section("png_save"):
                    if verbose:
                        print("      Creating static plots...")
                    
                    fig_dir = Path(output_dir) / "figs"
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    
                    png_file = fig_dir / f"{clean_name}_timeseries.png"
                    animator.create_static_plot(result, png_file, scenario_name)
                    
                    if verbose:
                        print(f"      Saved: {png_file}")
            
            if config.get('save_gif', True):
                with timer.time_section("gif_save"):
                    if verbose:
                        print("      Creating animation...")
                    
                    gif_dir = Path(output_dir) / "gifs"
                    gif_dir.mkdir(parents=True, exist_ok=True)
                    
                    gif_file = gif_dir / f"{clean_name}_animation.gif"
                    skip = config.get('animation_skip', 8)
                    animator.create_animation(result, gif_file, scenario_name, skip=skip)
                    
                    if verbose:
                        print(f"      Saved: {gif_file}")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        # Summary
        sim_time = timer.times.get('simulation', 0)
        total_time = timer.times.get('total', 0)
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Simulation time: {sim_time:.2f} s")
            print(f"  Total time: {total_time:.2f} s")
            
            if logger.warnings:
                print(f"  Warnings: {len(logger.warnings)}")
            if logger.errors:
                print(f"  Errors: {len(logger.errors)}")
            
            print(f"{'=' * 70}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"SIMULATION FAILED")
            print(f"  Error: {str(e)}")
            print(f"{'=' * 70}\n")
        
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description='kalimusada: Ma-Chen Financial Chaotic System Solver',
        epilog='Example: kalimusada case1'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run (case1-4)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    # Custom config
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose)
    
    # All cases
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose)
    
    # Single case
    elif args.case:
        case_map = {
            'case1': 'case1_standard_chaos',
            'case2': 'case2_high_sensitivity',
            'case3': 'case3_modified_parameters',
            'case4': 'case4_long_term'
        }
        
        cfg_name = case_map[args.case]
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
