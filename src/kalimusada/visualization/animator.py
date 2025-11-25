"""
Professional visualization for Ma-Chen Financial Chaotic System.

Creates static time series plots and animated 3D phase space visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')


class Animator:
    """
    Create professional visualizations for Ma-Chen chaotic dynamics.
    """
    
    # Dark theme color palette
    COLOR_BG = '#0D1117'
    COLOR_BG_LIGHTER = '#161B22'
    COLOR_A = '#00F5D4'
    COLOR_B = '#FF6B9D'
    COLOR_ACCENT = '#FFE66D'
    COLOR_GRID = '#30363D'
    COLOR_TEXT = '#E6EDF3'
    COLOR_TITLE = '#FFFFFF'
    
    def __init__(self, fps: int = 30, dpi: int = 150):
        """
        Initialize animator.
        
        Args:
            fps: Frames per second for animations
            dpi: Resolution for output images
        """
        self.fps = fps
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib dark theme styling."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': self.COLOR_BG,
            'axes.facecolor': self.COLOR_BG_LIGHTER,
            'axes.edgecolor': self.COLOR_GRID,
            'axes.labelcolor': self.COLOR_TEXT,
            'axes.titlecolor': self.COLOR_TITLE,
            'xtick.color': self.COLOR_TEXT,
            'ytick.color': self.COLOR_TEXT,
            'text.color': self.COLOR_TEXT,
            'grid.color': self.COLOR_GRID,
            'grid.alpha': 0.3,
            'legend.facecolor': self.COLOR_BG_LIGHTER,
            'legend.edgecolor': self.COLOR_GRID,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
        })
    
    def create_static_plot(self, result: Dict[str, Any], filepath: str,
                           title: str = "Ma-Chen Chaotic System"):
        """
        Create static time series plot.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Plot title
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(16, 14), facecolor=self.COLOR_BG)
        fig.suptitle(f'{title}\nSensitivity to Initial Conditions',
                    fontsize=22, fontweight='bold', color=self.COLOR_TITLE, y=0.97)
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1],
                             hspace=0.35, left=0.08, right=0.96, top=0.90, bottom=0.06)
        
        time = result['time']
        labels = [
            ('Interest Rate', '$x(t)$', 'x'),
            ('Investment Demand', '$y(t)$', 'y'),
            ('Price Index', '$z(t)$', 'z')
        ]
        
        for i, (name, ylabel, key) in enumerate(labels):
            ax = fig.add_subplot(gs[i, 0], facecolor=self.COLOR_BG_LIGHTER)
            
            # Glow effect
            ax.plot(time, result['sol_A'][key],
                   color=self.COLOR_A, lw=4, alpha=0.2)
            ax.plot(time, result['sol_B'][key],
                   color=self.COLOR_B, lw=4, alpha=0.2)
            
            # Main lines
            ax.plot(time, result['sol_A'][key],
                   color=self.COLOR_A, lw=1.8, label='Economy A', alpha=0.95)
            ax.plot(time, result['sol_B'][key],
                   color=self.COLOR_B, lw=1.8, label='Economy B (Perturbed)', alpha=0.95)
            
            ax.set_title(f'{name} - Chaotic Dynamics', fontsize=16,
                        fontweight='bold', color=self.COLOR_TITLE, pad=12)
            ax.set_ylabel(ylabel, fontsize=15, fontweight='bold', color=self.COLOR_TEXT)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=12, width=1.5, length=6)
            ax.grid(True, alpha=0.3, color=self.COLOR_GRID, linestyle='-', linewidth=0.8)
            
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(1.5)
            
            if i == 0:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=12,
                         facecolor=self.COLOR_BG_LIGHTER, edgecolor=self.COLOR_GRID)
            if i == 2:
                ax.set_xlabel('Time', fontsize=15, fontweight='bold', color=self.COLOR_TEXT)
        
        plt.savefig(filepath, dpi=self.dpi, facecolor=self.COLOR_BG,
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    def create_animation(self, result: Dict[str, Any], filepath: str,
                         title: str = "Ma-Chen Chaotic System", skip: int = 8):
        """
        Create animated 3D phase space visualization.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Animation title
            skip: Frame skip factor (use every nth point)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        time = result['time']
        sol_A = result['sol_A']
        sol_B = result['sol_B']
        
        # Setup figure
        fig = plt.figure(figsize=(14, 12), facecolor=self.COLOR_BG)
        ax = fig.add_subplot(111, projection='3d', facecolor=self.COLOR_BG)
        
        # Fixed axis limits
        pad = 0.15
        x_range = [min(sol_A['x'].min(), sol_B['x'].min()) - pad,
                  max(sol_A['x'].max(), sol_B['x'].max()) + pad]
        y_range = [min(sol_A['y'].min(), sol_B['y'].min()) - pad,
                  max(sol_A['y'].max(), sol_B['y'].max()) + pad]
        z_range = [min(sol_A['z'].min(), sol_B['z'].min()) - pad,
                  max(sol_A['z'].max(), sol_B['z'].max()) + pad]
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Aesthetics
        ax.set_xlabel("Interest Rate ($x$)", fontsize=14, fontweight='bold',
                     color=self.COLOR_TEXT, labelpad=14)
        ax.set_ylabel("Investment ($y$)", fontsize=14, fontweight='bold',
                     color=self.COLOR_TEXT, labelpad=14)
        ax.set_zlabel("Prices ($z$)", fontsize=14, fontweight='bold',
                     color=self.COLOR_TEXT, labelpad=14)
        ax.set_title(f"{title}\nButterfly Effect in Financial Dynamics",
                    fontsize=18, color=self.COLOR_TITLE, fontweight='bold', pad=20)
        
        # Dark 3D panes
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor(self.COLOR_BG_LIGHTER)
        ax.yaxis.pane.set_facecolor(self.COLOR_BG_LIGHTER)
        ax.zaxis.pane.set_facecolor(self.COLOR_BG_LIGHTER)
        ax.xaxis.pane.set_edgecolor(self.COLOR_GRID)
        ax.yaxis.pane.set_edgecolor(self.COLOR_GRID)
        ax.zaxis.pane.set_edgecolor(self.COLOR_GRID)
        ax.xaxis._axinfo['grid']['color'] = self.COLOR_GRID
        ax.yaxis._axinfo['grid']['color'] = self.COLOR_GRID
        ax.zaxis._axinfo['grid']['color'] = self.COLOR_GRID
        ax.tick_params(colors=self.COLOR_TEXT, labelsize=11, width=1.5, length=6)
        
        # Initialize plot elements
        glow_A, = ax.plot([], [], [], lw=4.0, color=self.COLOR_A, alpha=0.25)
        glow_B, = ax.plot([], [], [], lw=4.0, color=self.COLOR_B, alpha=0.25)
        trace_A, = ax.plot([], [], [], lw=1.2, color=self.COLOR_A, alpha=0.8, label='Economy A')
        trace_B, = ax.plot([], [], [], lw=1.2, color=self.COLOR_B, alpha=0.8, label='Economy B')
        trail_A, = ax.plot([], [], [], lw=3.0, color=self.COLOR_A, alpha=1.0)
        trail_B, = ax.plot([], [], [], lw=3.0, color=self.COLOR_B, alpha=1.0)
        head_A, = ax.plot([], [], [], 'o', color=self.COLOR_A, markersize=12,
                         markeredgecolor='white', markeredgewidth=2)
        head_B, = ax.plot([], [], [], 'o', color=self.COLOR_B, markersize=12,
                         markeredgecolor='white', markeredgewidth=2)
        
        ax.legend(loc='upper left', framealpha=0.9, fontsize=13,
                 facecolor=self.COLOR_BG_LIGHTER, edgecolor=self.COLOR_GRID)
        
        time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                             fontsize=14, fontweight='bold', color=self.COLOR_ACCENT)
        
        # Pre-compute frame indices
        n_points = len(time)
        frame_indices = list(range(0, n_points, skip))
        n_frames = len(frame_indices)
        
        print(f"      Preparing {n_frames} frames...")
        
        def init():
            for line in [glow_A, glow_B, trace_A, trace_B, trail_A, trail_B]:
                line.set_data([], [])
                line.set_3d_properties([])
            for head in [head_A, head_B]:
                head.set_data([], [])
                head.set_3d_properties([])
            time_text.set_text('')
            return (glow_A, glow_B, trace_A, trace_B, trail_A, trail_B,
                   head_A, head_B, time_text)
        
        def update(frame):
            idx = frame_indices[frame]
            
            # Glow layer
            glow_A.set_data(sol_A['x'][:idx], sol_A['y'][:idx])
            glow_A.set_3d_properties(sol_A['z'][:idx])
            glow_B.set_data(sol_B['x'][:idx], sol_B['y'][:idx])
            glow_B.set_3d_properties(sol_B['z'][:idx])
            
            # Full trace
            trace_A.set_data(sol_A['x'][:idx], sol_A['y'][:idx])
            trace_A.set_3d_properties(sol_A['z'][:idx])
            trace_B.set_data(sol_B['x'][:idx], sol_B['y'][:idx])
            trace_B.set_3d_properties(sol_B['z'][:idx])
            
            # Recent trail
            trail_len = 200
            start = max(0, idx - trail_len)
            trail_A.set_data(sol_A['x'][start:idx], sol_A['y'][start:idx])
            trail_A.set_3d_properties(sol_A['z'][start:idx])
            trail_B.set_data(sol_B['x'][start:idx], sol_B['y'][start:idx])
            trail_B.set_3d_properties(sol_B['z'][start:idx])
            
            # Current position heads
            head_A.set_data([sol_A['x'][idx]], [sol_A['y'][idx]])
            head_A.set_3d_properties([sol_A['z'][idx]])
            head_B.set_data([sol_B['x'][idx]], [sol_B['y'][idx]])
            head_B.set_3d_properties([sol_B['z'][idx]])
            
            # Time annotation
            time_text.set_text(f't = {time[idx]:.1f}')
            
            # Rotation
            ax.view_init(elev=20 + 5 * np.sin(frame * 0.02), azim=frame * 0.3)
            
            return (glow_A, glow_B, trace_A, trace_B, trail_A, trail_B,
                   head_A, head_B, time_text)
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                           blit=False, interval=1000//self.fps)
        
        # Progress tracking wrapper for saving
        class ProgressCallback:
            def __init__(self, total):
                self.pbar = tqdm(total=total, desc="      Saving",
                               ncols=70,
                               bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}')
            
            def __call__(self, current_frame, total_frames):
                self.pbar.update(1)
            
            def close(self):
                self.pbar.close()
        
        progress = ProgressCallback(n_frames)
        
        # Save GIF
        writer = PillowWriter(fps=self.fps)
        ani.save(str(filepath), writer=writer,
                savefig_kwargs={'facecolor': self.COLOR_BG, 'edgecolor': 'none'},
                progress_callback=progress)
        progress.close()
        
        plt.close(fig)
