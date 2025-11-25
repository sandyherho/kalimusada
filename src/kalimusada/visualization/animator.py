"""
Professional visualization for Ma-Chen Financial Chaotic System.

Creates static time series plots and animated 3D phase space visualizations.
Optimized for smooth, elegant animations with reasonable generation time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
import warnings

warnings.filterwarnings('ignore')

from PIL import Image
import io


class Animator:
    """
    Create professional visualizations for Ma-Chen chaotic dynamics.
    Smooth, elegant animations with optimized rendering.
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
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.linewidth': 1.2,
            'lines.antialiased': True,
        })
    
    def _get_time_tick_spacing(self, t_max: float) -> tuple:
        """Determine appropriate tick spacing based on time range."""
        if t_max <= 50:
            return 5, 5, '{:.0f}'
        elif t_max <= 100:
            return 10, 5, '{:.0f}'
        elif t_max <= 200:
            return 25, 5, '{:.0f}'
        elif t_max <= 300:
            return 50, 5, '{:.0f}'
        elif t_max <= 500:
            return 50, 5, '{:.0f}'
        elif t_max <= 1000:
            return 100, 5, '{:.0f}'
        else:
            return 200, 4, '{:.0f}'
    
    def _smooth_interpolate(self, start: float, end: float, n: int) -> np.ndarray:
        """Create smooth eased interpolation between two values."""
        t = np.linspace(0, 1, n)
        # Smooth ease-in-out curve
        smooth_t = t * t * (3 - 2 * t)
        return start + (end - start) * smooth_t
    
    def create_static_plot(self, result: Dict[str, Any], filepath: str,
                           title: str = "Ma-Chen Chaotic System"):
        """Create static time series plot with full time range."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 14), facecolor=self.COLOR_BG)
        fig.suptitle(f'{title}\nSensitivity to Initial Conditions',
                    fontsize=22, fontweight='bold', color=self.COLOR_TITLE, y=0.97)
        
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1],
                             hspace=0.35, left=0.08, right=0.96, top=0.90, bottom=0.06)
        
        time = result['time']
        t_min, t_max = time[0], time[-1]
        n_points = len(time)
        dt = (t_max - t_min) / (n_points - 1)
        
        major_spacing, minor_divs, tick_fmt = self._get_time_tick_spacing(t_max)
        xlabel = f'Time $t$ (0 → {t_max:.0f} dimensionless units, $\\Delta t$ = {dt:.2e})'
        
        labels = [
            ('Interest Rate', '$x(t)$', 'x'),
            ('Investment Demand', '$y(t)$', 'y'),
            ('Price Index', '$z(t)$', 'z')
        ]
        
        for i, (name, ylabel, key) in enumerate(labels):
            ax = fig.add_subplot(gs[i, 0], facecolor=self.COLOR_BG_LIGHTER)
            
            data_A = result['sol_A'][key]
            data_B = result['sol_B'][key]
            
            # Glow + main lines
            ax.plot(time, data_A, color=self.COLOR_A, lw=4, alpha=0.2)
            ax.plot(time, data_B, color=self.COLOR_B, lw=4, alpha=0.2)
            ax.plot(time, data_A, color=self.COLOR_A, lw=1.5, label='Economy A', alpha=0.95)
            ax.plot(time, data_B, color=self.COLOR_B, lw=1.5, label='Economy B (Perturbed)', alpha=0.95)
            
            ax.set_xlim(t_min, t_max)
            
            y_min = min(data_A.min(), data_B.min())
            y_max = max(data_A.max(), data_B.max())
            y_padding = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
            
            ax.xaxis.set_major_locator(MultipleLocator(major_spacing))
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_divs))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            
            ax.set_title(f'{name} - Chaotic Dynamics', fontsize=16,
                        fontweight='bold', color=self.COLOR_TITLE, pad=12)
            ax.set_ylabel(ylabel, fontsize=15, fontweight='bold', color=self.COLOR_TEXT)
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=12, width=1.5, length=6)
            ax.tick_params(which='minor', colors=self.COLOR_TEXT, width=1, length=3)
            
            ax.grid(True, which='major', alpha=0.4, color=self.COLOR_GRID, linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', alpha=0.15, color=self.COLOR_GRID, linestyle='-', linewidth=0.5)
            
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_linewidth(1.5)
            
            if i == 0:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=12,
                         facecolor=self.COLOR_BG_LIGHTER, edgecolor=self.COLOR_GRID)
            if i == 2:
                ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color=self.COLOR_TEXT)
        
        info_text = f'N = {n_points:,} points | Major ticks: Δ = {major_spacing:.0f}'
        fig.text(0.98, 0.01, info_text, fontsize=10, color=self.COLOR_TEXT,
                ha='right', va='bottom', alpha=0.7)
        
        plt.savefig(filepath, dpi=self.dpi, facecolor=self.COLOR_BG,
                   edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    
    def create_animation(self, result: Dict[str, Any], filepath: str,
                         title: str = "Ma-Chen Chaotic System", skip: int = None):
        """
        Create smooth, elegant animated 3D phase space visualization.
        
        Args:
            result: Simulation result dictionary
            filepath: Output file path
            title: Animation title
            skip: Frame skip (auto-calculated if None)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        time = result['time']
        sol_A = result['sol_A']
        sol_B = result['sol_B']
        t_max = time[-1]
        n_points = len(time)
        
        # Target 1000 frames
        target_frames = 1000
        skip = max(1, n_points // target_frames)
        
        # Frame indices
        frame_indices = np.arange(0, n_points, skip)
        n_frames = len(frame_indices)
        
        # Pre-extract arrays
        x_A, y_A, z_A = sol_A['x'], sol_A['y'], sol_A['z']
        x_B, y_B, z_B = sol_B['x'], sol_B['y'], sol_B['z']
        
        # Fixed axis limits with padding
        pad = 0.2
        x_range = [min(x_A.min(), x_B.min()) - pad, max(x_A.max(), x_B.max()) + pad]
        y_range = [min(y_A.min(), y_B.min()) - pad, max(y_A.max(), y_B.max()) + pad]
        z_range = [min(z_A.min(), z_B.min()) - pad, max(z_A.max(), z_B.max()) + pad]
        
        # Trail length for bright recent path
        trail_len = min(600, n_points // 6)
        
        # Animation settings - balanced quality and speed
        anim_dpi = 100
        fig_size = (11, 9)
        
        print(f"      Generating {n_frames} frames (smooth mode)...")
        
        # Smooth rotation - gentle oscillation
        elevs = 22 + 8 * np.sin(np.linspace(0, 1.5 * np.pi, n_frames))
        azims = self._smooth_interpolate(35, 155, n_frames)
        
        # Subsample factor for historical trace (keeps recent trail full res)
        hist_subsample = max(1, skip // 3)
        
        # Render frames
        frames = []
        
        for i, idx in enumerate(tqdm(frame_indices, 
                                      desc="      Rendering",
                                      ncols=70,
                                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}')):
            
            fig = plt.figure(figsize=fig_size, facecolor=self.COLOR_BG, dpi=anim_dpi)
            ax = fig.add_subplot(111, projection='3d', facecolor=self.COLOR_BG)
            
            # Smooth view angle
            ax.view_init(elev=elevs[i], azim=azims[i])
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            
            # Historical trace boundary
            trail_start = max(0, idx - trail_len)
            hist_end = trail_start
            
            # === LAYER 1: Historical trace (faded, subsampled) ===
            if hist_end > hist_subsample:
                hist_slice = slice(0, hist_end, hist_subsample)
                ax.plot(x_A[hist_slice], y_A[hist_slice], z_A[hist_slice],
                       lw=0.8, color=self.COLOR_A, alpha=0.25)
                ax.plot(x_B[hist_slice], y_B[hist_slice], z_B[hist_slice],
                       lw=0.8, color=self.COLOR_B, alpha=0.25)
            
            # === LAYER 2: Recent trail with glow effect ===
            if idx > trail_start:
                # Outer glow
                ax.plot(x_A[trail_start:idx], y_A[trail_start:idx], z_A[trail_start:idx],
                       lw=6, color=self.COLOR_A, alpha=0.15)
                ax.plot(x_B[trail_start:idx], y_B[trail_start:idx], z_B[trail_start:idx],
                       lw=6, color=self.COLOR_B, alpha=0.15)
                
                # Mid glow
                ax.plot(x_A[trail_start:idx], y_A[trail_start:idx], z_A[trail_start:idx],
                       lw=3.5, color=self.COLOR_A, alpha=0.4)
                ax.plot(x_B[trail_start:idx], y_B[trail_start:idx], z_B[trail_start:idx],
                       lw=3.5, color=self.COLOR_B, alpha=0.4)
                
                # Core line
                ax.plot(x_A[trail_start:idx], y_A[trail_start:idx], z_A[trail_start:idx],
                       lw=1.8, color=self.COLOR_A, alpha=0.95)
                ax.plot(x_B[trail_start:idx], y_B[trail_start:idx], z_B[trail_start:idx],
                       lw=1.8, color=self.COLOR_B, alpha=0.95)
            
            # === LAYER 3: Current position markers with glow ===
            # Outer glow
            ax.scatter([x_A[idx]], [y_A[idx]], [z_A[idx]],
                      s=250, c=self.COLOR_A, alpha=0.3, zorder=9)
            ax.scatter([x_B[idx]], [y_B[idx]], [z_B[idx]],
                      s=250, c=self.COLOR_B, alpha=0.3, zorder=9)
            # Core marker
            ax.scatter([x_A[idx]], [y_A[idx]], [z_A[idx]],
                      s=100, c=self.COLOR_A, edgecolors='white', linewidths=2, zorder=10)
            ax.scatter([x_B[idx]], [y_B[idx]], [z_B[idx]],
                      s=100, c=self.COLOR_B, edgecolors='white', linewidths=2, zorder=10)
            
            # === Labels and styling ===
            ax.set_xlabel("Interest Rate ($x$)", fontsize=11, fontweight='bold',
                         color=self.COLOR_TEXT, labelpad=8)
            ax.set_ylabel("Investment ($y$)", fontsize=11, fontweight='bold',
                         color=self.COLOR_TEXT, labelpad=8)
            ax.set_zlabel("Price Index ($z$)", fontsize=11, fontweight='bold',
                         color=self.COLOR_TEXT, labelpad=8)
            ax.set_title(f"{title}\nButterfly Effect in Financial Dynamics",
                        fontsize=14, color=self.COLOR_TITLE, fontweight='bold', pad=12)
            
            # Time annotation
            ax.text2D(0.02, 0.95, f't = {time[idx]:.1f} / {t_max:.0f}',
                     transform=ax.transAxes, fontsize=12, fontweight='bold',
                     color=self.COLOR_ACCENT,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=self.COLOR_BG_LIGHTER, 
                              edgecolor=self.COLOR_GRID, alpha=0.8))
            
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
            ax.tick_params(colors=self.COLOR_TEXT, labelsize=9)
            
            # Legend
            ax.plot([], [], [], lw=2.5, color=self.COLOR_A, label='Economy A')
            ax.plot([], [], [], lw=2.5, color=self.COLOR_B, label='Economy B (perturbed)')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9,
                     facecolor=self.COLOR_BG_LIGHTER, edgecolor=self.COLOR_GRID)
            
            # Render to buffer
            fig.tight_layout(pad=1.0)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=anim_dpi, 
                       facecolor=self.COLOR_BG, edgecolor='none')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)
        
        # Save GIF
        print(f"      Saving GIF ({n_frames} frames at {self.fps} fps)...")
        
        duration = int(1000 / self.fps)
        
        frames[0].save(
            str(filepath),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            optimize=False
        )
        
        print(f"      Done! Saved to {filepath.name}")
