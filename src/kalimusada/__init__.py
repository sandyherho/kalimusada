"""kalimusada: Ma-Chen Financial Chaotic System Solver"""

__version__ = "0.0.1"
__author__ = "Sandy H. S. Herho"
__license__ = "MIT"

from .core.solver import MaChenSolver
from .core.systems import MaChenSystem
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler

__all__ = [
    "MaChenSolver",
    "MaChenSystem",
    "ConfigManager",
    "DataHandler"
]
