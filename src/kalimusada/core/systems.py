"""
Ma-Chen Financial Chaotic System definition.

The Ma-Chen system (2001) models financial dynamics through three coupled ODEs:
    dx/dt = z + (y - a) * x    [Interest Rate]
    dy/dt = 1 - b * y - x^2    [Investment Demand]
    dz/dt = -x - c * z         [Price Index]

References:
    Ma, J. H., & Chen, Y. S. (2001). Study for the bifurcation topological
    structure and the global complicated character of a kind of nonlinear
    finance system. Applied Mathematics and Mechanics, 22(11), 1240-1251.
"""

import numpy as np
from typing import List, Tuple


class MaChenSystem:
    """
    Ma-Chen Financial Chaotic System.
    
    Attributes:
        a: Savings rate parameter (default: 0.9)
        b: Investment cost parameter (default: 0.2)
        c: Elasticity of demand parameter (default: 1.2)
    """
    
    def __init__(self, a: float = 0.9, b: float = 0.2, c: float = 1.2):
        """
        Initialize Ma-Chen system.
        
        Args:
            a: Savings rate parameter
            b: Investment cost parameter
            c: Elasticity of demand parameter
        """
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, t: float, state: List[float]) -> List[float]:
        """
        Compute derivatives for the Ma-Chen system.
        
        Args:
            t: Time (not used, system is autonomous)
            state: [x, y, z] state vector
        
        Returns:
            [dx/dt, dy/dt, dz/dt] derivatives
        """
        x, y, z = state
        
        dxdt = z + (y - self.a) * x
        dydt = 1 - self.b * y - x**2
        dzdt = -x - self.c * z
        
        return [dxdt, dydt, dzdt]
    
    def jacobian(self, state: List[float]) -> np.ndarray:
        """
        Compute Jacobian matrix at given state.
        
        Args:
            state: [x, y, z] state vector
        
        Returns:
            3x3 Jacobian matrix
        """
        x, y, z = state
        
        J = np.array([
            [y - self.a, x, 1],
            [-2*x, -self.b, 0],
            [-1, 0, -self.c]
        ])
        
        return J
    
    def get_equilibria(self) -> List[Tuple[float, float, float]]:
        """
        Find equilibrium points of the system.
        
        Returns:
            List of equilibrium points (x, y, z)
        """
        # At equilibrium: dz/dt = 0 => x = -c*z
        # dy/dt = 0 => y = (1 - x^2) / b
        # dx/dt = 0 => z + (y - a)*x = 0
        
        # This gives a cubic equation in z
        # For typical chaotic parameters, there are 1-3 real equilibria
        
        from scipy.optimize import fsolve
        
        equilibria = []
        
        # Try multiple starting points
        for x0 in [-2, 0, 2]:
            for y0 in [-2, 0, 2]:
                for z0 in [-2, 0, 2]:
                    try:
                        sol = fsolve(
                            lambda s: self(0, s),
                            [x0, y0, z0],
                            full_output=True
                        )
                        point = tuple(sol[0])
                        info = sol[1]
                        
                        # Check if solution is valid
                        residual = np.linalg.norm(self(0, list(point)))
                        if residual < 1e-8:
                            # Check if already found
                            is_new = True
                            for eq in equilibria:
                                if np.linalg.norm(np.array(point) - np.array(eq)) < 1e-6:
                                    is_new = False
                                    break
                            if is_new:
                                equilibria.append(point)
                    except:
                        pass
        
        return equilibria
    
    def __repr__(self) -> str:
        return f"MaChenSystem(a={self.a}, b={self.b}, c={self.c})"
