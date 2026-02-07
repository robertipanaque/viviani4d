"""
NURBS curve evaluation in arbitrary dimensions.
"""

import numpy as np
from .bspline_basis import BSplineBasis

class NURBSCurve:
    """
    NURBS curve evaluator for arbitrary dimensions.
    
    Attributes
    ----------
    control_points : numpy.ndarray
        Array of control points (n+1, dim)
    weights : numpy.ndarray
        Array of weights (n+1,)
    knots : numpy.ndarray
        Knot vector
    degree : int
        Degree of the NURBS curve
    """
    
    def __init__(self, control_points, weights, knots, degree):
        """
        Initialize a NURBS curve.
        
        Parameters
        ----------
        control_points : array-like
            Control points array of shape (n+1, dim)
        weights : array-like
            Weights array of shape (n+1,)
        knots : array-like
            Knot vector (non-decreasing)
        degree : int
            Degree of the curve
        """
        self.control_points = np.asarray(control_points, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.knots = np.asarray(knots, dtype=float)
        self.degree = degree
        
        # Validate dimensions
        n_control = len(self.control_points)
        n_weights = len(self.weights)
        if n_control != n_weights:
            raise ValueError(f"Number of control points ({n_control}) must match number of weights ({n_weights})")
    
    def evaluate(self, u):
        """
        Evaluate the NURBS curve at parameter u.
        
        Parameters
        ----------
        u : float or array
            Parameter value(s)
            
        Returns
        -------
        numpy.ndarray
            Point(s) on the curve
        """
        u = np.asarray(u, dtype=float)
        single_value = u.ndim == 0
        if single_value:
            u = np.array([u])
        
        results = []
        for u_val in u:
            # Find knot span
            n = len(self.control_points) - 1
            m = len(self.knots) - 1
            p = self.degree
            
            # Evaluate basis functions
            basis_vals = BSplineBasis.basis_functions_all(p, u_val, self.knots)
            
            # Compute weighted sum
            numerator = np.zeros(self.control_points.shape[1])
            denominator = 0.0
            
            for i, (point, weight) in enumerate(zip(self.control_points, self.weights)):
                basis_val = basis_vals[i] if i < len(basis_vals) else 0.0
                numerator += basis_val * weight * point
                denominator += basis_val * weight
            
            if denominator == 0:
                raise ValueError("Denominator is zero in NURBS evaluation")
            
            results.append(numerator / denominator)
        
        if single_value:
            return results[0]
        return np.array(results)
    
    def evaluate_grid(self, u_values):
        """
        Evaluate the curve at multiple parameter values.
        
        Parameters
        ----------
        u_values : array-like
            Array of parameter values
            
        Returns
        -------
        numpy.ndarray
            Array of points on the curve
        """
        return self.evaluate(u_values)
