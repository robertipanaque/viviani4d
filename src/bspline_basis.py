"""
B-spline basis function evaluation using Cox-de Boor recurrence relation.
"""

import numpy as np

class BSplineBasis:
    """
    B-spline basis function evaluator.
    
    Implements the Cox-de Boor algorithm for evaluating B-spline basis functions.
    """
    
    @staticmethod
    def basis_function(i, p, u, knots):
        """
        Evaluate the i-th B-spline basis function of degree p at parameter u.
        
        Parameters
        ----------
        i : int
            Basis function index (0 ≤ i ≤ n)
        p : int
            Degree of the basis function
        u : float or array
            Parameter value(s) at which to evaluate
        knots : array-like
            Knot vector (non-decreasing)
            
        Returns
        -------
        float or array
            Value of N_{i,p}(u)
        """
        knots = np.asarray(knots, dtype=float)
        
        # Base case: degree 0
        if p == 0:
            if i == len(knots) - 2:  # Last interval
                return np.where((knots[i] <= u) & (u <= knots[i+1]), 1.0, 0.0)
            else:
                return np.where((knots[i] <= u) & (u < knots[i+1]), 1.0, 0.0)
        
        # Recursive case: degree p > 0
        result = 0.0
        
        # First term
        denom1 = knots[i+p] - knots[i]
        if denom1 != 0:
            term1 = (u - knots[i]) / denom1
            result += term1 * BSplineBasis.basis_function(i, p-1, u, knots)
        
        # Second term
        denom2 = knots[i+p+1] - knots[i+1]
        if denom2 != 0:
            term2 = (knots[i+p+1] - u) / denom2
            result += term2 * BSplineBasis.basis_function(i+1, p-1, u, knots)
        
        return result
    
    @staticmethod
    def basis_functions_all(p, u, knots):
        """
        Evaluate all non-zero basis functions of degree p at parameter u.
        
        Parameters
        ----------
        p : int
            Degree of basis functions
        u : float
            Parameter value
        knots : array-like
            Knot vector
            
        Returns
        -------
        list
            Values of all non-zero basis functions at u
        """
        n = len(knots) - p - 1
        basis_vals = []
        
        for i in range(n):
            val = BSplineBasis.basis_function(i, p, u, knots)
            basis_vals.append(val)
        
        return basis_vals
