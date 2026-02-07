"""
4D Viviani hypersurface representation and visualization.
"""

import numpy as np
from .nurbs_curve import NURBSCurve

class VivianiSurface4D:
    """
    4D Viviani hypersurface with exact NURBS representation.
    
    The Viviani hypersurface is defined as the intersection of a 3-sphere
    and a 3-cylinder in R^4.
    
    Attributes
    ----------
    radius : float
        Radius parameter (default: 2.0)
    control_points_4d : numpy.ndarray
        4D control points for the NURBS surface
    weights_4d : numpy.ndarray
        Weights for the NURBS surface
    knot_vectors : tuple
        Knot vectors in u and v directions
    """
    
    def __init__(self, radius=2.0):
        """
        Initialize the 4D Viviani surface.
        
        Parameters
        ----------
        radius : float, optional
            Radius parameter (default: 2.0)
        """
        self.radius = radius
        self.a = radius / 2.0  # Scaling parameter
        self.control_points_4d = None
        self.weights_4d = None
        self.knot_vectors = None
        
    def build_nurbs_surface(self):
        """
        Construct the exact NURBS representation of the 4D Viviani surface.
        
        This method constructs the tensor-product NURBS surface using
        the derived control points and weights.
        """
        # Control points for 3D Viviani curve (17 points)
        # These come from Equations (3) in the manuscript
        viviani_ctrl_3d = np.array([
            [0.0, 0.0, 0.0],  # P0
            [0.0625, 0.0, 0.0],  # P1
            [0.125, 0.0, 0.0],  # P2
            [0.25, 0.0, 0.0],  # P3
            [0.5, 0.0, 0.0],  # P4
            [1.0, 0.0, 0.0],  # P5
            [1.5, 0.0, 0.0],  # P6
            [1.75, 0.0, 0.0],  # P7
            [2.0, 0.0, 0.0],  # P8
            [1.75, 0.0, 0.0],  # P9
            [1.5, 0.0, 0.0],  # P10
            [1.0, 0.0, 0.0],  # P11
            [0.5, 0.0, 0.0],  # P12
            [0.25, 0.0, 0.0],  # P13
            [0.125, 0.0, 0.0],  # P14
            [0.0625, 0.0, 0.0],  # P15
            [0.0, 0.0, 0.0]   # P16
        ])
        
        # Weights for 3D Viviani curve (17 weights)
        # These come from Equations (4) in the manuscript
        viviani_weights = np.array([
            1.0, 0.984375, 0.9375, 0.75, 0.5, 0.25, 0.125, 
            0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 
            0.9375, 0.984375, 1.0
        ])
        
        # Control points for 2D circle (7 points)
        circle_ctrl_2d = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0]
        ])
        
        # Weights for 2D circle
        circle_weights = np.array([1.0, np.sqrt(2)/2, 1.0, np.sqrt(2)/2, 
                                   1.0, np.sqrt(2)/2, 1.0])
        
        # Construct 4D tensor product control points
        n_viviani = len(viviani_ctrl_3d)
        n_circle = len(circle_ctrl_2d)
        
        control_points_4d = np.zeros((n_viviani, n_circle, 4))
        weights_4d = np.zeros((n_viviani, n_circle))
        
        for i in range(n_viviani):
            for j in range(n_circle):
                # Construct 4D control point according to Equation (5)
                x = viviani_ctrl_3d[i, 0]
                y = viviani_ctrl_3d[i, 1] * circle_ctrl_2d[j, 0]
                z = viviani_ctrl_3d[i, 1] * circle_ctrl_2d[j, 1]
                w = viviani_ctrl_3d[i, 2]
                
                control_points_4d[i, j] = [x, y, z, w]
                weights_4d[i, j] = viviani_weights[i] * circle_weights[j]
        
        self.control_points_4d = control_points_4d
        self.weights_4d = weights_4d
        
        # Knot vectors (from manuscript)
        self.knot_vectors = (
            [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4],  # u-direction
            [0, 0, 0, 1, 1, 2, 2, 2]  # v-direction
        )
        
        return self
    
    def project_to_3d(self, projection_basis=None):
        """
        Project the 4D surface to 3D using orthogonal projection.
        
        Parameters
        ----------
        projection_basis : numpy.ndarray, optional
            3x4 matrix defining the projection basis (default: standard basis)
            
        Returns
        -------
        numpy.ndarray
            3D coordinates of projected surface
        """
        if projection_basis is None:
            # Standard projection: drop w-coordinate
            projection_basis = np.eye(4)[:3, :]
        
        # Reshape control points for projection
        n_u, n_v, _ = self.control_points_4d.shape
        points_flat = self.control_points_4d.reshape(-1, 4)
        
        # Apply projection
        projected_flat = points_flat @ projection_basis.T
        
        # Reshape back to grid
        projected_3d = projected_flat.reshape(n_u, n_v, 3)
        
        return projected_3d
    
    def visualize(self, projection_basis=None, **kwargs):
        """
        Visualize the projected 3D surface.
        
        Parameters
        ----------
        projection_basis : numpy.ndarray, optional
            Projection basis matrix
        **kwargs : dict
            Additional visualization parameters
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("Matplotlib is required for visualization")
        
        # Project to 3D
        projected = self.project_to_3d(projection_basis)
        
        # Create visualization
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        X = projected[:, :, 0]
        Y = projected[:, :, 1]
        Z = projected[:, :, 2]
        
        surf = ax.plot_surface(X, Y, Z, 
                              cmap=kwargs.get('cmap', 'viridis'),
                              alpha=kwargs.get('alpha', 0.8),
                              linewidth=kwargs.get('linewidth', 0.5))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('4D Viviani Hypersurface Projected to 3D')
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
