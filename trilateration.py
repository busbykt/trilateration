"""
2D Trilateration Implementation

This module provides a class for performing trilateration in 2D space with three reference points.
The class can calculate positions of additional points based on distance measurements and provides
visualization and data analysis capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple, Optional
import warnings


class Trilateration2D:
    """
    A class for performing 2D trilateration with three reference points.
    
    Point A is fixed at origin [0, 0]
    Point B is placed at [distance_AB, 0]
    Point C coordinates are calculated from the three distance measurements
    """
    
    def __init__(self, distance_AB: float, distance_AC: float, distance_BC: float):
        """
        Initialize the trilateration system with three distance measurements.
        
        Args:
            distance_AB: Distance between point A and point B
            distance_AC: Distance between point A and point C
            distance_BC: Distance between point B and point C
        """
        self.distance_AB = distance_AB
        self.distance_AC = distance_AC
        self.distance_BC = distance_BC
        
        # Fixed positions for points A and B
        self.point_A = np.array([0.0, 0.0])
        self.point_B = np.array([distance_AB, 0.0])
        
        # Calculate position of point C
        self.point_C = self._calculate_point_C()
        
        # Store additional points
        self.additional_points = []  # List of tuples: (x, y, dist_A, dist_B, dist_C, error)
        
    def _calculate_point_C(self) -> np.ndarray:
        """
        Calculate the coordinates of point C using the law of cosines.
        
        Returns:
            numpy array with [x, y] coordinates of point C
        """
        # Using law of cosines to find angle at A
        # cos(A) = (b² + c² - a²) / (2bc)
        # where a = BC, b = AC, c = AB
        
        cos_angle_A = (self.distance_AC**2 + self.distance_AB**2 - self.distance_BC**2) / (2 * self.distance_AC * self.distance_AB)
        
        # Clamp to valid range for arccos
        cos_angle_A = np.clip(cos_angle_A, -1.0, 1.0)
        
        angle_A = np.arccos(cos_angle_A)
        
        # Calculate C coordinates
        x_C = self.distance_AC * np.cos(angle_A)
        y_C = self.distance_AC * np.sin(angle_A)
        
        return np.array([x_C, y_C])
    
    def _distance_error_function(self, position: np.ndarray, target_distances: Tuple[float, float, float]) -> float:
        """
        Calculate the sum of squared distance errors for optimization.
        
        Args:
            position: [x, y] coordinates to evaluate
            target_distances: (dist_A, dist_B, dist_C) target distances
            
        Returns:
            Sum of squared distance errors
        """
        x, y = position
        dist_A_target, dist_B_target, dist_C_target = target_distances
        
        # Calculate actual distances to reference points
        dist_A_actual = np.sqrt(x**2 + y**2)
        dist_B_actual = np.sqrt((x - self.point_B[0])**2 + y**2)
        dist_C_actual = np.sqrt((x - self.point_C[0])**2 + (y - self.point_C[1])**2)
        
        # Calculate squared errors
        error_A = (dist_A_actual - dist_A_target)**2
        error_B = (dist_B_actual - dist_B_target)**2
        error_C = (dist_C_actual - dist_C_target)**2
        
        return error_A + error_B + error_C
    
    def add_point(self, distance_A: float, distance_B: float, distance_C: float, 
                  initial_guess: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
        """
        Add a new point by minimizing distance errors from the three reference points.
        
        Args:
            distance_A: Distance from point A
            distance_B: Distance from point B
            distance_C: Distance from point C
            initial_guess: Optional initial guess for optimization (x, y)
            
        Returns:
            Tuple of (x, y, error) for the calculated point position and total error
        """
        # Default initial guess at centroid of reference points
        if initial_guess is None:
            initial_guess = ((self.point_A[0] + self.point_B[0] + self.point_C[0]) / 3,
                           (self.point_A[1] + self.point_B[1] + self.point_C[1]) / 3)
        
        # Optimize to find best position
        result = minimize(
            self._distance_error_function,
            initial_guess,
            args=((distance_A, distance_B, distance_C),),
            method='BFGS'
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        x, y = result.x
        error = result.fun
        
        # Store the point
        self.additional_points.append((x, y, distance_A, distance_B, distance_C, error))
        
        return x, y, error
    
    def plot_points(self, figsize: Tuple[int, int] = (10, 8), show_distances: bool = True) -> plt.Figure:
        """
        Plot all reference points and additional points.
        
        Args:
            figsize: Figure size (width, height)
            show_distances: Whether to show distance circles
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot reference points
        ax.plot(self.point_A[0], self.point_A[1], 'ro', markersize=12, label='Point A (Origin)')
        ax.plot(self.point_B[0], self.point_B[1], 'go', markersize=12, label='Point B')
        ax.plot(self.point_C[0], self.point_C[1], 'bo', markersize=12, label='Point C')
        
        # Add labels for reference points
        ax.annotate('A', (self.point_A[0], self.point_A[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
        ax.annotate('B', (self.point_B[0], self.point_B[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
        ax.annotate('C', (self.point_C[0], self.point_C[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Plot additional points
        if self.additional_points:
            for i, (x, y, _, _, _, error) in enumerate(self.additional_points):
                ax.plot(x, y, 'ko', markersize=8, alpha=0.7)
                ax.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
                
                # Show distance circles if requested
                if show_distances and i < 5:  # Limit to first 5 points to avoid clutter
                    _, _, dist_A, dist_B, dist_C, _ = self.additional_points[i]
                    
                    # Distance circles (semi-transparent)
                    circle_A = plt.Circle(self.point_A, dist_A, fill=False, 
                                        linestyle='--', alpha=0.3, color='red')
                    circle_B = plt.Circle(self.point_B, dist_B, fill=False, 
                                        linestyle='--', alpha=0.3, color='green')
                    circle_C = plt.Circle(self.point_C, dist_C, fill=False, 
                                        linestyle='--', alpha=0.3, color='blue')
                    
                    ax.add_patch(circle_A)
                    ax.add_patch(circle_B)
                    ax.add_patch(circle_C)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('2D Trilateration - Reference Points and Calculated Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame with all points and their properties.
        
        Returns:
            DataFrame with columns: Point, X, Y, Distance_A, Distance_B, Distance_C, Error
        """
        data = []
        
        # Add reference points
        data.append({
            'Point': 'A (Reference)',
            'X': self.point_A[0],
            'Y': self.point_A[1],
            'Distance_A': 0.0,
            'Distance_B': self.distance_AB,
            'Distance_C': self.distance_AC,
            'Error': 0.0
        })
        
        data.append({
            'Point': 'B (Reference)',
            'X': self.point_B[0],
            'Y': self.point_B[1],
            'Distance_A': self.distance_AB,
            'Distance_B': 0.0,
            'Distance_C': self.distance_BC,
            'Error': 0.0
        })
        
        data.append({
            'Point': 'C (Reference)',
            'X': self.point_C[0],
            'Y': self.point_C[1],
            'Distance_A': self.distance_AC,
            'Distance_B': self.distance_BC,
            'Distance_C': 0.0,
            'Error': 0.0
        })
        
        # Add calculated points
        for i, (x, y, dist_A, dist_B, dist_C, error) in enumerate(self.additional_points):
            data.append({
                'Point': f'P{i+1}',
                'X': x,
                'Y': y,
                'Distance_A': dist_A,
                'Distance_B': dist_B,
                'Distance_C': dist_C,
                'Error': error
            })
        
        return pd.DataFrame(data)
    
    def get_reference_points(self) -> dict:
        """
        Get the coordinates of the three reference points.
        
        Returns:
            Dictionary with reference point coordinates
        """
        return {
            'A': self.point_A.tolist(),
            'B': self.point_B.tolist(),
            'C': self.point_C.tolist()
        }
    
    def clear_additional_points(self):
        """Clear all additional points."""
        self.additional_points.clear()


def main():
    """
    Example usage of the Trilateration2D class.
    """
    # Example: Create a trilateration system
    print("Creating trilateration system with distances:")
    print("A to B: 10.0")
    print("A to C: 8.0") 
    print("B to C: 6.0")
    
    tril = Trilateration2D(distance_AB=10.0, distance_AC=8.0, distance_BC=6.0)
    
    print(f"\nReference points calculated:")
    ref_points = tril.get_reference_points()
    for point, coords in ref_points.items():
        print(f"Point {point}: [{coords[0]:.3f}, {coords[1]:.3f}]")
    
    # Add some example points
    print("\nAdding points with known distances:")
    
    # Add point 1
    x1, y1, error1 = tril.add_point(distance_A=5.0, distance_B=7.0, distance_C=4.0)
    print(f"Point 1: [{x1:.3f}, {y1:.3f}], Error: {error1:.6f}")
    
    # Add point 2
    x2, y2, error2 = tril.add_point(distance_A=3.0, distance_B=9.0, distance_C=6.0)
    print(f"Point 2: [{x2:.3f}, {y2:.3f}], Error: {error2:.6f}")
    
    # Display DataFrame
    print("\nAll points data:")
    df = tril.get_dataframe()
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Create plot
    print("\nGenerating plot...")
    fig = tril.plot_points()
    plt.show()


if __name__ == "__main__":
    main()
