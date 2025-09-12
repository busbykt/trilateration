"""
Test script for the Trilateration2D class.
This script demonstrates the usage and validates the implementation.
"""
# %%
from trilateration import Trilateration2D
import numpy as np
import matplotlib.pyplot as plt

# %%
def test_basic_functionality():
    """Test basic trilateration functionality."""
    print("=" * 50)
    print("Testing Basic Trilateration Functionality")
    print("=" * 50)
    
    # Create a trilateration system with known triangle
    # Using a 3-4-5 right triangle scaled by 2
    tril = Trilateration2D(distance_AB=28.75, distance_AC=39, distance_BC=27.75)
    
    # Get reference points
    ref_points = tril.get_reference_points()
    print(f"\nCalculated reference points:")
    for name, coords in ref_points.items():
        print(f"  Point {name}: [{coords[0]:.3f}, {coords[1]:.3f}]")

    return tril

def test_error_analysis():
    """Test error analysis with noisy measurements."""
    print("\n" + "=" * 50)
    print("Testing Error Analysis with Noisy Measurements")
    print("=" * 50)
    
    # Create trilateration system
    tril = Trilateration2D(distance_AB=28.75, distance_AC=39, distance_BC=27.75)
    
    # Add points with intentionally inconsistent measurements (simulating measurement noise)
    test_cases = [
        (22.25, 27.125, 19.125, "Small measurement errors")
    ]
    
    for dist_A, dist_B, dist_C, description in test_cases:
        x, y, error = tril.add_point(dist_A, dist_B, dist_C)
        print(f"{description}:")
        print(f"  Input distances: A={dist_A}, B={dist_B}, C={dist_C}")
        print(f"  Calculated position: [{x:.3f}, {y:.3f}]")
        print(f"  Total error: {error:.6f}")
        print()
    
    return tril

def test_visualization():
    """Test visualization capabilities."""
    print("=" * 50)
    print("Testing Visualization")
    print("=" * 50)
    
    # Create a trilateration system
    tril = Trilateration2D(distance_AB=28.75, distance_AC=39, distance_BC=27.75)
    
    # Add several points
    points_to_add = [
        (22.25, 27.125, 19.125),
        (11.875, 29.25, 30.25),
    ]
    
    print("Adding points for visualization:")
    for i, (dist_A, dist_B, dist_C) in enumerate(points_to_add):
        x, y, error = tril.add_point(dist_A, dist_B, dist_C)
        print(f"  Point {i+1}: [{x:.3f}, {y:.3f}], Error: {error:.6f}")
    
    # Generate plot
    print("\nGenerating visualization...")
    fig = tril.plot_points(figsize=(12, 10), show_distances=True)
    plt.savefig('trilateration_test.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'trilateration_test.png'")
    
    return tril


def test_dataframe_output():
    """Test DataFrame output functionality."""
    print("\n" + "=" * 50)
    print("Testing DataFrame Output")
    print("=" * 50)
    
    # Use the trilateration system from visualization test
    tril = test_visualization()
    
    # Get DataFrame
    df = tril.get_dataframe()
    
    print("Complete DataFrame:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    print(f"\nSummary statistics:")
    print(f"Total points: {len(df)}")
    print(f"Reference points: {len(df[df['Point'].str.contains('Reference')])}")
    print(f"Calculated points: {len(df[~df['Point'].str.contains('Reference')])}")
    print(f"Average error for calculated points: {df[~df['Point'].str.contains('Reference')]['Error'].mean():.6f}")
    print(f"Max error: {df['Error'].max():.6f}")
    
    return df

# %%
if __name__ == "__main__":
    print("Running Trilateration2D Tests")
    print("=" * 70)
    
    # Run all tests
    tril1 = test_basic_functionality()
    tril2 = test_error_analysis()
    tril3 = test_visualization()
    df = test_dataframe_output()
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("Check 'trilateration_test.png' for the visualization output.")
    
    # Show the plot
    plt.show()
