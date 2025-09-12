# 2D Trilateration Implementation

This Python script implements trilateration in 2D space using three reference points. It's designed for calculating positions of additional points based on distance measurements from the three reference points.

## Features

- **Reference Point Setup**: Point A at origin [0,0], Point B at [AB_distance, 0], Point C calculated automatically
- **Point Addition**: Add new points by specifying distances from A, B, and C
- **Error Minimization**: Uses scipy optimization to minimize distance measurement errors
- **Visualization**: Plot all points with optional distance circles
- **Data Export**: Export all points to a pandas DataFrame with coordinates and error metrics

## Class: Trilateration2D

### Initialization
```python
tril = Trilateration2D(distance_AB, distance_AC, distance_BC)
```

**Parameters:**
- `distance_AB`: Distance between reference points A and B
- `distance_AC`: Distance between reference points A and C  
- `distance_BC`: Distance between reference points B and C

### Key Methods

#### `add_point(distance_A, distance_B, distance_C, initial_guess=None)`
Adds a new point by minimizing distance errors from the three reference points.

**Parameters:**
- `distance_A`: Distance from point A
- `distance_B`: Distance from point B
- `distance_C`: Distance from point C
- `initial_guess`: Optional initial position guess (x, y)

**Returns:** `(x, y, error)` - calculated coordinates and total error

#### `plot_points(figsize=(10, 8), show_distances=True)`
Creates a visualization of all points.

**Parameters:**
- `figsize`: Figure size as (width, height)
- `show_distances`: Whether to show distance circles for first 5 points

**Returns:** matplotlib Figure object

#### `get_dataframe()`
Returns a pandas DataFrame with all points and their properties.

**Columns:**
- `Point`: Point identifier (A, B, C for reference points, P1, P2, etc. for calculated points)
- `X`, `Y`: Coordinates
- `Distance_A`, `Distance_B`, `Distance_C`: Distances to reference points
- `Error`: Total distance error (0 for reference points)

#### `get_reference_points()`
Returns dictionary with coordinates of reference points A, B, and C.

#### `clear_additional_points()`
Removes all calculated points, keeping only reference points.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib scipy
```

## Usage Examples

### Basic Usage
```python
from trilateration import Trilateration2D

# Create trilateration system
tril = Trilateration2D(distance_AB=10.0, distance_AC=8.0, distance_BC=6.0)

# Add a point
x, y, error = tril.add_point(distance_A=5.0, distance_B=7.0, distance_C=4.0)
print(f"Point calculated at: [{x:.3f}, {y:.3f}] with error: {error:.6f}")

# Plot results
tril.plot_points()

# Get data as DataFrame
df = tril.get_dataframe()
print(df)
```

### Advanced Usage with Error Analysis
```python
import matplotlib.pyplot as plt

# Create system
tril = Trilateration2D(12.0, 10.0, 8.0)

# Add multiple points with varying measurement accuracy
points = [
    (6.0, 8.0, 5.0),   # Accurate measurements
    (6.1, 7.9, 5.1),   # Small errors
    (6.5, 7.5, 5.5),   # Larger errors
]

for i, (dA, dB, dC) in enumerate(points):
    x, y, error = tril.add_point(dA, dB, dC)
    print(f"Point {i+1}: [{x:.3f}, {y:.3f}], Error: {error:.6f}")

# Visualize with distance circles
fig = tril.plot_points(show_distances=True)
plt.show()

# Analyze errors
df = tril.get_dataframe()
calculated_points = df[~df['Point'].str.contains('Reference')]
print(f"Average error: {calculated_points['Error'].mean():.6f}")
```

## Mathematical Background

The implementation uses:

1. **Law of Cosines** to calculate Point C coordinates from the three distance measurements
2. **Scipy Optimization** (BFGS method) to minimize sum of squared distance errors when adding new points
3. **Error Metric**: Sum of squared differences between target and actual distances to reference points

## Error Handling

- Triangle inequality violations are handled by clipping cosine values
- Optimization warnings are issued if convergence fails
- Graceful handling of edge cases and degenerate triangles


