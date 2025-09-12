"""
Trilateration for measuring the transom of a moore 24
"""
# %%
from trilateration import Trilateration2D
import numpy as np
import matplotlib.pyplot as plt

# %%
tril = Trilateration2D(
    distance_AB=,
    distance_AC=
    distance_BC=
)

# add points with measured distances
points = [
    (, , ),
    (, , ),
    (, , ),
    (, , ),
    (, , ),
    (, , ),
    (, , )
]

for dist_A, dist_B, dist_C in points:
    x, y, error = tril.add_point(dist_A, dist_B, dist_C)
    print(f"Added point at [{x:.3f}, {y:.3f}] with error {error:.6f}")

# plot the results
fig = tril.plot_points(figsize=(10, 8), show_distances=True)
plt.savefig('transom_trilateration.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Print DataFrame of results
df = tril.get_dataframe()
df.head(10)