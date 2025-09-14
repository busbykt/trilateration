"""
Trilateration for measuring the transom of a moore 24
"""
# %%
from trilateration import Trilateration2D
import numpy as np
import matplotlib.pyplot as plt

# %%
tril = Trilateration2D(
    distance_AB=914,
    distance_AC=715,
    distance_BC=631.5
)

# add points with measured distances
points = [
    (160,815,559), # along top edge
    (296,742,419),
    (429,686,285),
    (564,647,150), 
    (167,793,714), # along hull edge
    (323,648,685),
    (520,436,627),
    (734,198,600),
    (727,561,73),
    (725,405,229),
    (739,350,283),
    (751,260,374),
    (779,196,438),
    (745,175,505),
    (749,167,543),
    (772,146,572),
    (148,803,569), # elliptical cutout
    (221,726,504),
    (266,656,504),
    (268,646,556),
    (235,684,599),
    (190,731,624),
    (146,770,634),
    (114,801,631),
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
