import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

# Define the Beale function
def f(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

# Grid for plotting
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Evaluate constraints on the grid
ineq1 = X**2 + Y**2 <= 12

feasible = ineq1

# Plotting
plt.figure(figsize=(10, 8))
levels = np.logspace(-1, 5, 30)  # Log-spaced levels to better capture large variations
contours = plt.contour(X, Y, Z, levels=levels, colors='black')


# Mask Z outside feasible region
Z_masked = np.ma.masked_where(~feasible, Z)

# Plot colored contours only in feasible region
from matplotlib.colors import LogNorm

plt.contourf(X, Y, Z_masked, levels=20, cmap='viridis', norm=LogNorm(), alpha=0.7)

# Plot constraint boundary
plt.contour(X, Y, X**2 + Y**2 - 12, levels=[0], colors='red', linestyles='--', linewidths=2)



# Plot constraint lines/bounds
plt.contour(X, Y, X**2 + Y**2 - 12, levels=[0], colors='red', linestyles='--', linewidths=2)

# Masked array: 1 where constraint is satisfied, np.nan elsewhere
red_region = np.where(X**2 + Y**2 <= 12, np.nan, 1)

# Now plot them
plt.contourf(
    X, Y,
    red_region,
    colors=['#ff0000'],  # pure red
    alpha=1,
    extend='neither',
    label='Inequality'
)
# Labels for constraints
# plt.text(-3, 2, r'$\mathbf{(x_1 + 2)^2 - x_2 \geq 0}$', color='white', fontsize=14)
# plt.text(0, -3, r'$\mathbf{-4x_1 + 10x_2 \geq 0}$', color='white', fontsize=14)
# plt.text(0.6, 1.5, r'$\mathbf{x_1 - \frac{3}{2} x_2 = 0}$', color='orange', fontsize=14)



name = "x_tr_bfgs_True_beale"
x_coord = np.loadtxt(f'data/{name}.txt')

#plot the path of solution
plt.plot(x_coord[:, 0], x_coord[:, 1], 'b-', label='Path of solution', linewidth=2)
for x in x_coord[:-1]:
    plt.plot(x[0], x[1], 'bo', markersize=5)
plt.plot(x_coord[-1, 0], x_coord[-1, 1], '*', color ='purple', markersize=8, label='Final point')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Beale\'s Function with Constraints')

# Custom legend handles
red_constraint = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Inequality Constraint')

green_point = Line2D([0], [0], marker='o', color='blue', linestyle='None', label='Solution Path', markersize=8)
yellow_point = Line2D([0], [0], marker='*', color='purple', linestyle='None', label='Final Point', markersize=8)

# Add them to the legend
plt.legend(handles=[red_constraint, green_point, yellow_point], loc='upper right', fontsize=12)
plt.grid(True)
plt.savefig(f'figures/{name}.png', dpi=300)
# plt.show()
