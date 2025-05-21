import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

# Define the Himmelblau function
def f(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def constraint_eq(x):
    return x[0] - 3/2 * x[1]

def constraint_ineq1(x):
    return (x[0] + 2)**2 - x[1]

def constraint_ineq2(x):
    return -4*x[0] + 10*x[1]

# Grid for plotting
x = np.linspace(1, 5, 400)
y = np.linspace(0, 4, 400)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

# Evaluate constraints on the grid
ineq1 = (X + 2)**2 - Y >= 0
ineq2 = -4*X + 10*Y >= 0
eq = np.isclose(X - 3/2 * Y, 0)

feasible = ineq1 & ineq2 & eq

# Plotting
plt.figure(figsize=(10, 8))
contours = plt.contour(X, Y, Z, levels=50, colors='black')

# Highlight feasible region
plt.contourf(X, Y, feasible, levels=1, colors=['#e7f776'], alpha=0.1)

# Plot constraint lines/bounds
plt.contour(X, Y, (X + 2)**2 - Y, levels=[0], colors='red', linestyles='--', linewidths=2)
plt.contour(X, Y, -4*X + 10*Y, levels=[0], colors='red', linestyles='--', linewidths=2)
plt.contour(X, Y, constraint_eq([X, Y]), levels=[0], colors='orange', linestyles='--', linewidths=2)

# Masked array: 1 where constraint is satisfied, np.nan elsewhere
red_region = np.where((X + 2)**2 - Y >= 0, np.nan, 1)
blue_region = np.where(-4*X + 10*Y >= 0, np.nan, 1)

# Now plot them
plt.contourf(
    X, Y,
    red_region,
    colors=['#ff0000'],  # pure red
    alpha=1,
    extend='neither',
    label='Inequality'
)
plt.contourf(
    X, Y,
    blue_region,
    colors=['#ff0000'],  # pure red
)

# Labels for constraints
# plt.text(-3, 2, r'$\mathbf{(x_1 + 2)^2 - x_2 \geq 0}$', color='white', fontsize=14)
# plt.text(0, -3, r'$\mathbf{-4x_1 + 10x_2 \geq 0}$', color='white', fontsize=14)
# plt.text(0.6, 1.5, r'$\mathbf{x_1 - \frac{3}{2} x_2 = 0}$', color='orange', fontsize=14)

# Find stationary point(s) with constraints
constraints = [
    {'type': 'eq', 'fun': constraint_eq},
    {'type': 'ineq', 'fun': constraint_ineq1},
    {'type': 'ineq', 'fun': constraint_ineq2}
]

name = "x_ls_bfgs_False"
x_coord = np.loadtxt(f'data/{name}.txt')

#plot the path of solution
plt.plot(x_coord[:, 0], x_coord[:, 1], 'g-', label='Path of solution', linewidth=2)
for x in x_coord[:-1]:
    plt.plot(x[0], x[1], 'go', markersize=5)
plt.plot(x_coord[-1, 0], x_coord[-1, 1], '*', color ='purple', markersize=8, label='Final point')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Himmelblau\'s Function with Constraints')

# Custom legend handles
red_constraint = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Inequality Constraint')
black_constraint = Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Equality Constraint')
green_point = Line2D([0], [0], marker='o', color='green', linestyle='None', label='Solution Path', markersize=8)
yellow_point = Line2D([0], [0], marker='*', color='purple', linestyle='None', label='Final Point', markersize=8)

# Add them to the legend
plt.legend(handles=[red_constraint, black_constraint, green_point, yellow_point], loc='upper right', fontsize=12)
plt.grid(True)
plt.savefig(f'figures/{name}.png', dpi=300)
