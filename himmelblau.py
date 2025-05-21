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
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
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
plt.text(-3, 2, r'$\mathbf{(x_1 + 2)^2 - x_2 \geq 0}$', color='white', fontsize=14)
plt.text(0, -3, r'$\mathbf{-4x_1 + 10x_2 \geq 0}$', color='white', fontsize=14)
plt.text(0.6, 1.5, r'$\mathbf{x_1 - \frac{3}{2} x_2 = 0}$', color='orange', fontsize=14)

# Find stationary point(s) with constraints
constraints = [
    {'type': 'eq', 'fun': constraint_eq},
    {'type': 'ineq', 'fun': constraint_ineq1},
    {'type': 'ineq', 'fun': constraint_ineq2}
]

# List of initial guesses
initial_guesses = [
    np.array([1.0, -1.0]),
    np.array([-2.0, 3.0]),
    np.array([3.0, 2.0]),
    np.array([-3.0, -3.0])
]

# Run optimization for each initial guess and plot the results
for x0 in initial_guesses:
    res = minimize(f, x0, method='SLSQP')
    if res.success:
        x_opt = res.x
        plt.plot(x_opt[0], x_opt[1], 'go', label='Stationary Points')
        plt.text(
            x_opt[0] + 0.2, x_opt[1],
            f'({x_opt[0]:.2f}, {x_opt[1]:.2f})',
            color='green',
            fontsize=13,
            fontweight='bold',
            ha='left',
            va='center'
        )
    else:
        print("Optimization failed:", res.message)

#Run optimization for constraints
x0 = np.array([0.0, 0.0])  # initial guess
res = minimize(f, x0, method='SLSQP', constraints=constraints)
if res.success:
    x_opt = res.x
    plt.plot(x_opt[0], x_opt[1], 'bo', label='Minimum Point')
    plt.text(
        x_opt[0] + 0.2, x_opt[1],
        f'({x_opt[0]:.2f}, {x_opt[1]:.2f})',
        color='blue',
        fontsize=13,
        fontweight='bold',
        ha='left',
        va='center'
    )

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Himmelblau\'s Function with Constraints')

# Custom legend handles
red_constraint = Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Inequality Constraint')
black_constraint = Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Equality Constraint')
green_point = Line2D([0], [0], marker='o', color='green', linestyle='None', label='Stationary Point')
blue_point = Line2D([0], [0], marker='o', color='blue', linestyle='None', label='Minimizer')

# Add them to the legend
plt.legend(handles=[red_constraint, black_constraint, green_point, blue_point])
plt.grid(True)
plt.show()
