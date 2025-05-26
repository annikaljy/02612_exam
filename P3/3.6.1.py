import numpy as np
import matplotlib.pyplot as plt

# Example: replace with your actual computed solutions
x_gurobi = np.load("x_gurobi.npy")
x_simplex = np.load("x_simplex.npy")

#print(x_gurobi)
diff = x_gurobi - x_simplex



print(diff[30:38])

plt.figure(figsize=(10, 4))
plt.bar(range(len(diff)), diff)
plt.title("Difference between gurobi and simplex solution")
plt.xlabel("variable index")
plt.ylabel("difference")
plt.grid(True)
plt.savefig("gurobi_vs_simplex.png", bbox_inches='tight')
plt.show()