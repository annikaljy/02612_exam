import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# --- Load real data ---
data = scipy.io.loadmat('LP_test.mat')
U = data['U'].flatten().astype(float)
C = data['C'].flatten().astype(float)
Pd = data['Pd_max'].flatten().astype(float)
Pg = data['Pg_max'].flatten().astype(float)

n_demand = len(U)
n_gen = len(C)

# --- Create model ---
model = gp.Model("market_clearing")

# Decision variables
pd = model.addVars(n_demand, lb=0, ub=Pd.tolist(), name="pd")
pg = model.addVars(n_gen, lb=0, ub=Pg.tolist(), name="pg")

# Objective
model.setObjective(
    gp.quicksum(-U[i] * pd[i] for i in range(n_demand)) +
    gp.quicksum(C[j] * pg[j] for j in range(n_gen)),
    GRB.MINIMIZE
)

# Power balance constraint
model.addConstr(
    gp.quicksum(pd[i] for i in range(n_demand)) -
    gp.quicksum(pg[j] for j in range(n_gen)) == 0,
    "power_balance"
)

# --- Solve ---
model.optimize()

# --- Results ---
if model.status == GRB.OPTIMAL:
    print("\nOptimal Objective Value:", model.objVal)

    p_demand = np.array([pd[i].X for i in range(n_demand)])
    p_generation = np.array([pg[j].X for j in range(n_gen)])

    for constr in model.getConstrs():
        if constr.ConstrName == "power_balance":
            market_price = constr.Pi
            print("\nMarket Clearing Price: €", round(market_price, 2))

    print("\nPower Demand:")
    print(p_demand)
    print("\nPower Generation:")
    print(p_generation)

    # Plot Supply vs Demand
    plt.figure(figsize=(10, 6))
    sorted_demand = np.sort(U)[::-1]
    sorted_supply = np.sort(C)
    cum_demand = np.cumsum(Pd[np.argsort(-U)])
    cum_supply = np.cumsum(Pg[np.argsort(C)])
    plt.step(cum_demand, sorted_demand, where='post', label='Demand Curve')
    plt.step(cum_supply, sorted_supply, where='post', label='Supply Curve')
    plt.axhline(market_price, color='red', linestyle='--', label=f'Market Price €{round(market_price,2)}')
    plt.xlabel('Quantity (MW)')
    plt.ylabel('Price (€/MW)')
    plt.title('Market Clearing Supply vs Demand')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("\nOptimization was not successful.")

x_gurobi = np.concatenate([p_demand,p_generation])
np.save("x_gurobi.npy", x_gurobi)

