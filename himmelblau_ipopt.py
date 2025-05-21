import numpy as np
import cyipopt

class HimmelBlauProblem(cyipopt.Problem):

    def objective(self, x):
        x1, x2 = x
        return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

    def gradient(self, x):
        x1, x2 = x
        df_dx1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
        df_dx2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
        return np.array([df_dx1, df_dx2])

    def constraints(self, x):
        x1, x2 = x
        return np.array([
            x1 + x2,
            (x1 + 2)**2 - x2,
            -4*x1 + 10*x2
        ])

    def jacobian(self, x):
        x1, x2 = x
        return np.array([
            [1.0, 1.0],
            [2*(x1 + 2), -1.0],
            [-4.0, 10.0]
        ])
    
    def hessianstructure(self):
        return (np.array([0, 1, 1]), np.array([0, 0, 1]))

    def hessian(self, x, lagrange, obj_factor):
        x1, x2 = x

        h11 = 12*x1**2 + 4*x2 - 42
        h12 = 4*x1 + 4*x2
        h22 = 4*x1 + 12*x2**2 - 26
        return np.array([obj_factor * h11,
                        obj_factor * h12,
                        obj_factor * h22])
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                 mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        print(f"Iteration {iter_count}: Objective = {obj_value:.6f}, "
          f"Infeasibility = {inf_pr:.2e}, Dual infeasibility = {inf_du:.2e}, Step = {alpha_pr:.2f}")


# Bounds on variables
x_l = [-np.inf, -np.inf]
x_u = [np.inf, np.inf]

g_l = [0.0, 0.0, 0.0]
g_u = [0.0, np.inf, np.inf]

nlp = HimmelBlauProblem(
    n=2,
    m=3,
    lb=x_l,
    ub=x_u,
    cl=g_l,
    cu=g_u
)

# Solve
x0 = np.array([0.0, 0.0])  # initial guess
x_opt, info = nlp.solve(x0)

# Output
print("Optimal x:", x_opt)
print("Objective value:", info)
