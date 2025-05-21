from cvxopt import matrix, solvers
import cvxpy as cp
import numpy as np

def solve_qp_eq_ineq(H, g, A=None, b=None, C=None, d=None, delta=None):
    """
    Wrapper to call cvxopt QP solver with proper input formatting.
    """
    # Ensure double precision
    P = matrix(H, tc='d')
    q = matrix(g, tc='d')

    # Handle equality constraints
    if A is not None and b is not None:
        A = matrix(np.asarray(A, dtype=np.double))
        b = matrix(np.asarray(b, dtype=np.double))
    else:
        A = None
        b = None

    # Handle inequality constraints
    if C is not None and d is not None:
        G = matrix(np.asarray(C, dtype=np.double))
        h = matrix(np.asarray(d, dtype=np.double))
    else:
        G = None
        h = None

    sol = solvers.qp(P, q, G, h, A, b)

    lambdas = np.array(sol['y']).flatten() if 'y' in sol else None
    mus = np.array(sol['z']).flatten() if 'z' in sol else None

    lagrange_multipliers = np.hstack([lambdas, mus]) if lambdas is not None and mus is not None else None
    if lambdas is None or mus is None:
        if lambdas is not None:
            lagrange_multipliers = lambdas
        elif mus is not None:
            lagrange_multipliers = mus
    x = np.array(sol['x']).flatten()
    return x, lagrange_multipliers

def solve_qp_eq_ineq_with_trust_region(H, g, delta, A=None, b=None, C=None, d=None):
    """
    Solve QP with equality, inequality, and trust region constraint ||x|| <= delta using CVXPY.
    """
    n = H.shape[0]
    x = cp.Variable(n)

    obj = cp.Minimize(0.5 * cp.quad_form(x, H) + g @ x)
    constraints = []

    if A is not None and b is not None:
        constraints.append(A @ x == b)
    if C is not None and d is not None:
        constraints.append(C @ x + d.flatten() >= 0)
    if delta is not None:
        constraints.append(cp.norm(x, 2) <= delta)

    prob = cp.Problem(obj, constraints)
    prob.solve()
    langrange_multipliers = np.concatenate([np.array(constraints[i].dual_value).flatten() for i in range(len(constraints[:-1]))])
    return x.value, langrange_multipliers

if __name__ == "__main__":
    # Example usage
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-2, -5])
    A = np.array([[1, 1]])
    b = np.array([1])
    C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    d = np.array([1, 1, 0, 0])

    x = solve_qp_eq_ineq(H, g, A=A, b=b, C=C, d=d)
    print("Optimal solution:", x)