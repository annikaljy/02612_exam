import numpy as np
from function import QuadraticFunction as function

def wolfe_line_search(func, x, p, grad, alpha, beta, c1=1e-3, c2=0.9, max_iter=1000, verbose=True):
    phi_0 = func(x)
    phi_prime_0 = grad @ p

    for _ in range(max_iter):
        x_new = x + alpha * p.reshape(-1)
        phi_alpha = func(x_new)
        grad_new = func.gradient(x_new).reshape(-1)
        if grad_new.shape == p.shape:
            phi_prime_alpha = grad_new.T @ p
        else:
            phi_prime_alpha = grad_new @ p

        # Wolfe conditions
        if phi_alpha <= phi_0 + c1 * alpha * phi_prime_0 and -phi_prime_alpha <= -c2 * phi_prime_0:
            return alpha, grad_new

        alpha *= beta
    
    if verbose:
        print(f"Wolfe line search failed at alpha={alpha:.4e}, phi_alpha={phi_alpha}, expected upper bound={phi_0 + c1 * alpha * phi_prime_0}")
    return None, grad

def BFGS(func, x_0, tol=1e-6, max_iter=100, alpha=1, beta=0.5, verbose=True):
    x = x_0.copy()
    n = len(x)
    Bk = np.eye(n)

    for i in range(max_iter):
        grad = func.gradient(x).reshape(-1)
        grad_norm = np.linalg.norm(grad)
        if verbose:
            print(f"Iter {i}: x = {x}, ||grad|| = {grad_norm}")

        if grad_norm < tol:
            if verbose:
                print(f"Converged in {i} iterations.")
            break
        p = -np.linalg.solve(Bk, grad).reshape(-1)
        
        alpha_i, grad_new = wolfe_line_search(func, x, p, grad, alpha, beta, verbose=verbose)
        if alpha_i is None:
            return x, Bk        

        x_new = x + alpha_i * p
        s = x_new - x
        y = grad_new - grad
        sBs = s.T @ Bk @ s

        if y.T @ s >= 0.2 * sBs:
            theta = 1.0
        else:
            theta = (0.8 * sBs) / (sBs - y.T @ s)

        r = theta * y + (1 - theta) * Bk@s
        Bk = Bk - (Bk @ np.outer(s, s) @ Bk.T) / sBs + np.outer(r, r) / (r.T @ s)

        x = x_new
        if i == max_iter - 1:
            if verbose:
                print(f"Max iterations reached without convergence.")
    return x, Bk



if __name__ == "__main__":
    H = np.array([[2, 1], [1, 2]])
    g = np.array([-2, -5])
    A = np.array([[1, 1], [1, -1]])
    b = np.array([1, 0])
    f = function(H, g, A, b)

    x_0 = np.array([10, 10])
    x_opt_bfgs, H_bfgs = BFGS(f, x_0)

    print("BFGS Hessian approximation:\n", H_bfgs)
    print("True Hessian:\n", H)
    print("Optimal x:", x_opt_bfgs)
    print("True x*:", -np.linalg.inv(H) @ g)
