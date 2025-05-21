from himmelblau_function import SQPHimmelblau, SQPHimmelblau_lagrangian, SQPBeale_lagrangian, SQPBeale
import numpy as np
from QP_solver import solve_qp_eq_ineq
from BFGS import BFGS
import os
import argparse


def merit(x, SQPfunc, rho=10.0, function='himmelblau'):
    penalty = 0
    if function == 'himmelblau':
        penalty = np.abs(SQPfunc.h(x)).sum()
        penalty += np.maximum(0, -SQPfunc.g1(x)).sum()
        penalty += np.maximum(0, -SQPfunc.g2(x)).sum()
    elif function == 'beale':
        penalty += np.maximum(0, -SQPfunc.g(x)).sum()
    return SQPfunc(x) + rho * penalty

def project_to_psd(H, epsilon=1e-4):
    # Ensure symmetry
    H_sym = (H + H.T) / 2
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(H_sym)
    # Threshold negative eigenvalues
    eigvals[eigvals < epsilon] = epsilon
    # Reconstruct the PSD matrix
    H_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return H_psd

def SequentialQuadraticProgramming(SQPfunc, x0, tol=1e-5, max_iter=200, alpha=1.0, lambdas=None, use_bfgs=False, function='himmelblau'):
    """
    Sequential Quadratic Programming (SQP) algorithm.
    """
    x = x0.copy()
    n = len(x)
    data = {
        'x': [x.copy()],
        'grad_norm': [],
    }

    for i in range(max_iter):
        if use_bfgs:
            # Use BFGS to approximate the Hessian
            if function == 'himmelblau':
                lagrangian = SQPHimmelblau_lagrangian(lambdas)
            elif function == 'beale':
                lagrangian = SQPBeale_lagrangian(lambdas)
                x = x.reshape(-1)
            H = BFGS(lagrangian, x, verbose=False)[1]
        
        g = SQPfunc.gradient(x)
        A = SQPfunc.h_diff(x)
        b = -SQPfunc.h(x).reshape(-1, 1) if SQPfunc.h(x) is not None else None
        if function == 'himmelblau':
            C = np.vstack([SQPfunc.g1_diff(x), SQPfunc.g2_diff(x)])
            d = np.array([SQPfunc.g1(x), SQPfunc.g2(x)]).reshape(-1, 1)
        elif function == 'beale':
            C = SQPfunc.g_diff(x).reshape(-1, 2)
            d = SQPfunc.g(x).reshape(-1, 1)

        d_x, d_l = solve_qp_eq_ineq(H, g, A=A, b=b, C=C, d=d)

        # Update the solution
        alpha = 1.0
        while alpha > 1e-4:
            x_new = x + alpha * d_x
            if merit(x_new, SQPfunc, function=function) < merit(x, SQPfunc, function=function):
                break
            alpha *= 0.7
        print(alpha)

        x_new = x + alpha * d_x
        if d_l.size > 0:
            lambdas = lambdas + alpha * d_l

        grad = SQPfunc.gradient(x_new)
        grad_norm = np.linalg.norm(grad)
        data['x'].append(x_new.copy())
        data['grad_norm'].append(grad_norm)
        print(f"Iter {i}: x = {x}, ||grad|| = {grad_norm}")
        if grad_norm < tol:
            print(f"Converged in {i} iterations.")
            break
        x = x_new
    return x, data
        
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential Quadratic Programming (SQP) for optimization.')
    parser.add_argument('--function', type=str, choices=['himmelblau', 'beale'], default='himmelblau', help='Function to optimize (default: himmelblau)')
    parser.add_argument('--save', action='store_true', help='Save the results to files')
    args = parser.parse_args()

    if args.function == 'himmelblau':
        function = SQPHimmelblau()
        lambda_0 = np.array([0.0, 0.0, 0.0])  # initial Lagrange multipliers
    elif args.function == 'beale':
        function = SQPBeale()
        lambda_0 = np.array([0.0])

    x0 = np.array([0, 0])  # initial guess
        
    use_bfgs = True

    x_opt, data = SequentialQuadraticProgramming(function, x0, lambdas=lambda_0, use_bfgs=use_bfgs, function=args.function)

    if not os.path.exists('data'):
        os.makedirs('data')

    grad_norms = data['grad_norm']
    iterations = np.arange(1, len(grad_norms) + 1)
    grad_data = np.column_stack((iterations, grad_norms))

    if args.save:
        np.savetxt(f'data/grad_norm_ls_bfgs_{use_bfgs}_{args.function}.txt', grad_data, header='iter grad', comments='')
        np.savetxt(f'data/x_ls_bfgs_{use_bfgs}_{args.function}.txt', data['x'])

    print("Optimal solution:", x_opt)