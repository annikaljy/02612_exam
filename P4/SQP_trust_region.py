from himmelblau_function import SQPHimmelblau, SQPHimmelblau_lagrangian, SQPBeale, SQPBeale_lagrangian
import numpy as np
from QP_solver import solve_qp_eq_ineq_with_trust_region
import os
import argparse
from BFGS import BFGS

def objective(x, lambdas, d,SQPfunc):
    """
    Objective function for SQP.
    """
    return SQPfunc(x) + SQPfunc.gradient(x).T @ d + 0.5 * d.T @ SQPfunc.lagrangian_hessian(x, lambdas) @ d

def SequentialQuadraticProgramming(SQPfunc, x0, tol=1e-5, max_iter=100, lambdas=None, use_bfgs=False, function='himmelblau'):
    """
    Sequential Quadratic Programming (SQP) algorithm with Trust Region.
    """
    x = x0.copy()
    x = x.astype(np.float64)
    n = len(x)
    delta = 1.0
    delta_max = 100.0
    eta1 = 0.75
    eta2 = 1 - eta1
    gamma_inc = 2.0
    gamma_dec = 0.5
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

        d_x, d_l = solve_qp_eq_ineq_with_trust_region(H, g, delta, A=A, b=b, C=C, d=d)

        if d_x is None:
            print("QP solver failed to find a solution.")
            break

        num = SQPfunc(x) - SQPfunc(x + d_x)
        denom = objective(x, lambdas, np.zeros_like(d_x), SQPfunc) - objective(x, lambdas, d_x, SQPfunc)
        ratio = num / denom

        if ratio > eta1:
            delta = min(gamma_inc * delta, delta_max)
            x += d_x
            lambdas += d_l
        elif ratio < eta2:
            delta = max(gamma_dec * delta, 1e-5)
        else:
            x += d_x
            lambdas += d_l
        
        grad = SQPfunc.gradient(x)
        grad_norm = np.linalg.norm(grad)
        data['x'].append(x.copy())
        data['grad_norm'].append(grad_norm)
        print(f"Iter {i}: x = {x}, ||grad|| = {grad_norm}, delta = {delta:.4f}")
        if grad_norm < tol:
            print(f"Converged in {i} iterations.")
            break

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
        np.savetxt(f'data/grad_norm_tr_bfgs_{use_bfgs}_{args.function}.txt', grad_data, header='iter grad', comments='')
        np.savetxt(f'data/x_tr_bfgs_{use_bfgs}_{args.function}.txt', data['x'])

    print("Optimal solution:", x_opt)