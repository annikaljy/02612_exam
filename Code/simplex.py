import numpy as np
from scipy import linalg, sparse
import time
import scipy.io
import matplotlib.pyplot as plt 

 # convert lp to standard form
def lp_standard_form(g, A, b, l=None, u=None):
     '''
     Transform problem of form:
     min g'x
     st A'x=b  <- Note: A here is n x m as per original doc comment
     l≤x≤u
     To form:
     min h'y
     st Cy=d
     y≥0

     Input A is expected n x m (original problem), converts to m x n_std (standard form C)
     '''
     n_orig = len(g) # number of original variables
     m_eq = len(b)   # number of equality constraints

     # Ensure A is n_orig x m_eq
     if A.shape != (n_orig, m_eq):
          # Try transposing if it matches the other dimension
         if A.T.shape == (n_orig, m_eq):
             A = A.T
         else:
             raise ValueError(f"Input A shape {A.shape} inconsistent with g ({n_orig}) and b ({m_eq})")


     # Default bounds if not provided
     if l is None:
         l = np.full(n_orig, -np.inf)
     if u is None:
         u = np.full(n_orig, np.inf)

     # Identify fixed variables (l==u)
     fixed_vars = l == u
     free_vars = ~fixed_vars & ~np.isfinite(l) & ~np.isfinite(u)
     lower_bounded_vars = ~fixed_vars & np.isfinite(l) & ~np.isfinite(u) & (l != 0) # Exclude x>=0 unless l>0 explicitly
     upper_bounded_vars = ~fixed_vars & ~np.isfinite(l) & np.isfinite(u)
     interval_vars = ~fixed_vars & np.isfinite(l) & np.isfinite(u)
     non_neg_vars = (l == 0) & ~np.isfinite(u) # Already standard non-negative


     C_parts = []
     h_parts = []
     d_final = b.copy() # Start with original equality constraints RHS
     current_vars_map = {} # Keep track of where original vars map

     col_idx = 0

     # 1. Handle fixed variables
     if np.any(fixed_vars):
         A_fixed = A[fixed_vars, :]
         l_fixed = l[fixed_vars]
         # Adjust RHS of equality constraints
         d_final -= A_fixed.T @ l_fixed
         # Remove fixed variables from A for subsequent steps
         A = A[~fixed_vars, :]
         g = g[~fixed_vars] # Remove from objective as well
         # Update other index masks
         free_vars = free_vars[~fixed_vars]
         lower_bounded_vars = lower_bounded_vars[~fixed_vars]
         upper_bounded_vars = upper_bounded_vars[~fixed_vars]
         interval_vars = interval_vars[~fixed_vars]
         non_neg_vars = non_neg_vars[~fixed_vars]
         n_orig = A.shape[0] # Update n_orig


     # 2. Handle free variables: x = x+ - x-
     if np.any(free_vars):
         n_free = np.sum(free_vars)
         A_free = A[free_vars, :]
         g_free = g[free_vars]
         C_parts.append(A_free.T)
         C_parts.append(-A_free.T)
         h_parts.append(g_free)
         h_parts.append(-g_free)
         current_vars_map['free'] = (col_idx, col_idx + 2 * n_free)
         col_idx += 2 * n_free
         # Remove from A and g for subsequent steps
         A = A[~free_vars, :]
         g = g[~free_vars]
         lower_bounded_vars = lower_bounded_vars[~free_vars]
         upper_bounded_vars = upper_bounded_vars[~free_vars]
         interval_vars = interval_vars[~free_vars]
         non_neg_vars = non_neg_vars[~free_vars]


     # 3. Handle non-negative variables: Already in standard form
     if np.any(non_neg_vars):
          n_non_neg = np.sum(non_neg_vars)
          A_non_neg = A[non_neg_vars, :]
          g_non_neg = g[non_neg_vars]
          C_parts.append(A_non_neg.T)
          h_parts.append(g_non_neg)
          current_vars_map['non_neg'] = (col_idx, col_idx + n_non_neg)
          col_idx += n_non_neg
          A = A[~non_neg_vars, :]
          g = g[~non_neg_vars]
          lower_bounded_vars = lower_bounded_vars[~non_neg_vars]
          upper_bounded_vars = upper_bounded_vars[~non_neg_vars]
          interval_vars = interval_vars[~non_neg_vars]


     # --- Variables requiring transformation and slack variables ---
     C_slack_parts = []
     d_slack_parts = []


     # 4. Handle lower bounded variables (x >= l, l!=0): x = x' + l, x' >= 0
     if np.any(lower_bounded_vars):
         n_lb = np.sum(lower_bounded_vars)
         A_lb = A[lower_bounded_vars, :]
         g_lb = g[lower_bounded_vars]
         l_lb = l[lower_bounded_vars]
         # Adjust RHS
         d_final -= A_lb.T @ l_lb
         # Add transformed variable x' to C
         C_parts.append(A_lb.T)
         # Adjust objective h
         h_parts.append(g_lb)
         # Constant part added to objective (ignored in minimization)
         # obj_const += g_lb @ l_lb
         current_vars_map['lower_bounded'] = (col_idx, col_idx + n_lb)
         col_idx += n_lb
         A = A[~lower_bounded_vars, :]
         g = g[~lower_bounded_vars]
         upper_bounded_vars = upper_bounded_vars[~lower_bounded_vars]
         interval_vars = interval_vars[~lower_bounded_vars]


     # 5. Handle upper bounded variables (x <= u): x + s = u, x>=??, s >= 0
     #    Need to consider if x was free or lower-bounded before this.
     #    Assuming x was free: x = x+ - x-, so x+ - x- + s = u
     #    Assuming x >= 0: x + s = u, x>=0, s>=0 -> standard form
     #    Assuming x >= l: x' + l + s = u -> x' + s = u - l
     #    For simplicity, let's assume the variables reaching here are effectively x >= 0
     #    after previous transformations or originally x<=u with implicit x>=0.
     #    So we add slack s: x + s = u, x>=0, s>=0
     if np.any(upper_bounded_vars):
         n_ub = np.sum(upper_bounded_vars)
         A_ub = A[upper_bounded_vars, :]
         g_ub = g[upper_bounded_vars]
         u_ub = u[upper_bounded_vars]

         # Add x to main constraint matrix C
         C_parts.append(A_ub.T)
         h_parts.append(g_ub)
         x_indices = list(range(col_idx, col_idx + n_ub))
         current_vars_map['upper_bounded_x'] = (col_idx, col_idx + n_ub)
         col_idx += n_ub

         # Add slack variable s to C
         C_parts.append(np.zeros((m_eq, n_ub))) # Slack doesn't affect original equalities
         h_parts.append(np.zeros(n_ub)) # Slack has 0 cost
         s_indices = list(range(col_idx, col_idx + n_ub))
         current_vars_map['upper_bounded_s'] = (col_idx, col_idx + n_ub)
         col_idx += n_ub

         # Add the constraint x + s = u as new rows in C and d
         row = np.zeros((n_ub, col_idx))
         for i in range(n_ub):
             row[i, x_indices[i]] = 1
             row[i, s_indices[i]] = 1
         C_slack_parts.append(row)
         d_slack_parts.append(u_ub)
         A = A[~upper_bounded_vars, :]
         g = g[~upper_bounded_vars]
         interval_vars = interval_vars[~upper_bounded_vars]


     # 6. Handle interval variables (l <= x <= u): x' + l + s = u -> x' + s = u - l
     if np.any(interval_vars):
         n_iv = np.sum(interval_vars)
         A_iv = A[interval_vars, :]
         g_iv = g[interval_vars]
         l_iv = l[interval_vars]
         u_iv = u[interval_vars]

         # Adjust RHS for x = x' + l substitution
         d_final -= A_iv.T @ l_iv

         # Add x' to main constraint matrix C
         C_parts.append(A_iv.T)
         h_parts.append(g_iv)
         xp_indices = list(range(col_idx, col_idx + n_iv))
         current_vars_map['interval_xp'] = (col_idx, col_idx + n_iv)
         col_idx += n_iv

         # Add slack variable s to C
         C_parts.append(np.zeros((m_eq, n_iv))) # Slack doesn't affect original equalities
         h_parts.append(np.zeros(n_iv)) # Slack has 0 cost
         s_indices_iv = list(range(col_idx, col_idx + n_iv))
         current_vars_map['interval_s'] = (col_idx, col_idx + n_iv)
         col_idx += n_iv

         # Add the constraint x' + s = u - l
         row_iv = np.zeros((n_iv, col_idx))
         for i in range(n_iv):
             row_iv[i, xp_indices[i]] = 1
             row_iv[i, s_indices_iv[i]] = 1
         C_slack_parts.append(row_iv)
         d_slack_parts.append(u_iv - l_iv)


     # Combine parts
     C_final = np.hstack(C_parts)
     h_final = np.concatenate(h_parts)

     if C_slack_parts:
         C_final_rows = C_final.shape[0]
         C_slack_full = np.vstack(C_slack_parts)
         # Ensure slack constraint rows have correct total columns
         if C_slack_full.shape[1] < C_final.shape[1]:
              padding = np.zeros((C_slack_full.shape[0], C_final.shape[1] - C_slack_full.shape[1]))
              C_slack_full = np.hstack((C_slack_full, padding))
         elif C_slack_full.shape[1] > C_final.shape[1]: # Should not happen with current logic
              C_slack_full = C_slack_full[:, :C_final.shape[1]]

         C_final = np.vstack((C_final, C_slack_full))
         d_final = np.concatenate([d_final] + d_slack_parts)


     # Add basic checks
     if C_final.shape[1] != len(h_final):
         raise ValueError(f"Dimension mismatch: C columns {C_final.shape[1]} != h length {len(h_final)}")
     if C_final.shape[0] != len(d_final):
          raise ValueError(f"Dimension mismatch: C rows {C_final.shape[0]} != d length {len(d_final)}")

     return h_final, C_final, d_final # Return h, C, d for standard form
def lp_simplex_phase1_form(A, b):
     '''
     Linear program for simplex.
     takes an LP of form:
     min g'x
     st. Ax=b
     x≥0

     and converts it to:
     min 1't
     st. [A E] [x t]' = b  (where E is diagonal +/-1 to make b>=0 initially)
     [x t]'≥ 0

     initial feasible points are given by:
     x0 = 0
     t_i = |b_i| (adjusted for E)
     '''
     m, n = A.shape # A is C_std (m_std x n_std)

     # Ensure RHS is non-negative by potentially flipping rows
     b_nonneg = b.copy()
     A_nonneg = A.copy()
     flip_rows = b < 0
     if np.any(flip_rows):
         A_nonneg[flip_rows, :] *= -1
         b_nonneg[flip_rows] *= -1


     # Add artificial variables t
     # E is identity since we made b_nonneg >= 0
     E = np.eye(m)
     A_init = np.hstack((A_nonneg, E)) # Phase 1 matrix [C_std E]

     # Initial feasible point for Phase 1
     x0_init = np.zeros(n + m)
     x0_init[n:] = b_nonneg # Artificial variables start at b_nonneg

     # Objective for Phase 1: minimize sum of artificial variables
     g_init = np.zeros(n + m)
     g_init[n:] = 1 # Cost is 1 for artificial vars, 0 for original vars

     # Initial basis for phase 1 is the set of artificial variables
     b_ind = np.arange(n, n + m)
     n_ind = np.arange(n)

     return g_init, A_init, b_nonneg, x0_init, b_ind, n_ind


 # Revised Simplex algorithm
def lp_simplex_alg(g, A, b, x0, b_ind, n_ind, max_iter=100, verbose=False, tol=1e-8):
     '''
     simplex algorithm on matrix form:
     assumes standard form structure, ie.
     min g'x
     st. Ax = b
     x ≥ 0
     where g -> R^n, A -> R^mxn, b -> R^m

     also assumes knowledge of basic and nonbasic vars.

     params:
     g: vector, linear objective coefficients (h_std or g_p1)
     A: Equality constraint coeffs (C_std or A_p1)
     b: Equality constraints rhs (d_std or b_p1)
     x0: initial feasible solution (y_std or x_p1)
     b_ind: initial basic indices
     n_ind: initial non-basic indices
     max_iter: max iterations
     verbose: print iteration info
     tol: tolerance for optimality checks

     returns:
     vars: optimal solution vector
     b_ind: final basic indices
     n_ind: final non-basic indices
     iter: iterations performed
     '''
     if not np.allclose(A @ x0, b):
        print("Warning: Initial point x0 is not feasible for Ax=b.")
        # Depending on the context (Phase 1 vs Phase 2), this might be acceptable or an error.

     m, n = A.shape
     n_basic = len(b_ind)
     n_nonbasic = len(n_ind)

     if n_basic != m:
        raise ValueError(f"Number of basic variables {n_basic} does not match number of constraints {m}")
     if n_basic + n_nonbasic != n:
         raise ValueError(f"Sum of basic ({n_basic}) and non-basic ({n_nonbasic}) variables does not match total variables ({n})")


     vars_current = x0.copy() # Use current state naming
     b_ind_current = b_ind.copy()
     n_ind_current = n_ind.copy()


     STOP = False
     iter_count = 0

     while not STOP and iter_count < max_iter:
         xB = vars_current[b_ind_current]
         # xN = vars_current[n_ind_current] # Should be zero by simplex logic

         B = A[:, b_ind_current]
         N = A[:, n_ind_current]

         try:
             # Factorize B using LU factorization
             lu, piv = linalg.lu_factor(B)
         except linalg.LinAlgError:
             print("Error: Singular matrix B encountered in LU factorization. Cannot proceed.")
             # This indicates linear dependence in basic columns, potentially due to problem formulation or numerical issues.
             return None, b_ind_current, n_ind_current, iter_count


         # Solve B^T * mu = g_B for mu (Lagrange multipliers / dual variables)
         gB = g[b_ind_current]
         mu = linalg.lu_solve((lu, piv), gB, trans=1)

         # Calculate reduced costs (lambda_N = g_N - N^T * mu)
         gN = g[n_ind_current]
         lambdaN = gN - N.T @ mu

         # Check for optimality: lambda_N >= 0
         if np.all(lambdaN >= -tol):
             print(f"Optimal solution found after {iter_count} iterations.")
             STOP = True
             return vars_current, b_ind_current, n_ind_current, iter_count

         # Select entering variable: choose s with most negative reduced cost
         s_idx_in_N = np.argmin(lambdaN)
         entering_var_idx = n_ind_current[s_idx_in_N]

         # Calculate search direction h = B^{-1} * A_s (column of entering variable)
         As = A[:, entering_var_idx]
         h = linalg.lu_solve((lu, piv), As)

         # Check for unboundedness: if h <= 0
         if np.all(h <= tol):
             print("Problem is unbounded.")
             STOP = True
             # Indicate unboundedness, maybe return specific value or raise error
             return None, b_ind_current, n_ind_current, iter_count

         # Ratio test to find leaving variable
         ratios = np.full(n_basic, np.inf) # Initialize ratios to infinity
         positive_h_indices = h > tol # Indices where direction component is positive
         if not np.any(positive_h_indices):
              print("Warning: No positive direction found in ratio test, should have been caught by unbounded check.")
              # This case should ideally not be reached if unbounded check is correct
              return None, b_ind_current, n_ind_current, iter_count

         ratios[positive_h_indices] = xB[positive_h_indices] / h[positive_h_indices]

         # Find index of minimum ratio among basic variables
         j_idx_in_B = np.argmin(ratios)
         leaving_var_idx = b_ind_current[j_idx_in_B]

         # Calculate step size alpha
         alpha = ratios[j_idx_in_B]
         if alpha < 0: # Should not happen if xB >= 0 and h[j] > 0
             print(f"Warning: Negative step size alpha = {alpha} calculated.")
             alpha = 0 # Prevent moving in wrong direction

         # Update variable values
         vars_current[b_ind_current] -= alpha * h
         vars_current[entering_var_idx] = alpha # New basic variable value

         # Ensure leaving basic variable is numerically zero (can drift slightly)
         vars_current[leaving_var_idx] = 0.0


         # Update basis: swap entering and leaving variables
         b_ind_current[j_idx_in_B], n_ind_current[s_idx_in_N] = n_ind_current[s_idx_in_N], b_ind_current[j_idx_in_B]


         iter_count += 1

         if verbose:
             print(f"--- Iteration {iter_count} ---")
             # print(f"xB: {vars_current[b_ind_current]}")
             print(f"Entering variable index: {entering_var_idx} (from non-basic index {s_idx_in_N})")
             print(f"Most negative reduced cost: {lambdaN[s_idx_in_N]}")
             print(f"Leaving variable index: {leaving_var_idx} (from basic index {j_idx_in_B})")
             print(f"Step size alpha: {alpha}")


     if not STOP:
         print(f"Max iterations ({max_iter}) reached without convergence.")

     return vars_current, b_ind_current, n_ind_current, iter_count
def lp_simplex(gx, Aeq, beq, lb, ub, max_iter=1000, tol=1e-8, verbose=False, run_phase1=False):
     '''
     Revised simplex algorithm implementation (Algorithm 15).

     Solves LP of form:
         min gx'x
         st. Aeq'x = beq  (Note: Aeq is n x m)
             lb ≤ x ≤ ub

     Converts to standard form, runs Phase 1 if needed, then Phase 2.

     params:
         gx: objective coefficients (n_orig,)
         Aeq: equality constraints matrix (n_orig x m_eq)
         beq: rhs of equality constraints (m_eq,)
         lb: lower bound on original x (n_orig,)
         ub: upper bound on original x (n_orig,)
         max_iter: maximum iterations for each phase
         tol: tolerance for optimality and feasibility checks
         verbose: print iteration details
         run_phase1: force Phase 1 even if bypass seems possible

     returns:
         x_optimal: solution vector for original variables (n_orig,) or None if infeasible/unbounded/error
         total_iters: total iterations across Phase 1 and Phase 2
         status: string indicating 'optimal', 'infeasible', 'unbounded', 'max_iterations', 'error'
     '''
     print("Attempting to solve LP using revised simplex algorithm (Algorithm 15):")
     t1_t = time.perf_counter()
     n_vars_orig = len(gx)

     #Convert to Standard Form ---
     print("Converting to standard form...")
     try:
         # Expects Aeq.T (m_eq x n_orig) as input A for lp_standard_form
         h_std, C_std, d_std = lp_standard_form(gx, Aeq, beq, lb, ub)
         m_std, n_std = C_std.shape
         print(f"Standard form: {m_std} constraints, {n_std} variables.")
     except Exception as e:
         print(f"Error during standard form conversion: {e}")
         return None, 0, "error"

     #Step 2: Phase 1 (Find Initial Feasible Basis) ---
     iter_p1 = 0
     iter_p2 = 0
     status = "unknown"

     print("Starting Phase 1 (finding initial feasible solution)...")
     t1_p1 = time.perf_counter()
     try:
         # lp_simplex_phase1_form sets up the auxiliary problem
         g_p1, A_p1, b_p1, x0_p1, Bx_p1_init, Nx_p1_init = lp_simplex_phase1_form(C_std, d_std)

         # Solve the auxiliary problem using the simplex algorithm
         sol_p1_full, Bix_p1, Nix_p1, iter_p1 = lp_simplex_alg(g_p1, A_p1, b_p1,
                                                          x0=x0_p1,
                                                          b_ind=Bx_p1_init.copy(),
                                                          n_ind=Nx_p1_init.copy(),
                                                          max_iter=max_iter,
                                                          verbose=verbose,
                                                          tol=tol)
         t2_p1 = time.perf_counter()
         print(f"Phase 1 finished in {t2_p1 - t1_p1:.4f}s and {iter_p1} iterations.")

         if sol_p1_full is None:
             # Simplex alg returned None, could be unbounded (not possible for Phase 1 if original feasible) or error
             print("Phase 1 failed (likely unbounded auxiliary problem or numerical error).")
             status = "error" # Or potentially infeasible if Phase 1 objective > 0 but simplex fails
             return None, iter_p1, status

         # Check Phase 1 objective value
         obj_p1 = g_p1.T @ sol_p1_full
         print(f"Phase 1 final objective value: {obj_p1}")

         if not np.allclose(obj_p1, 0, atol=tol):
             print("Original problem is infeasible (Phase 1 objective > 0).")
             status = "infeasible"
             # Return info indicating infeasibility, maybe sol_p1_full if useful
             return None, iter_p1, status
         else:
             print("Phase 1 successful: Found initial feasible solution for standard form.")
             # Need to extract a feasible basis for Phase 2
             # Check if any artificial variables are in the basis Bix_p1 at zero level
             artificial_vars_indices = np.arange(n_std, A_p1.shape[1])
             basic_artificial = np.intersect1d(Bix_p1, artificial_vars_indices)

             if len(basic_artificial) > 0:
                  
                  print("Warning: Artificial variables still in Phase 1 basis. Pivoting needed (not implemented).")
                  # Try to proceed, but might fail in Phase 2 if basis is degenerate/invalid
                  # Extract basis containing only original standard form variables
                  b_ind_phase2 = np.setdiff1d(Bix_p1, artificial_vars_indices, assume_unique=True)
                  if len(b_ind_phase2) != m_std:
                       print("Error: Could not recover a valid basis for Phase 2 after Phase 1.")
                       return None, iter_p1, "error"
                  n_ind_phase2 = np.setdiff1d(np.arange(n_std), b_ind_phase2, assume_unique=True)
                  x0_phase2 = sol_p1_full[:n_std]

             else:
                  # No artificial variables in basis, use the result directly
                  b_ind_phase2 = Bix_p1.copy()
                  n_ind_phase2 = np.setdiff1d(np.arange(n_std), b_ind_phase2, assume_unique=True)
                  x0_phase2 = sol_p1_full[:n_std]


     except Exception as e:
         print(f"Error during Phase 1: {e}")
         return None, iter_p1, "error"

     #Phase 2 (Solve Original Problem in Standard Form) ---
     print("Starting Phase 2 (solving original problem)...")
     t1_p2 = time.perf_counter()
     try:
         sol_p2_std, Bix_p2, Nix_p2, iter_p2 = lp_simplex_alg(h_std, C_std, d_std,
                                                         x0=x0_phase2,
                                                         b_ind=b_ind_phase2,
                                                         n_ind=n_ind_phase2,
                                                         max_iter=max_iter,
                                                         verbose=verbose,
                                                         tol=tol)
         t2_p2 = time.perf_counter()
         print(f"Phase 2 finished in {t2_p2 - t1_p2:.4f}s and {iter_p2} iterations.")

         if sol_p2_std is None:
             # Simplex alg returned None, could be unbounded or error
             # Need to distinguish based on simplex internal state if possible
             print("Phase 2 failed (problem might be unbounded or numerical error occurred).")
             # Assuming unbounded if Phase 1 was feasible and Phase 2 returns None this way
             status = "unbounded" 
             return None, iter_p1 + iter_p2, status

         else:
              status = "optimal"
              if iter_p1 + iter_p2 >= max_iter * 2: # Check total iterations
                   status = "max_iterations"

              # Map Standard Form Solution Back to Original Variables ---
              # This is crucial and depends heavily on the lp_standard_form implementation.
              # Need a way to reconstruct x_orig from y_std (sol_p2_std)
              # Example placeholder logic:
              x_optimal = np.zeros(n_vars_orig)
              print("Warning: Mapping standard solution back to original variables is not fully implemented.")
              # Placeholder:
              try:

                  if n_vars_orig == len(gx): # Check if original problem size matches gx
                       x_optimal = sol_p2_std[:n_vars_orig]
                  else:
                       print("Cannot reliably map back solution due to unknown standard form structure.")
                       x_optimal = None # Indicate failure to map back
                       status = "error"

              except IndexError:
                  print("Error during solution mapping: Index out of bounds.")
                  x_optimal = None
                  status = "error"


              t2_t = time.perf_counter()
              total_time = t2_t - t1_t
              total_iters = iter_p1 + iter_p2

              if x_optimal is not None:
                  print(f"Optimal objective: {gx.T @ x_optimal:.6f}")
                  print(f"Solution status: {status}")
                  print(f"Total iterations: {total_iters}")
                  print(f"Total run time: {total_time:.4f}s")
              else:
                   print("Failed to reconstruct optimal solution for original variables.")

              return x_optimal, total_iters, status

     except Exception as e:
         print(f"Error during Phase 2: {e}")
         return None, iter_p1 + iter_p2, "error" 
def plot_simplex_results(x_sol, gx, Pd_max, Pg_max, U, C_prices, status, iters, save_plots=True):
    """
    Plot the results of the simplex LP solver for a market clearing problem.
    
    Parameters:
    -----------
    x_sol : numpy array
        Solution vector from simplex algorithm
    gx : numpy array
        Objective function coefficients
    Pd_max : numpy array
        Maximum demand values
    Pg_max : numpy array
        Maximum generation capacity values
    U : numpy array
        Utility values for demand
    C_prices : numpy array
        Cost prices for generators
    status : str
        Solution status from simplex
    iters : int
        Number of iterations performed
    save_plots : bool
        Whether to save plots to files
    """
    if x_sol is None:
        print(f"Cannot plot results: Solution is None (status: {status})")
        return
    
    n_demands = len(Pd_max)
    n_generators = len(Pg_max)
    
    # Extract demand and generation from solution
    P_d = x_sol[:n_demands]
    P_g = x_sol[n_demands:]
    
    # Calculate objective value
    obj_value = gx.T @ x_sol
    social_welfare = -obj_value  # Social welfare is negative of objective
    
    # Calculate market price (simplified - in reality would come from dual variables)
    # For this implementation, we'll use the price of the marginal generator
    active_gens = P_g > 1e-4
    if np.any(active_gens):
        # Marginal generator is the most expensive one that's active
        active_prices = C_prices[active_gens]
        market_price = np.max(active_prices) if len(active_prices) > 0 else 0
    else:
        market_price = 0
    
    # Print results summary
    print("\nOptimal Solution:")
    print(f"Objective value: {obj_value:.4f}")
    print(f"Social welfare: {social_welfare:.4f}")
    print(f"Market clearing price: {market_price:.4f}")
    print(f"Total iterations: {iters}")
    print(f"Status: {status}")
    
    # Display generation schedule
    print(f"\nGeneration schedule:")
    total_gen = 0
    for i in range(n_generators):
        if P_g[i] > 1e-4:  # Only show active generators
            print(f"Generator {i+1}: {P_g[i]:.4f} MW (Cost: {C_prices[i]:.4f}, Capacity: {Pg_max[i]:.4f})")
            total_gen += P_g[i]
    
    # Display demand schedule
    print(f"\nDemand schedule:")
    total_dem = 0
    for i in range(n_demands):
        if P_d[i] > 1e-4:  # Only show active demands
            print(f"Demand {i+1}: {P_d[i]:.4f} MW (Utility: {U[i]:.4f}, Max Demand: {Pd_max[i]:.4f})")
            total_dem += P_d[i]
    
    print(f"\nTotal generation: {total_gen:.4f} MW")
    print(f"Total demand: {total_dem:.4f} MW")
    print(f"Generation-demand balance: {total_gen - total_dem:.4f} MW")
    
    # Plot results
    # 1. Plot generation and demand
    plt.figure(figsize=(14, 10))
    
    # Plot generators
    plt.subplot(2, 1, 1)
    active_gen = P_g > 1e-4
    if np.any(active_gen):
        active_indices = np.where(active_gen)[0] + 1
        plt.bar(active_indices, P_g[active_gen], color='blue', alpha=0.7, label='Generation')
        plt.bar(active_indices, Pg_max[active_gen], color='lightblue', alpha=0.3, label='Max Capacity')
        for i, idx in enumerate(np.where(active_gen)[0]):
            plt.text(active_indices[i], P_g[idx] + 0.05 * max(Pg_max), f'${C_prices[idx]:.2f}', ha='center')
    plt.xlabel('Generator')
    plt.ylabel('Power (MW)')
    plt.title('Generator Output vs Capacity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot demands
    plt.subplot(2, 1, 2)
    active_dem = P_d > 1e-4
    if np.any(active_dem):
        active_indices = np.where(active_dem)[0] + 1
        plt.bar(active_indices, P_d[active_dem], color='green', alpha=0.7, label='Demand')
        plt.bar(active_indices, Pd_max[active_dem], color='lightgreen', alpha=0.3, label='Max Demand')
        for i, idx in enumerate(np.where(active_dem)[0]):
            plt.text(active_indices[i], P_d[idx] + 0.05 * max(Pd_max), f'${U[idx]:.2f}', ha='center')
    plt.xlabel('Demand')
    plt.ylabel('Power (MW)')
    plt.title('Demand vs Maximum Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('simplex_market_clearing.png', dpi=300)
    
    # 2. Plot supply curve and demand curve
    plt.figure(figsize=(12, 8))
    
    # Generators by cost for supply curve
    sorted_indices = np.argsort(C_prices)
    sorted_C = C_prices[sorted_indices]
    sorted_Pg_max = Pg_max[sorted_indices]
    
    # Supply curve (step function)
    cum_capacity = np.zeros(n_generators + 1)
    cum_capacity[1:] = np.cumsum(sorted_Pg_max)
    supply_x = np.repeat(cum_capacity, 2)[1:-1]
    supply_y = np.repeat(sorted_C, 2)
    
    # Demands by utility (descending) for demand curve
    sorted_dem_indices = np.argsort(-U)
    sorted_U = U[sorted_dem_indices]
    sorted_Pd_max = Pd_max[sorted_dem_indices]
    
    # Demand curve (step function)
    cum_demand = np.zeros(n_demands + 1)
    cum_demand[1:] = np.cumsum(sorted_Pd_max)
    demand_x = np.repeat(cum_demand, 2)[1:-1]
    demand_y = np.repeat(sorted_U, 2)
    
    plt.step(supply_x, supply_y, where='post', label='Supply Curve', color='blue', linewidth=2)
    plt.step(demand_x, demand_y, where='post', label='Demand Curve', color='green', linewidth=2)
    
    # Market clearing price and quantity
    total_cleared = np.sum(P_g)  # Should equal sum(P_d)
    plt.axhline(y=market_price, color='red', linestyle='--', label=f'Market Price: ${market_price:.2f}')
    plt.axvline(x=total_cleared, color='purple', linestyle='--', label=f'Cleared Quantity: {total_cleared:.2f} MW')
    
    # Active generators and demands
    active_gen = P_g > 1e-4
    if np.sum(active_gen) > 0:
        plt.scatter(cum_capacity[1:][active_gen[sorted_indices]], sorted_C[active_gen[sorted_indices]], 
                    color='red', s=100, zorder=5, label='Active Generators')
    
    active_dem = P_d > 1e-4
    if np.sum(active_dem) > 0:
        plt.scatter(cum_demand[1:][active_dem[sorted_dem_indices]], sorted_U[active_dem[sorted_dem_indices]], 
                    color='orange', s=100, zorder=5, label='Active Demands')
    
    plt.xlabel('Capacity/Demand (MW)')
    plt.ylabel('Price ($/MW)')
    plt.title('Supply Curve, Demand Curve, and Market Clearing')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig('simplex_supply_demand_curves.png', dpi=300)
    
    plt.show()
    
    return {
        'P_g': P_g,
        'P_d': P_d,
        'market_price': market_price,
        'total_generation': np.sum(P_g),
        'total_demand': np.sum(P_d),
        'objective_value': obj_value,
        'social_welfare': social_welfare
    }

if __name__ == '__main__':
    
    n_demands = 30
    n_generators = 15
    n_vars_orig = n_demands + n_generators
     
    data_file = "LP_Test.mat"
    data = scipy.io.loadmat(data_file)

    U = data["U"].flatten().astype(float)
    C_prices = data["C"].flatten().astype(float) # Offer prices
    Pd_max = data["Pd_max"].flatten().astype(float) # Demand caps
    Pg_max = data["Pg_max"].flatten().astype(float) # Generator caps

    # Original problem formulation variables
    gx = np.concatenate((-U, C_prices)) # min -U*Pd + C*Pg
    lb = np.zeros(n_vars_orig)
    ub = np.concatenate((Pd_max, Pg_max))

    # Equality constraint: sum(Pd) - sum(Pg) = 0
    Aeq = np.zeros((n_vars_orig, 1))
    Aeq[:n_demands, 0] = 1  # Coefficients for Pd
    Aeq[n_demands:, 0] = -1 # Coefficients for Pg
    beq = np.array([0.0])    # RHS

    print("\n--- Running Simplex Example ---")
    x_sol, iters, status = lp_simplex(gx, Aeq, beq, lb, ub, max_iter=500, verbose=False, run_phase1=True)

    print(f"\n--- Simplex Result ---")
    print(f"Status: {status}")
    if x_sol is not None:
        print(f"Solution (first few demands and generators):")
        print(f" Pd (first 5): {x_sol[:min(5, n_demands)]}")
        print(f" Pg (first 5): {x_sol[n_demands:n_demands+min(5, n_generators)]}")
        print(f"Iterations: {iters}")
        
        # Save solution for later reference
        np.save("x_simplex.npy", x_sol)
        
        # Plot the results with the new plotting function
        plot_results = plot_simplex_results(
            x_sol=x_sol,
            gx=gx,
            Pd_max=Pd_max,
            Pg_max=Pg_max,
            U=U,
            C_prices=C_prices,
            status=status,
            iters=iters,
            save_plots=True
        )
        
        # Print summary
        print("\nSummary:")
        print(f"Total generation: {plot_results['total_generation']:.4f} MW")
        print(f"Total demand: {plot_results['total_demand']:.4f} MW")
        print(f"Market clearing price: ${plot_results['market_price']:.4f}")
        print(f"Social welfare: ${plot_results['social_welfare']:.4f}")
    else:
        print("No solution found.")
