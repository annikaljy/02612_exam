import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from numpy.linalg import solve, norm

def primal_dual_interior_point(c, A, b, G=None, h=None, tol=1e-8, max_iter=100, verbose=True):
    """
    
    min  c^T x
    s.t. Ax = b
         Gx <= h  (converted to standard form with slack variables)
         x >= 0
    """
    # Convert inputs to numpy arrays
    c = np.array(c, dtype=float).flatten()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    
    # Handle inequality constraints
    if G is not None and h is not None:
        G = np.array(G, dtype=float)
        h = np.array(h, dtype=float).flatten()
        
        # Convert inequality constraints to standard form with slack variables
        n_orig = c.shape[0]      # Number of original variables
        n_slack = G.shape[0]     # Number of slack variables
        
        # Extended cost vector (original costs + zeros for slack variables)
        c_ext = np.concatenate([c, np.zeros(n_slack)])
        
        # Extended constraint matrix for equality constraints
        A_ext = np.hstack([A, np.zeros((A.shape[0], n_slack))])
        
        # Add constraints for inequalities: Gx + s = h, s >= 0
        G_ext = np.hstack([G, np.eye(n_slack)])
        
        # Combined equality constraints
        A_full = np.vstack([A_ext, G_ext])
        b_full = np.concatenate([b, h])
        
        # Set up for standard form
        c_std = c_ext
        A_std = A_full
        b_std = b_full
        n_orig_vars = n_orig
    else:
        # Only equality constraints
        c_std = c
        A_std = A
        b_std = b
        n_orig_vars = c.shape[0]
    
    # Problem dimensions
    m, n = A_std.shape  # m = number of constraints, n = number of variables
    
    # Store history of residuals
    residuals = {
        'primal': [],
        'dual': [],
        'complementarity': [],
        'mu': []
    }
    
    # Start with more conservative positive values
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)

    # Adjust starting point to better satisfy constraints
    # Try solving least squares for Ax = b
    try:
        x_ls = np.linalg.lstsq(A_std, b_std, rcond=None)[0]
        x0 = np.maximum(x_ls, 1.0)  # Make sure all components are positive
    except:
        # If least squares fails, use default values
        x0 = np.ones(n)

    # Compute initial z to make c - A^T y - z = 0
    z0 = c_std.copy()  # Start with c
    # Try to get a y0 that makes z0 positive
    try:
        # Solve for a y0 that minimizes ||c - A^T y0||
        y0 = np.linalg.lstsq(A_std.T, c_std, rcond=None)[0]
        z0 = np.maximum(c_std - A_std.T @ y0, 1e-6)  # ENSURE POSITIVE z0 HERE
    except:
        # If that fails, just ensure z0 is positive
        y0 = np.zeros(m)
        z0 = np.maximum(c_std.copy(), 1e-6) # ENSURE POSITIVE z0 HERE

    # Ensure z0 is positive
    z0 = np.maximum(z0, 1e-6) # ENSURE POSITIVE z0 HERE
    
    # Initialize variables
    x = x0
    y = y0
    z = z0
    
    # Initial mu
    mu = np.dot(x, z) / n
    
    # Compute initial residuals
    r_p = b_std - A_std @ x
    r_d = c_std - A_std.T @ y - z
    r_c = x * z  # Complementarity residual
    
    # Record initial residuals
    residuals['primal'].append(norm(r_p))
    residuals['dual'].append(norm(r_d))
    residuals['complementarity'].append(norm(r_c))
    residuals['mu'].append(mu)
    
    if verbose:
        print(f"Initial residuals: primal={norm(r_p):.6e}, dual={norm(r_d):.6e}, mu={mu:.6e}")
    
    # --------- Main algorithm loop ---------
    for iteration in range(max_iter):
        # Current duality gap
        mu = np.dot(x, z) / n
        
        # Compute residuals
        r_p = b_std - A_std @ x
        r_d = c_std - A_std.T @ y - z
        r_c = x * z  # Element-wise product
        
        # Record residuals
        residuals['primal'].append(norm(r_p))
        residuals['dual'].append(norm(r_d))
        residuals['complementarity'].append(norm(r_c))
        residuals['mu'].append(mu)
        
        # Check for convergence
        if norm(r_p) < tol and norm(r_d) < tol and mu < tol:
            if verbose:
                print(f"Converged in {iteration} iterations")
            break
        
        # Print progress
        if verbose and iteration % 5 == 0:
            print(f"Iteration {iteration}: primal_res={norm(r_p):.6e}, dual_res={norm(r_d):.6e}, mu={mu:.6e}")
        
        # Check for numerical problems
        if np.isnan(mu) or np.isinf(mu) or mu > 1e10:
            if verbose:
                print(f"Numerical issues detected. Attempting recovery...")
            
            x = np.maximum(x, 1.0)  # Ensure positivity
            z = np.maximum(z, 1.0)  # Ensure positivity
            
            # Reset mu
            mu = np.dot(x, z) / n
            
            # If still problematic, terminate
            if np.isnan(mu) or np.isinf(mu) or mu > 1e10:
                if verbose:
                    print(f"Recovery failed. Terminating at iteration {iteration}.")
                break
        
        # Compute scaling parameter with better numerical properties
        X = np.diag(x)
        Z = np.diag(z)
        
        # Use X^(1/2) Z^(-1/2) as scaling
        d = np.sqrt(x / z)  # diagonal of X^(1/2) Z^(-1/2)
        
        # Limit scaling to reasonable values
        d = np.clip(d, 1e-6, 1e6)
        
        # Form normal equations matrix
        # M = A D D^T A^T
        D = np.diag(d)
        AD = A_std @ D
        M = AD @ AD.T
        
        # Add small regularization for stability
        reg = 1e-8 * (1 + np.trace(M)/M.shape[0])
        M = M + reg * np.eye(M.shape[0])
        
        # Compute right-hand side for normal equations
        # rhs = r_p - A D D^T (r_c ./ z - r_d)
        rhs_p = r_p
        rhs_z = r_d
        rhs_c = -r_c
        
        # Combine terms for normal equations RHS
        temp = d * d * (rhs_c/z - rhs_z)
        rhs = rhs_p - A_std @ temp
        
        # Solve for predictor direction
        try:
            dy_aff = solve(M, rhs)
        except np.linalg.LinAlgError:
            # If direct solve fails, use a more robust approach
            dy_aff = np.linalg.lstsq(M, rhs, rcond=None)[0]
        
        # Back-substitution to get dz and dx
        dz_aff = rhs_z + A_std.T @ dy_aff
        dx_aff = D @ D @ (dz_aff - rhs_c/z)
        
        # Find step sizes that maintain positivity
        alpha_p_aff = min(0.99, 1.0 / max(0.0, -np.min(dx_aff/x)) if np.any(dx_aff < 0) else 1.0)
        alpha_d_aff = min(0.99, 1.0 / max(0.0, -np.min(dz_aff/z)) if np.any(dz_aff < 0) else 1.0)
        
        # Compute affine duality gap
        mu_aff = np.dot(x + alpha_p_aff * dx_aff, z + alpha_d_aff * dz_aff) / n
        
        # Compute centering parameter
        sigma = (mu_aff / mu) ** 3
        sigma = min(max(sigma, 0.1), 0.9)  # Keep between 0.1 and 0.9
        
        # Compute corrector term
        dxdz_aff = dx_aff * dz_aff
        corrector = sigma * mu - dxdz_aff
        
        # Compute right-hand side with corrector
        rhs_c_corr = rhs_c - corrector
        
        # Combine terms for normal equations RHS
        temp_corr = d * d * (rhs_c_corr/z - rhs_z)
        rhs_corr = rhs_p - A_std @ temp_corr
        
        # Solve for corrected direction
        try:
            dy = solve(M, rhs_corr)
        except np.linalg.LinAlgError:
            # If direct solve fails, use a more robust approach
            dy = np.linalg.lstsq(M, rhs_corr, rcond=None)[0]
        
        # Back-substitution to get dz and dx
        dz = rhs_z + A_std.T @ dy
        dx = D @ D @ (dz - rhs_c_corr/z)
        
        # Find step sizes that maintain positivity
        alpha_p = min(0.95, 1.0 / max(0.0, -np.min(dx/x)) if np.any(dx < 0) else 0.95)
        alpha_d = min(0.95, 1.0 / max(0.0, -np.min(dz/z)) if np.any(dz < 0) else 0.95)
        
        # Make step sizes even more conservative for stability
        alpha_p = min(alpha_p, 0.9)
        alpha_d = min(alpha_d, 0.9)
        
        # Update variables
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        z = z + alpha_d * dz
        
        # Ensure variables stay positive
        x = np.maximum(x, 1e-10)
        z = np.maximum(z, 1e-10)
    
    # Extract final solution components
    if G is not None:
        # Extract original variables
        x_orig = x[:n_orig_vars]
    else:
        x_orig = x
        
    obj_val = np.dot(c, x_orig)
    
    # Determine status
    if iteration >= max_iter - 1:
        status = "max_iterations"
    elif np.isnan(mu) or np.isinf(mu):
        status = "numerical_issues"
    else:
        status = "optimal"
    
    # Return results
    return {
        'x': x_orig,
        'y': y,
        'z': z,
        'obj_value': obj_val,
        'iterations': iteration + 1,
        'residuals': residuals,
        'status': status
    }

def solve_market_clearing_problem(data_file="LP_test.mat", verbose=True, plot_results=True):
    """
    The problem is formulated as:
    min    Σ C_g * P_g - Σ U_d * P_d
    s.t.   Σ P_g = Σ P_d   (power balance)
           0 <= P_g <= Pg_max  for all g
           0 <= P_d <= Pd_max  for all d
    """
    # Load data from the .mat file
    try:
        data = scipy.io.loadmat(data_file)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None
    
    # Extract parameters
    U = data["U"].flatten().astype(float)  # Utility (negative cost) for demand
    C = data["C"].flatten().astype(float)  # Cost of generation
    Pd_max = data["Pd_max"].flatten().astype(float)  # Maximum demand
    Pg_max = data["Pg_max"].flatten().astype(float)  # Maximum generation capacity
    
    num_generators = len(C)
    num_demands = len(U)
    
    # Print problem information
    if verbose:
        print(f"Market Clearing Problem:")
        print(f"Number of generators: {num_generators}")
        print(f"Number of demands: {num_demands}")
    
    
    # Decision variables: [P_g1, P_g2, ..., P_gG, P_d1, P_d2, ..., P_dD]
    # Objective: minimize Σ C_g * P_g - Σ U_d * P_d
    c = np.concatenate([C, -U])
    
    # Power balance constraint: Σ P_g = Σ P_d
    A_balance = np.concatenate([-np.ones(num_generators),np.ones(num_demands)]).reshape(1, -1)
    b_balance = np.array([0.0])
    
    # Inequality constraints: Bounds
    # Upper bounds for generators: P_g <= Pg_max
    G_gen_upper = np.zeros((num_generators, num_generators + num_demands))
    for i in range(num_generators):
        G_gen_upper[i, i] = 1.0
    h_gen_upper = Pg_max
    
    # Upper bounds for demands: P_d <= Pd_max
    G_dem_upper = np.zeros((num_demands, num_generators + num_demands))
    for i in range(num_demands):
        G_dem_upper[i, num_generators + i] = 1.0
    h_dem_upper = Pd_max
    
    # Lower bounds for generators: 0 <= P_g
    G_gen_lower = np.zeros((num_generators, num_generators + num_demands))
    for i in range(num_generators):
        G_gen_lower[i, i] = -1.0
    h_gen_lower = np.zeros(num_generators)

    # Lower bounds for demands: 0 <= P_d
    G_dem_lower = np.zeros((num_demands, num_generators + num_demands))
    for i in range(num_demands):
        G_dem_lower[i, num_generators + i] = -1.0
    h_dem_lower = np.zeros(num_demands)

    # Combine all inequality constraints
    G = np.vstack([G_gen_upper, G_dem_upper, G_gen_lower, G_dem_lower])
    h = np.concatenate([h_gen_upper, h_dem_upper, h_gen_lower, h_dem_lower])
    
    # Solve the LP using our improved primal-dual interior-point method
    if verbose:
        print("\nSolving the market clearing problem...")
    
    # Use tighter tolerance and more iterations
    result = primal_dual_interior_point(c, A_balance, b_balance, G, h, tol=1e-5, max_iter=1000, verbose=verbose)
    
    # Extract results
    x_opt = result['x']
    y_opt = result['y']  # Shadow price (Lagrange multiplier) of the power balance constraint
    P_g = x_opt[:num_generators]  # Optimal generation
    P_d = x_opt[num_generators:]  # Optimal demand
    
    # Calculate market clearing price (shadow price of the power balance constraint)
    market_price = abs(y_opt[0])  # Take absolute value as the sign depends on formulation
    
    # Print results
    if verbose:
        print("\nOptimal Solution:")
        print(f"Objective value: {result['obj_value']:.4f}")
        print(f"Market clearing price: {market_price:.4f}")
        print(f"Total iterations: {result['iterations']}")
        print(f"Status: {result['status']}")
        
        print(f"\nGeneration schedule:")
        total_gen = 0
        for i in range(num_generators):
            if P_g[i] > 1e-4:  # Only show active generators
                print(f"Generator {i+1}: {P_g[i]:.4f} MW (Cost: {C[i]:.4f}, Capacity: {Pg_max[i]:.4f})")
                total_gen += P_g[i]
                
        print(f"\nDemand schedule:")
        total_dem = 0
        for i in range(num_demands):
            if P_d[i] > 1e-4:  # Only show active demands
                print(f"Demand {i+1}: {P_d[i]:.4f} MW (Utility: {U[i]:.4f}, Max Demand: {Pd_max[i]:.4f})")
                total_dem += P_d[i]
                
        print(f"\nTotal generation: {total_gen:.4f} MW")
        print(f"Total demand: {total_dem:.4f} MW")
        print(f"Generation-demand balance: {total_gen - total_dem:.4f} MW")
    
    
    if plot_results:
        # Get residuals from the result
        residuals = result['residuals']
        
        # Plot residuals
        plt.figure(figsize=(12, 6))
        plt.semilogy(residuals['primal'], label='Primal Residual')
        plt.semilogy(residuals['dual'], label='Dual Residual')
        plt.semilogy(residuals['complementarity'], label='Complementarity')
        plt.semilogy(residuals['mu'], label='Duality Gap')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (log scale)')
        plt.title('Convergence of Residuals')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('residuals.png', dpi=300)
        
        # Plot generation and demand
        plt.figure(figsize=(14, 10))
        
        # Plot generators
        plt.subplot(2, 1, 1)
        active_gen = P_g > 1e-4
        if np.any(active_gen):
            active_indices = np.where(active_gen)[0] + 1
            plt.bar(active_indices, P_g[active_gen], color='blue', alpha=0.7, label='Generation')
            plt.bar(active_indices, Pg_max[active_gen], color='lightblue', alpha=0.3, label='Max Capacity')
            for i, idx in enumerate(np.where(active_gen)[0]):
                plt.text(active_indices[i], P_g[idx] + 0.05 * max(Pg_max), f'${C[idx]:.2f}', ha='center')
        plt.xlabel('Generator')
        plt.ylabel('Power (MW)')
        plt.title('Generator Output vs Capacity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        #demands
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
        plt.savefig('market_clearing.png', dpi=300)
        
        #supply curve and demand curve
        plt.figure(figsize=(12, 8))
        
        #generators by cost for supply curve
        sorted_indices = np.argsort(C)
        sorted_C = C[sorted_indices]
        sorted_Pg_max = Pg_max[sorted_indices]
        
        #supply curve (step function)
        cum_capacity = np.zeros(num_generators + 1)
        cum_capacity[1:] = np.cumsum(sorted_Pg_max)
        supply_x = np.repeat(cum_capacity, 2)[1:-1]
        supply_y = np.repeat(sorted_C, 2)
        
        #demands by utility (descending) for demand curve
        sorted_dem_indices = np.argsort(-U)
        sorted_U = U[sorted_dem_indices]
        sorted_Pd_max = Pd_max[sorted_dem_indices]
        
        #demand curve (step function)
        cum_demand = np.zeros(num_demands + 1)
        cum_demand[1:] = np.cumsum(sorted_Pd_max)
        demand_x = np.repeat(cum_demand, 2)[1:-1]
        demand_y = np.repeat(sorted_U, 2)
        
        
        plt.step(supply_x, supply_y, where='post', label='Supply Curve', color='blue', linewidth=2)
        plt.step(demand_x, demand_y, where='post', label='Demand Curve', color='green', linewidth=2)
        
        #market clearing price and quantity
        total_cleared = np.sum(P_g)  # Should equal sum(P_d)
        plt.axhline(y=market_price, color='red', linestyle='--', label=f'Market Price: ${market_price:.2f}')
        plt.axvline(x=total_cleared, color='purple', linestyle='--', label=f'Cleared Quantity: {total_cleared:.2f} MW')
        
        #active generators and demands
        active_gen = P_g > 1e-4
        plt.scatter(cum_capacity[1:][active_gen[sorted_indices]], sorted_C[active_gen[sorted_indices]], 
                    color='red', s=100, zorder=5, label='Active Generators')
        
        active_dem = P_d > 1e-4
        plt.scatter(cum_demand[1:][active_dem[sorted_dem_indices]], sorted_U[active_dem[sorted_dem_indices]], 
                    color='orange', s=100, zorder=5, label='Active Demands')
        
        plt.xlabel('Capacity/Demand (MW)')
        plt.ylabel('Price ($/MW)')
        plt.title('Supply Curve, Demand Curve, and Market Clearing')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('supply_demand_curves.png', dpi=300)
        
        plt.show()
    
    return {
        'result': result,
        'P_g': P_g,
        'P_d': P_d,
        'market_price': market_price,
        'total_generation': np.sum(P_g),
        'total_demand': np.sum(P_d),
        'objective_value': result['obj_value']
    }

# Example usage
if __name__ == "__main__":
    solution = solve_market_clearing_problem("LP_test.mat", verbose=True, plot_results=True)
    
    # Print summary
    print("\nSummary:")
    print(f"Total generation: {solution['total_generation']:.4f} MW")
    print(f"Total demand: {solution['total_demand']:.4f} MW")
    print(f"Market clearing price: ${solution['market_price']:.4f}")
    print(f"Social welfare (negative objective): ${-solution['objective_value']:.4f}")