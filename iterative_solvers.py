import numpy as np
from multiprocessing import Pool, cpu_count
from time import time

# Decorator to measure the execution time of a function
def timer_func(func):
    """Decorator to measure the execution time of a function."""
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result
    return wrap_func


def residual(Phi_old, Phi_new):
    """Calculate the maximum residual (error) between two iterations."""
    error = np.linalg.norm(Phi_new - Phi_old, ord=np.inf)
    return error

@timer_func
def Gauss_Seidel(A, b, Phi, tol, max_iterations):
    """
    Generic Gauss-Seidel solver for the system of linear equations A * Phi = b.
    
    Parameters:
    A               : Coefficient matrix
    b               : Right-hand side vector
    Phi             : Initial guess for the solution
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    Phi             : Updated solution after convergence
    residuals       : Residuals (errors) over the iterations
    """
    num_points = len(b)
    residuals = []
    
    for iteration in range(max_iterations):
        Phi_old = np.copy(Phi)  # Save the previous solution for convergence check

        # Perform Gauss-Seidel iteration
        for i in range(num_points):
            sum_A_Phi = np.dot(A[i, :i], Phi[:i]) + np.dot(A[i, i+1:], Phi[i+1:])
            Phi[i] = (b[i] - sum_A_Phi) / A[i, i]

        # Calculate and store the residual
        res = residual(Phi_old, Phi)
        residuals.append(res)

        # Check convergence
        if res < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")

    return Phi, residuals

@timer_func
def Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations):
    """
    Optimized Gauss-Seidel solver for a tridiagonal system.
    
    Parameters:
    lower_diag      : Lower diagonal of the tridiagonal matrix
    main_diag       : Main diagonal of the tridiagonal matrix
    upper_diag      : Upper diagonal of the tridiagonal matrix
    b               : Right-hand side vector
    Phi             : Initial guess for the solution
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    Phi             : Updated solution after convergence
    residuals       : Residuals (errors) over the iterations
    """
    num_points = len(b)
    residuals = []
    
    for iteration in range(max_iterations):
        Phi_old = np.copy(Phi)  # Save the previous solution for convergence check

        # Perform Gauss-Seidel iteration
        for i in range(num_points):
            if i == 0:
                # Left boundary (no need for lower diagonal)
                sum_A_Phi = upper_diag[i] * Phi[i+1]
            elif i == num_points - 1:
                # Right boundary (no need for upper diagonal)
                sum_A_Phi = lower_diag[i-1] * Phi[i-1]
            else:
                # Internal grid points
                sum_A_Phi = lower_diag[i-1] * Phi[i-1] + upper_diag[i] * Phi[i+1]

            # Update solution
            Phi[i] = (b[i] - sum_A_Phi) / main_diag[i]

        # Calculate and store the residual
        res = np.linalg.norm(Phi - Phi_old, ord=np.inf)
        residuals.append(res)

        # Check convergence
        if res < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")

    return Phi, residuals

@timer_func
def Jacobi_solver(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations):
    """
    Jacobi solver for a tridiagonal system.
    
    Parameters:
    lower_diag      : Lower diagonal of the tridiagonal matrix
    main_diag       : Main diagonal of the tridiagonal matrix
    upper_diag      : Upper diagonal of the tridiagonal matrix
    b               : Right-hand side vector
    Phi             : Initial guess for the solution
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    Phi             : Updated solution after convergence
    residuals       : Residuals (errors) over the iterations
    """
    num_points = len(b)
    residuals = []
    
    for iteration in range(max_iterations):
        Phi_old = np.copy(Phi)  # Keep previous iteration values for Jacobi updates

        # Perform Jacobi iteration
        for i in range(num_points):
            if i == 0:
                # Left boundary (no need for lower diagonal)
                sum_A_Phi = upper_diag[i] * Phi_old[i+1]
            elif i == num_points - 1:
                # Right boundary (no need for upper diagonal)
                sum_A_Phi = lower_diag[i-1] * Phi_old[i-1]
            else:
                # Internal grid points
                sum_A_Phi = lower_diag[i-1] * Phi_old[i-1] + upper_diag[i] * Phi_old[i+1]

            # Update solution based on previous values only
            Phi[i] = (b[i] - sum_A_Phi) / main_diag[i]

        # Calculate and store the residual
        res = np.linalg.norm(Phi - Phi_old, ord=np.inf)
        residuals.append(res)

        # Check convergence
        if res < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")

    return Phi, residuals




def jacobi_update(i, lower_diag, main_diag, upper_diag, b, Phi_old):
    """Function to compute the updated value for a single entry in the Jacobi method."""
    if i == 0:
        # Left boundary (no need for lower diagonal)
        sum_A_Phi = upper_diag[i] * Phi_old[i + 1]
    elif i == len(b) - 1:
        # Right boundary (no need for upper diagonal)
        sum_A_Phi = lower_diag[i - 1] * Phi_old[i - 1]
    else:
        # Internal grid points
        sum_A_Phi = lower_diag[i - 1] * Phi_old[i - 1] + upper_diag[i] * Phi_old[i + 1]
    
    # Return the updated Phi[i] value
    return (b[i] - sum_A_Phi) / main_diag[i]

@timer_func
def Jacobi_solver_parallel(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations):
    """
    Parallelized Jacobi solver for a tridiagonal system using multiprocessing.
    
    Parameters:
    lower_diag      : Lower diagonal of the tridiagonal matrix
    main_diag       : Main diagonal of the tridiagonal matrix
    upper_diag      : Upper diagonal of the tridiagonal matrix
    b               : Right-hand side vector
    Phi             : Initial guess for the solution
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    Phi             : Updated solution after convergence
    residuals       : Residuals (errors) over the iterations
    """
    num_points = len(b)
    residuals = []
    
    # Use a multiprocessing pool to parallelize the update
    with Pool(cpu_count()) as pool:
        for iteration in range(max_iterations):
            Phi_old = np.copy(Phi)  # Keep previous iteration values for Jacobi updates
            
            # Parallelize the updates for all points
            new_Phi = pool.starmap(jacobi_update, [(i, lower_diag, main_diag, upper_diag, b, Phi_old) for i in range(num_points)])
            Phi = np.array(new_Phi)

            # Calculate and store the residual
            res = np.linalg.norm(Phi - Phi_old, ord=np.inf)
            residuals.append(res)

            # Check convergence
            if res < tol:
                print(f"Converged in {iteration + 1} iterations.")
                break
        else:
            print("Did not converge within the maximum number of iterations.")

    return Phi, residuals
@timer_func
def Jacobi_solver_vectorized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations):
    """
    Vectorized Jacobi solver for a tridiagonal system.
    
    Parameters:
    lower_diag      : Lower diagonal of the tridiagonal matrix
    main_diag       : Main diagonal of the tridiagonal matrix
    upper_diag      : Upper diagonal of the tridiagonal matrix
    b               : Right-hand side vector
    Phi             : Initial guess for the solution
    tol             : Convergence tolerance
    max_iterations  : Maximum number of iterations allowed
    
    Returns:
    Phi             : Updated solution after convergence
    residuals       : Residuals (errors) over the iterations
    """
    num_points = len(b)
    residuals = []
    
    for iteration in range(max_iterations):
        Phi_old = np.copy(Phi)
        
        # Vectorized update: calculate new Phi without explicit loops
        Phi[1:-1] = (b[1:-1] - lower_diag[:-1] * Phi_old[:-2] - upper_diag[1:] * Phi_old[2:]) / main_diag[1:-1]
        
        # Left boundary (only uses the upper_diag)
        Phi[0] = (b[0] - upper_diag[0] * Phi_old[1]) / main_diag[0]

        # Right boundary (only uses the lower_diag)
        Phi[-1] = (b[-1] - lower_diag[-1] * Phi_old[-2]) / main_diag[-1]
        
        # Calculate and store the residual
        res = np.linalg.norm(Phi - Phi_old, ord=np.inf)
        residuals.append(res)

        # Check convergence
        if res < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")

    return Phi, residuals

def jacobi_step(lower_diag, main_diag, upper_diag, b, Phi, iterations=1):
    """Perform a few Jacobi iterations."""
    num_points = len(b)
    for _ in range(iterations):
        Phi_old = np.copy(Phi)
        for i in range(1, num_points - 1):
            Phi[i] = (b[i] - lower_diag[i - 1] * Phi_old[i - 1] - upper_diag[i] * Phi_old[i + 1]) / main_diag[i]
        # Boundary points
        Phi[0] = (b[0] - upper_diag[0] * Phi_old[1]) / main_diag[0]
        Phi[-1] = (b[-1] - lower_diag[-1] * Phi_old[-2]) / main_diag[-1]
    return Phi

def restrict(residual):
    """Restrict the residual to a coarser grid by averaging adjacent points."""
    return 0.5 * (residual[::2] + residual[1::2])

def prolongate(correction):
    """Prolongate the correction from coarse grid to fine grid using linear interpolation."""
    fine_points = 2 * len(correction) - 1
    fine_correction = np.zeros(fine_points)
    fine_correction[::2] = correction  # Direct copy for the coarse points
    fine_correction[1::2] = 0.5 * (correction[:-1] + correction[1:])  # Average for interpolated points
    return fine_correction

def compute_residual(lower_diag, main_diag, upper_diag, b, Phi):
    """Compute the residual on the current grid."""
    residual = np.copy(b)
    residual[1:-1] -= (lower_diag * Phi[:-2] + main_diag[1:-1] * Phi[1:-1] + upper_diag * Phi[2:])
    residual[0] -= main_diag[0] * Phi[0] + upper_diag[0] * Phi[1]
    residual[-1] -= lower_diag[-1] * Phi[-2] + main_diag[-1] * Phi[-1]
    return residual
@timer_func
def multigrid_v_cycle(lower_diag, main_diag, upper_diag, b, Phi, tol, max_depth, smooth_iterations=3):
    """
    Multigrid V-cycle to solve the system of equations using Jacobi smoothing.
    
    Parameters:
    lower_diag        : Lower diagonal of the tridiagonal matrix
    main_diag         : Main diagonal of the tridiagonal matrix
    upper_diag        : Upper diagonal of the tridiagonal matrix
    b                 : Right-hand side vector
    Phi               : Current solution guess
    tol               : Convergence tolerance
    max_depth         : Maximum depth of the V-cycle (controls the coarsest grid)
    smooth_iterations : Number of Jacobi iterations for smoothing at each level
    
    Returns:
    Phi               : Updated solution after one V-cycle
    """
    # Base case: if we reach the coarsest level, solve directly using a few smoothing steps
    if len(b) <= 3 or max_depth == 0:
        return jacobi_step(lower_diag, main_diag, upper_diag, b, Phi, iterations=10)

    # Pre-smoothing on the current grid
    Phi = jacobi_step(lower_diag, main_diag, upper_diag, b, Phi, iterations=smooth_iterations)

    # Compute the residual
    residual = compute_residual(lower_diag, main_diag, upper_diag, b, Phi)

    # Restrict the residual to a coarser grid
    coarse_b = restrict(residual)
    coarse_Phi = np.zeros_like(coarse_b)  # Initial guess on the coarse grid

    # Restrict the operator to the coarse grid (halve the number of points)
    coarse_lower_diag = 0.25 * lower_diag[::2]
    coarse_main_diag = main_diag[::2]
    coarse_upper_diag = 0.25 * upper_diag[::2]

    # Recursive call on the coarse grid (V-cycle)
    coarse_Phi = multigrid_v_cycle(coarse_lower_diag, coarse_main_diag, coarse_upper_diag, coarse_b, coarse_Phi, tol, max_depth - 1, smooth_iterations)

    # Prolongate the correction back to the fine grid
    fine_correction = prolongate(coarse_Phi)

    # Update the solution on the fine grid
    Phi += fine_correction

    # Post-smoothing on the fine grid
    Phi = jacobi_step(lower_diag, main_diag, upper_diag, b, Phi, iterations=smooth_iterations)

    return Phi