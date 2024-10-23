import numpy as np

def residual(Phi_old, Phi_new):
    """Calculate the maximum residual (error) between two iterations."""
    error = np.linalg.norm(Phi_new - Phi_old, ord=np.inf)
    return error

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