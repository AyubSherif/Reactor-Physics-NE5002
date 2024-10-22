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