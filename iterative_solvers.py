import numpy as np

# Jacobi Solver
def jacobi(A, b, x0=None, tol=1e-6, max_iterations=10000):
    """
    Solves the system of linear equations Ax = b via the Jacobi iterative method.
    
    Parameters:
        A (numpy.ndarray): Coefficient matrix (must be square).
        b (numpy.ndarray): Right-hand side vector.
        x0 (numpy.ndarray): Initial guess for the solution. Defaults to a zero vector if not provided.
        tol (float): Convergence criterion. The iteration stops when the error is less than tol.
        max_iterations (int): Maximum number of iterations.

    Returns:
        x (numpy.ndarray): Approximate solution vector.
        errors (list): List of errors at each iteration.
        iterations (int): The number of iterations performed.
        converged (bool): Whether the method converged.
    """
    n = len(b)
    
    # Initial guess
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    
    x_old = x0.copy()
    errors = []

    for iteration in range(max_iterations):
        x_new = np.zeros_like(x_old)
        
        # Perform the Jacobi update
        for i in range(n):
            sum_terms = np.dot(A[i, :i], x_old[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x_new[i] = (b[i] - sum_terms) / A[i, i]
        
        # Calculate error and check for convergence
        error = np.linalg.norm(x_new - x_old, ord=np.inf)
        errors.append(error)
        
        if error < tol:
            return x_new, errors, iteration + 1, True
        
        x_old = x_new.copy()
    
    return x_old, errors, max_iterations, False  # Did not converge


# Gauss-Seidel Solver
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iterations=10000):
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel iterative method.
    
    Parameters:
        A (numpy.ndarray): Coefficient matrix (must be square).
        b (numpy.ndarray): Right-hand side vector.
        x0 (numpy.ndarray): Initial guess for the solution. Defaults to a zero vector if not provided.
        tol (float): Convergence criterion. The iteration stops when the error is less than tol.
        max_iterations (int): Maximum number of iterations.

    Returns:
        x (numpy.ndarray): Approximate solution vector.
        errors (list): List of errors at each iteration.
        iterations (int): The number of iterations performed.
        converged (bool): Whether the method converged.
    """
    n = len(b)

    # Initial guess
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x0.astype(float)

    errors = []

    for k in range(max_iterations):
        x_old = x.copy()
        
        # Perform Gauss-Seidel update
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Calculate error and check for convergence
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        
        if error < tol:
            return x, errors, k + 1, True

    return x, errors, max_iterations, False  # Did not converge


# SOR Solver
def sor_solver(A, b, omega, x0=None, tol=1e-6, max_iterations=10000):
    """
    Solves the system of linear equations Ax = b using the Successive Over-Relaxation (SOR) method.
    
    Parameters:
        A (numpy.ndarray): Coefficient matrix (must be square).
        b (numpy.ndarray): Right-hand side vector.
        omega (float): Relaxation factor (0 < omega < 2).
        x0 (numpy.ndarray): Initial guess for the solution. Defaults to a zero vector if not provided.
        tol (float): Convergence criterion. The iteration stops when the error is less than tol.
        max_iterations (int): Maximum number of iterations.

    Returns:
        x (numpy.ndarray): Approximate solution vector.
        errors (list): List of errors at each iteration.
        iterations (int): The number of iterations performed.
        converged (bool): Whether the method converged.
    """
    n = len(b)

    # Initial guess
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x0.astype(float)
    
    errors = []

    for k in range(max_iterations):
        x_old = x.copy()
        
        # Perform SOR update
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sum1 - sum2) / A[i, i]

        # Calculate error and check for convergence
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        
        if error < tol:
            return x, errors, k + 1, True

    return x, errors, max_iterations, False  # Did not converge
