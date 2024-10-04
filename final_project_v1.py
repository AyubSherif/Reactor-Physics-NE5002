import numpy as np
import matplotlib.pyplot as plt

def finite_volume_1d_diffusion(Nx=100, Lx=1.0, D=0.1, dt=0.01, T=0.5, tol=1e-6, max_iter=1000):
    """
    Solves the 1D diffusion equation using the finite volume method and Gauss-Seidel iteration.

    Parameters:
    Nx : int
        Number of grid points.
    Lx : float
        Length of the domain.
    D : float
        Diffusion coefficient.
    dt : float
        Time step.
    T : float
        Total simulation time.
    tol : float
        Convergence tolerance for Gauss-Seidel iteration.
    max_iter : int
        Maximum number of iterations for Gauss-Seidel.

    Returns:
    x : np.array
        The spatial grid.
    u : np.array
        The solution at the final time.
    """

    # Calculate grid spacing and number of time steps
    dx = Lx / (Nx - 1)
    Nt = int(T / dt)

    # Setup the grid
    x = np.linspace(0, Lx, Nx)
    u = np.zeros(Nx)  # Initial solution (e.g., u = 0 everywhere)

    # Initial condition: Gaussian pulse
    u = np.exp(-50 * (x - Lx / 2) ** 2)

    # Plot the initial condition
    plt.plot(x, u, label="Initial Condition")
    plt.title("Finite Volume Method - 1D Diffusion")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()

    # Define the finite volume coefficients
    alpha = D * dt / dx**2  # Diffusion coefficient

    def gauss_seidel_step(u_old, u_new, alpha, Nx, tol, max_iter):
        """
        Performs one Gauss-Seidel iteration to solve the system of equations for the next time step.

        Parameters:
        u_old : np.array
            The solution from the previous time step.
        u_new : np.array
            The solution to be updated.
        alpha : float
            The coefficient in the finite volume method.
        Nx : int
            Number of grid points.
        tol : float
            Tolerance for convergence.
        max_iter : int
            Maximum number of iterations.

        Returns:
        u_new : np.array
            The updated solution after Gauss-Seidel iteration.
        """
        for iteration in range(max_iter):
            u_old_iter = u_new.copy()  # Save old values to check for convergence
            for i in range(1, Nx - 1):  # Interior points only
                u_new[i] = (1 - 2 * alpha) * u_old[i] + alpha * (u_new[i-1] + u_old[i+1])

            # Check for convergence
            if np.linalg.norm(u_new - u_old_iter, ord=np.inf) < tol:
                break
        return u_new

    # Time stepping loop
    u_new = u.copy()  # Placeholder for the next time step
    for n in range(Nt):
        u_new = gauss_seidel_step(u, u_new, alpha, Nx, tol, max_iter)
        u = u_new.copy()  # Update the solution for the next time step

    # Plot the final solution
    plt.plot(x, u, label="Final Solution")
    plt.title(f"1D Diffusion at t = {T}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.show()

    return x, u

# Example usage
x, u_final = finite_volume_1d_diffusion(Nx=100, Lx=1.0, D=0.1, dt=0.01, T=0.5)
