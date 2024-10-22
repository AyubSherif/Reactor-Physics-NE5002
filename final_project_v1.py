import numpy as np
from iterative_solvers import Gauss_Seidel
import matplotlib.pyplot as plt



def diffusion_solver_1D(slab_thickness, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations):
    """
    Solves the 1D diffusion equation with vacuum boundary extrapolation using the Gauss-Seidel method.
    
    Parameters:
    slab_thickness  : Thickness of the slab
    num_mesh_points : Number of mesh points
    sigma_t         : Total cross-section
    sigma_s_ratio   : Ratio of sigma_s to sigma_t
    Q               : Fixed source
    
    Returns:
    x               : Spatial domain (mesh points)
    phi             : Neutron flux solution
    residuals       : Convergence history (residuals per iteration)
    """
    # Step 1: Calculate extrapolated distance (delta) based on diffusion coefficient
    D = 1 / (3 * sigma_t)  # Diffusion coefficient  
    new_slab_thickness = slab_thickness + 2*D # Extrapolated boundary distance
    
    # Step 2: Recalculate number of mesh points to maintain the same dx
    dx = slab_thickness / (num_mesh_points - 1)
    new_num_mesh_points = int(new_slab_thickness / dx) + 1
    
    # Step 3: Discretize the new spatial domain
    x = np.linspace(-D, slab_thickness + D, new_num_mesh_points)
    
    # Step 4: Calculate cross-sections
    sigma_s = sigma_s_ratio * sigma_t
    sigma_a = sigma_t - sigma_s
    
    # Step 5: Initial guess for the neutron flux (phi)
    Phi = np.zeros(new_num_mesh_points)
    
    
    Phi, residuals = Gauss_Seidel(new_num_mesh_points, tol, LB, RB, Phi, new_slab_thickness, sigma_t, sigma_s, D,  Q, max_iterations)
    
    return x, Phi, residuals

# Example of using the solver with vacuum boundary extrapolation
slab_thickness = 10.0  # in cm
num_mesh_points = 50
sigma_t = 1.0  # total cross-section
sigma_s_ratio = 0.9  # ratio of sigma_s / sigma_t
Q = 1.0  # fixed source
# Vacuum boundary conditions: flux = 0 at extrapolated boundaries
LB = 0  # Left boundary (vacuum)
RB = 0  # Right boundary (vacuum)
# Step 7: Solve the diffusion equation using the Gauss-Seidel method
tol = 1e-6  # Convergence tolerance
max_iterations = 1000  # Maximum number of iterations

x, phi, residuals = diffusion_solver_1D(slab_thickness, num_mesh_points, LB, RB, sigma_t, sigma_s_ratio, Q, tol, max_iterations)

# Display the solution
plt.figure(figsize=(10, 5))
plt.plot(x, phi, label='Neutron Flux $\phi$', marker='o')
plt.xlabel('Position (cm)')
plt.ylabel('Neutron Flux $\phi$')
plt.title('1D Diffusion Solution with Extrapolated Vacuum Boundary')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residuals
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Residual (Error)', marker='x')
plt.title('Residual vs Iteration (Gauss-Seidel)')
plt.xlabel('Iteration')
plt.ylabel('Residual (Error)')
plt.yscale('log')  # Log scale to observe convergence
plt.legend()
plt.grid(True)
plt.show()
