import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel_optimized, Jacobi_solver, Jacobi_solver_parallel, Jacobi_solver_vectorized

def main():
    # Get user inputs
    t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol, max_iterations, source_type = get_user_input()

    # Run the solver
    if source_type == "f":
        x, Phi, k, residuals = diffusion_eigenvalue_solver_1D(t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol, max_iterations)
        # Plot the neutron flux
        plt.figure(figsize=(10, 6))
        plt.plot(x, Phi, label='Neutron Flux', color='b')
        plt.xlabel("Position")
        plt.ylabel("Neutron Flux")
        plt.title(f"1D Diffusion Eigenvalue Solver Result (k = {k:.6f})")
    else:
        x, Phi, residuals = diffusion_solver_1D(t, num_mesh_points, material_props, left_boundary, right_boundary, phi_initial, tol, max_iterations)
        # Plot the neutron flux
        plt.figure(figsize=(10, 6))
        plt.plot(x, Phi, label='Neutron Flux', color='b')
        plt.xlabel("Position")
        plt.ylabel("Neutron Flux")
        plt.title(f"1D Diffusion Solver")

    
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the convergence history (residuals)
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(residuals)), residuals, label='Residual', color='r')
    plt.title('Residual vs Iteration (Eigenvalue Solver)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (Error)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

def get_user_input():
    """Prompt user to input main parameters, boundary conditions, and validated material properties."""

    # Validate slab thickness (t)
    while True:
        try:
            t = float(input("Enter the total slab thickness: "))
            if t <= 0:
                raise ValueError("Slab thickness must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Validate boundary conditions with "r" for reflective and default to vacuum for any other input
    left_boundary = input("Enter the left boundary condition (r for reflective, anything else for vacuum): ").strip().lower()
    if left_boundary != "r":
        left_boundary = "v"

    right_boundary = input("Enter the right boundary condition (r for reflective, anything else for vacuum): ").strip().lower()
    if right_boundary != "r":
        right_boundary = "v"

    # Validate number of media
    while True:
        try:
            num_media = int(input("Enter the number of media: "))
            if num_media <= 0:
                raise ValueError("Number of media must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Determine the type of problem
    while True:
        source_type = input("Enter the type of problem (s for fixed source, anything else for fission source): ").strip().lower()
        if source_type != "s":
            source_type = "f"
            break
        else:
            break
            
    material_props = []
    last_end = 0  # Track the end position of the last medium

    for i in range(num_media):
        print(f"\nEnter properties for medium {i + 1}:")

        # Set start position for each medium
        start = last_end

        # For the last material, it ends at the total thickness `t`
        if i == num_media - 1:
            end = t
        else:
            # Validate end position
            while True:
                try:
                    end = float(input(f"  End position for medium {i + 1} (cm): "))
                    if end <= start or end > t:
                        raise ValueError(f"End position must be greater than {start} and within the slab thickness up to {t}.")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}")

        # Validate cross-sections
        while True:
            try:
                sigma_t = float(input("  Total cross-section (sigma_t): "))
                if sigma_t <= 0:
                    raise ValueError("Total cross-section (sigma_t) must be greater than 0.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}")

        while True:
            try:
                sigma_s_to_sigma_t_ratio = float(input("  Scattering to total cross-section ratio (sigma_s_to_sigma_t_ratio): "))
                if not (0 <= sigma_s_to_sigma_t_ratio <= 1):
                    raise ValueError("Scattering ratio must be between 0 and 1.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}")
        if source_type == "f":
            # Validate fission to absorption cross-section ratio
            while True:
                try:
                    sigma_f_to_sigma_a_ratio = float(
                        input("  Fission to absorption cross-section ratio (sigma_f to sigma_a ratio): ")
                    )
                    if not (0 <= sigma_f_to_sigma_a_ratio <= 1):
                        raise ValueError("The ratio must be between 0 and 1.")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}")

            if sigma_f_to_sigma_a_ratio == 0:  # Automatically set nu to 0 if sigma_f is 0
                nu = 0.0
            else:
                # Validate neutron yield (v, nu)
                while True:
                    try:
                        nu = float(input(f"  Neutron yield (v) for medium {i + 1}: "))
                        if nu <= 0:
                            raise ValueError("Neutron yield (v) must be greater than 0.")
                        break
                    except ValueError as e:
                        print(f"Invalid input: {e}")
        else:
            # Validate fixed source strength for the medium
            while True:
                try:
                    S_i = float(input(f"  Fixed source strength (S) for medium {i + 1}: "))
                    if S_i < 0:
                        raise ValueError("Fixed source strength (S) must be greater than or equal to 0.")
                    break
                except ValueError:
                    print("Invalid input: Fixed source strength must be a number.")
        
        # Append material properties
        if source_type == "f":
            
            material_props.append({
                'start': start, 
                'end': end, 
                'sigma_t': sigma_t, 
                'sigma_s_to_sigma_t_ratio': sigma_s_to_sigma_t_ratio, 
                'sigma_f_to_sigma_a_ratio': sigma_f_to_sigma_a_ratio,
                'nu': nu
            })
        else:
            material_props.append({
                'start': start, 
                'end': end, 
                'sigma_t': sigma_t, 
                'sigma_s_to_sigma_t_ratio': sigma_s_to_sigma_t_ratio, 
                'S': S_i
            })

        last_end = end  # Update end for the next material's start

    # Validate number of mesh points
    while True:
        try:
            num_mesh_points = int(input("Enter the number of mesh points: "))
            if num_mesh_points <= 0:
                raise ValueError("Number of mesh points must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Validate initial guess for flux distribution (phi)
    while True:
        try:
            phi_initial = float(input("Enter initial guess for flux distribution (phi): "))
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Validate initial guess for eigenvalue k
    while True:
        try:
            k_initial = float(input("Enter the initial guess for eigenvalue k: "))
            if k_initial <= 0:
                raise ValueError("Eigenvalue k must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Validate convergence tolerance
    while True:
        try:
            tol = float(input("Enter the convergence tolerance: "))
            if tol <= 0:
                raise ValueError("Tolerance must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Validate maximum number of iterations
    while True:
        try:
            max_iterations = int(input("Enter the maximum number of iterations: "))
            if max_iterations <= 0:
                raise ValueError("Maximum number of iterations must be greater than 0.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    return t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol, max_iterations, source_type

def diffusion_solver_1D(t, num_mesh_points, material_props, left_boundary, right_boundary, phi_initial, tol=1e-6, max_iterations=1000000):
    """
    1D diffusion solver for heterogeneous media with user-defined boundary conditions.
    
    Parameters:
    t               : float, thickness of the slab
    num_mesh_points : int, number of mesh points
    material_props  : list of dict, properties for each medium with keys
                      'start', 'end', 'sigma_t', 'sigma_s_to_sigma_t_ratio'
    Q               : float, fixed source strength
    left_boundary   : str, 'v' for vacuum or 'r' for reflective (left boundary condition)
    right_boundary  : str, 'v' for vacuum or 'r' for reflective (right boundary condition)
    tol             : float, convergence tolerance
    max_iterations  : int, maximum number of iterations allowed

    Returns:
    x               : numpy.ndarray, spatial domain (mesh points)
    phi             : numpy.ndarray, neutron flux solution
    residuals       : list, convergence history (residuals per iteration)
    """

    dx = t / (num_mesh_points - 1)

    # Calculate extrapolated distances for vacuum boundaries
    D_left = (1 / 3 / material_props[0]['sigma_t']) // dx * dx if left_boundary == "v" else 0
    D_right = (1 / 3 / material_props[-1]['sigma_t']) // dx * dx if right_boundary == "v" else 0

    # Adjusted slab thickness and mesh points
    new_t = t + D_left + D_right
    new_num_mesh_points = int(new_t / dx) + 1
    x = np.linspace(-D_left if left_boundary == "v" else 0,
                    t + D_right if right_boundary == "v" else t,
                    new_num_mesh_points)

    # Initialize arrays for cross-sections and source term
    sigma_t = np.zeros(new_num_mesh_points)
    sigma_a = np.zeros(new_num_mesh_points)
    D = np.zeros(new_num_mesh_points)
    # Initialize the source array `b` with zeros
    b = np.zeros(new_num_mesh_points)

    # Populate material properties and source for each region
    for i in range(new_num_mesh_points):
        position = i * dx - D_left  # Adjust position for vacuum left side
    
        # Iterate through the materials to find which region the current position falls into
        for material in material_props:
            if material['start'] - D_left <= position <= material['end'] + (D_right if material == material_props[-1] else 0):
                # Populate material properties
                sigma_t[i] = material['sigma_t']
                sigma_s = material['sigma_s_to_sigma_t_ratio'] * material['sigma_t']
                sigma_a[i] = material['sigma_t'] - sigma_s
                D[i] = 1 / (3 * material['sigma_t'])
                
                # Assign the fixed source strength for the region
                b[i] =material['S']
                break


    # Initialize tridiagonal matrix components
    lower_diag = np.zeros(new_num_mesh_points - 1)
    main_diag = np.zeros(new_num_mesh_points)
    upper_diag = np.zeros(new_num_mesh_points - 1)

    # Interior points setup for the tridiagonal matrix
    for i in range(1, new_num_mesh_points - 1):
        D_left_avg = (D[i] + D[i - 1]) / 2
        D_right_avg = (D[i] + D[i + 1]) / 2

        lower_diag[i - 1] = -D_left_avg / dx**2
        main_diag[i] = (D_left_avg + D_right_avg) / dx**2 + sigma_a[i]
        upper_diag[i] = -D_right_avg / dx**2

    # Apply boundary conditions
    if left_boundary == "r":  # Reflective left boundary
        main_diag[0] = 2 * D[1] / dx**2 + sigma_a[1]
        upper_diag[0] = -2 * D[1] / dx**2
    else:  # Vacuum left boundary
        main_diag[0] = 1
        b[0] = 0

    if right_boundary == "r":  # Reflective right boundary
        main_diag[-1] = 2 * D[-2] / dx**2 + sigma_a[-2]
        lower_diag[-1] = -2 * D[-2] / dx**2
    else:  # Vacuum right boundary
        main_diag[-1] = 1
        b[-1] = 0

    # Initial guess for neutron flux
    Phi = np.full(new_num_mesh_points, phi_initial)
    A = np.zeros((new_num_mesh_points, new_num_mesh_points))
    np.fill_diagonal(A, main_diag)  # Fill main diagonal
    np.fill_diagonal(A[1:], lower_diag)  # Fill lower diagonal
    np.fill_diagonal(A[:, 1:], upper_diag)  # Fill upper diagonal

    # Solve using Gauss-Seidel method
    Phi, residuals = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b, Phi, tol, max_iterations)

    return x, Phi, residuals

def diffusion_eigenvalue_solver_1D(
    t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol=1e-6, max_iterations=1000000
):
    """
    1D diffusion eigenvalue solver for heterogeneous media with user-defined boundary conditions.
    
    Parameters:
    t               : float, thickness of the slab
    num_mesh_points : int, number of mesh points
    left_boundary   : str, 'v' for vacuum or 'r' for reflective (left boundary condition)
    right_boundary  : str, 'v' for vacuum or 'r' for reflective (right boundary condition)
    material_props  : list of dict, properties for each medium with keys
                      'start', 'end', 'sigma_t', 'sigma_s_to_sigma_t_ratio', 'sigma_f_to_sigma_a_ratio', 'nu', 'S'
    phi_initial     : float, initial guess for neutron flux
    k_initial       : float, initial guess for eigenvalue
    tol             : float, convergence tolerance
    max_iterations  : int, maximum number of iterations allowed

    Returns:
    x               : numpy.ndarray, spatial domain (mesh points)
    phi             : numpy.ndarray, neutron flux solution
    k               : float, eigenvalue
    residuals       : list, convergence history (residuals per iteration)
    """

    dx = t / (num_mesh_points - 1)

    # Calculate extrapolated distances for vacuum boundaries
    D_left = (1 / 3 / material_props[0]['sigma_t']) if left_boundary == "v" else 0
    D_right = (1 / 3 / material_props[-1]['sigma_t']) if right_boundary == "v" else 0

    # Adjusted slab thickness and mesh points
    new_t = t + D_left + D_right
    new_num_mesh_points = int(new_t / dx) + 1
    x = np.linspace(
        -D_left if left_boundary == "v" else 0,
        t + D_right if right_boundary == "v" else t,
        new_num_mesh_points,
    )

    # Initialize arrays for cross-sections and source terms
    sigma_t = np.zeros(new_num_mesh_points)
    sigma_a = np.zeros(new_num_mesh_points)
    sigma_f = np.zeros(new_num_mesh_points)
    nu = np.zeros(new_num_mesh_points)
    D = np.zeros(new_num_mesh_points)

    # Populate material properties
    for i in range(new_num_mesh_points):
        position = i * dx - D_left
        for material in material_props:
            if material['start'] - D_left <= position <= material['end'] + (D_right if material == material_props[-1] else 0):
                sigma_t[i] = material['sigma_t']
                sigma_s = material['sigma_s_to_sigma_t_ratio'] * material['sigma_t']
                sigma_a[i] = sigma_t[i] - sigma_s
                sigma_f[i] = material['sigma_f_to_sigma_a_ratio'] * sigma_a[i]
                nu[i] = material['nu']
                D[i] = 1 / (3 * sigma_t[i])
                break

    # Initial guesses
    Phi = np.full(new_num_mesh_points, phi_initial)
    Phi_new = np.zeros(new_num_mesh_points)
    k = k_initial
    residuals = []

    for iteration in range(max_iterations):
        # Update the source term
        fission_source = np.zeros(new_num_mesh_points)
        for i in range(1, new_num_mesh_points - 1):
            fission_source[i] = ((nu[i] * sigma_f[i] + nu[i + 1] * sigma_f[i + 1]) / 2.0) * Phi[i]

        b_new = fission_source / k

        # Initialize tridiagonal matrix components
        lower_diag = np.zeros(new_num_mesh_points - 1)
        main_diag = np.zeros(new_num_mesh_points)
        upper_diag = np.zeros(new_num_mesh_points - 1)

        # Setup tridiagonal matrix for interior points
        for i in range(1, new_num_mesh_points - 1):
            D_left_avg = (D[i] + D[i - 1]) / 2
            D_right_avg = (D[i] + D[i + 1]) / 2

            lower_diag[i - 1] = -D_left_avg / dx**2
            main_diag[i] = (D_left_avg + D_right_avg) / dx**2 + sigma_a[i]
            upper_diag[i] = -D_right_avg / dx**2

        # Apply boundary conditions
        if left_boundary == "r":
            main_diag[0] = 2 * D[1] / dx**2 + sigma_a[1]
            upper_diag[0] = -2 * D[1] / dx**2
        else:
            main_diag[0] = 1
            b_new[0] = 0

        if right_boundary == "r":
            main_diag[-1] = 2 * D[-2] / dx**2 + sigma_a[-2]
            lower_diag[-1] = -2 * D[-2] / dx**2
        else:
            main_diag[-1] = 1
            b_new[-1] = 0

        # Solve the linear system
        Phi_new, res = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b_new, Phi, tol, max_iterations)
        residuals.append(res)

        new_fission_source = np.zeros(new_num_mesh_points)
        for i in range(1, new_num_mesh_points - 1):
            new_fission_source[i] = ((nu[i] * sigma_f[i] + nu[i + 1] * sigma_f[i + 1]) / 2.0) * Phi_new[i]

        # Update k
        k_new = k * np.sum(new_fission_source) / np.sum(fission_source)
        if abs(k_new - k) < tol:
            k = k_new
            break
        k = k_new

        # Update flux
        Phi = Phi_new

    return x, Phi, k, residuals


# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()