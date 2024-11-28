import numpy as np
import matplotlib.pyplot as plt
from iterative_solvers import Gauss_Seidel_optimized, Jacobi_solver, Jacobi_solver_parallel, Jacobi_solver_vectorized
import datetime

def write_to_file(content):
        """Helper function to write content to the file."""
        with open("data.txt", "a") as file:
            file.write(content + "\n")

def version_data():
    # Define the details
    code_name = "Monoenergetic 1D Diffusion Solver for Eigenvalue Problems"
    version_number = 5
    author_name = "Ayub Sherif"
    execution_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the version data as a formatted string
    version_data = (
        f"Code Name: {code_name}\n"
        f"Version: {version_number}\n"
        f"Author: {author_name}\n"
        f"Date and Time of Execution: {execution_time}\n"
    )

    # Print to screen
    print(version_data)

    # Write to an output file
    write_to_file(version_data)

    print(f"Input and output data is written to data.txt.")



def plot_results(x, Phi, residuals, solver_type, k=None):
    """Plots the neutron flux and residuals."""
    # Plot the neutron flux
    plt.figure(figsize=(10, 6))
    plt.plot(x, Phi, label='Neutron Flux', color='b')
    plt.xlabel("Position")
    plt.ylabel("Neutron Flux")
    title = f"1D Diffusion Eigenvalue Solver (k = {k:.6f})" if solver_type == "fission" else "1D Diffusion Solver"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the convergence history (residuals)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(residuals)), residuals, label='Residual', color='r')
    plt.title('Residual vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (Error)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    version_data()

    # Determine the type of problem
    source_type = input("Enter the type of problem (f for fission source, anything else for fixed source): ").strip().lower()
    source_type = "f" if source_type == "f" else "s"

    # Get user inputs
    if source_type == "f":
        t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol, max_iterations = get_user_input(source_type)
        # Solve the eigenvalue problem
        x, Phi, k, residuals = diffusion_eigenvalue_solver_1D(
            t, num_mesh_points, left_boundary, right_boundary,
            material_props, phi_initial, k_initial, tol, max_iterations
        )
        # Plot results
        plot_results(x, Phi, residuals, solver_type="fission", k=k)
    else:
        t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, tol, max_iterations = get_user_input(source_type)
        # Solve the fixed source problem
        x, Phi, residuals = diffusion_solver_1D(
            t, num_mesh_points, material_props, left_boundary, right_boundary,
            phi_initial, tol, max_iterations
        )
        # Plot results
        plot_results(x, Phi, residuals, solver_type="fixed")

    
def get_user_input(source_type):
    """Prompt user to input main parameters, boundary conditions, and validated material properties."""

    def get_positive_float(prompt, error_msg="Value must be greater than 0."):
        while True:
            try:
                value = float(input(prompt))
                if value <= 0:
                    raise ValueError(error_msg)
                return value
            except ValueError as e:
                print(f"Invalid input: {e}")

    def get_positive_int(prompt, error_msg="Value must be greater than 0."):
        while True:
            try:
                value = int(input(prompt))
                if value <= 0:
                    raise ValueError(error_msg)
                return value
            except ValueError as e:
                print(f"Invalid input: {e}")

    def get_ratio(prompt, error_msg="Value must be between 0 and 1."):
        while True:
            try:
                value = float(input(prompt))
                if not (0 <= value <= 1):
                    raise ValueError(error_msg)
                return value
            except ValueError as e:
                print(f"Invalid input: {e}")

    def get_boundary_condition(prompt):
        condition = input(prompt).strip().lower()
        return "r" if condition == "r" else "v"

    t = get_positive_float("Enter the total slab thickness: ")
    left_boundary = get_boundary_condition("Enter the left boundary condition (r for reflective, anything else for vacuum): ")
    right_boundary = get_boundary_condition("Enter the right boundary condition (r for reflective, anything else for vacuum): ")
    num_media = get_positive_int("Enter the number of media: ")

    material_props = []
    last_end = 0
    material_details = ""

    for i in range(num_media):
        print(f"\nEnter properties for medium {i + 1}:")
        start = last_end
        end = t if i == num_media - 1 else get_positive_float(
            f"  End position for medium {i + 1} (greater than {start} and up to {t}): ",
            error_msg=f"End position must be greater than {start} and within the slab thickness ({t}).",
        )

        sigma_t = get_positive_float("  Total cross-section (sigma_t): ")
        sigma_s_to_sigma_t_ratio = get_ratio("  Scattering to total cross-section ratio (0 ≤ value ≤ 1): ")

        if source_type == "f":
            sigma_f_to_sigma_a_ratio = get_ratio("  Fission to absorption cross-section ratio (0 ≤ value ≤ 1): ")
            nu = 0.0 if sigma_f_to_sigma_a_ratio == 0 else get_positive_float("  Neutron yield (v): ")
            material_props.append({
                'start': start,
                'end': end,
                'sigma_t': sigma_t,
                'sigma_s_to_sigma_t_ratio': sigma_s_to_sigma_t_ratio,
                'sigma_f_to_sigma_a_ratio': sigma_f_to_sigma_a_ratio,
                'nu': nu
            })
            material_details += (
                f"Medium {i + 1}:\n"
                f"  Start: {start}\n"
                f"  End: {end}\n"
                f"  sigma_t: {sigma_t}\n"
                f"  Scattering Ratio: {sigma_s_to_sigma_t_ratio}\n"
                f"  Fission Ratio: {sigma_f_to_sigma_a_ratio}\n"
                f"  Neutron Yield: {nu}\n"
            )
        else:
            S_i = get_positive_float("  Fixed source strength (S): ", error_msg="Value must be greater than or equal to 0.")
            material_props.append({
                'start': start,
                'end': end,
                'sigma_t': sigma_t,
                'sigma_s_to_sigma_t_ratio': sigma_s_to_sigma_t_ratio,
                'S': S_i
            })
            material_details += (
                f"Medium {i + 1}:\n"
                f"  Start: {start}\n"
                f"  End: {end}\n"
                f"  sigma_t: {sigma_t}\n"
                f"  Scattering Ratio: {sigma_s_to_sigma_t_ratio}\n"
                f"  Source Strength: {S_i}\n"
            )

        last_end = end

    num_mesh_points = get_positive_int("Enter the number of mesh points: ")
    phi_initial = get_positive_float("Enter initial guess for flux distribution (phi): ")

    if source_type == "f":
        k_initial = get_positive_float("Enter the initial guess for eigenvalue k: ")
        tol = get_positive_float("Enter the convergence tolerance: ")
        max_iterations = get_positive_int("Enter the maximum number of iterations: ")
        user_input = (
            f"Problem Type: Fission\n"
            f"Slab Thickness: {t}\n"
            f"Left Boundary: {'Reflective' if left_boundary == 'r' else 'Vacuum'}\n"
            f"Right Boundary: {'Reflective' if right_boundary == 'r' else 'Vacuum'}\n"
            f"Number of Media: {num_media}\n"
            f"{material_details}"
            f"Mesh Points: {num_mesh_points}\n"
            f"Initial Flux: {phi_initial}\n"
            f"Initial Eigenvalue (k): {k_initial}\n"
            f"Tolerance: {tol}\n"
            f"Max Iterations: {max_iterations}\n"
        )
        write_to_file(user_input)
        return t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol, max_iterations
    else:
        tol = get_positive_float("Enter the convergence tolerance: ")
        max_iterations = get_positive_int("Enter the maximum number of iterations: ")
        user_input = (
            f"Problem Type: Fixed Source\n"
            f"Slab Thickness: {t}\n"
            f"Left Boundary: {'Reflective' if left_boundary == 'r' else 'Vacuum'}\n"
            f"Right Boundary: {'Reflective' if right_boundary == 'r' else 'Vacuum'}\n"
            f"Number of Media: {num_media}\n"
            f"{material_details}"
            f"Mesh Points: {num_mesh_points}\n"
            f"Initial Flux: {phi_initial}\n"
            f"Tolerance: {tol}\n"
            f"Max Iterations: {max_iterations}\n"
        )
        write_to_file(user_input)
        return t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, tol, max_iterations


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

        # Prepare output data
    output_data = (
        f"Results:\n"
        f"Neutron Flux (Phi):\n{Phi}\n"
        f"Spatial Domain (x):\n{x}\n"
    )

    # Write the output data to a file
    write_to_file(output_data)


    return x, Phi, residuals

def diffusion_eigenvalue_solver_1D(t, num_mesh_points, left_boundary, right_boundary, material_props, phi_initial, k_initial, tol=1e-6, max_iterations=1000000):
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
            fission_source[i] = nu[i] * sigma_f[i] * Phi[i]

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

        # Solve using Gauss-Seidel method
        Phi_new, res = Gauss_Seidel_optimized(lower_diag, main_diag, upper_diag, b_new, Phi, tol, max_iterations)
        residuals.append(res)

        # Calculate new fission source
        new_fission_source = np.zeros(new_num_mesh_points)
        for i in range(1, new_num_mesh_points - 1):
            new_fission_source[i] = nu[i] * sigma_f[i] * Phi_new[i]

        # Update k
        k_new = k * np.sum(new_fission_source) / np.sum(fission_source)
        if abs(k_new - k) < tol:
            k = k_new
            break
        k = k_new

        # Update flux
        Phi = Phi_new
    
    # Prepare output data
    output_data = (
        f"Results:\n"
        f"Final Eigenvalue (k): {k:.6f}\n"
        f"Neutron Flux (Phi):\n{Phi}\n"
        f"Spatial Domain (x):\n{x}\n"
    )

    # Write the output data to a file
    write_to_file(output_data)

    return x, Phi, k, residuals[-1]


# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()