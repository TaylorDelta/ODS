import matplotlib.pyplot as plt
import numpy as np

# =======================
# INITIAL SETUP & PARAMETERS
# =======================
# Grid definition for contour plot
t1 = np.linspace(-1.5, 5, 500)
t2 = np.linspace(-0.5, 4, 500)
theta_1, theta_2 = np.meshgrid(t1, t2)

# Define the objective function J
#J = (theta_1 - 1) ** 2 + (theta_2 - 2.5) ** 2
theta = np.stack([theta_1, theta_2], axis=-1)  # Shape (500, 500, 2)

# Quadratic and linear terms for J
G = np.array([[2, 1], [1, 2]])  # Quadratic coefficient matrix
d = np.array([2, 1])  # Linear coefficient vector
r = 7.25  # Constant term

# Efficient computation of the quadratic and linear terms
quad_term = 0.5 * np.einsum('...i,ij,...j', theta, G, theta)
linear_term = np.einsum('...i,i', theta, d)

# Final scalar field J_2
J_2 = quad_term + linear_term + r

# Constraints: A matrix and b vector
A = np.array([
    [1, 0],
    [0, 1],
    [-1, -1]
])
b = np.array([1,0,-4.5])


# ============================
# CORE FUNCTION DEFINITIONS
# ============================

def solve_EQP(W_k, theta_k):
    """
    Solve the quadratic programming problem using the active set W_k.

    Parameters:
        W_k (list): Indices of active constraints
        theta_k (np.array): Current values of theta

    Returns:
        p (np.array): Optimal direction
        lambda_k (np.array): Lagrange multipliers for active constraints
    """
    A_active = A[W_k]
    m_active = A_active.shape[0]

    # Construct matrix K
    K = np.block([
        [G, A_active.T],
        [A_active, np.zeros((m_active, m_active))]
    ])

    g = d + G @ theta_k.T
    h = A_active @ theta_k.T - b[W_k]

    # Combine g and h
    gh = np.concatenate([g, h]) if len(h) > 0 else g

    # Solve for p_lambda and apply thresholding
    p_lambda = np.linalg.inv(K) @ gh
    p_lambda[np.abs(p_lambda) < 1e-10] = 0

    p = -p_lambda[:2]
    lambda_k = p_lambda[2:]
    return p, lambda_k


def compute_alpha(A, p, b, theta_k):
    """
    Compute the step size alpha for the update of theta.

    Parameters:
        A (np.array): Constraint matrix
        p (np.array): Direction vector
        b (np.array): Right-hand side of constraints
        theta_k (np.array): Current theta values

    Returns:
        alpha (float): Computed step size
    """
    alpha = [1]
    for i, ai in enumerate(A):
        atp = ai.T @ p
        if np.abs(atp) >= 1e-10 and atp < 0:
            alpha.append((b[i] - ai.T @ theta_k.T) / atp)
    return min(alpha)


def check_blocking_constraints(A, theta_k, b):
    """
    Check which constraints are violated by the current theta.

    Parameters:
        A (np.array): Constraint matrix
        theta_k (np.array): Current values of theta
        b (np.array): Right-hand side of constraints

    Returns:
        blocking_indices (list): Indices of constraints that are violated
    """
    constraints = A @ theta_k.T - b
    return np.where(constraints <= 0)[0].tolist()


# =======================
# MAIN OPTIMIZATION LOOP
# =======================
def main_optimization(theta_0, k_max=10):
    """
    Run the optimization procedure, updating theta values and active constraints.

    Parameters:
        theta_0 (np.array): Initial theta values
        k_max (int): Maximum number of iterations for the optimization

    Returns:
        theta_list (list): List of theta values at each step
        p_list (list): List of direction vectors at each step
    """
    W_0 = check_blocking_constraints(A, theta_0, b)
    theta_k, W_k = theta_0, W_0
    theta_list = [theta_k]
    p_list = []

    for k in range(k_max):
        print(f"\n--- Iteration {k} ---")
        print(f"Current theta: {theta_k.flatten()}")
        print(f"Active constraints (W_k): {W_k}")

        p, lambda_k = solve_EQP(W_k, theta_k)
        p_list.append(p)

        print(f"Direction p: {p.flatten()}")
        print(f"Lagrange multipliers: {lambda_k}")

        if np.all(p == 0):
            if np.all(lambda_k >= 0):
                print("All Lagrange multipliers are non-negative. Optimization complete.")
                break
            else:
                idx = np.argmin(lambda_k)
                print(f"Removing constraint {W_k[idx]} due to negative multiplier.")
                W_k = np.delete(W_k, idx)
        else:
            alpha = compute_alpha(A, p, b, theta_k)
            print(f"Step size alpha: {alpha}")
            theta_k = theta_k + alpha * p
            print(f"Updated theta: {theta_k.flatten()}")
            blocking_indices = check_blocking_constraints(A, theta_k, b)
            if blocking_indices:
                print(f"Adding blocking constraints: {blocking_indices}")
                for idx in blocking_indices:
                    idx = int(idx)  # Convert each idx to an integer
                    if idx not in W_k:
                        W_k.append(idx)  # Use list append directly
        theta_list.append(theta_k)

    return theta_list, p_list


# =======================
# PLOTTING THE RESULTS
# =======================
def plot_results(theta_list):
    """
    Plot the optimization process and constraints.

    Parameters:
        theta_list (list): List of theta values at each step
    """
    # Prepare for plotting
    plt.figure(figsize=(8, 6))

    # Contour plot for objective function J
    plt.contour(theta_1, theta_2, J_2, levels=15, cmap='viridis', alpha=0.8)
    plt.colorbar(label="Objective Function J")

    # Plot each constraint
    for i, (ai, bi) in enumerate(zip(A, b)):
        if ai[0] != 0 and ai[1] != 0:
            t1_vals = (bi - ai[1] * t2) / ai[0]
            plt.plot(t1_vals, t2, label=f'Constraint {i + 1}')
        elif ai[0] == 0:
            plt.axhline(bi / ai[1], label=f'Constraint {i + 1}')
        elif ai[1] == 0:
            plt.axvline(bi / ai[0], label=f'Constraint {i + 1}')

    # Plot trajectory of theta_k points
    theta_array = np.array(theta_list)
    plt.plot(theta_array[:, 0], theta_array[:, 1], marker='o', linestyle='-', color='b')

    # Set plot labels and legend
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.xlim([-1.5, 5])
    plt.ylim([-0.5, 4])
    plt.title('Contour Plot of Objective Function with Constraints')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Show plot
    plt.show()


# =======================
# RUN THE MAIN CODE
# =======================
if __name__ == "__main__":
    # Initial theta and optimization execution
    theta_0 = np.array([2, 1])
    theta_list, p_list = main_optimization(theta_0)

    # Plot the results after optimization
    plot_results(theta_list)
