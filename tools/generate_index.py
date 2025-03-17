import json
import numpy as np
import matplotlib.pyplot as plt


def arcsinh_space_scaled(min_val, max_val, num_points, alpha=1.0):
    """
    Returns an array of 'num_points' values between min_val and max_val,
    spaced uniformly in asinh-space but with a scaling factor alpha.

    If alpha is large, points near zero become more spread out.

    Parameters:
      min_val (float): lower bound of the interval
      max_val (float): upper bound of the interval
      num_points (int): number of points to generate
      alpha (float): scale factor to reduce clustering near zero

    Returns:
      numpy.ndarray: values spanning [min_val, max_val] with arcsinh-based spacing
    """
    # Transform bounds using arcsinh(x/alpha)
    t1 = np.arcsinh(min_val / alpha)
    t2 = np.arcsinh(max_val / alpha)
    # Uniformly sample in that transformed domain
    t = np.linspace(t1, t2, num_points)
    # Transform back: x = alpha * sinh(t)
    return alpha * np.sinh(t)


def get_scaled_arcsinh_grid(m_min, m_max, n_min, n_max, m_points, n_points, alpha_m=1.0, alpha_n=1.0):
    # Generate boundaries using arcsinh with scaling factors
    m_boundaries = arcsinh_space_scaled(m_min, m_max, m_points, alpha=alpha_m)
    n_boundaries = arcsinh_space_scaled(n_min, n_max, n_points, alpha=alpha_n)
    """
    Create grid boundaries for the m-axis and n-axis after shifting the origin to (0, 0).

    For the m-axis:
      - Negative side: uniformly spaced from -0.1*m to 0 (y points).
      - Positive side: logarithmically spaced from 0 to (m - 0.1*m) (x points, with 0 inserted manually).

    For the n-axis:
      - Negative side: logarithmically spaced on the absolute values from 0.5*n to a small epsilon,
        then negated and reversed to yield boundaries from -0.5*n to 0 (s points, with 0 manually inserted).
      - Positive side: logarithmically spaced from epsilon to (n - 0.5*n) (s points, with 0 inserted).

    Returns:
      m_boundaries: 1D numpy array of sorted boundary values along the m-axis.
      n_boundaries: 1D numpy array of sorted boundary values along the n-axis.
    """
    return m_boundaries, n_boundaries


def create_local2token_ndarray(m_boundaries, n_boundaries):
    """
    Build:
      1) local2token: a 2D ndarray of shape (M, N), where local2token[i, j] = token_id
      2) token2local: a dict mapping token_id -> [center_x, center_y]

    Parameters:
      m_boundaries (np.ndarray): shape [M+1], sorted x-axis boundaries
      n_boundaries (np.ndarray): shape [N+1], sorted y-axis boundaries

    Returns:
      local2token (np.ndarray): shape (M, N), each entry is an integer token_id
      token2local (dict): {token_id: [center_x, center_y]}
    """
    M = len(m_boundaries) - 1
    N = len(n_boundaries) - 1

    local2token = np.zeros((M, N), dtype=np.int32)
    token2local = {}

    token_id = 0
    for i in range(M):
        x0, x1 = m_boundaries[i], m_boundaries[i + 1]
        for j in range(N):
            y0, y1 = n_boundaries[j], n_boundaries[j + 1]

            # get token_id
            local2token[i, j] = token_id

            # cal center
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            # save as  [float(cx), float(cy)]
            token2local[token_id] = [float(cx), float(cy)]

            token_id += 1

    return local2token, token2local


# Example usage:
if __name__ == "__main__":
    m_max, m_min = 90, -10
    n_max, n_min = 10, -10
    M_points = 32 + 1
    N_points = 16 + 1

    # Create grid boundaries with the origin shifted to (0, 0)
    m_boundaries_max = arcsinh_space_scaled(0, m_max, M_points - 7, alpha=3.0)
    m_boundaries_min = arcsinh_space_scaled(m_min, 0, 8, alpha=3.0)
    n_boundaries = arcsinh_space_scaled(n_min, n_max, N_points, alpha=1.5)
    m_boundaries = np.concatenate((m_boundaries_max, m_boundaries_min[:-1]))
    m_boundaries = np.sort(m_boundaries)
    print(m_boundaries)
    print(n_boundaries)
    # Create token map for the grid cells
    local2token, token2local = create_local2token_ndarray(m_boundaries, n_boundaries)
    print("local2token shape:", local2token.shape)  # (32, 16)
    print("Total tokens:", local2token.size)  # 32*16=512

    with open("/home/nio/reparke2e/configs/token2local.json", "w") as f:
        json.dump(token2local, f, indent=2)

    np.save("/home/nio/reparke2e/configs/local2token.npy", local2token)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw vertical lines for each m-axis boundary
    for x in m_boundaries:
        ax.axvline(x=x, color='blue', linestyle='--', linewidth=0.5)

    # Draw horizontal lines for each n-axis boundary
    for y in n_boundaries:
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

    ax.set_xlim(m_boundaries[0], m_boundaries[-1])
    ax.set_ylim(n_boundaries[0], n_boundaries[-1])
    ax.set_xlabel("m-axis")
    ax.set_ylabel("n-axis")
    ax.set_title("Grid Visualization")
    plt.show()

    fig.savefig("/home/nio/reparke2e/configs/token_plot.png", dpi=300)
