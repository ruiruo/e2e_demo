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


def create_token_dict_and_centers(m_boundaries, n_boundaries):
    """
    Create:
      1) A 4-level dict: token_dict[x0][x1][y0][y1] = token_id
      2) A dict mapping token_id -> (center_x, center_y)

    Parameters:
      m_boundaries: 1D array of boundary values along m-axis
      n_boundaries: 1D array of boundary values along n-axis

    Returns:
      token_dict: a nested dictionary (4 levels) mapping boundary quadruples to token_id
      token_center: a dictionary mapping token_id -> (center_x, center_y)
    """
    token_dict = {}
    token_center = {}

    token_id = 0
    # Loop over each pair of adjacent boundaries on the m-axis
    for i in range(len(m_boundaries) - 1):
        # Convert to float16 then to Python float
        x0 = float(np.float16(m_boundaries[i]))
        x1 = float(np.float16(m_boundaries[i + 1]))

        # Ensure the nested dict structure exists
        if x0 not in token_dict:
            token_dict[x0] = {}
        if x1 not in token_dict[x0]:
            token_dict[x0][x1] = {}

        # Loop over each pair of adjacent boundaries on the n-axis
        for j in range(len(n_boundaries) - 1):
            y0 = float(np.float16(n_boundaries[j]))
            y1 = float(np.float16(n_boundaries[j + 1]))

            # Ensure deeper nested structure
            if y0 not in token_dict[x0][x1]:
                token_dict[x0][x1][y0] = {}

            # Assign the final level to token_id
            token_dict[x0][x1][y0][y1] = token_id

            # Compute center of the box
            center_x = (x0 + x1) / 2.0
            center_y = (y0 + y1) / 2.0

            # Store the center in a separate dict keyed by token_id
            token_center[token_id] = (center_x, center_y)

            token_id += 1

    return token_dict, token_center


# Example usage:
if __name__ == "__main__":
    m_max, m_min = 90, -10
    n_max, n_min = 10, -10
    x = 32 + 1 # Number of logarithmically spaced points on m-axis positive side
    y = 16 + 1 # Number of logarithmically spaced points on each side of the n-axis

    # Create grid boundaries with the origin shifted to (0, 0)
    m_boundaries, n_boundaries = get_scaled_arcsinh_grid(m_min, m_max, n_min, n_max, x, y, 6, 2)
    # Create token map for the grid cells
    token_search, tokenizer = create_token_dict_and_centers(m_boundaries, n_boundaries)
    with open("/home/nio/reparke2e/configs/token2local.json", "w") as f:
        f.write(json.dumps(tokenizer))

    with open("/home/nio/reparke2e/configs/local2token.json", "w") as f:
        f.write(json.dumps(token_search))

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw vertical lines for each m-axis boundary
    for x in m_boundaries:
        ax.axvline(x=x, color='blue', linestyle='--', linewidth=0.5)

    # Draw horizontal lines for each n-axis boundary
    for y in n_boundaries:
        ax.axhline(y=y, color='red', linestyle='--', linewidth=0.5)

    print(len(tokenizer))

    # Set axis limits based on the boundaries
    ax.set_xlim(m_boundaries[0], m_boundaries[-1])
    ax.set_ylim(n_boundaries[0], n_boundaries[-1])

    ax.set_xlabel("m-axis")
    ax.set_ylabel("n-axis")
    fig.savefig("/home/nio/reparke2e/configs/token_plot.png", dpi=300)