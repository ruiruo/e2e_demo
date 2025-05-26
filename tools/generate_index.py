import json
import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory_utils import parallel_find_bin


def arcsinh_space_scaled(min_val, max_val, num_points, alpha=1.0):
    """
    Returns an array of 'num_points' values between min_val and max_val,
    spaced uniformly in asinh-space with a scaling factor alpha.
    """
    t1 = np.arcsinh(min_val / alpha)
    t2 = np.arcsinh(max_val / alpha)
    t = np.linspace(t1, t2, num_points)
    return alpha * np.sinh(t)


def piecewise_boundaries_1d(
        overall_min, overall_max,
        uniform_left, uniform_right,
        uniform_segments,
        arcsinh_alpha=3.0,
        arcsinh_points_neg=10,
        arcsinh_points_pos=10,
):
    """
    Build a 1D boundary array by combining three regions:
      1) arcsinh-spaced boundaries from [overall_min, uniform_left] (excluding uniform_left)
      2) uniform boundaries from [uniform_left, uniform_right] (with uniform_segments intervals)
      3) arcsinh-spaced boundaries from [uniform_right, overall_max] (excluding uniform_right)

    If the uniform region covers 0, the interval that straddles 0 is adjusted so that its center is exactly 0.
    """
    # arcsinh negative side: include one extra point and drop the last (duplicate uniform_left)
    if uniform_left > overall_min:
        neg_arcsinh_full = arcsinh_space_scaled(overall_min, uniform_left, arcsinh_points_neg + 1, arcsinh_alpha)
        neg_arcsinh = neg_arcsinh_full[:-1]
    else:
        neg_arcsinh = np.array([], dtype=np.float64)

    # uniform middle: uniform_segments intervals yield uniform_segments+1 boundaries
    middle = np.linspace(uniform_left, uniform_right, uniform_segments + 1, dtype=np.float64)

    # If the uniform region covers 0, adjust the interval straddling 0 so its center becomes exactly 0.
    if uniform_left < 0 and uniform_right > 0:
        for i in range(len(middle) - 1):
            if middle[i] < 0 and middle[i + 1] > 0:
                # Compute half-width of the current cell.
                d = (middle[i + 1] - middle[i]) / 2
                # Force the boundaries to be symmetric around 0.
                middle[i] = -d
                middle[i + 1] = d
                break

    # arcsinh positive side: include extra point and drop the first (duplicate uniform_right)
    if uniform_right < overall_max:
        pos_arcsinh_full = arcsinh_space_scaled(uniform_right, overall_max, arcsinh_points_pos + 1, arcsinh_alpha)
        pos_arcsinh = pos_arcsinh_full[1:]
    else:
        pos_arcsinh = np.array([], dtype=np.float64)

    # Merge the three regions and sort the boundaries.
    combined = np.concatenate([neg_arcsinh, middle, pos_arcsinh])
    combined.sort()
    return combined


def compute_boundaries_by_token_count(overall_min, overall_max, desired_cells,
                                      uniform_left, uniform_right,
                                      uniform_segments, arcsinh_alpha=3.0,
                                      neg_weight=0.5, pos_weight=0.5):
    """
    Compute a 1D boundaries array such that the total number of cells (intervals) is exactly desired_cells.
    The total number of boundaries will be desired_cells + 1.

    The uniform (linear) region between uniform_left and uniform_right is divided into
    'uniform_segments' intervals (uniform_segments must be odd so that 0 is centered).

    The remaining cells are allocated to arcsinh spacing on the negative and positive sides.
    The allocation between negative and positive sides is unbalanced,
    controlled by neg_weight and pos_weight.

    Raises a ValueError if desired_cells + 2 is less than uniform_segments.
    """
    if desired_cells + 2 < uniform_segments:
        raise ValueError(
            f"desired_cells + 2 must be at least uniform_segments; "
            f"got desired_cells={desired_cells} (=> {desired_cells + 2}) and uniform_segments={uniform_segments}"
        )
    remaining = desired_cells + 2 - uniform_segments
    total_weight = neg_weight + pos_weight
    arcsinh_points_neg = int(np.ceil(remaining * (neg_weight / total_weight)))
    arcsinh_points_pos = int(remaining - arcsinh_points_neg)
    boundaries = piecewise_boundaries_1d(
        overall_min=overall_min,
        overall_max=overall_max,
        uniform_left=uniform_left,
        uniform_right=uniform_right,
        uniform_segments=uniform_segments,
        arcsinh_alpha=arcsinh_alpha,
        arcsinh_points_neg=arcsinh_points_neg,
        arcsinh_points_pos=arcsinh_points_pos,
    )
    return boundaries


def compute_2d_boundaries_by_token_count(x_min, x_max, y_min, y_max,
                                         desired_cells_x, desired_cells_y,
                                         x_uniform_left, x_uniform_right, x_uniform_segments,
                                         y_uniform_left, y_uniform_right, y_uniform_segments,
                                         x_arcsinh_alpha=3.0, y_arcsinh_alpha=3.0,
                                         x_neg_weight=0.5, x_pos_weight=0.5,
                                         y_neg_weight=0.5, y_pos_weight=0.5):
    """
    Compute 2D boundaries for x and y axes based on desired cell counts.

    Returns:
      x_boundaries, y_boundaries: each is a sorted 1D numpy array.
    """
    x_boundaries = compute_boundaries_by_token_count(
        overall_min=x_min,
        overall_max=x_max,
        desired_cells=desired_cells_x,
        uniform_left=x_uniform_left,
        uniform_right=x_uniform_right,
        uniform_segments=x_uniform_segments,
        arcsinh_alpha=x_arcsinh_alpha,
        neg_weight=x_neg_weight,
        pos_weight=x_pos_weight,
    )

    y_boundaries = compute_boundaries_by_token_count(
        overall_min=y_min,
        overall_max=y_max,
        desired_cells=desired_cells_y,
        uniform_left=y_uniform_left,
        uniform_right=y_uniform_right,
        uniform_segments=y_uniform_segments,
        arcsinh_alpha=y_arcsinh_alpha,
        neg_weight=y_neg_weight,
        pos_weight=y_pos_weight,
    )

    return x_boundaries, y_boundaries


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


if __name__ == "__main__":
    # Overall ranges:
    x_min, x_max = -20, 180
    y_min, y_max = -10, 10

    # To obtain 2048 tokens in the grid,
    # note that the total number of tokens will be:
    #   (desired_cells_x + 2) * (desired_cells_y + 2)
    desired_cells_x = 80
    desired_cells_y = 30

    # Uniform region parameters:
    x_uniform_left = -5
    x_uniform_right = 100
    x_uniform_segments = 60


    y_uniform_left = -10
    y_uniform_right = 10
    y_uniform_segments = 31  # must be odd, and <= desired_cells_y + 2

    # Unbalanced arcsinh allocation:
    x_neg_weight = 0
    x_pos_weight = 1

    y_neg_weight = 0.5
    y_pos_weight = 0.5

    # Compute boundaries
    x_boundaries, y_boundaries = compute_2d_boundaries_by_token_count(
        x_min, x_max, y_min, y_max,
        desired_cells_x, desired_cells_y,
        x_uniform_left, x_uniform_right, x_uniform_segments,
        y_uniform_left, y_uniform_right, y_uniform_segments,
        x_arcsinh_alpha=5.0, y_arcsinh_alpha=3.0,
        x_neg_weight=x_neg_weight, x_pos_weight=x_pos_weight,
        y_neg_weight=y_neg_weight, y_pos_weight=y_pos_weight
    )

    print("x_boundaries:\n", x_boundaries.astype(np.float16).tolist())
    print("y_boundaries:\n", y_boundaries.astype(np.float16).tolist())

    # Identify the token (cell) with center at (0,0)
    token_x = None
    token_y = None
    # This search for (0,0) might need adjustment if boundary definitions change significantly
    # or if a cell isn't perfectly centered at (0,0) due to segment counts.
    for i in range(len(x_boundaries) - 1):
        center_x = (x_boundaries[i] + x_boundaries[i + 1]) / 2
        if np.isclose(center_x, 0, atol=1e-8):  # Default atol is 1e-8
            token_x = (x_boundaries[i], x_boundaries[i + 1])
            break

    for j in range(len(y_boundaries) - 1):
        center_y = (y_boundaries[j] + y_boundaries[j + 1]) / 2
        if np.isclose(center_y, 0, atol=1e-8):  # Default atol is 1e-8
            token_y = (y_boundaries[j], y_boundaries[j + 1])
            break

    if token_x is not None and token_y is not None:
        print(f"Cell containing (0,0): X_bounds={token_x}, Y_bounds={token_y}")
    else:
        print("Could not find a cell perfectly centered at (0,0).")

    # Visualize the boundaries:
    fig, ax = plt.subplots(figsize=(10, 8))
    for xb in x_boundaries:
        ax.axvline(x=xb, color='blue', linestyle='--', linewidth=0.5)
    for yb in y_boundaries:
        ax.axhline(y=yb, color='red', linestyle='--', linewidth=0.5)

    ax.set_xlim(x_boundaries[0], x_boundaries[-1])
    ax.set_ylim(y_boundaries[0], y_boundaries[-1])
    ax.set_title(
        "Computed 2D Boundaries\n(x: {} cells, y: {} cells)".format(len(x_boundaries) - 1, len(y_boundaries) - 1))
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.show()

    # Create token map for the grid cells
    local2token, token2local = create_local2token_ndarray(x_boundaries, y_boundaries)

    print("BOS for (0,0):",
          parallel_find_bin(np.array([[0.0, 0.0]]), x_boundaries, y_boundaries))  # Ensure searching for 0.0
    print("local2token shape:", local2token.shape)
    print("Total tokens:", local2token.size)  # This should be 2048

    # Ensure the paths are correct for your system
    base_path = "/home/nio/reparke2e/configs/"  # Example base path, adjust if needed

    with open(f"{base_path}token2local_{len(token2local)}.json", "w") as f:
        json.dump(token2local, f, indent=2)

    np.save(f"{base_path}local2token_{len(token2local)}.npy", local2token)
    print(f"Saved token maps with {len(token2local)} tokens.")

