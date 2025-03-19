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
    Build a 1D boundary array with:
      1) arcsinh from [overall_min, uniform_left]
      2) uniform from [uniform_left, uniform_right] in 'uniform_segments' intervals
      3) arcsinh from [uniform_right, overall_max]

    Ensures that 0 is inside the uniform region if 0 in [uniform_left, uniform_right].
    If 'uniform_segments' is odd, 0 will be the exact center of the middle interval.
    """
    # arcsinh negative side
    if uniform_left > overall_min:
        neg_arcsinh = arcsinh_space_scaled(overall_min, uniform_left, arcsinh_points_neg, arcsinh_alpha)
    else:
        neg_arcsinh = np.array([uniform_left], dtype=np.float64)

    # uniform middle
    # uniform_segments intervals => uniform_segments+1 boundaries
    middle = np.linspace(uniform_left, uniform_right, uniform_segments + 1, dtype=np.float64)

    # arcsinh positive side
    if uniform_right < overall_max:
        pos_arcsinh = arcsinh_space_scaled(uniform_right, overall_max, arcsinh_points_pos, arcsinh_alpha)
    else:
        pos_arcsinh = np.array([uniform_right], dtype=np.float64)

    # merge
    combined = np.concatenate([neg_arcsinh, middle, pos_arcsinh])
    combined = np.unique(combined)  # remove duplicates if any
    combined.sort()
    return combined


def build_2d_boundaries(x_min, x_max, y_min, y_max):
    """
    Example: we want:
      - x in [overall_min_x, overall_max_x] = [-10, 40]
        with a uniform region [-5, 5] subdivided into 5 intervals (=> center interval around 0).
      - y in [overall_min_y, overall_max_y] = [-5, 5]
        with a uniform region [-2, 2] subdivided into 5 intervals (=> center interval around 0).
      - arcsinh outside the uniform region for both axes.
    """
    # x-axis: piecewise from [-10, -5, 5, 40]
    x_boundary = piecewise_boundaries_1d(
        overall_min=x_min, overall_max=x_max,
        uniform_left=-5, uniform_right=5,
        uniform_segments=21,  # must be odd => the middle interval center is x=0
        arcsinh_alpha=3.0,
        arcsinh_points_neg=9,
        arcsinh_points_pos=22,
    )

    # y-axis: piecewise from [-5, -2, 2, 5]
    y_boundary = piecewise_boundaries_1d(
        overall_min=y_min, overall_max=y_max,
        uniform_left=-2, uniform_right=2,
        uniform_segments=11,  # also odd => center interval is y=0
        arcsinh_alpha=3.0,
        arcsinh_points_neg=7,
        arcsinh_points_pos=8,
    )

    return x_boundary, y_boundary


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
    m_max, m_min = 40, -10
    n_max, n_min = 5, -5

    x_boundaries, y_boundaries = build_2d_boundaries(m_min, m_max, n_min, n_max)

    print("x_boundaries:\n", x_boundaries.astype(np.float16).tolist())
    print("y_boundaries:\n", y_boundaries.astype(np.float16).tolist())

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    for xb in x_boundaries:
        ax.axvline(x=xb, color='blue', linestyle='--', linewidth=0.5)
    for yb in y_boundaries:
        ax.axhline(y=yb, color='red', linestyle='--', linewidth=0.5)

    ax.set_xlim(x_boundaries[0], x_boundaries[-1])
    ax.set_ylim(y_boundaries[0], y_boundaries[-1])
    ax.set_title("Piecewise Boundaries with 0,0 as Middle Interval Center")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    plt.show()
    fig.savefig("/home/nio/reparke2e/configs/token_plot.png", dpi=300)

    # Create token map for the grid cells
    local2token, token2local = create_local2token_ndarray(x_boundaries, y_boundaries)
    bos_local = parallel_find_bin(np.array([[0, 0]]), x_boundaries, y_boundaries)
    bos_local = (int(bos_local[0]), int(bos_local[1]))
    print("BOS:", bos_local, bos_local[0] * (len(y_boundaries)-1) + bos_local[1])
    print("local2token shape:", local2token.shape)  # (32, 16)
    print("Total tokens:", local2token.size)  # 32*16=512

    with open("/home/nio/reparke2e/configs/token2local.json", "w") as f:
        json.dump(token2local, f, indent=2)

    np.save("/home/nio/reparke2e/configs/local2token.npy", local2token)
