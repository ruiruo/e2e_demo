import json
import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory_utils import parallel_find_bin


def arcsinh_space_scaled(min_val, max_val, num_points, alpha=1.0):
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
    if uniform_left > overall_min:
        neg_full = arcsinh_space_scaled(overall_min, uniform_left,
                                        arcsinh_points_neg + 1, arcsinh_alpha)
        neg_part = neg_full[:-1]
    else:
        neg_part = np.array([], dtype=np.float64)

    middle = np.linspace(uniform_left, uniform_right,
                         uniform_segments + 1, dtype=np.float64)

    if uniform_left < 0 < uniform_right:
        for i in range(len(middle) - 1):
            if middle[i] < 0 < middle[i + 1]:
                d = (middle[i + 1] - middle[i]) / 2
                middle[i], middle[i + 1] = -d, d
                break

    if uniform_right < overall_max:
        pos_full = arcsinh_space_scaled(uniform_right, overall_max,
                                        arcsinh_points_pos + 1, arcsinh_alpha)
        pos_part = pos_full[1:]
    else:
        pos_part = np.array([], dtype=np.float64)

    combined = np.concatenate([neg_part, middle, pos_part])
    combined.sort()
    return combined


def compute_boundaries_by_token_count(overall_min, overall_max, desired_cells,
                                      uniform_left, uniform_right,
                                      uniform_segments, arcsinh_alpha=3.0,
                                      neg_weight=0.5, pos_weight=0.5):
    if desired_cells + 2 < uniform_segments:
        raise ValueError(
            f"desired_cells + 2 must be ≥ uniform_segments; "
            f"got desired_cells={desired_cells} (→ {desired_cells + 2}) vs {uniform_segments}"
        )
    remaining = desired_cells + 2 - uniform_segments
    total_weight = neg_weight + pos_weight
    neg_pts = int(np.ceil(remaining * neg_weight / total_weight))
    pos_pts = remaining - neg_pts

    return piecewise_boundaries_1d(
        overall_min, overall_max,
        uniform_left, uniform_right,
        uniform_segments,
        arcsinh_alpha, neg_pts, pos_pts
    )


def compute_2d_boundaries_by_token_count(x_min, x_max, y_min, y_max,
                                         desired_cells_x, desired_cells_y,
                                         x_uniform_left, x_uniform_right, x_uniform_segments,
                                         y_uniform_left, y_uniform_right, y_uniform_segments,
                                         x_arcsinh_alpha=3.0, y_arcsinh_alpha=3.0,
                                         x_neg_weight=0.5, x_pos_weight=0.5,
                                         y_neg_weight=0.5, y_pos_weight=0.5):
    xb = compute_boundaries_by_token_count(
        x_min, x_max, desired_cells_x,
        x_uniform_left, x_uniform_right, x_uniform_segments,
        x_arcsinh_alpha, x_neg_weight, x_pos_weight)

    yb = compute_boundaries_by_token_count(
        y_min, y_max, desired_cells_y,
        y_uniform_left, y_uniform_right, y_uniform_segments,
        y_arcsinh_alpha, y_neg_weight, y_pos_weight)

    return xb, yb


def create_local2token_ndarray(m_boundaries, n_boundaries):
    M, N = len(m_boundaries) - 1, len(n_boundaries) - 1
    local2token = np.zeros((M, N), dtype=np.int32)
    token2local = {}
    tid = 0
    for i in range(M):
        x0, x1 = m_boundaries[i], m_boundaries[i + 1]
        for j in range(N):
            y0, y1 = n_boundaries[j], n_boundaries[j + 1]
            local2token[i, j] = tid
            token2local[tid] = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
            tid += 1
    return local2token, token2local


if __name__ == "__main__":
    # ------------------------ grid definition ------------------------
    x_min, x_max = -20, 180
    y_min, y_max = -10, 10

    desired_cells_x = 80
    desired_cells_y = 30

    x_uniform_left, x_uniform_right = -5, 100
    x_uniform_segments = 60

    y_uniform_left, y_uniform_right = -10, 10
    y_uniform_segments = 31

    x_neg_weight, x_pos_weight = 0, 1
    y_neg_weight, y_pos_weight = 0.5, 0.5

    # --------------------- compute boundaries -----------------------
    x_boundaries, y_boundaries = compute_2d_boundaries_by_token_count(
        x_min, x_max, y_min, y_max,
        desired_cells_x, desired_cells_y,
        x_uniform_left, x_uniform_right, x_uniform_segments,
        y_uniform_left, y_uniform_right, y_uniform_segments,
        x_arcsinh_alpha=5.0, y_arcsinh_alpha=3.0,
        x_neg_weight=x_neg_weight, x_pos_weight=x_pos_weight,
        y_neg_weight=y_neg_weight, y_pos_weight=y_pos_weight)

    # ------------------- build token maps (regular) -----------------
    local2token, token2local = create_local2token_ndarray(x_boundaries, y_boundaries)

    # ----------- locate the token whose cell contains (0,0) ----------
    bins_x, bins_y = parallel_find_bin(
        np.array([[0.0, 0.0]], dtype=np.float64), x_boundaries, y_boundaries
    )
    token_00 = int(local2token[bins_x[0], bins_y[0]])
    print(f"Token ID for (0,0): {token_00}")

    # -------------------- append special tokens ----------------------
    REGULAR_TOKENS = len(token2local)
    BOS_TOKEN_ID = REGULAR_TOKENS
    EOS_TOKEN_ID = REGULAR_TOKENS + 1
    PAD_TOKEN_ID = REGULAR_TOKENS + 2

    token2local[BOS_TOKEN_ID] = [np.nan, np.nan]
    token2local[EOS_TOKEN_ID] = [np.nan, np.nan]
    token2local[PAD_TOKEN_ID] = [np.nan, np.nan]

    TOKEN_NUMS = len(token2local)              # regular + 3

    print(f"BOS token ID: {BOS_TOKEN_ID}")
    print(f"EOS token ID: {EOS_TOKEN_ID}")
    print(f"PAD token ID: {PAD_TOKEN_ID}")
    print(f"Total tokens (incl. special): {TOKEN_NUMS}")

    # --------------------- plot (optional) ---------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    for xb in x_boundaries:
        ax.axvline(x=xb, color='blue', linestyle='--', linewidth=0.5)
    for yb in y_boundaries:
        ax.axhline(y=yb, color='red', linestyle='--', linewidth=0.5)

    ax.set_xlim(x_boundaries[0], x_boundaries[-1])
    ax.set_ylim(y_boundaries[0], y_boundaries[-1])
    ax.set_title(f"Computed 2D Boundaries\n"
                 f"(x: {len(x_boundaries) - 1} cells, "
                 f"y: {len(y_boundaries) - 1} cells)")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")

    base_path = "/home/nio/reparke2e/configs/"   # adjust if needed
    plt.savefig(f"{base_path}token_plot_{TOKEN_NUMS}.png")

    # --------------------- save mapping files ------------------------
    with open(f"{base_path}token2local_{TOKEN_NUMS}.json", "w") as f:
        json.dump(token2local, f, indent=2)

    np.save(f"{base_path}local2token_{TOKEN_NUMS}.npy", local2token)
    print(f"Saved token maps with {TOKEN_NUMS} tokens.")
