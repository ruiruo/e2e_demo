import math


def quantize_to_step(value: float, step: float = 0.2, method: str = 'round') -> float:
    """
    Quantize `value` to the nearest multiple of `step`.

    Args:
        value: the original float.
        step: the grid size (default 0.2).
        method: one of:
            - 'round': round to nearest multiple
            - 'floor': round down (toward -∞)
            - 'ceil' : round up   (toward +∞)

    Returns:
        A float equal to step * n for some integer n.
    """
    if method == 'round':
        n = round(value / step)
    elif method == 'floor':
        n = math.floor(value / step)
    elif method == 'ceil':
        n = math.ceil(value / step)
    else:
        raise ValueError("`method` must be 'round', 'floor', or 'ceil'")
    return step * n
