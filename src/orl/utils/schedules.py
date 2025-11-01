def linear_eps_decay(step: int, eps_start: float, eps_end: float, eps_decay: int) -> float:
    span = max(1, eps_decay)
    frac = max(0.0, 1.0 - step / span)
    return eps_end + (eps_start - eps_end) * frac
