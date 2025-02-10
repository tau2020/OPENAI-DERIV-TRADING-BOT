import numpy as np
import pandas as pd
import logging

# ---------------------------
# ATR Computation Function
# ---------------------------
def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int, logger: logging.Logger = None) -> np.ndarray:
    """
    Computes the Average True Range (ATR) using Wilder's smoothing.

    Parameters:
      - highs: Array of high prices.
      - lows: Array of low prices.
      - closes: Array of close prices.
      - window: ATR window.
      - logger: Optional logger for debug output.

    Returns:
      The ATR as a NumPy array.
    """
    if len(highs) < window:
        if logger:
            logger.warning("Insufficient data: Required %d, Got %d", window, len(highs))
        return np.array([])

    tr = np.empty(len(highs))
    tr[0] = highs[0] - lows[0]
    h_l = highs[1:] - lows[1:]
    h_pc = np.abs(highs[1:] - closes[:-1])
    l_pc = np.abs(lows[1:] - closes[:-1])
    # Compute elementwise maximum for TR starting at index 1.
    np.maximum(h_l, h_pc, out=tr[1:])
    np.maximum(tr[1:], l_pc, out=tr[1:])

    atr = np.empty(len(tr) - window + 1)
    atr[0] = tr[:window].mean()
    for i in range(1, len(atr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i + window - 1]) / window

    if logger:
        logger.debug("TR: %s", tr)
        logger.debug("Valid ATR(%d): %s", window, atr)
    
    return atr