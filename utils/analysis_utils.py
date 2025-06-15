# utils/analysis_utils.py

# utils/analysis_utils.py

def analyze_volume(volume_series, lookback=5):
    """
    Returns a measure of current candle volume relative to an average
    in the last 'lookback' intervals. The higher the ratio,
    the more significant the volume spike.
    """
    if not volume_series:
        return 1.0
    window = volume_series[-lookback:] if len(volume_series) >= lookback else volume_series
    avg_volume = sum(window) / len(window)
    current_volume = volume_series[-1]
    if avg_volume == 0:
        return 1.0
    return current_volume / avg_volume

