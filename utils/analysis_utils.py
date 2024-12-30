# utils/analysis_utils.py

def analyze_volume(volume_data, lookback=5):
    """
    Returns a measure of current volume relative to an average
    in the last 'lookback' intervals. The higher the ratio,
    the more significant the volume spike.
    """
    if not volume_data:
        return 1.0
    window = volume_data[-lookback:] if len(volume_data) >= lookback else volume_data
    avg_volume = sum(window) / len(window)
    current_volume = volume_data[-1]
    if avg_volume == 0:
        return 1.0
    return current_volume / avg_volume
