from collections import deque
from datetime import datetime, timedelta

class VolumeEstimator:
    """
    Simple class to track tick timestamps and approximate 'volume' by
    counting ticks in a rolling time window or a rolling tick window.
    """
    def __init__(self, max_size=200):
        self.tick_timestamps = deque(maxlen=max_size)

    def update(self, timestamp: datetime):
        # Store new tick time
        self.tick_timestamps.append(timestamp)
    
    def get_tick_count(self, lookback_seconds=60) -> int:
        """
        Returns how many ticks occurred within the last 'lookback_seconds' seconds.
        """
        now = datetime.now()
        # Count how many are within the timeframe
        count = 0
        for t in reversed(self.tick_timestamps):
            if (now - t).total_seconds() <= lookback_seconds:
                count += 1
            else:
                break
        return count

    def get_tick_volume_time_series(self, intervals=5, interval_seconds=60):
        """
        Returns a list of tick counts for the last 'intervals' time blocks.
        E.g., for intervals=5 and interval_seconds=60, returns tick counts
        in each of the last 5 minute-blocks.
        """
        now = datetime.now()
        result = []
        for block in range(intervals):
            block_end = now - timedelta(seconds=block * interval_seconds)
            block_start = block_end - timedelta(seconds=interval_seconds)
            # Count ticks in this block
            count = sum(1 for t in self.tick_timestamps 
                        if block_start <= t < block_end)
            result.append(count)
        result.reverse()
        return result
