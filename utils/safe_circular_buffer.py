# utils/safe_circular_buffer.py
import numpy as np
from datetime import datetime

class SafeCircularBuffer:
    """
    A safe circular buffer that holds numeric data along with timestamps.
    This implementation avoids misalignment issues when the buffer wraps.
    """
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.full(size, np.nan, dtype=float)
        self.timestamps = np.empty(size, dtype='datetime64[ns]')
        self.index = 0
        self.full = False

    def add(self, value: float, timestamp: datetime):
        self.buffer[self.index] = value
        self.timestamps[self.index] = np.datetime64(timestamp)
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def get_recent(self, length: int):
        if not self.full and self.index < length:
            raise ValueError("Not enough data in buffer")
        indices = [(self.index - i - 1) % self.size for i in range(length)]
        # Return data in chronological order
        data = self.buffer[indices][::-1]
        ts = self.timestamps[indices][::-1]
        return data, ts

    def get_all(self):
        if self.full:
            return self.buffer.copy(), self.timestamps.copy()
        else:
            return self.buffer[:self.index].copy(), self.timestamps[:self.index].copy()
