import numpy as np
import logging
from datetime import datetime
from config import config

from utils.compute_atr import compute_atr
from utils.safe_circular_buffer import SafeCircularBuffer
# Optionally, you can use from collections import deque for buffers if a maximum length is known.


class RuleBasedSignals:
    """
    Aggregates market ticks into candles and computes technical indicators such as EMA, MACD, RSI, and ATR.
    Generates a trading signal (BUY, SELL, or HOLD) with an associated confidence level and risk parameters.
    """
    
    def __init__(self, logger: logging.Logger, candle_interval: int = 1, rsi_window: int = 14, buffer_size: int = 500):
        self.logger = logger
        self.candle_interval = candle_interval  # seconds per candle
        self.rsi_window = rsi_window

        # Current candle state
        self.current_candle_start: datetime = None
        self.current_candle_open: float = None
        self.current_candle_high: float = None
        self.current_candle_low: float = None
        self.current_candle_close: float = None
        self.current_candle_volume: int = 0

        # Fixed‐size buffer for close prices
        self.close_buffer = SafeCircularBuffer(size=buffer_size)
        # You could consider using deques (from collections import deque) for opens, highs, etc., if a maximum length is known.
        self.opens = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.min_confidence = 1.0  # minimum confidence threshold

        # Indicator cache (for example, ATR)
        self.indicator_cache = {"atr": np.array([])}
        
        
    def update_buffers_batch(self, opens: list, highs: list, lows: list, closes: list, volumes: list):
        for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
            self.opens.append(o)
            self.highs.append(h)
            self.lows.append(l)
            self.volumes.append(v)
            self.close_buffer.add(c, datetime.now())
    

    def process_tick(self, timestamp: datetime, price: float):
        """
        Aggregates ticks into candles. If the tick falls within the current candle interval,
        the candle is updated; otherwise, the current candle is finalized and a new candle is started.
        """
        try:
            if self.current_candle_start is None:
                # Initialize new candle
                self.current_candle_start = timestamp
                self.current_candle_open = price
                self.current_candle_high = price
                self.current_candle_low = price
                self.current_candle_close = price
                self.current_candle_volume = 1
                self.logger.debug("Started new candle at %s with price %.2f", timestamp, price)
            else:
                elapsed = (timestamp - self.current_candle_start).total_seconds()
                if elapsed < self.candle_interval:
                    self.current_candle_close = price
                    self.current_candle_high = max(self.current_candle_high, price)
                    self.current_candle_low = min(self.current_candle_low, price)
                    self.current_candle_volume += 1
                    self.logger.debug("Updated candle: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%d",
                                      self.current_candle_open, self.current_candle_high,
                                      self.current_candle_low, self.current_candle_close,
                                      self.current_candle_volume)
                else:
                    # Finalize current candle
                    self.logger.info("Finalizing candle starting at %s: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%d",
                                     self.current_candle_start, self.current_candle_open,
                                     self.current_candle_high, self.current_candle_low,
                                     self.current_candle_close, self.current_candle_volume)
                    self.opens.append(self.current_candle_open)
                    self.highs.append(self.current_candle_high)
                    self.lows.append(self.current_candle_low)
                    self.volumes.append(self.current_candle_volume)
                    self.close_buffer.add(self.current_candle_close, self.current_candle_start)
                    # Start new candle
                    self.current_candle_start = timestamp
                    self.current_candle_open = price
                    self.current_candle_high = price
                    self.current_candle_low = price
                    self.current_candle_close = price
                    self.current_candle_volume = 1
                    self.logger.debug("Started new candle at %s with price %.2f", timestamp, price)
        except Exception:
            self.logger.exception("Error processing tick at %s", timestamp)

    def _calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Computes the Exponential Moving Average (EMA) using the standard recursive formula.
        (For larger datasets, consider using pandas.Series.ewm for vectorized computation.)
        """
        if len(data) == 0:
            return np.array([])
        alpha = 2 / (window + 1)
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def calculate_macd(self):
        """
        Computes the MACD indicator:
          - MACD = EMA(12) - EMA(26)
          - Signal line = EMA(9) of the MACD
        Returns a tuple (macd, signal_line) as numpy arrays or None if insufficient data.
        """
        data, _ = self.close_buffer.get_all()
        if len(data) < 26:
            self.logger.warning("Insufficient data for MACD calculation.")
            return None
        ema12 = self._calculate_ema(data, 12)
        ema26 = self._calculate_ema(data, 26)
        macd = ema12 - ema26
        signal_line = self._calculate_ema(macd, 9)
        return macd, signal_line

    def calculate_rsi(self):
        """
        Computes the Relative Strength Index (RSI) using Wilder's smoothing method.
        Returns the most recent RSI value or None if insufficient data.
        """
        data, _ = self.close_buffer.get_all()
        if len(data) < self.rsi_window + 1:
            self.logger.warning("Insufficient data for RSI calculation.")
            return None
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.zeros_like(gains)
        avg_loss = np.zeros_like(losses)
        avg_gain[0] = np.mean(gains[:self.rsi_window])
        avg_loss[0] = np.mean(losses[:self.rsi_window])
        for i in range(1, len(gains)):
            avg_gain[i] = (avg_gain[i - 1] * (self.rsi_window - 1) + gains[i]) / self.rsi_window
            avg_loss[i] = (avg_loss[i - 1] * (self.rsi_window - 1) + losses[i]) / self.rsi_window
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi[-1]

    def calculate_atr(self, window: int = 14):
        """
        Computes the Average True Range (ATR) using the external compute_atr function.
        Returns the latest ATR value or None if insufficient data.
        """
        highs_arr = np.array(self.highs)
        lows_arr = np.array(self.lows)
        closes_arr, _ = self.close_buffer.get_all()
        atr_arr = compute_atr(highs_arr, lows_arr, closes_arr, window, logger=self.logger)
        return atr_arr[-1] if atr_arr.size > 0 else None

    def _default_hold_signal(self) -> dict:
        """
        Helper method to return a default "HOLD" signal when there is insufficient data.
        """
        return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0, "indicator_contrib": {}}

    def generate_signal(self, adaptive_weights=None) -> dict:
        data, _ = self.close_buffer.get_all()
        if len(data) < 26:
            self.logger.info("Not enough data for signal generation. Holding position.")
            return self._default_hold_signal()

        # Calculate indicator components:
        # Trend component (T) using EMA difference.
        ema_fast = self._calculate_ema(data, 9)
        ema_slow = self._calculate_ema(data, 21)
        price = data[-1]
        # Guard against division by zero by adding a small epsilon if price is zero.
        epsilon = 1e-8
        T = (ema_fast[-1] - ema_slow[-1]) / (price if price != 0 else epsilon)

        # MACD component (M).
        macd_data = self.calculate_macd()
        if macd_data is None:
            self.logger.info("Not enough data for MACD-based momentum. Holding position.")
            return self._default_hold_signal()
        macd, signal_line = macd_data
        scale_factor = 5.0
        M = (macd[-1] - signal_line[-1]) / scale_factor

        # Oscillator component (O) using RSI.
        rsi_val = self.calculate_rsi()
        if rsi_val is None:
            self.logger.info("Not enough data for RSI. Holding position.")
            return self._default_hold_signal()
        O = (50 - rsi_val) / 50.0  # Center RSI around 50

        # Use adaptive weights if provided; otherwise use defaults.
        if adaptive_weights is None:
            w_T, w_M, w_O = 0.4, 0.35, 0.25
        else:
            weights = adaptive_weights.weights
            w_T = weights.get("T", 0.4)
            w_M = weights.get("M", 0.35)
            w_O = weights.get("O", 0.25)

        # Compute the combined signal strength.
        S = w_T * T + w_M * M + w_O * O

        # Compute signal confidence using a sigmoid function.
        k, c = 10.0, 0.0
        confidence = 1 / (1 + np.exp(-k * (S - c)))
        threshold = 1.5
        if S > threshold:
            signal = "BUY"
        elif S < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
            confidence = 0.0

        # Set risk parameters based on instrument type.
        if getattr(config, "INSTRUMENT_TYPE", "underlying") == "volatility_option":
            base_stake = 2000  # or use dynamic sizing
            stop_loss = 0.2 * base_stake
            take_profit = 0.3 * base_stake
        else:
            atr_val = self.calculate_atr(window=14)
            atr_val = atr_val if atr_val is not None and atr_val > 0 else 1.0
            stop_loss = atr_val * 2.0
            take_profit = atr_val * 3.0

        # Calculate indicator contributions (normalized absolute values).
        sum_abs = abs(T) + abs(M) + abs(O)
        if sum_abs == 0:
            indicator_contrib = {"T": 0.0, "M": 0.0, "O": 0.0}
        else:
            indicator_contrib = {
                "T": abs(T) / sum_abs,
                "M": abs(M) / sum_abs,
                "O": abs(O) / sum_abs
            }

        self.logger.info("Signal Generation: %s, S=%.4f, Conf=%.2f, RSI=%.2f, StopLoss=%.2f, TakeProfit=%.2f",
                         signal, S, confidence, rsi_val, stop_loss, take_profit)
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "indicator_contrib": indicator_contrib
        }
            
    def reset_buffers(self):
        """
        Resets internal buffers and indicator cache.
        """
        self.indicator_cache = {"atr": np.array([])}
        self.opens.clear()
        self.highs.clear()
        self.lows.clear()
        self.volumes.clear()
        self.current_candle_start = None
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.current_candle_volume = 0
        self.close_buffer.reset()
        self.logger.debug("Signal buffers reset.")
            
    @property
    def closes(self) -> np.ndarray:
        """
        Exposes the array of close prices from the circular buffer.
        """
        data, _ = self.close_buffer.get_all()
        return data      
    
    def update_buffers(self, open_price: float, high: float, low: float, close: float, volume: float):
        """
        Updates the internal candle buffers with new candle data.
        """
        self.opens.append(open_price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.close_buffer.add(close, datetime.now())
        self.logger.debug("Updated buffers with candle: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%.2f",
                          open_price, high, low, close, volume)
