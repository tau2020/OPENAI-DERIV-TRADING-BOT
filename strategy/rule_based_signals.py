# strategy/rule_based_signals.py
import numpy as np
import logging
from datetime import datetime
from config import config
from utils.safe_circular_buffer import SafeCircularBuffer

class RuleBasedSignals:

    
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
        self.opens = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.min_confidence = 0.07  # minimum confidence threshold
        
        # Indicator cache (for example, ATR)
        self.indicator_cache = {"atr": np.array([])}

    def process_tick(self, timestamp: datetime, price: float):
        """
        Aggregates ticks into candles. If the tick falls within the current candle interval,
        the candle is updated; otherwise, the current candle is finalized and a new candle is started.
        """
        if self.current_candle_start is None:
            self.current_candle_start = timestamp
            self.current_candle_open = price
            self.current_candle_high = price
            self.current_candle_low = price
            self.current_candle_close = price
            self.current_candle_volume = 5
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
                self.current_candle_volume = 5
                self.logger.debug("Started new candle at %s with price %.2f", timestamp, price)

    def _calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Computes the Exponential Moving Average (EMA) using the standard recursive formula.
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
        Computes the MACD indicator as:
          MACD = EMA(12) - EMA(26)
          Signal line = EMA(9) of the MACD
        Returns a tuple (macd, signal_line) as numpy arrays.
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
        Returns the most recent RSI value.
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
        Computes the Average True Range (ATR) using completed candle data.
        Returns the latest ATR value.
        """
        if len(self.opens) < window + 1:
            self.logger.warning("Insufficient candle data for ATR calculation.")
            return None
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        closes, _ = self.close_buffer.get_all()
        if len(closes) < window + 1:
            self.logger.warning("Insufficient close data for ATR calculation.")
            return None
        tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
        atr = np.empty_like(tr)
        atr[0] = np.mean(tr[:window])
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
        return atr[-1]

    def generate_signal(self) -> dict:
        
    
        # Retrieve historical close data.
        data, _ = self.close_buffer.get_all()
        if len(data) < 26:
            self.logger.info("Not enough data for signal generation. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}

        # --- Calculate Indicator Components (Trend, Momentum, Oscillator) ---
        ema_fast = self._calculate_ema(data, 9)
        ema_slow = self._calculate_ema(data, 21)
        price = data[-1]
        T = (ema_fast[-1] - ema_slow[-1]) / price

        macd_data = self.calculate_macd()
        if macd_data is None:
            self.logger.info("Not enough data for MACD-based momentum. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        macd, signal_line = macd_data
        scale_factor = 5.0
        M = (macd[-1] - signal_line[-1]) / scale_factor

        rsi_val = self.calculate_rsi()
        if rsi_val is None:
            self.logger.info("Not enough data for RSI. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        O = (50 - rsi_val) / 50.0  # Center RSI around 50

        w_T, w_M, w_O = 0.4, 0.35, 0.25
        S = w_T * T + w_M * M + w_O * O
        self.logger.debug("Indicator Components: Trend=%.4f, Momentum=%.4f, Oscillator=%.4f; Combined S=%.4f", T, M, O, S)

        k, c = 10.0, 0.0
        confidence = 1 / (1 + np.exp(-k * (S - c)))

        threshold = 0.08
        if S > threshold:
            signal = "BUY"
        elif S < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
            confidence = 0.0

        # --- Risk Parameters ---
        if getattr(config, "INSTRUMENT_TYPE", "underlying") == "volatility_option":
            # For volatility options, use fixed percentage-based risk parameters
            base_stake = 2000  # or use self.dynamic_position_size if set
            stop_loss = 0.2 * base_stake    # e.g., 20% of the stake
            take_profit = 0.3 * base_stake  # e.g., 30% of the stake
        else:
            atr_val = self.calculate_atr(window=14)
            atr_val = atr_val if atr_val and atr_val > 0 else 1.0
            stop_loss = atr_val * 2.0
            take_profit = atr_val * 3.0

        self.logger.info("Signal Generation: %s, S=%.4f, Conf=%.2f, RSI=%.2f, StopLoss=%.2f, TakeProfit=%.2f", 
                        signal, S, confidence, rsi_val, stop_loss, take_profit)
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    
    def reset_buffers(self):
        """Resets internal buffers and indicator cache."""
        self.indicator_cache = {"atr": np.array([])}
        self.opens = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.current_candle_start = None
        self.current_candle_open = None
        self.current_candle_high = None
        self.current_candle_low = None
        self.current_candle_close = None
        self.current_candle_volume = 0
        self.close_buffer = SafeCircularBuffer(size=self.close_buffer.size)
        self.logger.debug("Signal buffers reset.")
            
    @property
    def closes(self) -> np.ndarray:
        """Exposes the array of close prices from the circular buffer."""
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
