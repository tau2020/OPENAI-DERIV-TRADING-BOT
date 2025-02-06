import numpy as np
import logging
from datetime import datetime
from utils.safe_circular_buffer import SafeCircularBuffer

class RuleBasedSignals:
    """
    Generates trading signals based on a fusion of technical indicators computed from candle data.
    
    Ticks are aggregated into candles (OHLCV) over a configurable interval.
    
    The signal logic integrates:
      1. Trend analysis via an EMA crossover (fast vs. slow).
      2. Momentum analysis via MACD (the difference between a fast and slow EMA, and its signal line).
      3. Oscillator analysis via RSI (using Wilder's smoothing) centered around a neutral level.
      
    These individual components are normalized, weighted, and then fused via a logistic transformation 
    to yield a continuous confidence level. Risk parameters (stop loss and take profit) are dynamically set 
    based on ATR.
    """
    
    def __init__(self, logger: logging.Logger, candle_interval: int = 60, rsi_window: int = 14, buffer_size: int = 500):
        self.logger = logger
        self.candle_interval = candle_interval      # Duration (in seconds) of one candle.
        self.rsi_window = rsi_window                # Window length for RSI computation.
        
        # Internal state for aggregating ticks into a candle.
        self.current_candle_start: datetime = None
        self.current_candle_open: float = None
        self.current_candle_high: float = None
        self.current_candle_low: float = None
        self.current_candle_close: float = None
        self.current_candle_volume: int = 0

        # Use a SafeCircularBuffer to store close prices (ensuring robust fixed-size history).
        self.close_buffer = SafeCircularBuffer(size=buffer_size)
        # For open, high, low and volume, we use lists.
        self.opens = []
        self.highs = []
        self.lows = []
        self.volumes = []
        # Set a minimum confidence threshold for signals.
        self.min_confidence = 0.6
        
        # For indicator caching, e.g. ATR.
        self.indicator_cache = {"atr": np.array([])}

    def process_tick(self, timestamp: datetime, price: float):
        """
        Aggregates ticks into candles. If the tick's timestamp is within the current candle interval,
        update the candle; otherwise, finalize and store the candle and start a new one.
        """
        if self.current_candle_start is None:
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
                self.logger.info("Finalizing candle starting at %s: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%d",
                                 self.current_candle_start, self.current_candle_open,
                                 self.current_candle_high, self.current_candle_low,
                                 self.current_candle_close, self.current_candle_volume)
                self.opens.append(self.current_candle_open)
                self.highs.append(self.current_candle_high)
                self.lows.append(self.current_candle_low)
                self.volumes.append(self.current_candle_volume)
                self.close_buffer.add(self.current_candle_close, self.current_candle_start)
                # Start new candle.
                self.current_candle_start = timestamp
                self.current_candle_open = price
                self.current_candle_high = price
                self.current_candle_low = price
                self.current_candle_close = price
                self.current_candle_volume = 1
                self.logger.debug("Started new candle at %s with price %.2f", timestamp, price)

    def _calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """
        Computes the Exponential Moving Average (EMA) using the standard recursive formula.
        
        Formula:
          EMA[0] = data[0]
          EMA[i] = α * data[i] + (1 - α) * EMA[i-1]  
        where α = 2 / (window + 1)
        """
        alpha = 2 / (window + 1)
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def calculate_macd(self):
        """
        Computes the MACD indicator.
        
        MACD is defined as:
          MACD = EMA(12) - EMA(26)
          Signal line = EMA(9) of the MACD
        
        Returns:
          Tuple (macd, signal_line) as numpy arrays.
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
        
        Steps:
          - Compute deltas between successive close prices.
          - Separate positive gains and negative losses.
          - Compute the smoothed average gain and loss.
          - RSI = 100 - (100 / (1 + RS)), where RS = avg_gain / (avg_loss + eps)
        
        Returns:
          The most recent RSI value.
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
        # Initialize with simple average for the first window.
        avg_gain[0] = np.mean(gains[:self.rsi_window])
        avg_loss[0] = np.mean(losses[:self.rsi_window])
        for i in range(1, len(gains)):
            avg_gain[i] = (avg_gain[i - 1] * (self.rsi_window - 1) + gains[i]) / self.rsi_window
            avg_loss[i] = (avg_loss[i - 1] * (self.rsi_window - 1) + losses[i]) / self.rsi_window
        rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi[-1]

    def calculate_atr(self, window: int = 14):
        """
        Computes the Average True Range (ATR) using Wilder's smoothing on completed candle data.
        
        True Range (TR) for each candle is the maximum of:
          - Current High - Current Low
          - |Current High - Previous Close|
          - |Current Low - Previous Close|
        
        ATR is the smoothed average of the TR values.
        
        Returns:
          The most recent ATR value.
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

    def generate_signal(self):
        """
        Produces a trading signal by fusing several technical indicator components.
        
        The signal is built from three main components:
        
          1. Trend Component (T):
             - Derived from an EMA crossover.
             - Normalized as: T = (EMA_fast - EMA_slow) / Price.
             - A positive T indicates bullish trend; negative T indicates bearish.
          
          2. Momentum Component (M):
             - Derived from MACD.
             - Defined as the normalized difference: M = (MACD - Signal_line) / Scale,
               where the scale factor (e.g., 5) adjusts the sensitivity.
          
          3. Oscillator Component (O):
             - Derived from RSI.
             - Center RSI around 50: O = (50 - RSI) / 50.
             - This yields positive values for oversold conditions (bullish) and negative for overbought.
        
        These components are combined using weights (w_T, w_M, w_O):
        
            S = w_T * T + w_M * M + w_O * O
        
        The combined score S is then transformed via a logistic (sigmoid) function:
        
            Confidence = 1 / (1 + exp(-k * (S - c)))
        
        where k determines the steepness and c is the center threshold.
        
        Finally, a decision threshold is applied:
          - If S > threshold, signal = BUY.
          - If S < -threshold, signal = SELL.
          - Otherwise, signal = HOLD.
        
        Risk parameters (stop loss and take profit) are set as multiples of ATR.
        
        Returns:
          A dictionary with keys: signal, confidence, stop_loss, take_profit.
        """
        # Retrieve historical close data.
        data, _ = self.close_buffer.get_all()
        if len(data) < 26:
            self.logger.info("Not enough data for signal generation. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}

        # Trend Component: EMA crossover.
        ema_fast = self._calculate_ema(data, 9)
        ema_slow = self._calculate_ema(data, 21)
        price = data[-1]
        T = (ema_fast[-1] - ema_slow[-1]) / price  # Normalized difference
        
        # Momentum Component: MACD difference.
        macd_data = self.calculate_macd()
        if macd_data is None:
            self.logger.info("Not enough data for MACD-based momentum. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        macd, signal_line = macd_data
        scale_factor = 5.0  # Empirically chosen scaling factor.
        M = (macd[-1] - signal_line[-1]) / scale_factor

        # Oscillator Component: RSI centered at 50.
        rsi_val = self.calculate_rsi()
        if rsi_val is None:
            self.logger.info("Not enough data for RSI. Holding position.")
            return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0, "take_profit": 0.0}
        O = (50 - rsi_val) / 50.0

        # Define component weights.
        w_T = 0.4  # Weight for Trend.
        w_M = 0.35 # Weight for Momentum.
        w_O = 0.25 # Weight for Oscillator.

        # Compute combined score.
        S = w_T * T + w_M * M + w_O * O
        self.logger.debug("Indicator Components: Trend=%.4f, Momentum=%.4f, Oscillator=%.4f; Combined S=%.4f", T, M, O, S)
        
        # Apply logistic (sigmoid) transformation.
        k = 10.0  # Steepness of the sigmoid.
        c = 0.0   # Center of the sigmoid (threshold).
        confidence = 1 / (1 + np.exp(-k * (S - c)))
        
        # Decision thresholds.
        threshold = 0.10
        if S > threshold:
            signal = "BUY"
        elif S < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
            confidence = 0.0  # No decisive signal.

        # Determine risk parameters using ATR.
        atr_val = self.calculate_atr(window=14)
        atr_val = atr_val if atr_val and atr_val > 0 else 1.0
        stop_loss = atr_val * 2.0
        take_profit = atr_val * 3.0

        self.logger.info("Signal Generation: %s, S=%.4f, Conf=%.2f, RSI=%.2f, ATR=%.2f", 
                         signal, S, confidence, rsi_val, atr_val)
        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
    def reset_buffers(self):
    

            self.indicator_cache = {"atr": np.array([])}
            self.logger.debug("Signal buffers reset.")
            
    @property
    def closes(self) -> np.ndarray:
        """
        Exposes the array of close prices from the circular buffer.
        This property allows external code (e.g., indicator calculations) to access close data.
        """
        data, _ = self.close_buffer.get_all()
        return data      
    
    
    def update_buffers(self, open_price: float, high: float, low: float, close: float, volume: float):
        """
        Updates the internal candle buffers with new candle data.
        This method is used to backfill or update the internal state from historical data.
        """
        self.opens.append(open_price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        # We use the current datetime as a timestamp; modify if you have an actual timestamp.
        self.close_buffer.add(close, datetime.now())
        self.logger.debug("Updated buffers with candle: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%.2f",
                          open_price, high, low, close, volume)  