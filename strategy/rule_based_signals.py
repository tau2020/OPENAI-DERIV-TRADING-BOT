import numpy as np
import logging
from datetime import datetime
from config import config
from utils.compute_atr import compute_atr
from utils.safe_circular_buffer import SafeCircularBuffer

# --- AdaptiveWeights Class ---
class AdaptiveWeights:
    def __init__(self, initial_weights: dict):
        """
        initial_weights: A dict such as {"T": 0.5, "M": 0.25, "O": 0.25} representing
                         the initial weights for Trend (T), MACD (M), and Oscillator (O).
        """
        self.weights = initial_weights.copy()
        self.indicator_performance = {"T": [], "M": [], "O": []}

    def record_trade(self, indicator_contributions: dict, profit: float):
        """
        Record a trade’s performance.
        
        Parameters:
          - indicator_contributions: dict with keys 'T', 'M', 'O' representing the relative
                                     contribution of each indicator.
          - profit: The realized profit (or loss) from the trade.
        """
        for key, contrib in indicator_contributions.items():
            self.indicator_performance[key].append(profit * contrib)

    def update_weights(self):
        """
        Update indicator weights based on recent performance.
        Computes an average score per indicator and uses a softmax transformation to produce
        normalized weights. A smoothing factor prevents abrupt changes.
        """
        scores = {}
        for key, perf in self.indicator_performance.items():
            scores[key] = np.mean(perf) if perf else 0.0

        # Softmax transformation.
        exp_scores = np.exp(np.array(list(scores.values())))
        if np.sum(exp_scores) == 0:
            norm_weights = np.array([1/3] * 3)
        else:
            norm_weights = exp_scores / np.sum(exp_scores)
        new_weights = dict(zip(scores.keys(), norm_weights))

        # Smoothing factor to avoid abrupt changes.
        smoothing = 0.1
        for key in self.weights.keys():
            self.weights[key] = (1 - smoothing) * self.weights[key] + smoothing * new_weights[key]

        # Optionally clear performance history (or use a rolling window).
        self.indicator_performance = {key: [] for key in self.indicator_performance}
        return self.weights

# --- RuleBasedSignals Class ---
class RuleBasedSignals:
    """
    Aggregates market ticks into candles and computes technical indicators (EMA, MACD, RSI, ATR).
    Generates a trading signal (BUY, SELL, or HOLD) with associated confidence, risk parameters,
    and position sizing. This version integrates adaptive weights so that indicator contributions
    are recorded and used to update the weighting over time.
    """
    
    def __init__(self, logger: logging.Logger, candle_interval: int = 60, rsi_window: int = 14, buffer_size: int = 500):
        self.logger = logger
        self.candle_interval = candle_interval  # seconds per candle
        self.rsi_window = rsi_window

        # Current candle state.
        self.current_candle_start: datetime = None
        self.current_candle_open: float = None
        self.current_candle_high: float = None
        self.current_candle_low: float = None
        self.current_candle_close: float = None
        self.current_candle_volume: int = 0

        # Fixed‐size buffer for close prices.
        self.close_buffer = SafeCircularBuffer(size=buffer_size)
        self.opens = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.min_confidence = 1.5  # minimum confidence threshold

        # Indicator cache (e.g., for ATR).
        self.indicator_cache = {"atr": np.array([])}
        
        # History for S values (for dynamic threshold calculation).
        self.s_history = []
        
        # Initialize adaptive weights with default weights.
        self.adaptive_weights = AdaptiveWeights({"T": 0.5, "M": 0.25, "O": 0.25})
        self.last_indicator_contrib = None

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
        the candle is updated; otherwise, the current candle is finalized and a new one is started.
        """
        try:
            if self.current_candle_start is None:
                # Initialize new candle.
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
                    # Finalize the current candle.
                    self.logger.info("Finalizing candle starting at %s: O=%.2f, H=%.2f, L=%.2f, C=%.2f, V=%d",
                                     self.current_candle_start, self.current_candle_open,
                                     self.current_candle_high, self.current_candle_low,
                                     self.current_candle_close, self.current_candle_volume)
                    self.opens.append(self.current_candle_open)
                    self.highs.append(self.current_candle_high)
                    self.lows.append(self.current_candle_low)
                    self.volumes.append(self.current_candle_volume)
                    self.close_buffer.add(self.current_candle_close, self.current_candle_start)
                    # Start a new candle.
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
        if len(data) == 0:
            return np.array([])
        alpha = 2 / (window + 1)
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def calculate_macd(self):
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
        highs_arr = np.array(self.highs)
        lows_arr = np.array(self.lows)
        closes_arr, _ = self.close_buffer.get_all()
        atr_arr = compute_atr(highs_arr, lows_arr, closes_arr, window, logger=self.logger)
        return atr_arr[-1] if atr_arr.size > 0 else None

    def _default_hold_signal(self) -> dict:
        return {"signal": "HOLD", "confidence": 0.0, "stop_loss": 0.0,
                "take_profit": 0.0, "indicator_contrib": {}}

    def generate_signal(self) -> dict:
        # Retrieve historical price data.
        data, _ = self.close_buffer.get_all()
        if len(data) < 26:
            self.logger.info("Not enough data for signal generation. Holding position.")
            return self._default_hold_signal()

        # === Indicator Calculations ===
        # 1. Trend component (T): difference between fast and slow EMA (9 and 21 periods).
        ema_fast = self._calculate_ema(data, 9)
        ema_slow = self._calculate_ema(data, 21)
        price = data[-1]
        epsilon = 1e-8
        T = (ema_fast[-1] - ema_slow[-1]) / (price if price != 0 else epsilon)

        # 2. MACD component (M)
        macd_data = self.calculate_macd()
        if macd_data is None:
            self.logger.info("Not enough data for MACD-based momentum. Holding position.")
            return self._default_hold_signal()
        macd, signal_line = macd_data
        scale_factor = 5.0
        M = (macd[-1] - signal_line[-1]) / scale_factor

        # 3. Oscillator component (O) using RSI.
        rsi_val = self.calculate_rsi()
        if rsi_val is None:
            self.logger.info("Not enough data for RSI. Holding position.")
            return self._default_hold_signal()
        # Note: The RSI component is scaled with a target of 60.
        O = (60 - rsi_val) / 50.0

        # Use the adaptive weights stored in the instance.
        weights = self.adaptive_weights.weights
        w_T = weights.get("T", 0.5)
        w_M = weights.get("M", 0.25)
        w_O = weights.get("O", 0.25)

        # Combined signal strength.
        S = w_T * T + w_M * M + w_O * O

        # === Revised Confidence Calculation ===
        # Use lower steepness and a center shifted to 1.5.
        k = 3.0   # reduced steepness
        c = 1.5   # center shifted
        confidence = 2 * (1 / (1 + np.exp(-k * (abs(S) - c))) - 0.5)

        # === Dynamic Threshold Calculation Based on Historical S ===
        self.s_history.append(S)
        max_history = 100
        if len(self.s_history) > max_history:
            self.s_history = self.s_history[-max_history:]
        min_history_length = 10
        default_threshold = 0.1
        if len(self.s_history) < min_history_length:
            dynamic_threshold = default_threshold
        else:
            s_std = np.std(self.s_history)
            min_threshold = 0.1
            dynamic_threshold = s_std if s_std >= min_threshold else min_threshold

        # === Signal Determination ===
        if S > dynamic_threshold:
            signal = "BUY"
        elif S < -dynamic_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
            confidence = 0.0

        # === Refined Entry Filter for BUY Signals ===
        if signal == "BUY":
            if not (S > 1.5 and S <= 2.08 and confidence >= 0.99 and 50 <= rsi_val <= 70 and dynamic_threshold < 0.50):
                self.logger.info("BUY signal does not meet the refined entry filter criteria; overriding to HOLD.")
                signal = "HOLD"
                confidence = 0.0

        # === Risk Parameter and Position Sizing Setup ===
        if getattr(config, "INSTRUMENT_TYPE", "underlying") == "volatility_option":
            atr_val = self.calculate_atr(window=14)
            atr_val = atr_val if atr_val is not None and atr_val > 0 else 1.0
            account_balance = getattr(config, "STARTING_BALANCE", 10000)
            # Risk 2% per trade divided by (2 * ATR).
            position_size = account_balance * 0.02 / (atr_val * 2)
            stop_loss = atr_val * 1.0   # tighter stop-loss
            take_profit = atr_val * 3.5 # higher reward
        else:
            atr_val = self.calculate_atr(window=14)
            atr_val = atr_val if atr_val is not None and atr_val > 0 else 1.0
            stop_loss = atr_val * 2.0
            take_profit = atr_val * 3.0
            position_size = 1.0

        # === Indicator Contribution (for adaptive weight updates) ===
        sum_abs = abs(T) + abs(M) + abs(O)
        if sum_abs == 0:
            indicator_contrib = {"T": 0.0, "M": 0.0, "O": 0.0}
        else:
            indicator_contrib = {
                "T": abs(T) / sum_abs,
                "M": abs(M) / sum_abs,
                "O": abs(O) / sum_abs
            }
        
        # Store the last indicator contributions.
        self.last_indicator_contrib = indicator_contrib

        self.logger.info("Signal Generation: %s, S=%.4f, Conf=%.4f, RSI=%.2f, DynThreshold=%.4f, StopLoss=%.2f, TakeProfit=%.2f, PositionSize=%.2f",
                         signal, S, confidence, rsi_val, dynamic_threshold, stop_loss, take_profit, position_size)

        return {
            "signal": signal,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
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
                          
    def record_trade_result(self, profit: float):
        """
        Records the outcome of a trade by updating the adaptive weights based on the last indicator
        contributions. This method should be called after a trade is closed.
        """
        if self.last_indicator_contrib is not None:
            self.adaptive_weights.record_trade(self.last_indicator_contrib, profit)
            updated_weights = self.adaptive_weights.update_weights()
            self.logger.info("Adaptive weights updated: %s", updated_weights)
