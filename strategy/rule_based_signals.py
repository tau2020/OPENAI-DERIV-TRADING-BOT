import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging # Added import for type hinting logger if needed elsewhere

# Aligned version with added DEBUG loggers
class RuleBasedSignals:
    """
    Enhanced Rule-Based Signal Engine for Deriv Synthetic Indices (Aligned for TradingStrategy)
    ----
    • Tuned parameters for higher volatility (Synthetics R_10, R_100 etc.)
    • Adjusted indicator settings for sensitivity & accuracy trade-off.
    • Refined risk management (dynamic SL/TP, position sizing).
    • Improved signal strength and confidence scoring logic.
    • Aligned `generate_candle_signal` method and return keys for compatibility.
    • Volatility calculation adjusted for 24/7 markets.
    • **Includes detailed DEBUG log statements.**
    """

    # ------------------------------------------------------------------ #
    #  INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, logger: Optional[logging.Logger] = None): # More specific type hint
        self.logger = logger

        # --- Tunable Parameters (Adjusted for Synthetics) ---
        self.atr_sl_multiplier: float = 1.8
        self.atr_tp_multiplier: float = 3.5
        self.volatility_window: int   = 20
        self.atr_window: int          = 14
        self.rsi_window: int          = 14
        self.rsi_oversold: float      = 25
        self.rsi_overbought: float    = 75
        self.macd_fast: int           = 12
        self.macd_slow: int           = 26
        self.macd_signal: int         = 9
        self.trend_ema_fast: int      = 9
        self.trend_ema_slow: int      = 21
        self.volume_sma_window: int   = 20
        self.volume_spike_multiplier: float = 2.5
        self.min_confidence: float    = 0.75
        self.max_confidence: float    = 0.95
        self.risk_per_trade: float    = 0.015
        self.max_risk_percent_per_trade: float = 0.05
        self.min_position_size: float = 0.01

        # --- Rolling Buffers ---
        self.buffer_size: int = 750
        self.price_data: deque = deque(maxlen=self.buffer_size)
        self.high_data: deque  = deque(maxlen=self.buffer_size)
        self.low_data: deque   = deque(maxlen=self.buffer_size)
        self.volume_data: deque= deque(maxlen=self.buffer_size)
        self.trade_history: deque = deque(maxlen=100)

        # --- Indicator Cache ---
        self._cache: Dict[str, Dict[Tuple, np.ndarray]] = {
            "ema":   {}, "rsi":   {}, "atr":   {}, "macd":  {}, "sma":   {},
        }

        # --- Account State ---
        self.account_balance: float = 1_000.0

    # ------------------------------------------------------------------ #
    #  UTILITY HELPERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_np(data: deque) -> np.ndarray:
        return np.array(data, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  INDICATOR CALCULATIONS (Core logic unchanged)
    # ------------------------------------------------------------------ #
    def _sma(self, data_deque: deque, window: int) -> np.ndarray:
        length = len(data_deque)
        if length < window: return np.full(length, np.nan)
        cache_key = (window, length); cached = self._cache["sma"].get(cache_key)
        if cached is not None: return cached
        arr = self._to_np(data_deque); weights = np.repeat(1.0, window) / window
        sma = np.convolve(arr, weights, 'valid')
        sma_padded = np.concatenate((np.full(window - 1, np.nan), sma))
        self._cache["sma"][cache_key] = sma_padded; return sma_padded

    def _ema(self, data_deque: deque, window: int) -> np.ndarray:
        length = len(data_deque)
        if length == 0: return np.array([])
        if length < 2: return self._to_np(data_deque)
        cache_key = (window, length); cached = self._cache["ema"].get(cache_key)
        if cached is not None: return cached
        arr = self._to_np(data_deque); alpha = 2.0 / (window + 1)
        ema = np.empty_like(arr); ema[0] = arr[0]
        for i in range(1, arr.size): ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
        self._cache["ema"][cache_key] = ema; return ema

    def _wilder_smooth(self, arr: np.ndarray, window: int) -> np.ndarray:
        alpha = 1.0 / window; smoothed = np.empty_like(arr)
        if arr.size >= window:
            smoothed[:window] = np.mean(arr[:window])
            for i in range(window, arr.size): smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
        elif arr.size > 0:
             smoothed[0] = arr[0]
             for i in range(1, arr.size): smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    def _rsi(self, prices_deque: deque) -> np.ndarray:
        length = len(prices_deque)
        if length < self.rsi_window + 1: return np.full(length, np.nan)
        cache_key = (length,); cached = self._cache["rsi"].get(cache_key)
        if cached is not None: return cached
        arr = self._to_np(prices_deque); delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0.0); loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = self._wilder_smooth(gain, self.rsi_window); avg_loss = self._wilder_smooth(loss, self.rsi_window)
        rs = np.full_like(avg_gain, np.inf); non_zero_loss_idx = avg_loss != 0
        rs[non_zero_loss_idx] = avg_gain[non_zero_loss_idx] / avg_loss[non_zero_loss_idx]
        rsi = 100.0 - (100.0 / (1.0 + rs)); rsi_padded = np.concatenate((np.full(1, np.nan), rsi))
        self._cache["rsi"][cache_key] = rsi_padded; return rsi_padded

    def _macd(self, prices_deque: deque) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(prices_deque)
        if length < self.macd_slow: nan_arr = np.full(length, np.nan); return nan_arr, nan_arr, nan_arr
        cache_key = (length,); cached = self._cache["macd"].get(cache_key)
        if cached is not None: return cached[0], cached[1], cached[2]
        ema_fast = self._ema(prices_deque, self.macd_fast); ema_slow = self._ema(prices_deque, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line_deque = deque(macd_line.tolist(), maxlen=self.buffer_size)
        signal_line = self._ema(signal_line_deque, self.macd_signal); histogram = macd_line - signal_line
        out = (macd_line, signal_line, histogram); self._cache["macd"][cache_key] = out
        return macd_line, signal_line, histogram

    def _atr(self, highs_deque: deque, lows_deque: deque, closes_deque: deque) -> np.ndarray:
        length = len(closes_deque)
        if length < self.atr_window: return np.full(length, np.nan)
        cache_key = (length,); cached = self._cache["atr"].get(cache_key)
        if cached is not None: return cached
        h = self._to_np(highs_deque); l = self._to_np(lows_deque); c = self._to_np(closes_deque)
        high_minus_low = h - l; high_minus_prev_close = np.abs(h - np.roll(c, 1))
        low_minus_prev_close = np.abs(l - np.roll(c, 1)); high_minus_prev_close[0] = np.nan
        low_minus_prev_close[0] = np.nan
        true_range = np.nanmax([high_minus_low, high_minus_prev_close, low_minus_prev_close], axis=0)
        true_range[0] = high_minus_low[0]; atr = self._wilder_smooth(true_range, self.atr_window)
        self._cache["atr"][cache_key] = atr; return atr

    def _calculate_volatility(self, prices_deque: deque) -> float:
        arr = self._to_np(prices_deque)
        if arr.size < self.volatility_window + 1: return 0.0
        log_returns = np.log(arr[1:] / arr[:-1])
        if log_returns.size < self.volatility_window: return 0.0
        vol = float(np.std(log_returns[-self.volatility_window:], ddof=0))
        return vol

    # ------------------------------------------------------------------ #
    #  SIGNAL LOGIC COMPONENTS
    # ------------------------------------------------------------------ #

    def _get_market_context(self, closes_deque: deque, highs_deque: deque, lows_deque: deque, volumes_deque: deque) -> Dict:
        context = {"valid": False}
        min_len = max(self.trend_ema_slow, self.macd_slow, self.atr_window, self.rsi_window, self.volatility_window + 1)
        if len(closes_deque) < min_len:
             context["reason"] = f"Insufficient data: Need {min_len}, have {len(closes_deque)}"
             return context # Exit early if not enough data

        # Calculate indicators
        ema_fast_trend = self._ema(closes_deque, self.trend_ema_fast)
        ema_slow_trend = self._ema(closes_deque, self.trend_ema_slow)
        rsi = self._rsi(closes_deque)
        macd_line, signal_line, macd_hist = self._macd(prices_deque=closes_deque) # Corrected arg name
        atr = self._atr(highs_deque, lows_deque, closes_deque)
        vol_sma = self._sma(volumes_deque, self.volume_sma_window)
        volatility = self._calculate_volatility(prices_deque=closes_deque) # Corrected arg name

        # --- ADDED DEBUG LOGGER ---
        if self.logger:
             # Check if latest values exist before logging to avoid index error on very short data
             if len(closes_deque) > 0 and len(ema_fast_trend) > 0 and len(ema_slow_trend) > 0 and \
                len(rsi) > 0 and len(macd_line) > 0 and len(signal_line) > 0 and \
                len(atr) > 0 and len(vol_sma) > 0:
                 self.logger.debug(
                     f"Latest Indicators - Price: {closes_deque[-1]:.4f}, "
                     f"EMA_F: {ema_fast_trend[-1]:.4f}, EMA_S: {ema_slow_trend[-1]:.4f}, "
                     f"RSI: {rsi[-1]:.2f}, MACD_L: {macd_line[-1]:.4f}, "
                     f"MACD_S: {signal_line[-1]:.4f}, ATR: {atr[-1]:.4f}, Vol: {volatility:.5f}"
                 )
             else:
                 self.logger.debug("Could not log latest indicators: insufficient data length.")


        # Check for NaNs in the latest values needed for decisions
        latest_indicators = [ema_fast_trend[-1], ema_slow_trend[-1], rsi[-1], macd_line[-1], signal_line[-1], atr[-1], vol_sma[-1]]
        if np.isnan(latest_indicators).any():
             context["reason"] = "Latest indicator values contain NaN"
             return context # Exit early if core indicators are NaN

        # Determine states (Trend, RSI, MACD, Volume) - Logic remains the same
        trend = "neutral"; rsi_state = "neutral"; macd_crossover = "none" # Defaults
        if ema_fast_trend[-1] > ema_slow_trend[-1] and ema_fast_trend[-2] <= ema_slow_trend[-2]: trend = "bullish_crossover"
        elif ema_fast_trend[-1] < ema_slow_trend[-1] and ema_fast_trend[-2] >= ema_slow_trend[-2]: trend = "bearish_crossover"
        elif ema_fast_trend[-1] > ema_slow_trend[-1]: trend = "bullish"
        elif ema_fast_trend[-1] < ema_slow_trend[-1]: trend = "bearish"

        if rsi[-1] > self.rsi_overbought: rsi_state = "overbought"
        elif rsi[-1] < self.rsi_oversold: rsi_state = "oversold"

        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]: macd_crossover = "bullish"
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]: macd_crossover = "bearish"
        macd_trend = "bullish" if macd_line[-1] > signal_line[-1] else "bearish"

        current_volume = volumes_deque[-1] if volumes_deque else 0
        avg_volume = vol_sma[-1]
        volume_spike = current_volume > avg_volume * self.volume_spike_multiplier if avg_volume > 0 else False
        volume_trend = "rising" if current_volume > avg_volume else "falling" if avg_volume > 0 else "flat"

        # Update context dictionary
        context.update({
            "valid": True, "price": closes_deque[-1], "high": highs_deque[-1], "low": lows_deque[-1],
            "volume": current_volume, "ema_fast": ema_fast_trend[-1], "ema_slow": ema_slow_trend[-1],
            "rsi": rsi[-1], "macd_line": macd_line[-1], "macd_signal": signal_line[-1], "macd_hist": macd_hist[-1],
            "atr": atr[-1], "volatility": volatility, "avg_volume": avg_volume, "trend": trend,
            "rsi_state": rsi_state, "macd_crossover": macd_crossover, "macd_trend": macd_trend,
            "volume_spike": volume_spike, "volume_trend": volume_trend,
        })
        return context

    def _calculate_risk_parameters(self, context: Dict) -> Dict:
        # --- No loggers added here, focuses on calculation ---
        risk_params = {"sl": 0.0, "tp": 0.0, "size": 0.0, "valid": False}
        price = context["price"]; atr = context["atr"]
        if atr <= 0 or price <= 0: risk_params["reason"] = "Invalid price or ATR"; return risk_params
        vol_factor = 1 + min(context["volatility"] * 0.5, 0.5)
        sl_dist = min(atr * self.atr_sl_multiplier * vol_factor, price * self.max_risk_percent_per_trade)
        tp_dist = atr * self.atr_tp_multiplier * vol_factor
        sl_buy = price - sl_dist; tp_buy = price + tp_dist
        sl_sell = price + sl_dist; tp_sell = price - tp_dist
        risk_amt = self.account_balance * self.risk_per_trade
        size_buy = max(self.min_position_size, round(risk_amt / sl_dist, 2)) if sl_dist > 0 else 0.0
        size_sell = max(self.min_position_size, round(risk_amt / sl_dist, 2)) if sl_dist > 0 else 0.0
        risk_params.update({ "sl_buy": sl_buy, "tp_buy": tp_buy, "size_buy": size_buy, "sl_sell": sl_sell,
                             "tp_sell": tp_sell, "size_sell": size_sell, "sl_distance": sl_dist, "valid": True })
        return risk_params

    def _get_performance_factor(self) -> float:
        # --- No loggers added here, simple calculation ---
        num_trades = len(self.trade_history);
        if num_trades < 5: return 1.0
        recent = list(self.trade_history)[-10:]; wins = sum(1 for pnl in recent if pnl > 0)
        win_rate = wins / len(recent); perf_factor = 0.8 + (win_rate * 0.4)
        return perf_factor

    def _calculate_confidence(self, context: Dict, signal_direction: str, perf_factor: float) -> float:
        weights = { "trend_match": 0.30, "trend_crossover": 0.15, "macd_match": 0.25,
                    "macd_crossover": 0.15, "rsi_confirm": 0.10, "volume_confirm": 0.05 }
        raw_score = 0.0
        # Calculate raw score based on signal direction and context (logic unchanged)
        if signal_direction == "BUY":
            if context["trend"] in ["bullish", "bullish_crossover"]: raw_score += weights["trend_match"]
            if context["trend"] == "bullish_crossover": raw_score += weights["trend_crossover"]
            if context["macd_trend"] == "bullish": raw_score += weights["macd_match"]
            if context["macd_crossover"] == "bullish": raw_score += weights["macd_crossover"]
            if context["rsi_state"] != "overbought": raw_score += weights["rsi_confirm"]
            if context["volume_spike"] and context["volume_trend"] == "rising": raw_score += weights["volume_confirm"]
        elif signal_direction == "SELL":
            if context["trend"] in ["bearish", "bearish_crossover"]: raw_score += weights["trend_match"]
            if context["trend"] == "bearish_crossover": raw_score += weights["trend_crossover"]
            if context["macd_trend"] == "bearish": raw_score += weights["macd_match"]
            if context["macd_crossover"] == "bearish": raw_score += weights["macd_crossover"]
            if context["rsi_state"] != "oversold": raw_score += weights["rsi_confirm"]
        # --- ADDED DEBUG LOGGER ---
        if self.logger:
             self.logger.debug(f"Confidence Calc - Raw Score: {raw_score:.4f}, Perf Factor: {perf_factor:.4f}")

        adjusted_score = raw_score * perf_factor
        confidence = float(np.clip(adjusted_score, self.min_confidence, self.max_confidence))
        if raw_score < 0.3: confidence = 0.0 # Filter low raw conviction
        return confidence

    def _generate_signal(self, context: Dict) -> Dict:
        signal = {"direction": "HOLD", "strength": 0.0}
        # Check BUY conditions
        is_buy_trend = context["trend"] in ["bullish", "bullish_crossover"]
        is_buy_macd = context["macd_crossover"] == "bullish" or (context["macd_trend"] == "bullish" and context["trend"] != "bearish")
        is_rsi_ok_buy = context["rsi_state"] != "overbought"
        if is_buy_trend and is_buy_macd and is_rsi_ok_buy:
            signal["direction"] = "BUY"
            signal["strength"] = ((1 if context["trend"] == "bullish_crossover" else 0.5) + (1 if context["macd_crossover"] == "bullish" else 0.5) + (1 if context["rsi_state"] == "oversold" else 0.5)) / 3.0

        # Check SELL conditions
        is_sell_trend = context["trend"] in ["bearish", "bearish_crossover"]
        is_sell_macd = context["macd_crossover"] == "bearish" or (context["macd_trend"] == "bearish" and context["trend"] != "bullish")
        is_rsi_ok_sell = context["rsi_state"] != "oversold"
        if is_sell_trend and is_sell_macd and is_rsi_ok_sell:
            signal["direction"] = "SELL"
            signal["strength"] = ((1 if context["trend"] == "bearish_crossover" else 0.5) + (1 if context["macd_crossover"] == "bearish" else 0.5) + (1 if context["rsi_state"] == "overbought" else 0.5)) / 3.0

        # ATR/Volatility Filter
        min_atr_threshold = context["price"] * 0.0005
        min_volatility_threshold = 0.001
        low_vol_filter_active = context["atr"] < min_atr_threshold or context["volatility"] < min_volatility_threshold

        if low_vol_filter_active:
             if signal["direction"] != "HOLD" and self.logger:
                 self.logger.info(f"Signal {signal['direction']} suppressed due to low ATR/Volatility.")
             signal = {"direction": "HOLD", "strength": 0.0} # Override

        # --- ADDED DEBUG LOGGER (before returning) ---
        if self.logger:
             self.logger.debug(f"Signal Checks - is_buy_trend: {is_buy_trend}, is_buy_macd: {is_buy_macd}, is_rsi_ok_buy: {is_rsi_ok_buy}")
             self.logger.debug(f"Signal Checks - is_sell_trend: {is_sell_trend}, is_sell_macd: {is_sell_macd}, is_rsi_ok_sell: {is_rsi_ok_sell}")
             self.logger.debug(f"ATR/Vol Filter Check - ATR: {context['atr']:.4f} > {min_atr_threshold:.4f}? Vol: {context['volatility']:.5f} > {min_volatility_threshold:.5f}? Filter Active: {low_vol_filter_active}")
             self.logger.debug(f"Final Signal Generated by _generate_signal: {signal['direction']}")

        return signal

    def _suggested_holding_time(self, context: Dict, signal_strength: float) -> int:
        # --- No loggers added here, simple calculation ---
        base_minutes = 1; max_minutes = 5
        vol_factor = 1 / (1 + context["volatility"] * 10)
        hold_time = base_minutes + (signal_strength * 3)
        hold_time *= vol_factor
        return int(max(1, min(max_minutes, round(hold_time))))

    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #
    def generate_candle_signal(self, candle_data: List[Dict[str, float]]) -> Dict[str, object]:
        # --- Data Population ---
        self.price_data.clear(); self.high_data.clear(); self.low_data.clear(); self.volume_data.clear()
        has_volume = all("volume" in c for c in candle_data)
        valid_candles_count = 0
        for candle in candle_data:
             if not all(k in candle for k in ["close", "high", "low"]):
                 if self.logger: self.logger.warning(f"Skipping candle due to missing keys: {candle}")
                 continue
             self.price_data.append(float(candle["close"]))
             self.high_data.append(float(candle["high"]))
             self.low_data.append(float(candle["low"]))
             self.volume_data.append(float(candle.get("volume", 1.0)))
             valid_candles_count += 1
        if not has_volume and self.logger: self.logger.debug("Volume data missing in input, using default value 1.0.")

        # --- ADDED DEBUG LOGGER ---
        if self.logger:
             self.logger.debug(f"Input candle count: {len(candle_data)}, Populated deque length (valid candles): {valid_candles_count}")

        # Minimum length check
        min_len = max(self.trend_ema_slow, self.macd_slow, self.atr_window, self.rsi_window, self.volatility_window + 1)
        if valid_candles_count < min_len:
             return self._hold_payload(f"Insufficient valid data points: Need {min_len}, have {valid_candles_count}")

        # 1. Get Market Context
        context = self._get_market_context(self.price_data, self.high_data, self.low_data, self.volume_data)
        if not context["valid"]: return self._hold_payload(context.get("reason", "Invalid market context"))

        # 2. Generate Base Signal
        signal = self._generate_signal(context)
        if signal["direction"] == "HOLD": return self._hold_payload("Signal logic determined HOLD")

        # 3. Calculate Risk Parameters
        risk_params = self._calculate_risk_parameters(context)
        if not risk_params["valid"]: return self._hold_payload(risk_params.get("reason", "Risk parameter calculation failed"))

        # 4. Calculate Performance Factor & Confidence
        perf_factor = self._get_performance_factor()
        confidence = self._calculate_confidence(context, signal["direction"], perf_factor)

        # --- ADDED DEBUG LOGGER ---
        if self.logger:
             self.logger.debug(f"Pre-Confidence Check - Signal: {signal['direction']}, Confidence Calculated: {confidence:.4f} (Min Required: {self.min_confidence})")

        # 5. Final Confidence Filter
        if confidence < self.min_confidence:
             # Reason logged internally by _hold_payload if logger exists
             return self._hold_payload(f"Confidence {confidence:.2f} below threshold {self.min_confidence}")

        # --- Assemble Final Signal Payload ---
        final_signal = { "signal": signal["direction"], "confidence": confidence }
        if signal["direction"] == "BUY":
            final_signal["stop_loss"] = round(risk_params["sl_buy"], 5)
            final_signal["take_profit"] = round(risk_params["tp_buy"], 5)
            final_signal["position_size"] = risk_params["size_buy"]
        elif signal["direction"] == "SELL":
            final_signal["stop_loss"] = round(risk_params["sl_sell"], 5)
            final_signal["take_profit"] = round(risk_params["tp_sell"], 5)
            final_signal["position_size"] = risk_params["size_sell"]
        final_signal["max_holding_time"] = self._suggested_holding_time(context, signal["strength"])
        final_signal["trailing_stop"] = round(context["atr"] * self.atr_sl_multiplier, 5)

        # --- ADDED DEBUG LOGGER ---
        pos_size = final_signal.get("position_size", 0.0)
        if self.logger:
             self.logger.debug(f"Pre-Size Check - Position Size: {pos_size}")

        # Filter out zero size trades
        if pos_size <= 0:
            # Reason logged internally by _hold_payload if logger exists
            return self._hold_payload(f"Calculated position size is zero ({pos_size})")

        return final_signal # Return the final BUY/SELL signal dictionary

    # ------------------------------------------------------------------ #
    #  HELPER: HOLD PAYLOAD
    # ------------------------------------------------------------------ #
    def _hold_payload(self, reason: str) -> Dict[str, object]:
        if self.logger:
            # Log HOLD reason at DEBUG level
            self.logger.debug(f"HOLD Signal – Reason: {reason}")
        return { "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
                 "position_size": 0.0, "max_holding_time": 0, "trailing_stop": 0.0, }

    # ------------------------------------------------------------------ #
    #  EXTERNAL STATE UPDATE
    # ------------------------------------------------------------------ #
    def record_trade_result(self, pnl: float):
         self.trade_history.append(pnl)