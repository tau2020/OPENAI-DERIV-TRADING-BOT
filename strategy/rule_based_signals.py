import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

class RuleBasedSignals:
    """
    Enhanced Rule-Based Signal Engine for Deriv Synthetic Indices (R_10 Scalping Focus)
    ----
    • Tuned parameters for higher sensitivity scalping on R_10.
    • Adjusted indicator settings for quicker reaction to price changes.
    • Refined risk management (tighter SL/TP for scalping, adjusted R:R).
    • Optimized for rapid entry and exit for small profits.
    • Aligned `generate_candle_signal` method and return keys for compatibility.
    • **Critical fix attempt for MACD signal line calculation.**
    • **Adjustable ATR/Volatility filter parameters for fine-tuning.**
    """

    # ------------------------------------------------------------------ #
    #  INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

        # --- Parameters for R_10 Scalping (Adjust as needed) ---
        self.atr_sl_multiplier: float = 1.0
        self.atr_tp_multiplier: float = 1.2 # Aim for >1:1 R:R

        self.volatility_window: int   = 10
        self.atr_window: int          = 10
        self.rsi_window: int          = 9
        self.rsi_oversold: float      = 30
        self.rsi_overbought: float    = 70
        self.macd_fast: int           = 5
        self.macd_slow: int           = 13
        self.macd_signal: int         = 5 # EMA of the MACD line
        self.trend_ema_fast: int      = 7
        self.trend_ema_slow: int      = 14
        self.volume_sma_window: int   = 15
        self.volume_spike_multiplier: float = 2.0

        self.min_confidence: float    = 0.60
        self.max_confidence: float    = 0.95
        
        self.risk_per_trade: float    = 0.01
        self.max_risk_percent_per_trade: float = 0.02
        self.min_position_size: float = 0.01 # Check Deriv's min for R_10

        # --- ATR/Volatility Filter Parameters (CRITICAL TO TUNE FOR R_10) ---
        # Adjust these based on live R_10 M1 data observation
        self.min_atr_threshold_factor: float = 0.00015 # Example: If price is 6300, min ATR = 6300*0.00015 = 0.945
        self.min_volatility_threshold: float = 0.00005 # Log return standard deviation

        self.buffer_size: int = 750 # Should be ample
        self.price_data: deque = deque(maxlen=self.buffer_size)
        self.high_data: deque  = deque(maxlen=self.buffer_size)
        self.low_data: deque   = deque(maxlen=self.buffer_size)
        self.volume_data: deque= deque(maxlen=self.buffer_size)
        self.trade_history: deque = deque(maxlen=100)

        self._cache: Dict[str, Dict[Tuple, np.ndarray]] = {
            "ema": {}, "rsi": {}, "atr": {}, "macd": {}, "sma": {},
        }
        self.account_balance: float = 1_000.0

    @staticmethod
    def _to_np(data: deque) -> np.ndarray:
        return np.array(data, dtype=np.float64)

    def _sma(self, data_deque: deque, window: int) -> np.ndarray:
        length = len(data_deque)
        if length < window: return np.full(length, np.nan)
        cache_key = ("sma", tuple(data_deque)[-1], window, length) # More specific cache key
        if cache_key in self._cache["sma"]: return self._cache["sma"][cache_key]
        
        arr = self._to_np(data_deque)
        weights = np.repeat(1.0, window) / window
        sma_values = np.convolve(arr, weights, 'valid')
        
        result = np.full(length, np.nan)
        result[window - 1:] = sma_values # Align to the end
        self._cache["sma"][cache_key] = result
        return result

    def _ema(self, data_deque: deque, window: int, source_for_debug: str = "price") -> np.ndarray:
        length = len(data_deque)
        if length == 0: return np.array([])
        
        # More specific cache key: include source and first/last few elements if data_deque is short
        # For simplicity, using source and last element as a proxy.
        # A proper hash of the deque content would be more robust but costly.
        last_val = data_deque[-1] if length > 0 else np.nan
        cache_key = ("ema", source_for_debug, last_val, window, length)
        if cache_key in self._cache["ema"]:
            return self._cache["ema"][cache_key]

        arr = self._to_np(data_deque)
        if arr.size < 1 : return np.full(length, np.nan) # Should be caught by length check already

        alpha = 2.0 / (window + 1)
        ema_values = np.empty_like(arr)
        
        # Find first valid (non-NaN) value to start EMA
        first_valid_idx = 0
        while first_valid_idx < arr.size and np.isnan(arr[first_valid_idx]):
            ema_values[first_valid_idx] = np.nan
            first_valid_idx += 1

        if first_valid_idx == arr.size: # All NaNs
            self._cache["ema"][cache_key] = ema_values
            return ema_values

        ema_values[first_valid_idx] = arr[first_valid_idx] # Seed EMA
        for i in range(first_valid_idx + 1, arr.size):
            if np.isnan(arr[i]): # Propagate NaN if data is NaN (though deque should be filtered)
                ema_values[i] = ema_values[i-1] # Or np.nan - depends on desired behavior
            else:
                ema_values[i] = alpha * arr[i] + (1 - alpha) * ema_values[i - 1]
        
        self._cache["ema"][cache_key] = ema_values
        return ema_values

    def _wilder_smooth(self, arr: np.ndarray, window: int) -> np.ndarray:
        alpha = 1.0 / window
        smoothed = np.empty_like(arr)
        if arr.size == 0: return arr

        first_valid_idx = 0
        while first_valid_idx < arr.size and np.isnan(arr[first_valid_idx]):
            smoothed[first_valid_idx] = np.nan
            first_valid_idx += 1
        
        if first_valid_idx == arr.size: return smoothed # All NaNs

        if arr.size - first_valid_idx >= window :
            # Initial sum for Wilder's (some use mean, some use sum / N, which is mean)
            smoothed[first_valid_idx + window - 1] = np.nanmean(arr[first_valid_idx : first_valid_idx + window])
            for i in range(first_valid_idx + window, arr.size):
                if np.isnan(arr[i]): smoothed[i] = smoothed[i-1] # Propagate if NaN encountered
                else: smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
            # Fill initial part that doesn't have full window for smoothing
            for i in range(first_valid_idx, first_valid_idx + window - 1):
                smoothed[i] = np.nan 
        elif (arr.size - first_valid_idx) > 0 : # Less data than window, use simple EMA like smoothing from first valid
            smoothed[first_valid_idx] = arr[first_valid_idx]
            for i in range(first_valid_idx + 1, arr.size):
                if np.isnan(arr[i]): smoothed[i] = smoothed[i-1]
                else: smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    def _rsi(self, prices_deque: deque) -> np.ndarray:
        length = len(prices_deque)
        if length < self.rsi_window + 1: return np.full(length, np.nan)
        
        last_price = prices_deque[-1] if length > 0 else np.nan
        cache_key = ("rsi", last_price, self.rsi_window, length)
        if cache_key in self._cache["rsi"]: return self._cache["rsi"][cache_key]

        arr = self._to_np(prices_deque)
        delta = np.diff(arr)
        if delta.size == 0: return np.full(length, np.nan)

        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = self._wilder_smooth(gain, self.rsi_window)
        avg_loss = self._wilder_smooth(loss, self.rsi_window)

        # Ensure avg_gain and avg_loss are not all NaNs
        if np.all(np.isnan(avg_loss)) or np.all(np.isnan(avg_gain)):
            return np.full(length, np.nan)
            
        rs = np.full_like(avg_gain, np.nan) # Initialize with NaN
        non_zero_loss_idx = (avg_loss != 0) & (~np.isnan(avg_loss)) & (~np.isnan(avg_gain))
        zero_loss_positive_gain_idx = (avg_loss == 0) & (avg_gain > 0) & (~np.isnan(avg_loss)) & (~np.isnan(avg_gain))

        rs[non_zero_loss_idx] = avg_gain[non_zero_loss_idx] / avg_loss[non_zero_loss_idx]
        rs[zero_loss_positive_gain_idx] = np.inf # RSI should be 100

        rsi_vals = 100.0 - (100.0 / (1.0 + rs))
        rsi_vals[np.isinf(rs)] = 100.0 # Handle np.inf for rs -> RSI 100
        
        result = np.full(length, np.nan)
        # rsi_vals is calculated based on delta, so it's 1 shorter than price, and wilder smooth introduces further latency
        # The first valid RSI appears after rsi_window periods of gains/losses are smoothed.
        # Example: if rsi_window = 14, first RSI value is at index 14 of price array (after 14 price changes).
        # The length of rsi_vals from wilder_smooth will be len(delta) - (rsi_window -1)
        # So, result[1 + rsi_window -1 + (rsi_window-1) : ] approximately
        # Simpler: align to the end.
        if len(rsi_vals) > 0:
            nan_padding_count = length - len(rsi_vals)
            if nan_padding_count >= 0:
                 result[nan_padding_count:] = rsi_vals[-len(result[nan_padding_count:]):] # Align to end carefully
            else: # rsi_vals is somehow longer than prices, indicates error
                if self.logger: self.logger.error("RSI calculation error: rsi_vals longer than price array.")


        self._cache["rsi"][cache_key] = result
        return result

    def _macd(self, prices_deque: deque) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(prices_deque)
        if length < self.macd_slow:
            nan_arr = np.full(length, np.nan)
            return nan_arr, nan_arr, nan_arr

        last_price = prices_deque[-1] if length > 0 else np.nan
        cache_key_macd = ("macd_line", last_price, self.macd_fast, self.macd_slow, length)
        cache_key_signal = ("macd_signal", last_price, self.macd_signal, length) # Assume signal depends on macd_line's state

        if cache_key_macd in self._cache["macd"] and cache_key_signal in self._cache["macd"]:
            cached_data = self._cache["macd"][cache_key_macd]
            cached_signal = self._cache["macd"][cache_key_signal]
            # Ensure histogram is derived from potentially cached independent macd_line and signal_line
            histogram = cached_data - cached_signal 
            return cached_data, cached_signal, histogram

        ema_fast_arr = self._ema(prices_deque, self.macd_fast, source_for_debug="macd_ema_fast")
        ema_slow_arr = self._ema(prices_deque, self.macd_slow, source_for_debug="macd_ema_slow")

        if np.all(np.isnan(ema_fast_arr)) or np.all(np.isnan(ema_slow_arr)):
            nan_arr = np.full(length, np.nan)
            return nan_arr, nan_arr, nan_arr

        macd_line = ema_fast_arr - ema_slow_arr
        self._cache["macd"][cache_key_macd] = macd_line # Cache macd_line

        # Now calculate signal line based on the macd_line
        # We need a deque of the macd_line itself for the _ema function
        # Important: The `_ema` function expects a deque of values it should smooth.
        # It does not inherently know it's for a MACD signal line unless we feed it MACD line values.
        
        # Create a deque from macd_line, preserving NaNs at the start if _ema handles them
        # or filter them if _ema expects only valid numbers. The current _ema seeds with first valid.
        macd_line_deque_for_signal = deque(macd_line.tolist(), maxlen=self.buffer_size)

        if self.logger:
            valid_macd_values_for_signal = [v for v in macd_line_deque_for_signal if not np.isnan(v)]
            if valid_macd_values_for_signal:
                self.logger.debug(f"MACD_DEBUG: Input to signal EMA (sample): {valid_macd_values_for_signal[-5:]}, Length: {len(valid_macd_values_for_signal)}")
            else:
                self.logger.debug("MACD_DEBUG: Input to signal EMA is empty or all NaNs.")

        # Pass 'macd_signal_line_calc' to differentiate this EMA calculation in caching/debugging
        signal_line = self._ema(macd_line_deque_for_signal, self.macd_signal, source_for_debug="macd_signal_line_calc")
        
        if self.logger and len(signal_line) > 0 and not np.all(np.isnan(signal_line)):
             self.logger.debug(f"MACD_DEBUG: Raw Signal Line (sample from _ema): {signal_line[~np.isnan(signal_line)][-5:] if np.any(~np.isnan(signal_line)) else 'All NaNs'}")


        self._cache["macd"][cache_key_signal] = signal_line # Cache signal_line

        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _atr(self, highs_deque: deque, lows_deque: deque, closes_deque: deque) -> np.ndarray:
        length = len(closes_deque)
        if length < self.atr_window + 1 : return np.full(length, np.nan)
        
        last_close = closes_deque[-1] if length > 0 else np.nan
        cache_key = ("atr", last_close, self.atr_window, length)
        if cache_key in self._cache["atr"]: return self._cache["atr"][cache_key]

        h = self._to_np(highs_deque); l = self._to_np(lows_deque); c = self._to_np(closes_deque)
        
        high_minus_low = h - l
        high_minus_prev_close = np.abs(h - np.roll(c, 1))
        low_minus_prev_close = np.abs(l - np.roll(c, 1))
        
        high_minus_prev_close[0] = np.nan
        low_minus_prev_close[0] = np.nan

        tr_components = np.vstack([high_minus_low, high_minus_prev_close, low_minus_prev_close])
        true_range = np.nanmax(tr_components, axis=0)
        if len(true_range)>0: true_range[0] = high_minus_low[0] if len(high_minus_low)>0 else np.nan
        else: return np.full(length, np.nan)


        # Wilder smooth expects an array of valid TRs
        # NaNs in true_range can occur if h,l,c have NaNs at the start
        # Find first valid TR to pass to smoother
        first_valid_tr_idx = 0
        while first_valid_tr_idx < len(true_range) and np.isnan(true_range[first_valid_tr_idx]):
            first_valid_tr_idx +=1
        
        if first_valid_tr_idx == len(true_range) or (len(true_range) - first_valid_tr_idx) < self.atr_window :
            return np.full(length, np.nan) # Not enough valid TRs

        valid_tr_segment = true_range[first_valid_tr_idx:]
        atr_raw = self._wilder_smooth(valid_tr_segment, self.atr_window)
        
        atr_result = np.full(length, np.nan)
        # Align atr_raw back into atr_result.
        # atr_raw starts effectively at index (first_valid_tr_idx + atr_window - 1) of the original price series
        start_idx_in_result = first_valid_tr_idx + self.atr_window -1
        
        len_atr_raw = len(atr_raw) - (self.atr_window -1) # Effective length of smoothed values after window
        valid_atr_raw_values = atr_raw[self.atr_window-1:]


        if len(valid_atr_raw_values) > 0 and start_idx_in_result < length:
            num_elements_to_copy = min(len(valid_atr_raw_values), length - start_idx_in_result)
            atr_result[start_idx_in_result : start_idx_in_result + num_elements_to_copy] = valid_atr_raw_values[:num_elements_to_copy]

        self._cache["atr"][cache_key] = atr_result
        return atr_result

    def _calculate_volatility(self, prices_deque: deque) -> float:
        arr = self._to_np(prices_deque)
        if arr.size < self.volatility_window + 1: return 0.0 
        
        # Ensure no NaNs in the segment used for log returns
        segment_for_log = arr[-(self.volatility_window + 1):]
        if np.isnan(segment_for_log).any(): return 0.0

        log_returns = np.log(segment_for_log[1:] / segment_for_log[:-1])
        if np.isnan(log_returns).any() or log_returns.size < self.volatility_window : return 0.0 # Should not happen if segment_for_log is clean
        
        vol = float(np.std(log_returns[-self.volatility_window:], ddof=0))
        return vol

    def _get_market_context(self, closes_deque: deque, highs_deque: deque, lows_deque: deque, volumes_deque: deque) -> Dict:
        context = {"valid": False}
        min_len_required = max(self.trend_ema_slow, self.macd_slow, self.atr_window + 1, self.rsi_window + 1, self.volatility_window + 1)
        if len(closes_deque) < min_len_required:
             context["reason"] = f"Insufficient data: Need {min_len_required}, have {len(closes_deque)}"
             return context

        # Calculate indicators
        ema_fast = self._ema(closes_deque, self.trend_ema_fast, "trend_ema_fast")
        ema_slow = self._ema(closes_deque, self.trend_ema_slow, "trend_ema_slow")
        rsi_val = self._rsi(closes_deque) # Corrected variable name
        macd_line, macd_signal, macd_hist = self._macd(prices_deque=closes_deque)
        atr_val = self._atr(highs_deque, lows_deque, closes_deque) # Corrected variable name
        volume_sma = self._sma(volumes_deque, self.volume_sma_window) # Corrected variable name
        current_volatility = self._calculate_volatility(prices_deque=closes_deque) # Corrected

        # Check for NaNs in the latest values needed for decisions
        # Ensure arrays are long enough before accessing [-1] and [-2]
        required_len_for_crossover = 2 # For trend[-2] etc.
        
        latest_values_dict = {}
        def get_latest(arr, name, precision=4):
            if len(arr) > 0 and not np.isnan(arr[-1]):
                latest_values_dict[name] = arr[-1]
                return arr[-1]
            latest_values_dict[name] = np.nan
            return np.nan

        def get_second_latest(arr, name):
            if len(arr) > 1 and not np.isnan(arr[-2]):
                latest_values_dict[name+"_prev"] = arr[-2]
                return arr[-2]
            latest_values_dict[name+"_prev"] = np.nan
            return np.nan

        price_latest = get_latest(closes_deque, "price")
        ema_fast_latest = get_latest(ema_fast, "ema_fast")
        ema_slow_latest = get_latest(ema_slow, "ema_slow")
        rsi_latest = get_latest(rsi_val, "rsi", 2)
        macd_line_latest = get_latest(macd_line, "macd_line")
        macd_signal_latest = get_latest(macd_signal, "macd_signal")
        macd_hist_latest = get_latest(macd_hist, "macd_hist")
        atr_latest = get_latest(atr_val, "atr")
        volume_sma_latest = get_latest(volume_sma, "volume_sma")
        
        ema_fast_prev = get_second_latest(ema_fast, "ema_fast")
        ema_slow_prev = get_second_latest(ema_slow, "ema_slow")
        macd_line_prev = get_second_latest(macd_line, "macd_line")
        macd_signal_prev = get_second_latest(macd_signal, "macd_signal")


        if self.logger:
             log_msg = "Latest Indicators - "
             for k, v in latest_values_dict.items():
                 log_msg += f"{k.upper()}: {v:.{2 if k=='rsi' else 4}f} " if not np.isnan(v) else f"{k.upper()}: NaN "
             log_msg += f"VOL: {current_volatility:.5f}"
             self.logger.debug(log_msg)
        
        # Check if any CRITICAL latest values are NaN
        critical_nans = np.isnan([
            price_latest, ema_fast_latest, ema_slow_latest, rsi_latest, 
            macd_line_latest, macd_signal_latest, atr_latest
        ])
        if critical_nans.any():
            context["reason"] = f"Critical latest indicator values contain NaN. NaN in: {[k for i,k in enumerate(['price', 'ema_fast', 'ema_slow', 'rsi', 'macd_line', 'macd_signal', 'atr']) if critical_nans[i]]}"
            return context


        # Determine states
        trend = "neutral"
        if not np.isnan(ema_fast_latest) and not np.isnan(ema_slow_latest):
            if not np.isnan(ema_fast_prev) and not np.isnan(ema_slow_prev): # Check prev values for crossover
                if ema_fast_latest > ema_slow_latest and ema_fast_prev <= ema_slow_prev: trend = "bullish_crossover"
                elif ema_fast_latest < ema_slow_latest and ema_fast_prev >= ema_slow_prev: trend = "bearish_crossover"
                elif ema_fast_latest > ema_slow_latest: trend = "bullish"
                elif ema_fast_latest < ema_slow_latest: trend = "bearish"
            else: # Not enough data for crossover, just check current state
                if ema_fast_latest > ema_slow_latest: trend = "bullish"
                elif ema_fast_latest < ema_slow_latest: trend = "bearish"


        rsi_state = "neutral"
        if not np.isnan(rsi_latest):
            if rsi_latest > self.rsi_overbought: rsi_state = "overbought"
            elif rsi_latest < self.rsi_oversold: rsi_state = "oversold"

        macd_crossover_val = "none" # Renamed to avoid conflict with context key
        macd_trend_val = "neutral"  # Renamed
        if not np.isnan(macd_line_latest) and not np.isnan(macd_signal_latest):
            if not np.isnan(macd_line_prev) and not np.isnan(macd_signal_prev): # Check prev values for crossover
                if macd_line_latest > macd_signal_latest and macd_line_prev <= macd_signal_prev: macd_crossover_val = "bullish"
                elif macd_line_latest < macd_signal_latest and macd_line_prev >= macd_signal_prev: macd_crossover_val = "bearish"
            # Determine trend regardless of crossover history for current bar
            if macd_line_latest > macd_signal_latest: macd_trend_val = "bullish"
            elif macd_line_latest < macd_signal_latest: macd_trend_val = "bearish"


        current_volume = volumes_deque[-1] if volumes_deque and len(volumes_deque)>0 else 0
        avg_volume_val = volume_sma_latest if not np.isnan(volume_sma_latest) else 0 # Renamed
        volume_spike = current_volume > avg_volume_val * self.volume_spike_multiplier if avg_volume_val > 0 else False
        volume_trend_val = "rising" if current_volume > avg_volume_val else "falling" if avg_volume_val > 0 else "flat" # Renamed

        context.update({
            "valid": True, "price": price_latest, 
            "high": highs_deque[-1] if len(highs_deque)>0 else np.nan, 
            "low": lows_deque[-1] if len(lows_deque)>0 else np.nan,
            "volume": current_volume, "ema_fast": ema_fast_latest, "ema_slow": ema_slow_latest,
            "rsi": rsi_latest, "macd_line": macd_line_latest, "macd_signal": macd_signal_latest, 
            "macd_hist": macd_hist_latest, "atr": atr_latest, "volatility": current_volatility, 
            "avg_volume": avg_volume_val, "trend": trend, "rsi_state": rsi_state, 
            "macd_crossover": macd_crossover_val, "macd_trend": macd_trend_val,
            "volume_spike": volume_spike, "volume_trend": volume_trend_val,
        })
        return context

    def _calculate_risk_parameters(self, context: Dict) -> Dict:
        risk_params = {"sl": 0.0, "tp": 0.0, "size": 0.0, "valid": False}
        price = context["price"]; atr_val = context["atr"]
        
        if atr_val <= 0 or np.isnan(atr_val) or price <= 0 or np.isnan(price):
            risk_params["reason"] = f"Invalid price or ATR (Price: {price}, ATR: {atr_val})"
            return risk_params
        
        vol_factor = 1 + min(context["volatility"] * 0.25, 0.25)
        sl_dist_atr = atr_val * self.atr_sl_multiplier * vol_factor
        sl_dist_price_cap = price * self.max_risk_percent_per_trade 
        sl_dist = min(sl_dist_atr, sl_dist_price_cap)
        tp_dist = atr_val * self.atr_tp_multiplier * vol_factor
        
        # Ensure SL distance is not excessively small
        min_sl_distance_points = 0.01 # R_10 min price fluctuation might be 0.001 or 0.01. ADJUST.
        if sl_dist < min_sl_distance_points :
            if self.logger: self.logger.warning(f"Calculated SL distance {sl_dist:.5f} is very small. May cause issues. Min SL set to {min_sl_distance_points}")
            sl_dist = min_sl_distance_points # Enforce a minimum practical SL distance
            # Re-evaluate TP based on this new SL to maintain some R:R logic, or cap TP also
            tp_dist = max(tp_dist, sl_dist * self.atr_tp_multiplier / self.atr_sl_multiplier)


        sl_buy = price - sl_dist; tp_buy = price + tp_dist
        sl_sell = price + sl_dist; tp_sell = price - tp_dist
        
        risk_amt = self.account_balance * self.risk_per_trade
        size_buy = 0.0
        if sl_dist > 1e-9: # Effectively sl_dist > 0
            size_buy = max(self.min_position_size, round(risk_amt / sl_dist, 2))
        
        size_sell = 0.0
        if sl_dist > 1e-9:
            size_sell = max(self.min_position_size, round(risk_amt / sl_dist, 2))

        risk_params.update({ "sl_buy": sl_buy, "tp_buy": tp_buy, "size_buy": size_buy, 
                             "sl_sell": sl_sell, "tp_sell": tp_sell, "size_sell": size_sell, 
                             "sl_distance": sl_dist, "tp_distance": tp_dist, "valid": True })
        return risk_params

    def _get_performance_factor(self) -> float:
        num_trades = len(self.trade_history)
        if num_trades < 5: return 1.0
        recent_trades_count = min(num_trades, 20)
        recent = list(self.trade_history)[-recent_trades_count:]
        wins = sum(1 for pnl in recent if pnl > 0)
        win_rate = wins / len(recent) if len(recent) > 0 else 0.5
        perf_factor = 0.7 + (win_rate * 0.4) 
        return float(np.clip(perf_factor, 0.7, 1.1))

    def _calculate_confidence(self, context: Dict, signal_direction: str, perf_factor: float) -> float:
        weights = { "trend_match": 0.25, "trend_crossover": 0.20, 
                    "macd_match": 0.20, "macd_crossover": 0.20, 
                    "rsi_confirm": 0.10, "volume_confirm": 0.05 }
        raw_score = 0.0
        # Null checks for context values used below
        trend = context.get("trend", "neutral")
        macd_trend = context.get("macd_trend", "neutral")
        macd_crossover = context.get("macd_crossover", "none")
        rsi_state = context.get("rsi_state", "neutral")
        volume_spike = context.get("volume_spike", False)
        volume_trend = context.get("volume_trend", "flat")

        if signal_direction == "BUY":
            if trend in ["bullish", "bullish_crossover"]: raw_score += weights["trend_match"]
            if trend == "bullish_crossover": raw_score += weights["trend_crossover"]
            if macd_trend == "bullish": raw_score += weights["macd_match"]
            if macd_crossover == "bullish": raw_score += weights["macd_crossover"]
            if rsi_state != "overbought": raw_score += weights["rsi_confirm"]
            if volume_spike and volume_trend == "rising": raw_score += weights["volume_confirm"]
        elif signal_direction == "SELL":
            if trend in ["bearish", "bearish_crossover"]: raw_score += weights["trend_match"]
            if trend == "bearish_crossover": raw_score += weights["trend_crossover"]
            if macd_trend == "bearish": raw_score += weights["macd_match"]
            if macd_crossover == "bearish": raw_score += weights["macd_crossover"]
            if rsi_state != "oversold": raw_score += weights["rsi_confirm"]
            if volume_spike and (volume_trend == "rising" or trend == "bearish_crossover"): raw_score += weights["volume_confirm"]

        if self.logger:
             self.logger.debug(f"Confidence Calc - Raw Score: {raw_score:.4f}, Perf Factor: {perf_factor:.4f}")

        adjusted_score = raw_score * perf_factor
        confidence = float(np.clip(adjusted_score, 0.0, self.max_confidence))
        if raw_score < 0.35: confidence = 0.0 
        return confidence

    def _generate_signal(self, context: Dict) -> Dict:
        signal = {"direction": "HOLD", "strength": 0.0}
        
        # Null checks for context values
        trend = context.get("trend", "neutral")
        macd_crossover = context.get("macd_crossover", "none")
        macd_trend = context.get("macd_trend", "neutral")
        macd_hist = context.get("macd_hist", np.nan)
        rsi_state = context.get("rsi_state", "neutral")
        # rsi_val = context.get("rsi", np.nan) # Actual rsi value, not strictly needed if using rsi_state for this simplified logic
        atr_val = context.get("atr", np.nan)
        price_val = context.get("price", np.nan)
        volatility_val = context.get("volatility", np.nan)

        # --- ATR/Volatility Filter (Applied first) ---
        if np.isnan(atr_val) or np.isnan(price_val) or np.isnan(volatility_val):
            low_vol_filter_active = True 
            if self.logger: self.logger.warning("ATR/Vol filter active due to NaN in ATR, Price, or Volatility for filter check.")
        else:
            min_atr_val_calc = price_val * self.min_atr_threshold_factor 
            low_vol_filter_active = atr_val < min_atr_val_calc or volatility_val < self.min_volatility_threshold
            if self.logger:
                 self.logger.debug(
                     f"ATR/Vol Filter Check - ATR: {atr_val:.5f} (Req > {min_atr_val_calc:.5f}), "
                     f"Vol: {volatility_val:.6f} (Req > {self.min_volatility_threshold:.6f}), "
                     f"Filter Active: {low_vol_filter_active}"
                 )
        
        if low_vol_filter_active:
             if self.logger:
                 self.logger.info(f"Signal generation skipped due to ATR/Vol filter.")
             return signal # Return HOLD if filter is active

        # --- Signal Logic (if filter passed) ---
        is_buy_trend = trend in ["bullish", "bullish_crossover"]
        is_buy_macd = macd_crossover == "bullish" or \
                      (macd_trend == "bullish" and trend != "bearish" and (not np.isnan(macd_hist) and macd_hist > 0))
        # SCALPING OPTIMIZATION: Simplified RSI condition. Only avoid buying if RSI is currently in overbought state.
        is_rsi_ok_buy = rsi_state != "overbought" 

        if self.logger:
            self.logger.debug(f"Signal Components (BUY): is_buy_trend: {is_buy_trend}, is_buy_macd: {is_buy_macd}, is_rsi_ok_buy: {is_rsi_ok_buy} (RSI State: {rsi_state})")

        if is_buy_trend and is_buy_macd and is_rsi_ok_buy:
            signal["direction"] = "BUY"
            strength_trend = 1 if trend == "bullish_crossover" else 0.6
            strength_macd = 1 if macd_crossover == "bullish" else (0.7 if macd_trend == "bullish" else 0.3)
            # RSI strength: stronger if coming from oversold, neutral otherwise if not overbought
            strength_rsi = 1 if rsi_state == "oversold" else (0.5 if rsi_state == "neutral" else 0.3) 
            signal["strength"] = np.mean([strength_trend, strength_macd, strength_rsi])

        # Check SELL conditions only if not already a BUY
        if signal["direction"] == "HOLD":
            is_sell_trend = trend in ["bearish", "bearish_crossover"]
            is_sell_macd = macd_crossover == "bearish" or \
                           (macd_trend == "bearish" and trend != "bullish" and (not np.isnan(macd_hist) and macd_hist < 0))
            # SCALPING OPTIMIZATION: Simplified RSI condition. Only avoid selling if RSI is currently in oversold state.
            is_rsi_ok_sell = rsi_state != "oversold"

            if self.logger:
                self.logger.debug(f"Signal Components (SELL): is_sell_trend: {is_sell_trend}, is_sell_macd: {is_sell_macd}, is_rsi_ok_sell: {is_rsi_ok_sell} (RSI State: {rsi_state})")

            if is_sell_trend and is_sell_macd and is_rsi_ok_sell:
                signal["direction"] = "SELL"
                strength_trend = 1 if trend == "bearish_crossover" else 0.6
                strength_macd = 1 if macd_crossover == "bearish" else (0.7 if macd_trend == "bearish" else 0.3)
                # RSI strength: stronger if coming from overbought, neutral otherwise if not oversold
                strength_rsi = 1 if rsi_state == "overbought" else (0.5 if rsi_state == "neutral" else 0.3)
                signal["strength"] = np.mean([strength_trend, strength_macd, strength_rsi])
        
        if self.logger:
             self.logger.debug(f"Final Signal Generated by _generate_signal: Direction: {signal['direction']}, Strength: {signal['strength']:.2f}")
        return signal

    def _suggested_holding_time(self, context: Dict, signal_strength: float) -> int:
        base_minutes = 1 
        max_minutes = 3  
        
        volatility = context.get("volatility", 0.0001) # Default to a low-ish vol if not present
        if np.isnan(volatility): volatility = 0.0001

        vol_scaling = 1.0 - np.clip(volatility * 5, -0.2, 0.2)
        strength_scaling = 1.0 + np.clip(signal_strength * 0.5 - 0.25, 0.0, 0.5)
        hold_time_float = base_minutes * strength_scaling * vol_scaling
        return int(max(base_minutes, min(max_minutes, round(hold_time_float))))

    def generate_candle_signal(self, candle_data: List[Dict[str, float]]) -> Dict[str, object]:
        self.price_data.clear(); self.high_data.clear(); self.low_data.clear(); self.volume_data.clear()
        valid_candles_count = 0
        for candle in candle_data:
             if not all(k in candle for k in ["close", "high", "low"]):
                 if self.logger: self.logger.warning(f"Skipping candle due to missing keys: {candle}")
                 continue
             try:
                self.price_data.append(float(candle["close"]))
                self.high_data.append(float(candle["high"]))
                self.low_data.append(float(candle["low"]))
                self.volume_data.append(float(candle.get("volume", 1.0)))
                valid_candles_count += 1
             except ValueError as e:
                if self.logger: self.logger.error(f"Error converting candle data to float: {e}, candle: {candle}")
                continue

        if self.logger:
             self.logger.debug(f"Input candle count: {len(candle_data)}, Populated deque length: {valid_candles_count}")

        min_len_required = max(self.trend_ema_slow, self.macd_slow, self.atr_window + 1, self.rsi_window + 1, self.volatility_window + 1)
        if valid_candles_count < min_len_required:
             return self._hold_payload(f"Insufficient valid data: Need {min_len_required}, have {valid_candles_count}")

        # Clear cache at the beginning of new data processing if it's candle based
        # Or make cache keys more specific if data is streaming tick-by-tick forming candles
        # For now, assuming candle_data represents a full new set for decision.
        self._cache = { "ema": {}, "rsi": {}, "atr": {}, "macd": {}, "sma": {} }


        context = self._get_market_context(self.price_data, self.high_data, self.low_data, self.volume_data)
        if not context["valid"]:
             return self._hold_payload(context.get("reason", "Invalid market context from _get_market_context"))

        signal_details = self._generate_signal(context) # Renamed variable
        if signal_details["direction"] == "HOLD":
             return self._hold_payload(f"Signal logic determined HOLD. Strength: {signal_details['strength']:.2f}")

        risk_params = self._calculate_risk_parameters(context)
        if not risk_params["valid"]:
             return self._hold_payload(risk_params.get("reason", "Risk parameter calculation failed"))

        perf_factor = self._get_performance_factor()
        confidence = self._calculate_confidence(context, signal_details["direction"], perf_factor)

        if self.logger:
             self.logger.debug(f"Pre-Confidence Check - Signal: {signal_details['direction']}, Confidence: {confidence:.4f} (Min Req: {self.min_confidence})")

        if confidence < self.min_confidence:
             return self._hold_payload(f"Confidence {confidence:.2f} < threshold {self.min_confidence}")

        final_signal_payload = { "signal": signal_details["direction"], "confidence": confidence } # Renamed
        # R_10 price precision typically needs more decimal places than default 2 for currency.
        # e.g., price 6307.7910 implies 4 decimal places after the point for price,
        # SL/TP might need similar or one more. Let's use 4 for R_10, but this should be verified.
        price_precision = 4 # ADJUST FOR R_10 ACTUAL PRECISION

        if signal_details["direction"] == "BUY":
            final_signal_payload["stop_loss"] = round(risk_params["sl_buy"], price_precision)
            final_signal_payload["take_profit"] = round(risk_params["tp_buy"], price_precision)
            final_signal_payload["position_size"] = risk_params["size_buy"]
        elif signal_details["direction"] == "SELL":
            final_signal_payload["stop_loss"] = round(risk_params["sl_sell"], price_precision)
            final_signal_payload["take_profit"] = round(risk_params["tp_sell"], price_precision)
            final_signal_payload["position_size"] = risk_params["size_sell"]
        
        final_signal_payload["max_holding_time"] = self._suggested_holding_time(context, signal_details["strength"])
        final_signal_payload["trailing_stop"] = round(risk_params["sl_distance"], price_precision) 

        pos_size = final_signal_payload.get("position_size", 0.0)
        if pos_size <= 1e-9: # Effectively pos_size <= 0
            return self._hold_payload(f"Calculated position size is zero or less ({pos_size}). SL Dist: {risk_params.get('sl_distance', 'N/A')}")

        if self.logger:
            self.logger.info(
                f"TRADE SIGNAL: {final_signal_payload['signal']} | Conf: {final_signal_payload['confidence']:.2f} | "
                f"SL: {final_signal_payload['stop_loss']:.{price_precision}f} | "
                f"TP: {final_signal_payload['take_profit']:.{price_precision}f} | "
                f"Size: {final_signal_payload['position_size']} | Hold: {final_signal_payload['max_holding_time']}m | "
                f"Trail: {final_signal_payload['trailing_stop']:.{price_precision}f}"
            )
        return final_signal_payload

    def _hold_payload(self, reason: str) -> Dict[str, object]:
        if self.logger:
            self.logger.debug(f"HOLD Signal – Reason: {reason}")
        return { "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
                 "position_size": 0.0, "max_holding_time": 0, "trailing_stop": 0.0, }

    def record_trade_result(self, pnl: float):
         self.trade_history.append(pnl)
         if self.logger:
             self.logger.info(f"Trade result: PNL = {pnl:.2f}. History size: {len(self.trade_history)}")

    def update_account_balance(self, balance: float):
        self.account_balance = balance
        if self.logger:
            self.logger.debug(f"Account balance updated: {self.account_balance}")