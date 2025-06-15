import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

class RuleBasedSignals:
    """
    Rule-Based Signal Engine - OPTIMIZED FOR R_10 SCALPING
    ----
    • Tuned parameters for high sensitivity R_10 scalping.
    • Corrected MACD calculations and robust NaN/index handling.
    • Refined exits: Tighter TP (1:1 R:R), slightly longer max hold time.
    • Handles R_10 tick volume (assumes it might be unreliable for strong confirmation).
    • ATR/Volatility filters tuned for R_10.
    """

    # ------------------------------------------------------------------ #
    #  INITIALISATION
    # ------------------------------------------------------------------ #
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

        # --- Parameters for R_10 Scalping ---
        self.atr_sl_multiplier: float = 1.0
        self.atr_tp_multiplier: float = 1.0 # For 1:1 R:R, aiming for quicker TP hits

        self.volatility_window: int   = 10
        self.atr_window: int          = 10
        self.rsi_window: int          = 9
        self.rsi_oversold: float      = 20   # More extreme levels for R_10 scalping
        self.rsi_overbought: float    = 80   # More extreme levels for R_10 scalping
        
        self.macd_fast: int           = 5
        self.macd_slow: int           = 13
        self.macd_signal: int         = 5 
        self.trend_ema_fast: int      = 7
        self.trend_ema_slow: int      = 14
        
        self.volume_sma_window: int   = 15 
        self.volume_spike_multiplier: float = 2.0 # Less critical if volume is just tick count

        # Set this based on your testing (e.g., 0.55 or 0.60 allowed trades)
        self.min_confidence: float    = 0.55
        self.max_confidence: float    = 0.95
        
        self.risk_per_trade: float    = 0.01
        self.max_risk_percent_per_trade: float = 0.02 # Max % of price for SL cap
        self.min_position_size: float = 0.01 # Verify Deriv's min for R_10 contracts/lots

        # --- ATR/Volatility Filter Parameters for R_10 ---
        self.min_atr_threshold_factor: float = 0.00015 
        self.min_volatility_threshold: float = 0.00005 

        # --- Volume Data Usage Control for R_10 ---
        # Set to False if R_10 volume is just tick count '1.0' or unreliable
        self.use_r10_volume_confirmation: bool = False 

        self.buffer_size: int = 750 
        self.price_data: deque = deque(maxlen=self.buffer_size)
        self.high_data: deque  = deque(maxlen=self.buffer_size)
        self.low_data: deque   = deque(maxlen=self.buffer_size)
        self.volume_data: deque= deque(maxlen=self.buffer_size) 
        self.trade_history: deque = deque(maxlen=100)

        self._cache: Dict[str, Dict[Tuple, np.ndarray]] = {
            "ema": {}, "rsi": {}, "atr": {}, "macd": {}, "sma": {},
        }
        self.account_balance: float = 10000.0 # Default, updated by TradingStrategy

    @staticmethod
    def _to_np(data: deque) -> np.ndarray:
        return np.array(data, dtype=np.float64)

    def _sma(self, data_deque: deque, window: int) -> np.ndarray:
        length = len(data_deque)
        if length < window: return np.full(length, np.nan)
        
        # Simplified cache key for when data_deque is the primary varying input
        # Convert deque to tuple for hashing; only use part if very long.
        tuple_data = tuple(data_deque)
        cache_key_tuple = tuple_data if len(tuple_data) < 50 else (tuple_data[0], tuple_data[-1]) # Avoid hashing huge tuples
        cache_key = ("sma", cache_key_tuple, window, length)
        
        if cache_key in self._cache["sma"]: return self._cache["sma"][cache_key]
        
        arr = self._to_np(data_deque)
        weights = np.repeat(1.0, window) / window
        sma_values = np.convolve(arr, weights, 'valid')
        
        result = np.full(length, np.nan)
        if len(sma_values) > 0:
            result[length - len(sma_values):] = sma_values # Align to end
        self._cache["sma"][cache_key] = result
        return result

    def _ema(self, data_deque: deque, window: int, source_for_debug: str = "price") -> np.ndarray:
        length = len(data_deque)
        if length == 0: return np.array([])
        
        tuple_data = tuple(data_deque)
        cache_key_tuple = tuple_data if len(tuple_data) < 50 else (tuple_data[0], tuple_data[-1])
        cache_key = ("ema", source_for_debug, cache_key_tuple, window, length)

        if cache_key in self._cache["ema"]:
            return self._cache["ema"][cache_key]

        arr = self._to_np(data_deque)
        if arr.size < 1 : return np.full(length, np.nan)

        alpha = 2.0 / (window + 1)
        ema_values = np.full_like(arr, np.nan) 
        
        first_valid_idx = 0
        while first_valid_idx < arr.size and np.isnan(arr[first_valid_idx]):
            first_valid_idx += 1

        if first_valid_idx == arr.size: # All NaNs
            self._cache["ema"][cache_key] = ema_values
            return ema_values

        ema_values[first_valid_idx] = arr[first_valid_idx] # Seed EMA
        for i in range(first_valid_idx + 1, arr.size):
            if np.isnan(arr[i]): 
                ema_values[i] = ema_values[i-1] # Propagate last valid EMA
            else:
                ema_values[i] = alpha * arr[i] + (1 - alpha) * ema_values[i - 1]
        
        self._cache["ema"][cache_key] = ema_values
        return ema_values

    def _wilder_smooth(self, arr: np.ndarray, window: int) -> np.ndarray:
        alpha = 1.0 / window
        smoothed = np.full_like(arr, np.nan)
        if arr.size == 0: return arr

        first_valid_idx = 0
        while first_valid_idx < arr.size and np.isnan(arr[first_valid_idx]):
            first_valid_idx += 1
        
        if first_valid_idx == arr.size: return smoothed

        if arr.size - first_valid_idx >= window :
            initial_mean = np.nanmean(arr[first_valid_idx : first_valid_idx + window])
            if np.isnan(initial_mean): 
                if self.logger: self.logger.warning(f"WilderSmooth: All NaNs in initial window. Input: {arr[first_valid_idx : first_valid_idx + window]}")
                return smoothed 

            smoothed[first_valid_idx + window - 1] = initial_mean
            for i in range(first_valid_idx + window, arr.size):
                if np.isnan(arr[i]): 
                    smoothed[i] = smoothed[i-1]
                else: 
                    smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
        elif (arr.size - first_valid_idx) > 0 : 
            smoothed[first_valid_idx] = arr[first_valid_idx]
            for i in range(first_valid_idx + 1, arr.size):
                if np.isnan(arr[i]): smoothed[i] = smoothed[i-1]
                else: smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    def _rsi(self, prices_deque: deque) -> np.ndarray:
        length = len(prices_deque)
        if length < self.rsi_window + 1: return np.full(length, np.nan)
        
        tuple_data = tuple(prices_deque)
        cache_key_tuple = tuple_data if len(tuple_data) < 50 else (tuple_data[0], tuple_data[-1])
        cache_key = ("rsi", cache_key_tuple, self.rsi_window, length)
        if cache_key in self._cache["rsi"]: return self._cache["rsi"][cache_key]

        arr = self._to_np(prices_deque)
        delta = np.diff(arr)
        
        if delta.size < self.rsi_window : return np.full(length, np.nan) 

        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = self._wilder_smooth(gain, self.rsi_window)
        avg_loss = self._wilder_smooth(loss, self.rsi_window)
            
        rs_values = np.full_like(avg_gain, np.nan) 
        
        valid_smooth_indices = ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
        zero_loss_indices = valid_smooth_indices & (avg_loss == 0) & (avg_gain > 0)
        rs_values[zero_loss_indices] = np.inf
        
        non_zero_loss_indices = valid_smooth_indices & (avg_loss != 0)
        # Ensure no division by zero or with NaN before calculation
        if np.any(non_zero_loss_indices):
            avg_gain_safe = avg_gain[non_zero_loss_indices]
            avg_loss_safe = avg_loss[non_zero_loss_indices]
            rs_values[non_zero_loss_indices] = avg_gain_safe / avg_loss_safe
        
        rsi_calculated = 100.0 - (100.0 / (1.0 + rs_values))
        
        result = np.full(length, np.nan)
        first_rsi_idx_in_delta_smooth = self.rsi_window -1 
        if len(rsi_calculated) > first_rsi_idx_in_delta_smooth:
            valid_rsi_segment = rsi_calculated[first_rsi_idx_in_delta_smooth:]
            start_idx_in_result = self.rsi_window 
            num_to_copy = min(len(valid_rsi_segment), length - start_idx_in_result)
            if num_to_copy > 0 and start_idx_in_result >=0: # ensure start_idx is valid
                 result[start_idx_in_result : start_idx_in_result + num_to_copy] = valid_rsi_segment[:num_to_copy]

        self._cache["rsi"][cache_key] = result
        return result

    def _macd(self, prices_deque: deque) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(prices_deque)
        if length < self.macd_slow: 
            nan_arr = np.full(length, np.nan)
            return nan_arr, nan_arr, nan_arr

        price_tuple_data = tuple(prices_deque)
        price_cache_key_tuple = price_tuple_data if len(price_tuple_data) < 50 else (price_tuple_data[0], price_tuple_data[-1])

        cache_key_base = (price_cache_key_tuple, self.macd_fast, self.macd_slow, self.macd_signal, length)
        cache_key_macd_line = ("macd_line",) + cache_key_base
        cache_key_signal_line = ("macd_signal_line",) + cache_key_base
        
        if cache_key_macd_line in self._cache["macd"] and cache_key_signal_line in self._cache["macd"]:
            macd_line = self._cache["macd"][cache_key_macd_line]
            signal_line = self._cache["macd"][cache_key_signal_line]
            # Histogram is always derived, not cached directly from these two.
            histogram = macd_line - signal_line 
            return macd_line, signal_line, histogram

        ema_fast_arr = self._ema(prices_deque, self.macd_fast, source_for_debug="r10_macd_ema_fast")
        ema_slow_arr = self._ema(prices_deque, self.macd_slow, source_for_debug="r10_macd_ema_slow")

        macd_line = ema_fast_arr - ema_slow_arr
        self._cache["macd"][cache_key_macd_line] = macd_line

        macd_line_deque_for_signal = deque(macd_line.tolist(), maxlen=self.buffer_size)
        
        if self.logger:
            valid_m_vals = [v for v in macd_line_deque_for_signal if not np.isnan(v)]
            self.logger.debug(f"R10_MACD_DEBUG: Input to signal EMA (sample): {valid_m_vals[-5:] if valid_m_vals else 'None'}, Valid Length: {len(valid_m_vals)}")

        signal_line = self._ema(macd_line_deque_for_signal, self.macd_signal, source_for_debug="r10_macd_signal_line")
        self._cache["macd"][cache_key_signal_line] = signal_line
        
        if self.logger and len(signal_line)>0 and not np.all(np.isnan(signal_line)):
             valid_signal_vals = signal_line[~np.isnan(signal_line)]
             self.logger.debug(f"R10_MACD_DEBUG: Raw Signal Line (sample from _ema): {valid_signal_vals[-5:] if len(valid_signal_vals)>0 else 'All NaNs or Empty'}")
        
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _atr(self, highs_deque: deque, lows_deque: deque, closes_deque: deque) -> np.ndarray:
        length = len(closes_deque)
        if length < self.atr_window + 1 : return np.full(length, np.nan) 
        
        # Simplified cache key for ATR
        cache_key_tuple_h = tuple(highs_deque) if len(highs_deque) < 25 else (highs_deque[0], highs_deque[-1])
        cache_key_tuple_l = tuple(lows_deque) if len(lows_deque) < 25 else (lows_deque[0], lows_deque[-1])
        cache_key_tuple_c = tuple(closes_deque) if len(closes_deque) < 25 else (closes_deque[0], closes_deque[-1])
        cache_key = ("atr", cache_key_tuple_h, cache_key_tuple_l, cache_key_tuple_c, self.atr_window, length)

        if cache_key in self._cache["atr"]: return self._cache["atr"][cache_key]

        h_arr = self._to_np(highs_deque); l_arr = self._to_np(lows_deque); c_arr = self._to_np(closes_deque)
        
        tr = np.full(length, np.nan)
        if length > 0 and not (np.isnan(h_arr[0]) or np.isnan(l_arr[0])):
            tr[0] = h_arr[0] - l_arr[0]
        
        for i in range(1, length):
            # Ensure indices are valid before access
            if i-1 < 0: continue # Should not happen with range(1,length)

            val1 = h_arr[i] - l_arr[i] if not (np.isnan(h_arr[i]) or np.isnan(l_arr[i])) else np.nan
            val2 = np.abs(h_arr[i] - c_arr[i-1]) if not (np.isnan(h_arr[i]) or np.isnan(c_arr[i-1])) else np.nan
            val3 = np.abs(l_arr[i] - c_arr[i-1]) if not (np.isnan(l_arr[i]) or np.isnan(c_arr[i-1])) else np.nan
            
            # Filter out NaNs before nanmax if all are NaN to avoid warning, though nanmax handles it
            components = [comp for comp in [val1, val2, val3] if not np.isnan(comp)]
            tr[i] = np.max(components) if components else np.nan # Use np.max if filtered, or np.nanmax if not
            
        atr_smoothed = self._wilder_smooth(tr, self.atr_window)

        self._cache["atr"][cache_key] = atr_smoothed
        return atr_smoothed

    def _calculate_volatility(self, prices_deque: deque) -> float:
        arr_full = self._to_np(prices_deque)
        if arr_full.size < self.volatility_window + 1: return 0.0
        
        arr = arr_full[-(self.volatility_window + 1):] 
        if np.isnan(arr).any():
            if self.logger: self.logger.warning("R10 Volatility: NaNs in price segment.")
            return 0.0

        log_returns = np.log(arr[1:] / arr[:-1])
        log_returns[np.isinf(log_returns)] = np.nan 
        
        valid_log_returns = log_returns[~np.isnan(log_returns)]
        if valid_log_returns.size < self.volatility_window:
            if self.logger: self.logger.debug(f"R10 Volatility: Not enough valid log returns ({valid_log_returns.size}) for window ({self.volatility_window}). Using 0.0.")
            return 0.0
        
        vol = float(np.std(valid_log_returns[-self.volatility_window:], ddof=1)) 
        return vol

    def _get_market_context(self, closes_deque: deque, highs_deque: deque, lows_deque: deque, volumes_deque: deque) -> Dict:
        context = {"valid": False}
        min_len_required = max(
            self.trend_ema_slow, 
            self.macd_slow, 
            self.atr_window + 1, 
            self.rsi_window + 1, 
            self.volatility_window + 1
        ) + self.macd_signal + 5 # Max lookback for base indicators + MACD signal + buffer

        if len(closes_deque) < min_len_required:
             context["reason"] = f"R10 Insufficient data: Need ~{min_len_required}, have {len(closes_deque)}"
             if self.logger: self.logger.info(context["reason"])
             return context

        ema_fast = self._ema(closes_deque, self.trend_ema_fast, "r10_trend_ema_fast")
        ema_slow = self._ema(closes_deque, self.trend_ema_slow, "r10_trend_ema_slow")
        rsi_val = self._rsi(prices_deque=closes_deque)
        macd_line, macd_signal, macd_hist = self._macd(prices_deque=closes_deque)
        atr_val = self._atr(highs_deque, lows_deque, closes_deque)
        calculated_volume_sma = self._sma(volumes_deque, self.volume_sma_window) 
        current_volatility = self._calculate_volatility(prices_deque=closes_deque)

        latest_values_dict = {}
        def get_latest(arr_or_deque, name):
            val = np.nan
            if arr_or_deque is None: return val # Guard clause
            arr_actual_len = len(arr_or_deque) if isinstance(arr_or_deque, deque) else (arr_or_deque.shape[0] if isinstance(arr_or_deque, np.ndarray) else 0)
            if arr_actual_len > 0:
                potential_val = arr_or_deque[-1]
                if not np.isnan(potential_val): val = potential_val
            latest_values_dict[name] = val
            return val

        def get_second_latest(arr_or_deque, name):
            val = np.nan
            if arr_or_deque is None: return val
            arr_actual_len = len(arr_or_deque) if isinstance(arr_or_deque, deque) else (arr_or_deque.shape[0] if isinstance(arr_or_deque, np.ndarray) else 0)
            if arr_actual_len > 1:
                potential_val = arr_or_deque[-2]
                if not np.isnan(potential_val): val = potential_val
            latest_values_dict[name+"_prev"] = val
            return val
        
        # Populate latest values, ensuring deques/arrays have enough elements
        price_latest = get_latest(closes_deque, "price")
        ema_fast_latest = get_latest(ema_fast, "ema_fast")
        ema_slow_latest = get_latest(ema_slow, "ema_slow")
        rsi_latest = get_latest(rsi_val, "rsi")
        macd_line_latest = get_latest(macd_line, "macd_line")
        macd_signal_latest = get_latest(macd_signal, "macd_signal")
        macd_hist_latest = get_latest(macd_hist, "macd_hist")
        atr_latest = get_latest(atr_val, "atr")
        volume_sma_latest = get_latest(calculated_volume_sma, "volume_sma")
        
        ema_fast_prev = get_second_latest(ema_fast, "ema_fast_prev_direct") # Different name to avoid dict key clash
        ema_slow_prev = get_second_latest(ema_slow, "ema_slow_prev_direct")
        macd_line_prev = get_second_latest(macd_line, "macd_line_prev_direct")
        macd_signal_prev = get_second_latest(macd_signal, "macd_signal_prev_direct")


        if self.logger:
             log_msg = "R10 Latest Indicators - "
             for k, v_val in latest_values_dict.items():
                 log_msg += f"{k.upper()}: {v_val:.{4}f} " if not np.isnan(v_val) else f"{k.upper()}: NaN "
             log_msg += f"VOL: {current_volatility:.6f}"
             self.logger.debug(log_msg)
        
        critical_nan_keys = []
        # Check all values that will be used in logic - ensure these are correctly populated
        check_nan_values = { # Using the direct latest values
            "price": price_latest, "ema_fast": ema_fast_latest, "ema_slow": ema_slow_latest,
            "rsi": rsi_latest, "macd_line": macd_line_latest, "macd_signal": macd_signal_latest,
            "atr": atr_latest, 
            "ema_fast_prev": ema_fast_prev, "ema_slow_prev": ema_slow_prev,
            "macd_line_prev": macd_line_prev, "macd_signal_prev": macd_signal_prev
        }
        for k, v_val in check_nan_values.items():
            if np.isnan(v_val): critical_nan_keys.append(k)

        if critical_nan_keys:
            context["reason"] = f"Critical R10 latest/prev indicator values NaN: {critical_nan_keys}"
            if self.logger: self.logger.warning(context["reason"])
            return context

        trend = "neutral"
        if ema_fast_latest > ema_slow_latest and ema_fast_prev <= ema_slow_prev: trend = "bullish_crossover"
        elif ema_fast_latest < ema_slow_latest and ema_fast_prev >= ema_slow_prev: trend = "bearish_crossover"
        elif ema_fast_latest > ema_slow_latest: trend = "bullish"
        elif ema_fast_latest < ema_slow_latest: trend = "bearish"

        rsi_state = "neutral"
        if rsi_latest > self.rsi_overbought: rsi_state = "overbought"
        elif rsi_latest < self.rsi_oversold: rsi_state = "oversold"

        macd_crossover_val = "none"; macd_trend_val = "neutral"
        if macd_line_latest > macd_signal_latest and macd_line_prev <= macd_signal_prev: macd_crossover_val = "bullish"
        elif macd_line_latest < macd_signal_latest and macd_line_prev >= macd_signal_prev: macd_crossover_val = "bearish"
        if macd_line_latest > macd_signal_latest: macd_trend_val = "bullish"
        elif macd_line_latest < macd_signal_latest: macd_trend_val = "bearish"
        
        current_volume_val = get_latest(volumes_deque, "current_volume_val")
        avg_volume_val = volume_sma_latest 
        volume_spike_val = False
        volume_trend_val = "flat"
        is_r10_volume_meaningful = False 

        if self.use_r10_volume_confirmation and len(volumes_deque) >= self.volume_sma_window and not np.isnan(avg_volume_val):
            recent_vols = np.array(list(volumes_deque)[-self.volume_sma_window:])
            if np.std(recent_vols) > 1e-5 and np.mean(recent_vols) > 1e-5 : 
                is_r10_volume_meaningful = True
            
            if is_r10_volume_meaningful:
                if avg_volume_val > 1e-9 : 
                    volume_spike_val = current_volume_val > avg_volume_val * self.volume_spike_multiplier
                    volume_trend_val = "rising" if current_volume_val > avg_volume_val else "falling"
                else: 
                    volume_spike_val = current_volume_val > 1.0 
                    volume_trend_val = "rising" if current_volume_val > 0 else "flat"
            else:
                if self.logger: self.logger.debug("R10 volume data deemed not meaningful. Using defaults for volume metrics.")
        elif self.logger and self.use_r10_volume_confirmation:
            self.logger.debug("R10 volume metrics skipped: use_r10_volume_confirmation is True but insufficient data or NaN SMA.")

        context.update({
            "valid": True, "price": price_latest, 
            "high": get_latest(highs_deque, "high"), "low": get_latest(lows_deque, "low"),
            "volume": current_volume_val, "ema_fast": ema_fast_latest, "ema_slow": ema_slow_latest,
            "rsi": rsi_latest, "macd_line": macd_line_latest, "macd_signal": macd_signal_latest, 
            "macd_hist": macd_hist_latest, "atr": atr_latest, "volatility": current_volatility, 
            "avg_volume": avg_volume_val, "trend": trend, "rsi_state": rsi_state, 
            "macd_crossover": macd_crossover_val, "macd_trend": macd_trend_val,
            "volume_spike": volume_spike_val, "volume_trend": volume_trend_val,
            "is_volume_meaningful_for_r10": is_r10_volume_meaningful 
        })
        return context

    def _calculate_risk_parameters(self, context: Dict) -> Dict:
        risk_params = {"sl": 0.0, "tp": 0.0, "size": 0.0, "valid": False}
        price = context.get("price", np.nan); atr_val = context.get("atr", np.nan)
        current_volatility = context.get("volatility", np.nan)

        if atr_val <= 1e-9 or np.isnan(atr_val) or price <= 1e-9 or np.isnan(price) or np.isnan(current_volatility):
            risk_params["reason"] = f"R10 Invalid price, ATR, or Volatility (P:{price}, A:{atr_val}, V:{current_volatility})"
            return risk_params
        
        vol_factor = 1 + min(current_volatility * 5, 0.2) # R10 vol factor
        sl_dist_atr = atr_val * self.atr_sl_multiplier * vol_factor
        sl_dist_price_cap = price * self.max_risk_percent_per_trade 
        sl_dist = min(sl_dist_atr, sl_dist_price_cap) 
        tp_dist = atr_val * self.atr_tp_multiplier * vol_factor
        
        min_sl_distance_points = 0.01 # For R10, this might be very small.
        if sl_dist < min_sl_distance_points :
            if self.logger: self.logger.debug(f"R10 SL distance {sl_dist:.5f} < min {min_sl_distance_points}. Using min.")
            sl_dist = min_sl_distance_points
            tp_dist = sl_dist * (self.atr_tp_multiplier / self.atr_sl_multiplier) # Maintain R:R

        sl_buy = price - sl_dist; tp_buy = price + tp_dist
        sl_sell = price + sl_dist; tp_sell = price - tp_dist
        
        risk_amt = self.account_balance * self.risk_per_trade
        size = 0.0
        if sl_dist > 1e-9: 
            # For R10, size might be notional or number of contracts, not directly 1/sl_dist if sl_dist is points
            # Assuming size is calculated based on account risk and SL distance in points
            # This part highly depends on how Deriv R10 position sizing works (points vs currency)
            # If SL distance is in points, risk_amt / sl_dist_in_currency_per_point.
            # Assuming sl_dist is in same currency unit as price for simplicity here.
            # If using multipliers, this will be taken care of by the TradingStrategy class usually.
            # For this RuleBasedSignal, it returns a 'size' that TradingStrategy will interpret.
            # Let's assume self.min_position_size is a contract/lot value for R10.
            # And risk_amt / sl_dist (if sl_dist is in currency) would give stake.
            # The provided code calculates size = risk_amt / sl_dist. This is currency/currency.
            # This 'size' might need to be multiplier if Deriv API expects that.
            # For now, keeping it as it was, which implies 'size' is a stake amount in account currency.
            calculated_stake_or_size = risk_amt / sl_dist # This is a ratio.
            # For R10, this might be a multiplier or stake. If stake, round to 2 decimals.
            # If it's lot size, min_position_size is critical.
            size = max(self.min_position_size, round(calculated_stake_or_size, 2))


        risk_params.update({ 
            "sl_buy": sl_buy, "tp_buy": tp_buy, "size_buy": size, 
            "sl_sell": sl_sell, "tp_sell": tp_sell, "size_sell": size,
            "sl_distance": sl_dist, "tp_distance": tp_dist, "valid": True 
        })
        return risk_params

    def _get_performance_factor(self) -> float:
        num_trades = len(self.trade_history)
        if num_trades < 5: return 1.0 # For R10, 5 trades might be quick
        recent_trades_count = min(num_trades, 20) 
        recent = list(self.trade_history)[-recent_trades_count:]
        if not recent: return 1.0
        wins = sum(1 for pnl in recent if pnl > 0)
        win_rate = wins / len(recent) if len(recent) > 0 else 0.5
        perf_factor = 0.7 + (win_rate * 0.4) 
        return float(np.clip(perf_factor, 0.7, 1.1)) # Tighter band for scalping

    def _calculate_confidence(self, context: Dict, signal_direction: str, perf_factor: float) -> float:
        base_weights = { "trend_match": 0.25, "trend_crossover": 0.20, 
                         "macd_match": 0.20, "macd_crossover": 0.20, 
                         "rsi_confirm": 0.10, "volume_confirm": 0.05 } 
        
        current_weights = base_weights.copy()
        is_volume_meaningful = context.get("is_volume_meaningful_for_r10", False)
        if not self.use_r10_volume_confirmation or not is_volume_meaningful:
            current_weights["volume_confirm"] = 0.0 
            if self.logger and self.use_r10_volume_confirmation and not is_volume_meaningful: # Log only if it was intended to be used
                self.logger.debug("R10 Confidence: Volume data unreliable/disabled, volume_confirm weight=0.")

        raw_score = 0.0
        trend = context.get("trend", "neutral")
        macd_trend = context.get("macd_trend", "neutral")
        macd_crossover = context.get("macd_crossover", "none")
        rsi_state = context.get("rsi_state", "neutral")
        volume_spike = context.get("volume_spike", False)
        
        if signal_direction == "BUY":
            if trend in ["bullish", "bullish_crossover"]: raw_score += current_weights["trend_match"]
            if trend == "bullish_crossover": raw_score += current_weights["trend_crossover"]
            if macd_trend == "bullish": raw_score += current_weights["macd_match"]
            if macd_crossover == "bullish": raw_score += current_weights["macd_crossover"]
            if rsi_state != "overbought": raw_score += current_weights["rsi_confirm"]
            if volume_spike and is_volume_meaningful: raw_score += current_weights["volume_confirm"]
        elif signal_direction == "SELL":
            if trend in ["bearish", "bearish_crossover"]: raw_score += current_weights["trend_match"]
            if trend == "bearish_crossover": raw_score += current_weights["trend_crossover"]
            if macd_trend == "bearish": raw_score += current_weights["macd_match"]
            if macd_crossover == "bearish": raw_score += current_weights["macd_crossover"]
            if rsi_state != "oversold": raw_score += current_weights["rsi_confirm"]
            if volume_spike and is_volume_meaningful: raw_score += current_weights["volume_confirm"]

        if self.logger:
             self.logger.debug(f"R10 Confidence Calc - Raw Score: {raw_score:.4f} (vol_meaningful: {is_volume_meaningful}, vol_weight: {current_weights['volume_confirm']:.2f}), Perf Factor: {perf_factor:.4f}")

        adjusted_score = raw_score * perf_factor
        confidence = float(np.clip(adjusted_score, 0.0, self.max_confidence))
        if raw_score < 0.35: confidence = 0.0 
        return confidence

    def _generate_signal(self, context: Dict) -> Dict:
        signal = {"direction": "HOLD", "strength": 0.0}
        
        trend = context.get("trend", "neutral")
        macd_crossover = context.get("macd_crossover", "none")
        macd_trend = context.get("macd_trend", "neutral")
        macd_hist = context.get("macd_hist", np.nan)
        rsi_state = context.get("rsi_state", "neutral")
        atr_val = context.get("atr", np.nan)
        price_val = context.get("price", np.nan)
        volatility_val = context.get("volatility", np.nan)

        if np.isnan(atr_val) or np.isnan(price_val) or np.isnan(volatility_val):
            low_vol_filter_active = True 
            if self.logger: self.logger.warning("R10 ATR/Vol filter active: NaN in critical values for filter.")
        else:
            min_atr_val_calc = price_val * self.min_atr_threshold_factor 
            low_vol_filter_active = atr_val < min_atr_val_calc or volatility_val < self.min_volatility_threshold
            if self.logger:
                 self.logger.debug(
                     f"R10 ATR/Vol Filter Check - ATR: {atr_val:.5f} (Req > {min_atr_val_calc:.5f}), "
                     f"Vol: {volatility_val:.6f} (Req > {self.min_volatility_threshold:.6f}), "
                     f"Filter Active: {low_vol_filter_active}"
                 )
        
        if low_vol_filter_active:
             if self.logger: self.logger.info(f"R10 Signal generation skipped: ATR/Vol filter active.")
             return signal 

        is_buy_trend = trend in ["bullish", "bullish_crossover"]
        is_buy_macd = macd_crossover == "bullish" or \
                      (macd_trend == "bullish" and trend != "bearish" and (not np.isnan(macd_hist) and macd_hist > 0))
        is_rsi_ok_buy = rsi_state != "overbought" # Using updated RSI levels (e.g. 20/80)

        if self.logger:
            self.logger.debug(f"R10 Signal Components (BUY): TrendOK: {is_buy_trend}, MACD_OK: {is_buy_macd}, RSI_OK: {is_rsi_ok_buy} (RSI State: {rsi_state})")

        if is_buy_trend and is_buy_macd and is_rsi_ok_buy:
            signal["direction"] = "BUY"
            strength_trend = 1 if trend == "bullish_crossover" else 0.6
            strength_macd = 1 if macd_crossover == "bullish" else (0.7 if macd_trend == "bullish" else 0.3)
            strength_rsi = 1 if rsi_state == "oversold" else (0.5 if rsi_state == "neutral" else 0.3) 
            signal["strength"] = np.mean([strength_trend, strength_macd, strength_rsi]) if not np.any(np.isnan([strength_trend, strength_macd, strength_rsi])) else 0.0

        if signal["direction"] == "HOLD":
            is_sell_trend = trend in ["bearish", "bearish_crossover"]
            is_sell_macd = macd_crossover == "bearish" or \
                           (macd_trend == "bearish" and trend != "bullish" and (not np.isnan(macd_hist) and macd_hist < 0))
            is_rsi_ok_sell = rsi_state != "oversold" # Using updated RSI levels

            if self.logger:
                self.logger.debug(f"R10 Signal Components (SELL): TrendOK: {is_sell_trend}, MACD_OK: {is_sell_macd}, RSI_OK: {is_rsi_ok_sell} (RSI State: {rsi_state})")

            if is_sell_trend and is_sell_macd and is_rsi_ok_sell:
                signal["direction"] = "SELL"
                strength_trend = 1 if trend == "bearish_crossover" else 0.6
                strength_macd = 1 if macd_crossover == "bearish" else (0.7 if macd_trend == "bearish" else 0.3)
                strength_rsi = 1 if rsi_state == "overbought" else (0.5 if rsi_state == "neutral" else 0.3)
                signal["strength"] = np.mean([strength_trend, strength_macd, strength_rsi]) if not np.any(np.isnan([strength_trend, strength_macd, strength_rsi])) else 0.0
        
        if self.logger:
             self.logger.debug(f"R10 Final Signal by _generate_signal: Dir: {signal['direction']}, Str: {signal['strength']:.2f}")
        return signal

    def _suggested_holding_time(self, context: Dict, signal_strength: float) -> int:
        base_minutes = 1 
        max_minutes = 4   # R10 Scalping: Max 4 minutes to give a bit more room
        
        volatility = context.get("volatility", 0.0001) 
        if np.isnan(volatility): volatility = 0.0001

        vol_scaling = 1.0 - np.clip(volatility * 5, -0.2, 0.2) 
        strength_scaling = 1.0 + np.clip(signal_strength * 0.5 - 0.25, 0.0, 0.5) 
        
        hold_time_float = base_minutes * strength_scaling * vol_scaling
        calculated_time = int(round(hold_time_float))
        final_hold_time = int(max(base_minutes, min(max_minutes, calculated_time)))
        
        if self.logger:
            self.logger.debug(f"R10 Holding Time Calc: Strength={signal_strength:.2f}, Volatility={volatility:.6f} -> Base={base_minutes}, Max={max_minutes}, CalcFloat={hold_time_float:.2f} -> Final={final_hold_time}m")
        return final_hold_time

    def generate_candle_signal(self, candle_data: List[Dict[str, float]]) -> Dict[str, object]:
        self.price_data.clear(); self.high_data.clear(); self.low_data.clear(); self.volume_data.clear()
        valid_candles_count = 0
        default_volume_for_r10 = 1.0 

        for candle in candle_data:
             if not all(k in candle for k in ["close", "high", "low"]):
                 if self.logger: self.logger.warning(f"R10 Skipping candle, missing c,h,l: {candle}")
                 continue
            
             current_candle_volume = float(candle.get("volume", default_volume_for_r10))

             try:
                self.price_data.append(float(candle["close"]))
                self.high_data.append(float(candle["high"]))
                self.low_data.append(float(candle["low"]))
                self.volume_data.append(current_candle_volume)
                valid_candles_count += 1
             except ValueError as e:
                if self.logger: self.logger.error(f"R10 Error converting candle data: {e}, candle: {candle}")
                continue

        if self.logger:
             self.logger.debug(f"R10 Input candles: {len(candle_data)}, Valid populated: {valid_candles_count}")

        min_len_required = max(
            self.trend_ema_slow, 
            self.macd_slow, 
            self.atr_window + 1, 
            self.rsi_window + 1, 
            self.volatility_window + 1
        ) + self.macd_signal + 5 
        
        if valid_candles_count < min_len_required:
             if self.logger: self.logger.info(f"R10 Insufficient data: Need ~{min_len_required}, have {valid_candles_count}. HOLDING.")
             return self._hold_payload(f"R10 Insufficient valid data: Need {min_len_required}, have {valid_candles_count}")

        self._cache = { "ema": {}, "rsi": {}, "atr": {}, "macd": {}, "sma": {} }

        context = self._get_market_context(self.price_data, self.high_data, self.low_data, self.volume_data)
        if not context["valid"]:
             return self._hold_payload(context.get("reason", "R10 Invalid market context"))

        signal_details = self._generate_signal(context)
        if signal_details["direction"] == "HOLD":
             return self._hold_payload(f"R10 Signal logic determined HOLD. Strength: {signal_details['strength']:.2f}")

        risk_params = self._calculate_risk_parameters(context)
        if not risk_params["valid"]:
             return self._hold_payload(risk_params.get("reason", "R10 Risk calc failed"))

        perf_factor = self._get_performance_factor()
        confidence = self._calculate_confidence(context, signal_details["direction"], perf_factor)

        if self.logger:
             self.logger.debug(f"R10 Pre-Conf Check - Sig: {signal_details['direction']}, Conf: {confidence:.4f} (Min: {self.min_confidence})")

        if confidence < self.min_confidence:
             return self._hold_payload(f"R10 Confidence {confidence:.2f} < threshold {self.min_confidence}")

        final_signal_payload = { "signal": signal_details["direction"], "confidence": confidence }
        price_precision = 4 # For R_10

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
        if pos_size <= 1e-9: 
            return self._hold_payload(f"R10 Calculated position size {pos_size} too small. SL Dist: {risk_params.get('sl_distance', 'N/A')}")

        if self.logger:
            self.logger.info(
                f"R10 TRADE SIGNAL: {final_signal_payload['signal']} | Conf: {final_signal_payload['confidence']:.2f} | "
                f"SL: {final_signal_payload['stop_loss']:.{price_precision}f} | "
                f"TP: {final_signal_payload['take_profit']:.{price_precision}f} | "
                f"Size: {final_signal_payload['position_size']} | Hold: {final_signal_payload['max_holding_time']}m | "
                f"Trail: {final_signal_payload['trailing_stop']:.{price_precision}f}"
            )
        return final_signal_payload

    def _hold_payload(self, reason: str) -> Dict[str, object]:
        if self.logger:
            self.logger.debug(f"R10 HOLD Signal – Reason: {reason}")
        return { "signal": "HOLD", "stop_loss": 0.0, "take_profit": 0.0, "confidence": 0.0,
                 "position_size": 0.0, "max_holding_time": 0, "trailing_stop": 0.0 }

    def record_trade_result(self, pnl: float):
         self.trade_history.append(pnl)
         if self.logger:
             self.logger.info(f"R10 Trade result: PNL = {pnl:.2f}. History size: {len(self.trade_history)}")

    def update_account_balance(self, balance: float):
        self.account_balance = balance
        if self.logger:
            self.logger.debug(f"R10 Account balance updated: {self.account_balance}")