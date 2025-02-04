import numpy as np
import logging
from typing import Dict, Any, Tuple
from collections import deque

class RuleBasedSignals:
    def __init__(self, logger: Any = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

        # Trading parameters
        self.base_sl = 0.0005
        self.base_tp = 0.0010
        self.volatility_window = 20
        self.atr_window = 7
        self.rsi_window = 7
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.volume_spike_multiplier = 2.0
        self.min_confidence = 0.65
        self.max_confidence = 0.9
        self.risk_per_trade = 0.02
        self.min_confidence = 0.75  # Example threshold
        self.opens = deque(maxlen=100)
        self.highs = deque(maxlen=100)
        self.lows = deque(maxlen=100)
        self.closes = deque(maxlen=100)
        self.volumes = deque(maxlen=100)

        # Optimized data storage
        self.price_buffer = np.zeros(500, dtype=np.float64)
        self.high_buffer = np.zeros(500, dtype=np.float64)
        self.low_buffer = np.zeros(500, dtype=np.float64)
        self.volume_buffer = np.zeros(500, dtype=np.float64)
        self.buffer_index = 0
        self.buffer_size = 0

        # Precomputed indicators cache
        self.indicator_cache = {
            'ema': {},
            'rsi': np.zeros(500, dtype=np.float64),
            'macd': np.zeros(500, dtype=np.float64),
            'signal': np.zeros(500, dtype=np.float64),
            'atr': np.zeros(500, dtype=np.float64),
            'volumes': np.zeros(500, dtype=np.float64)
        }

        self.trade_history = deque(maxlen=100)
        self.account_balance = 1000.0
        
    def reset_buffers(self):
        """Clear buffers to avoid shape conflicts."""
        self.opens.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()    

    def update_buffers(self, candle: Dict[str, float]) -> None:
        """Update circular buffers with new candle data"""
        idx = self.buffer_index % 500
        self.price_buffer[idx] = candle['close']
        self.high_buffer[idx] = candle['high']
        self.low_buffer[idx] = candle['low']
        self.volume_buffer[idx] = candle['volume']
        self.buffer_index = (self.buffer_index + 1) % 500
        self.buffer_size = min(self.buffer_size + 1, 500)
        self._precompute_indicators()
        
        
        
        

    def _get_recent_data(self, length: int) -> Dict[str, np.ndarray]:
        """Get aligned recent data window"""
        start_idx = (self.buffer_index - length) % 500
        if start_idx + length <= 500:
            return {
                'prices': self.price_buffer[start_idx:start_idx+length],
                'highs': self.high_buffer[start_idx:start_idx+length],
                'lows': self.low_buffer[start_idx:start_idx+length],
                'volumes': self.volume_buffer[start_idx:start_idx+length]
            }
        else:
            part1 = 500 - start_idx
            return {
                'prices': np.concatenate((self.price_buffer[start_idx:], 
                                       self.price_buffer[:length-part1])),
                'highs': np.concatenate((self.high_buffer[start_idx:], 
                                      self.high_buffer[:length-part1])),
                'lows': np.concatenate((self.low_buffer[start_idx:], 
                                     self.low_buffer[:length-part1])),
                'volumes': np.concatenate((self.volume_buffer[start_idx:], 
                                        self.volume_buffer[:length-part1]))
            }

    def _vectorized_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Vectorized EMA using numpy"""
        alpha = 2 / (window + 1)
        weights = (1 - alpha) ** np.arange(len(data))[::-1]
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')

    def _precompute_indicators(self) -> None:
        """Batch compute all indicators – only if enough data is available,
        and assign into the cache using the computed length."""
        length = min(self.buffer_size, 500)
        if length == 0:
            return  # Nothing to compute

        data = self._get_recent_data(length)
        
        # --- EMA Calculation ---
        for window in [9, 21, 12, 26]:
            if length < window:
                self.indicator_cache['ema'][window] = np.array([])
            else:
                ema = self._vectorized_ema(data['prices'], window)
                self.indicator_cache['ema'][window] = ema  # ema is of length (length - window + 1)

        # --- RSI Calculation ---
        if length < self.rsi_window:
            self.indicator_cache['rsi'] = np.array([])
        else:
            deltas = np.diff(data['prices'])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            # The "valid" mode produces an array of length: length - rsi_window
            avg_gain = np.convolve(gains, np.ones(self.rsi_window)/self.rsi_window, 'valid')
            avg_loss = np.convolve(losses, np.ones(self.rsi_window)/self.rsi_window, 'valid')
            rs = avg_gain / np.where(avg_loss == 0, 1e-9, avg_loss)
            computed_rsi = 100 - (100 / (1 + rs))
            # Assign the computed RSI into the cache at positions 0..(computed_rsi.shape[0]-1)
            self.indicator_cache['rsi'][:computed_rsi.shape[0]] = computed_rsi

        # --- MACD Calculation ---
        # We need both EMAs for MACD. Use the computed arrays (if not empty)
        ema12 = self.indicator_cache['ema'].get(12, np.array([]))
        ema26 = self.indicator_cache['ema'].get(26, np.array([]))
        if ema12.size == 0 or ema26.size == 0:
            self.indicator_cache['macd'] = np.array([])
            self.indicator_cache['signal'] = np.array([])
        else:
            mlen = min(ema12.shape[0], ema26.shape[0])
            macd = ema12[-mlen:] - ema26[-mlen:]
            # Store the computed MACD (its length is mlen)
            self.indicator_cache['macd'][:macd.shape[0]] = macd
            # For the signal line, ensure enough data is available:
            if macd.shape[0] >= self.macd_signal:
                signal_line = self._vectorized_ema(macd, self.macd_signal)
                self.indicator_cache['signal'][:signal_line.shape[0]] = signal_line
            else:
                self.indicator_cache['signal'] = np.array([])

        # --- ATR Calculation ---
        if length < self.atr_window + 1:
            self.indicator_cache['atr'] = np.array([])
        else:
            highs = data['highs']
            lows = data['lows']
            prices = data['prices']
            # Compute True Range for indices 1..length-1
            tr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - prices[:-1]),
                                    np.abs(lows[1:] - prices[:-1])))
            # Convolve with a window of atr_window; result length = (len(tr) - atr_window + 1)
            atr = np.convolve(tr, np.ones(self.atr_window)/self.atr_window, 'valid')
            self.indicator_cache['atr'][:atr.shape[0]] = atr

        # --- Volume Moving Average Calculation ---
        if length < 20:
            self.indicator_cache['volumes'] = np.array([])
        else:
            volumes = data['volumes']
            vol_ma = np.convolve(volumes, np.ones(20)/20, 'valid')
            self.indicator_cache['volumes'][:vol_ma.shape[0]] = vol_ma


    def _trend_direction(self, ema_fast: np.ndarray, ema_slow: np.ndarray) -> str:
        """Vectorized trend analysis"""
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return "neutral"
        
        current_fast = ema_fast[-1]
        current_slow = ema_slow[-1]
        prev_fast = ema_fast[-2]
        prev_slow = ema_slow[-2]
        
        crossover = np.sign(current_fast - current_slow) - np.sign(prev_fast - prev_slow)
        if crossover > 0:
            return "bullish_crossover"
        if crossover < 0:
            return "bearish_crossover"
        return "bullish" if current_fast > current_slow else "bearish"

    def _momentum_strength(self, rsi: np.ndarray, macd: np.ndarray, signal: np.ndarray) -> Dict[str, str]:
        """Vectorized momentum analysis"""
        momentum = {'rsi': 'neutral', 'macd_crossover': 'none'}
        
        # RSI analysis
        last_rsi = rsi[-1]
        if last_rsi > 70:
            momentum['rsi'] = 'overbought'
        elif last_rsi < 30:
            momentum['rsi'] = 'oversold'
        
        # MACD crossover
        if len(macd) >= 2 and len(signal) >= 2:
            current_diff = macd[-1] - signal[-1]
            prev_diff = macd[-2] - signal[-2]
            if current_diff > 0 and prev_diff <= 0:
                momentum['macd_crossover'] = 'bullish'
            elif current_diff < 0 and prev_diff >= 0:
                momentum['macd_crossover'] = 'bearish'
        
        return momentum

    def _volume_analysis(self, current_volume: float, volume_ma: float) -> Dict[str, Any]:
        """Vectorized volume analysis"""
        return {
            'volume_spike': current_volume > volume_ma * self.volume_spike_multiplier,
            'volume_trend': 'rising' if current_volume > volume_ma else 'falling'
        }

    def _dynamic_stop_loss_take_profit(self, price: float, atr: float, volatility: float) -> Tuple[float, float]:
        """Optimized SL/TP calculation"""
        multiplier_sl = 1.2 + volatility * 0.5
        multiplier_tp = 2.0 + volatility * 1.0
        sl = price - (atr * multiplier_sl)
        tp = price + (atr * multiplier_tp)
        return np.round(sl, 5), np.round(tp, 5)

    def _calculate_position_size(self, sl: float, price: float) -> float:
        """Vectorized position sizing"""
        price_risk = np.abs(price - sl)
        if price_risk < 0.0005:
            return 0.0
        risk_amount = self.account_balance * self.risk_per_trade
        position_size = np.clip(risk_amount / price_risk, 0, self.account_balance * 0.1)
        return np.round(position_size, 2)

    def _recent_performance(self) -> float:
        """Vectorized performance calculation"""
        if len(self.trade_history) < 3:
            return 0.0
        recent = np.array(list(self.trade_history)[-3:])
        win_rate = np.sum(recent > 0) / len(recent)
        return (win_rate - 0.5) * 2

    def _signal_strength(self, trend: str, momentum: Dict, volume: Dict) -> Dict[str, Any]:
        """Optimized signal aggregation"""
        score = 0.0
        
        # Trend scoring
        trend_scores = {
            "bullish_crossover": 1.0,
            "bearish_crossover": -1.0,
            "bullish": 0.5,
            "bearish": -0.5
        }
        score += trend_scores.get(trend, 0)
        
        # Momentum scoring
        momentum_scores = {
            'overbought': -0.5, 'oversold': 0.5,
            'bullish': 0.3, 'bearish': -0.3
        }
        score += momentum_scores.get(momentum['rsi'], 0)
        score += momentum_scores.get(momentum['macd_crossover'], 0)
        
        # Volume scoring
        if volume['volume_spike']:
            score += 0.3 if volume['volume_trend'] == 'rising' else -0.3
        
        # Normalize and return
        strength = np.clip(score / 1.6, -1, 1)  # Max possible score ±1.6
        return {
            'direction': 'BUY' if strength > 0 else 'SELL' if strength < 0 else 'HOLD',
            'strength': np.abs(strength)
        }

    def _holding_time(self, strength: float) -> int:
        """Optimized holding time calculation"""
        base_time = 2  # minutes
        return int(np.clip(base_time * (1 + strength), 1, 3))

    def generate_signal(self) -> Dict[str, Any]:
        """Generate complete trading signal"""
        if self.buffer_size < max(self.macd_slow, self.atr_window, self.rsi_window):
            return self._hold_signal("Insufficient data")
        
        data = self._get_recent_data(self.buffer_size)
        current_price = data['prices'][-1]
        atr_value = self.indicator_cache['atr'][self.buffer_size-1]
        volatility = np.std(np.diff(np.log(data['prices']))) * np.sqrt(252)
        
        # Calculate key components
        trend = self._trend_direction(
            self.indicator_cache['ema'][9], 
            self.indicator_cache['ema'][21]
        )
        momentum = self._momentum_strength(
            self.indicator_cache['rsi'],
            self.indicator_cache['macd'],
            self.indicator_cache['signal']
        )
        volume_analysis = self._volume_analysis(
            data['volumes'][-1], 
            self.indicator_cache['volumes'][self.buffer_size-1]
        )
        
        # Generate signal
        signal = self._signal_strength(trend, momentum, volume_analysis)
        sl, tp = self._dynamic_stop_loss_take_profit(current_price, atr_value, volatility)
        
        return {
            'signal': signal['direction'],
            'stop_loss': sl,
            'take_profit': tp,
            'confidence': np.clip(
                (signal['strength'] * 0.7) + (self._recent_performance() * 0.3),
                self.min_confidence, self.max_confidence
            ),
            'position_size': self._calculate_position_size(sl, current_price),
            'max_holding_time': self._holding_time(signal['strength']),
            'trailing_stop': atr_value * 2.0,
            'indicators': {
                'ema_fast': self.indicator_cache['ema'][9][-1],
                'ema_slow': self.indicator_cache['ema'][21][-1],
                'rsi': self.indicator_cache['rsi'][-1],
                'macd': self.indicator_cache['macd'][-1],
                'signal_line': self.indicator_cache['signal'][-1],
                'atr': atr_value
            }
        }

    def _hold_signal(self, reason: str) -> Dict[str, Any]:
        if self.logger:
            self.logger.debug(f"HOLD signal: {reason}")
        return {
            'signal': 'HOLD',
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'confidence': 0.0,
            'position_size': 0.0,
            'max_holding_time': 0,
            'trailing_stop': 0.0,
            'indicators': {}
        }