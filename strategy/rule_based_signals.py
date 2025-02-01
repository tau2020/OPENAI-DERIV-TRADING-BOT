import numpy as np
from typing import List, Dict
from collections import deque

class RuleBasedSignals:
    def __init__(self, logger=None):
        self.logger = logger
        
        # Dynamic parameters
        self.base_sl = 0.0010
        self.base_tp = 0.0015
        self.volatility_window = 20
        self.atr_window = 14
        self.rsi_window = 14
        self.macd_fast = 24
        self.macd_slow = 50
        self.macd_signal = 9
        self.volume_spike_multiplier = 3.0
        self.min_confidence = 0.7
        self.max_confidence = 0.9
        self.risk_per_trade = 0.02  # 2% of account balance
        
        # Data storage
        self.price_data = deque(maxlen=500)
        self.volume_data = deque(maxlen=500)
        self.trade_history = deque(maxlen=100)
        self.indicator_cache = {}
        
        # External account balance (can be set externally)
        self.account_balance = 1000.0

    def _sma(self, data: List[float], window: int) -> float:
        """Compute simple moving average over the last `window` values."""
        if len(data) < window:
            return np.mean(data)
        return np.mean(data[-window:])

    def _ema(self, data: List[float], window: int) -> List[float]:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='full')[:len(data)]
        ema[:window] = ema[window]
        return list(ema)

    def _rsi(self, prices: List[float]) -> List[float]:
        deltas = np.diff(prices)
        seed = deltas[:self.rsi_window+1]
        up = seed[seed >= 0].sum() / self.rsi_window
        down = -seed[seed < 0].sum() / self.rsi_window
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:self.rsi_window] = 100. - 100. / (1. + rs)
        for i in range(self.rsi_window, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up * (self.rsi_window - 1) + upval) / self.rsi_window
            down = (down * (self.rsi_window - 1) + downval) / self.rsi_window
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        return list(rsi)

    def _macd(self, prices: List[float]) -> tuple:
        ema12 = self._ema(prices, self.macd_fast)
        ema26 = self._ema(prices, self.macd_slow)
        macd = np.subtract(ema12, ema26)
        signal = self._ema(macd, self.macd_signal)
        return list(macd), list(signal)

    def _atr(self, highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        tr = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr.append(max(hl, hc, lc))
        atr = [sum(tr[:self.atr_window]) / self.atr_window]
        for i in range(self.atr_window, len(tr)):
            atr_val = (atr[-1] * (self.atr_window - 1) + tr[i]) / self.atr_window
            atr.append(atr_val)
        return atr

    def _calculate_volatility(self, prices: List[float]) -> float:
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(252)  # Annualized volatility

    def _trend_direction(self, ema_fast: List[float], ema_slow: List[float]) -> str:
        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] <= ema_slow[-2]:
            return "bullish_crossover"
        if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] >= ema_slow[-2]:
            return "bearish_crossover"
        if ema_fast[-1] > ema_slow[-1]:
            return "bullish"
        return "bearish"

    def _momentum_strength(self, rsi: List[float], macd: List[float], signal: List[float]) -> Dict:
        return {
            'rsi': 'overbought' if rsi[-1] > 70 else 'oversold' if rsi[-1] < 30 else 'neutral',
            'macd_crossover': 'bullish' if macd[-1] > signal[-1] and macd[-2] <= signal[-2] else 
                             'bearish' if macd[-1] < signal[-1] and macd[-2] >= signal[-2] else 'none'
        }

    def _volume_analysis(self, current_volume: float, volume_ma: float) -> Dict:
        """
        Compare the current volume to the simple moving average.
        Note: volume_ma is now a scalar.
        """
        return {
            'volume_spike': current_volume > volume_ma * self.volume_spike_multiplier,
            'volume_trend': 'rising' if current_volume > volume_ma else 'falling'
        }

    def _dynamic_stop_loss_take_profit(self, price: float, atr: List[float], volatility: float) -> tuple:
        sl = price - (atr[-1] * 1.5 * (1 + volatility))
        tp = price + (atr[-1] * 3.0 * (1 + volatility))
        return sl, tp

    def _calculate_position_size(self, sl: float, price: float, risk_percent: float) -> float:
        risk_amount = self.account_balance * risk_percent
        price_risk = abs(price - sl)
        
        # Add minimum risk distance filter
        min_risk_distance = 0.0005  # 0.5 pips
        if price_risk < min_risk_distance:
            return 0.0
            
        position_size = min(
            risk_amount / price_risk,
            self.account_balance * 0.1  # Max 10% per trade
        )
        return round(position_size, 2)

    def _recent_performance(self) -> float:
        if len(self.trade_history) < 3:
            return 0.0
        recent = list(self.trade_history)[-3:]
        win_rate = sum(1 for t in recent if t > 0) / len(recent)
        return (win_rate - 0.5) * 2

    def _signal_strength(self, trend: str, momentum: Dict, volume: Dict) -> Dict:
        buy_signals = 0
        sell_signals = 0
        
        if 'bullish' in trend:
            buy_signals += 1
        if momentum['macd_crossover'] == 'bullish':
            buy_signals += 1
        if momentum['rsi'] == 'oversold':
            buy_signals += 1
        if volume['volume_spike'] and volume['volume_trend'] == 'rising':
            buy_signals += 1
            
        if 'bearish' in trend:
            sell_signals += 1
        if momentum['macd_crossover'] == 'bearish':
            sell_signals += 1
        if momentum['rsi'] == 'overbought':
            sell_signals += 1
        if volume['volume_spike'] and volume['volume_trend'] == 'falling':
            sell_signals += 1
            
        if buy_signals > sell_signals:
            return {'direction': 'BUY', 'strength': buy_signals / 4}
        elif sell_signals > buy_signals:
            return {'direction': 'SELL', 'strength': sell_signals / 4}
        return {'direction': 'HOLD', 'strength': 0}

    def _holding_time(self, signal: Dict) -> int:
        base_time = 3  # minutes
        return max(1, min(5, base_time * (1 + signal['strength'])))

    def _hold_signal(self, reason: str) -> Dict:
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

    def _calculate_confidence(self, trend: str, momentum: Dict, volume: Dict, performance: float) -> float:
        confidence = 0.3
        if 'crossover' in trend:
            confidence += 0.25
        elif trend in ['bullish', 'bearish']:
            confidence += 0.1
        if momentum['rsi'] in ['overbought', 'oversold']:
            confidence += 0.2
        if momentum['macd_crossover'] != 'none':
            confidence += 0.1
        if volume['volume_spike']:
            confidence += 0.15
        if volume['volume_trend'] == 'rising':
            confidence += 0.05
        confidence *= (1 + performance)
        
        return np.clip(confidence, self.min_confidence, self.max_confidence)

    def generate_candle_signal(self, candle_data: List[Dict]) -> Dict:
        if len(candle_data) < max(self.macd_slow, self.atr_window, self.rsi_window):
            return self._hold_signal("Insufficient data")
        
        # Extract price components
        closes = [c['close'] for c in candle_data]
        opens = [c['open'] for c in candle_data]
        highs = [c['high'] for c in candle_data]
        lows = [c['low'] for c in candle_data]
        volumes = [c['volume'] for c in candle_data]
        
        
        
        
        
        # Calculate indicators
        ema_fast = self._ema(closes, 9)
        ema_slow = self._ema(closes, 21)
        rsi = self._rsi(closes)
        macd, signal = self._macd(closes)
        atr = self._atr(highs, lows, closes)
        volume_ma = self._sma(volumes, 20)  # returns a scalar
        current_volume = volumes[-1]
        
        volatility = self._calculate_volatility(closes)
        sl, tp = self._dynamic_stop_loss_take_profit(closes[-1], atr, volatility)
        
        trend_direction = self._trend_direction(ema_fast, ema_slow)
        momentum = self._momentum_strength(rsi, macd, signal)
        volume_analysis = self._volume_analysis(current_volume, volume_ma)
        
        confidence = self._calculate_confidence(
            trend_direction,
            momentum,
            volume_analysis,
            self._recent_performance()
        )
        
        position_size = self._calculate_position_size(sl, closes[-1], self.risk_per_trade)
        signal_strength = self._signal_strength(
            trend_direction,
            momentum,
            volume_analysis
        )
        
        return {
            'signal': signal_strength['direction'],
            'stop_loss': sl,
            'take_profit': tp,
            'confidence': confidence,
            'position_size': position_size,
            'max_holding_time': self._holding_time(signal_strength),
            'trailing_stop': atr[-1] * 2.0,  # use the latest ATR value
            'indicators': {
                'ema_fast': ema_fast[-1],
                'ema_slow': ema_slow[-1],
                'rsi': rsi[-1],
                'macd': macd[-1],
                'signal_line': signal[-1],
                'atr': atr[-1]
            }
        }
