import numpy as np

class AdaptiveWeights:
    def __init__(self, initial_weights: dict):
        """
        initial_weights: a dict like {"T": 0.4, "M": 0.35, "O": 0.25} representing the
                         initial weight for Trend (T), MACD (M), and Oscillator (O).
        """
        self.weights = initial_weights.copy()
        self.indicator_performance = {"T": [], "M": [], "O": []}  # list of weighted trade outcomes

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
        Update indicator weights based on their recent performance.
        Computes an average score per indicator, then uses a softmax transformation to produce
        normalized weights. A smoothing factor is applied to prevent abrupt changes.
        """
        scores = {}
        for key, perf in self.indicator_performance.items():
            scores[key] = np.mean(perf) if perf else 0.0

        # Softmax transformation
        exp_scores = np.exp(np.array(list(scores.values())))
        if np.sum(exp_scores) == 0:
            norm_weights = np.array([1/3] * 3)
        else:
            norm_weights = exp_scores / np.sum(exp_scores)
        new_weights = dict(zip(scores.keys(), norm_weights))

        # Smoothing to avoid rapid weight changes
        smoothing = 0.1
        for key in self.weights.keys():
            self.weights[key] = (1 - smoothing) * self.weights[key] + smoothing * new_weights[key]

        # Optionally, clear performance history or use a rolling window.
        self.indicator_performance = {key: [] for key in self.indicator_performance}
        return self.weights