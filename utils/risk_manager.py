from datetime import datetime

class RiskManager:
    def __init__(self, logger):
        self.logger = logger

    def apply_trailing_stop(self, current_price, trailing_stop, entry_price):
        """
        Adjust stop loss according to trailing_stop logic.
        Note: Deriv contracts may not support mid-contract modification.
        This function demonstrates the logic only.
        """
        adjusted_sl = max(entry_price - trailing_stop, current_price - trailing_stop)
        return adjusted_sl

    def exceeded_max_holding_time(self, trade_open_time, max_holding_time):
        """
        Check if max holding time is exceeded.
        """
        elapsed = (datetime.now() - trade_open_time).total_seconds()
        return elapsed > max_holding_time
