from datetime import datetime

class RiskManager:
    def __init__(self, logger, break_even_threshold=0.5, lock_in_profit_threshold=1.0, scale_out_threshold=1.5):
        """
        Initialize the RiskManager with thresholds for dynamic risk adjustments.
        
        Parameters:
            logger (logging.Logger): Logger instance for logging risk events.
            break_even_threshold (float): The multiple of initial risk at which to move stop to break-even. 
                                          Example: 0.5 means when half-way to the take profit is reached, move SL to entry.
            lock_in_profit_threshold (float): The multiple of initial risk at which to move stop loss into profit territory.
            scale_out_threshold (float): The multiple of initial risk at which to consider partial scaling out.
        """
        self.logger = logger
        self.break_even_threshold = break_even_threshold
        self.lock_in_profit_threshold = lock_in_profit_threshold
        self.scale_out_threshold = scale_out_threshold

    def apply_trailing_stop(self, current_price, trailing_stop, entry_price, original_stop_loss):
        """
        Adjust the stop loss dynamically according to a trailing stop logic.
        
        current_price: Current price of the asset.
        trailing_stop: The trailing stop value (pips, ticks, or price units).
        entry_price: The price at which the trade was entered.
        original_stop_loss: The original stop loss set at the trade outset.
        
        Returns:
            float: The newly adjusted stop loss.
        """
        # Calculate a standard trailing stop from current price
        new_stop = current_price - trailing_stop
        
        # The stop loss should never be worse than the original stop:
        new_stop = max(new_stop, original_stop_loss)
        return new_stop

    def dynamic_profit_safeguard(self, entry_price, current_price, take_profit, initial_stop_loss):
        """
        Dynamically update the stop loss level as profits increase.
        
        Parameters:
            entry_price (float): The trade entry price.
            current_price (float): The current market price.
            take_profit (float): The configured take profit level.
            initial_stop_loss (float): The initial stop loss level set at trade entry.
        
        Returns:
            float: The updated stop loss to safeguard profits.
        """

        # Calculate the total planned profit (TP distance) and initial risk (SL distance)
        total_planned_profit = take_profit - entry_price
        initial_risk = entry_price - initial_stop_loss
        if initial_risk <= 0:
            self.logger.warning("Initial risk is not positive, dynamic profit safeguard may not apply correctly.")
            return initial_stop_loss

        current_profit = current_price - entry_price
        if current_profit <= 0:
            # No profit to safeguard yet, return original SL
            return initial_stop_loss

        # Calculate multiples of initial risk achieved
        multiples_of_risk = current_profit / initial_risk

        # Start with default new stop
        new_stop = initial_stop_loss

        # If we have reached the break-even threshold, move SL to entry price (no more risk)
        if multiples_of_risk >= self.break_even_threshold and new_stop < entry_price:
            new_stop = entry_price
            self.logger.info("Break-even threshold reached, SL moved to entry.")

        # If we've reached the lock-in profit threshold, move SL above entry to lock in profits
        if multiples_of_risk >= self.lock_in_profit_threshold:
            # For example, lock in half of current profit:
            # Let's say we lock in 25% of the current profit as a starting point
            locked_in = entry_price + (current_profit * 0.25)
            new_stop = max(new_stop, locked_in)
            self.logger.info(f"Profit lock-in threshold reached, SL moved above entry to {new_stop:.5f} to secure some profit.")

        # If scale_out_threshold is reached and partial closes are possible:
        # We would trigger partial closing logic here, but since Deriv might not support mid-contract scale-outs,
        # we just tighten the stop more aggressively.
        if multiples_of_risk >= self.scale_out_threshold:
            # Lock in even more profit, say 50% of current profit:
            locked_in_more = entry_price + (current_profit * 0.5)
            new_stop = max(new_stop, locked_in_more)
            self.logger.info(f"Scale-out threshold reached, SL tightened further to {new_stop:.5f} to secure larger portion of profit.")

        return new_stop

    def exceeded_max_holding_time(self, trade_open_time, max_holding_time):
        """
        Check if max holding time is exceeded.
        
        Parameters:
            trade_open_time (datetime): The time when the trade was opened.
            max_holding_time (int): Maximum recommended holding time in seconds.
        
        Returns:
            bool: True if max holding time is exceeded, False otherwise.
        """
        elapsed = (datetime.now() - trade_open_time).total_seconds()
        return elapsed > max_holding_time
