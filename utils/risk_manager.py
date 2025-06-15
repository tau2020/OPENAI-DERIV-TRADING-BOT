from datetime import datetime

class RiskManager:
    """
    Handles trailing stops, dynamic stop-loss adjustments, maximum holding time checks,
    and optional trailing profit logic.
    """

    def __init__(
        self,
        logger,
        break_even_threshold: float = 0.3,
        lock_in_profit_threshold: float = 0.8,
        scale_out_threshold: float = 1.2,
        trailing_profit_buffer: float = 0.1,
    ):
        """
        Initialize the RiskManager with thresholds for dynamic risk adjustments.
        
        Args:
            logger (logging.Logger): Logger instance for logging risk events.
            break_even_threshold (float): Multiple of initial risk at which the stop-loss
                moves to breakeven (entry price).
            lock_in_profit_threshold (float): Multiple of initial risk at which the stop-loss
                moves above entry to lock in partial profit.
            scale_out_threshold (float): Multiple of initial risk at which you scale out or
                further tighten the stop.
            trailing_profit_buffer (float): If > 0, once the trade moves past lock_in_profit_threshold,
                apply a trailing stop that maintains a buffer behind the price.
        """
        self.logger = logger
        self.break_even_threshold = break_even_threshold
        self.lock_in_profit_threshold = lock_in_profit_threshold
        self.scale_out_threshold = scale_out_threshold
        self.trailing_profit_buffer = trailing_profit_buffer

    def apply_trailing_stop(self, current_price: float, trailing_stop: float,
                            entry_price: float, original_stop_loss: float) -> float:
        """
        Apply a standard trailing stop-loss offset from the current price (BUY case):
          new_stop = max(current_price - trailing_stop, original_stop_loss)

        For SELL, you'd invert the logic outside or adapt as needed.

        Args:
            current_price (float): Current market price.
            trailing_stop (float): Trailing stop distance (in price units, pips, or ticks).
            entry_price (float): Trade entry price (unused in a basic trailing stop, but available).
            original_stop_loss (float): Original stop-loss set at trade entry.

        Returns:
            float: The newly adjusted stop-loss level.
        """
        raw_stop = current_price - trailing_stop
        # Never loosen beyond original SL
        return max(raw_stop, original_stop_loss)

    def dynamic_profit_safeguard(self, entry_price: float, current_price: float,
                                 take_profit: float, initial_stop_loss: float) -> float:
        """
        Dynamically update the stop-loss to safeguard profits as the trade moves in your favor.
        Moves stop-loss to breakeven, then progressively locks in more profit
        as multiples of the initial risk are achieved. Optionally transitions
        to a trailing stop once a threshold is reached.

        Args:
            entry_price (float): The trade entry price.
            current_price (float): The current market price.
            take_profit (float): The originally configured take-profit level.
            initial_stop_loss (float): The initially set stop-loss level.

        Returns:
            float: The updated stop-loss to safeguard profits.
        """
        # Calculate initial risk
        initial_risk = entry_price - initial_stop_loss
        if initial_risk <= 0:
            self.logger.warning("Initial risk is not positive; profit safeguard may not apply correctly.")
            return initial_stop_loss

        # Current profit from entry
        current_profit = current_price - entry_price
        if current_profit <= 0:
            # No profit to safeguard yet
            return initial_stop_loss

        multiples_of_risk = current_profit / initial_risk

        # Start with the original stop
        new_stop = initial_stop_loss

        # 1) Move SL to breakeven if break-even threshold reached
        if multiples_of_risk >= self.break_even_threshold and new_stop < entry_price:
            new_stop = entry_price
            self.logger.info("Break-even threshold reached, SL moved to entry.")

        # 2) Lock in profit if above lock_in_profit_threshold
        if multiples_of_risk >= self.lock_in_profit_threshold:
            # For example, lock in 25% of the current profit
            locked_in = entry_price + (current_profit * 0.25)
            new_stop = max(new_stop, locked_in)
            self.logger.info(
                f"Profit lock-in threshold reached, SL moved above entry to {new_stop:.5f} to secure partial profit."
            )

            # 2a) If trailing_profit_buffer is set, apply a trailing approach
            #     that sets the stop to (current_price - trailing_profit_buffer),
            #     but never loosening below new_stop.
            if self.trailing_profit_buffer > 0:
                # This means from here onward, we run a trailing approach
                raw_trailing_stop = current_price - self.trailing_profit_buffer
                new_stop = max(new_stop, raw_trailing_stop)
                self.logger.info(
                    f"Trailing profit activated. SL updated to {new_stop:.5f} using buffer {self.trailing_profit_buffer}."
                )

        # 3) Scale-out or further tighten if scale_out_threshold is reached
        if multiples_of_risk >= self.scale_out_threshold:
            # For demonstration, lock in 50% of the current profit
            locked_in_more = entry_price + (current_profit * 0.5)
            new_stop_before = new_stop
            new_stop = max(new_stop, locked_in_more)
            self.logger.info(
                f"Scale-out threshold reached. SL tightened further from {new_stop_before:.5f} to {new_stop:.5f}."
            )

            # Also optionally re-check trailing logic here if you want to tighten further
            if self.trailing_profit_buffer > 0:
                trailing_stop_price = current_price - self.trailing_profit_buffer
                if trailing_stop_price > new_stop:
                    self.logger.info(
                        f"Additional trailing after scale-out. SL from {new_stop:.5f} to {trailing_stop_price:.5f}."
                    )
                    new_stop = trailing_stop_price

        return new_stop

    def exceeded_max_holding_time(self, trade_open_time: datetime, max_holding_time: int) -> bool:
        """
        Check if the maximum recommended holding time has been exceeded.
        
        Args:
            trade_open_time (datetime): The time when the trade was opened.
            max_holding_time (int): Maximum holding time in seconds.

        Returns:
            bool: True if holding time is exceeded, False otherwise.
        """
        elapsed_seconds = (datetime.now() - trade_open_time).total_seconds()
        return elapsed_seconds > max_holding_time
