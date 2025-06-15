# tests/test_reverse.py
import asyncio
import pytest
from strategy.trading_strategy import TradingStrategy

class DummyAPI:  # minimal stub
    async def buy(self, *_, **__): return {"buy": {"contract_id": "1"}}
    async def sell(self, *_, **__): return {"sell": {"profit": -3.10}}
    async def subscribe(self, *_):  # returns object w/ subscribe attr
        class _S:
            def subscribe(self, **kwargs): pass
        return _S()
    async def proposal(self, *_): return {"proposal": {"id": "x", "ask_price": 100}}
    async def balance(self): return {"balance": {"balance": 10_000}}
    async def ticks_history(self, *_): return {"history": {"times": [0,1], "prices":[1,1]}}
    async def get_account_status(self): return {"get_account_status": {"is_virtual":1}}

@pytest.mark.asyncio
async def test_opposite_bid():
    strat = TradingStrategy(logging.getLogger("test"), DummyAPI())  # type: ignore[arg-type]
    strat.trade_directions["1"] = "BUY"
    poc = {"proposal_open_contract": {"is_valid_to_sell": True, "profit": -3.05}}
    fut = asyncio.get_event_loop().create_future()
    await strat.handle_poc_update(poc, "1", fut)
    assert "1" in strat.reversed_trades
