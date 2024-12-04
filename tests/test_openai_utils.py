# tests/test_openai_utils.py

import pytest
from unittest.mock import MagicMock
from utils.openai_utils import get_trading_signal_from_openai
from datetime import datetime

@pytest.mark.asyncio
async def test_get_trading_signal_from_openai_buy():
    # Mock OpenAI client response
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = {'content': 'BUY'}
    mock_client.ChatCompletion.create.return_value = mock_response

    # Mock logger
    mock_logger = MagicMock()

    signal = await get_trading_signal_from_openai(
        client=mock_client,
        symbol='R_50',
        prices=[100, 101, 102],
        times=[datetime.now(), datetime.now(), datetime.now()],
        logger=mock_logger
    )

    assert signal == 'BUY'
    mock_logger.info.assert_called_with("OpenAI assistant recommendation: BUY")
