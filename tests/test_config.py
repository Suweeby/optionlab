"""The shared defaults in optionlab.config are valid and used consistently."""
import ast
from pathlib import Path

from optionlab import config, volatility
from optionlab.data import normalize_ticker


def test_presets_are_valid_unique_tickers_and_include_the_default():
    assert all(normalize_ticker(t) == t for t in config.PRESET_TICKERS)
    assert len(set(config.PRESET_TICKERS)) == len(config.PRESET_TICKERS)
    assert config.DEFAULT_TICKER in config.PRESET_TICKERS


def test_rate_is_a_decimal_not_a_percentage():
    assert 0.0 < config.DEFAULT_RISK_FREE_RATE < 0.2


def test_time_conventions_have_one_source():
    assert volatility.TRADING_DAYS is config.TRADING_DAYS == 252
    assert config.MIN_HISTORY_DAYS == config.TRADING_DAYS          # one year of returns
    assert 2 <= config.MIN_VOL_RETURNS < config.MIN_HISTORY_DAYS
    assert config.DEFAULT_LOOKBACK_DAYS >= 30 and config.EXPIRED_CHAIN_GRACE_DAYS >= 0


def test_config_imports_nothing_from_the_package():
    tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names]
    assert not any(name.startswith("optionlab") for name in imported)
