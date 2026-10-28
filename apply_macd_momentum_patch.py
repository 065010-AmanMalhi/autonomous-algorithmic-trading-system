from __future__ import annotations

import ast
import shutil
from datetime import datetime
from pathlib import Path

CREW = Path('crew/trading_crew.py')
INDICATORS = Path('mcp_servers/stock_trading/servers/technical_indicators_mcp_server.py')


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match for {description}; found {count}. No files written.")
    return source.replace(old, new, 1)


def main() -> None:
    for path in (CREW, INDICATORS):
        if not path.exists():
            raise SystemExit(f"Run from project root; missing required file: {path}")

    crew_source = CREW.read_text(encoding='utf-8')
    indicator_source = INDICATORS.read_text(encoding='utf-8')

    # Pass quantitative MACD components out of the MCP snapshot. Keep the existing
    # `macd` key as the crossover category for backward compatibility.
    indicator_source = replace_once(
        indicator_source,
        '                "macd": (indicators.get("macd") or {}).get("crossover"),\n'
        '                "bb_percent_b": (indicators.get("bollinger") or {}).get("percent_b"),',
        '                "macd": (indicators.get("macd") or {}).get("crossover"),\n'
        '                "macd_line": (indicators.get("macd") or {}).get("line"),\n'
        '                "macd_signal": (indicators.get("macd") or {}).get("signal"),\n'
        '                "macd_histogram": (indicators.get("macd") or {}).get("histogram"),\n'
        '                "bb_percent_b": (indicators.get("bollinger") or {}).get("percent_b"),',
        'portfolio snapshot MACD fields',
    )

    # Whitelist the added fields into the compact prompt and deterministic gate.
    crew_source = replace_once(
        crew_source,
        '            "macd": row.get("macd"),\n'
        '            "bb_percent_b": row.get("bb_percent_b"),',
        '            "macd": row.get("macd"),\n'
        '            "macd_line": row.get("macd_line"),\n'
        '            "macd_signal": row.get("macd_signal"),\n'
        '            "macd_histogram": row.get("macd_histogram"),\n'
        '            "bb_percent_b": row.get("bb_percent_b"),',
        'compact snapshot MACD whitelist',
    )

    crew_source = replace_once(
        crew_source,
        '    macd = str(row.get("macd", "")).strip().lower()\n'
        '    rsi = _number(row.get("rsi"), -1.0)',
        '    macd = str(row.get("macd", "")).strip().lower()\n'
        '    macd_hist_raw = row.get("macd_histogram")\n'
        '    macd_histogram = _number(macd_hist_raw, -1e12)\n'
        '    has_macd_histogram = macd_hist_raw is not None and abs(macd_histogram) < 1e11\n'
        '    macd_line = _number(row.get("macd_line"), -1e12)\n'
        '    macd_signal = _number(row.get("macd_signal"), -1e12)\n'
        '    rsi = _number(row.get("rsi"), -1.0)',
        'deterministic signal MACD parsing',
    )

    crew_source = replace_once(
        crew_source,
        '    if macd == "bullish":\n'
        '        buy_score += 2.0\n'
        '    elif macd == "bearish":\n'
        '        buy_score -= 2.0',
        '    if macd == "bullish":\n'
        '        buy_score += 2.0\n'
        '    elif macd == "bearish":\n'
        '        buy_score -= 2.0\n'
        '    elif has_macd_histogram and macd_histogram > 0:\n'
        '        # Histogram already positive without a fresh crossover: momentum is bullish.\n'
        '        buy_score += 1.5\n'
        '    elif has_macd_histogram and macd_histogram < 0:\n'
        '        # Mild penalty for negative momentum, not equivalent to a fresh bearish cross.\n'
        '        buy_score -= 0.5',
        'buy-score MACD momentum weighting',
    )

    crew_source = replace_once(
        crew_source,
        '    buy_confirmation = (macd == "bullish") or (0.15 <= bb <= 0.85)',
        '    macd_bullish_momentum = (macd == "bullish") or (has_macd_histogram and macd_histogram > 0)\n'
        '    buy_confirmation = macd_bullish_momentum or (0.15 <= bb <= 0.85)',
        'buy-confirmation condition',
    )

    crew_source = replace_once(
        crew_source,
        '    if macd == "bearish":\n'
        '        sell_score += 2.0\n'
        '    elif macd == "bullish":\n'
        '        sell_score -= 2.0\n'
        '    if rsi > 0 and rsi <= 40.0:',
        '    if macd == "bearish":\n'
        '        sell_score += 2.0\n'
        '    elif macd == "bullish":\n'
        '        sell_score -= 2.0\n'
        '    elif has_macd_histogram and macd_histogram < 0:\n'
        '        sell_score += 1.0\n'
        '    elif has_macd_histogram and macd_histogram > 0:\n'
        '        sell_score -= 0.5\n'
        '    if rsi > 0 and rsi <= 40.0:',
        'sell-score MACD momentum weighting',
    )

    crew_source = replace_once(
        crew_source,
        '        f"MACD={row.get(\'macd\', \'missing\')}, %B={round(bb, 3) if -10 < bb < 10 else \'missing\'}, "\n'
        '        f"relative_volume={relvol if relvol >= 0 else \'missing\'} (diagnostic only for 1Day bars)"',
        '        f"MACD crossover={row.get(\'macd\', \'missing\')}, "\n'
        '        f"hist={round(macd_histogram, 6) if has_macd_histogram else \'missing\'}, "\n'
        '        f"line={round(macd_line, 6) if abs(macd_line) < 1e11 else \'missing\'}, "\n'
        '        f"signal={round(macd_signal, 6) if abs(macd_signal) < 1e11 else \'missing\'}, "\n'
        '        f"%B={round(bb, 3) if -10 < bb < 10 else \'missing\'}, "\n'
        '        f"relative_volume={relvol if relvol >= 0 else \'missing\'} (diagnostic only for 1Day bars)"',
        'signal diagnostics MACD values',
    )

    crew_source = replace_once(
        crew_source,
        '    if not buy_confirmation:\n'
        '        why.append("no bullish MACD crossover or constructive Bollinger position")',
        '    if not buy_confirmation:\n'
        '        why.append("no positive MACD histogram/crossover or constructive Bollinger position")',
        'signal hold-reason MACD wording',
    )

    # Clarify the extra numeric fields in the model prompt where current wording matches V2.
    crew_source = replace_once(
        crew_source,
        '        "The macd field is a crossover category: \'Bullish\', \'Bearish\', or \'None\'; \'None\' means "\n'
        '        "no fresh crossover, not bearish. RSI is numeric and bb_percent_b is a Bollinger position "',
        '        "The macd field is a crossover category: \'Bullish\', \'Bearish\', or \'None\'; \'None\' means "\n'
        '        "no fresh crossover, not that momentum is missing. macd_line, macd_signal, and "\n'
        '        "macd_histogram are numeric; a positive histogram means MACD is above its signal line. "\n'
        '        "RSI is numeric and bb_percent_b is a Bollinger position "',
        'decision prompt MACD field explanation',
    )

    # Validate all content before changing disk.
    ast.parse(crew_source)
    ast.parse(indicator_source)

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    crew_backup = CREW.with_suffix(CREW.suffix + f'.{stamp}.bak')
    indicator_backup = INDICATORS.with_suffix(INDICATORS.suffix + f'.{stamp}.bak')
    shutil.copy2(CREW, crew_backup)
    shutil.copy2(INDICATORS, indicator_backup)
    CREW.write_text(crew_source, encoding='utf-8', newline='\n')
    INDICATORS.write_text(indicator_source, encoding='utf-8', newline='\n')
    # Parse what was actually written too.
    ast.parse(CREW.read_text(encoding='utf-8'))
    ast.parse(INDICATORS.read_text(encoding='utf-8'))
    print(f'Backup created: {crew_backup}')
    print(f'Backup created: {indicator_backup}')
    print('Updated crew/trading_crew.py and technical_indicators_mcp_server.py.')
    print('The snapshot now includes MACD line, signal and histogram; positive histogram is usable even without a fresh crossover.')
    print('Risk limits, sizing limits, Alpaca paper-only setting and model fallback order were not changed.')


if __name__ == '__main__':
    main()
