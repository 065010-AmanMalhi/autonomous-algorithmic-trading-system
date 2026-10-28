from pathlib import Path
import re
import shutil
import sys
from datetime import datetime

ROOT = Path.cwd()
CREW = ROOT / "crew" / "trading_crew.py"

FALLBACK_FUNCTIONS = r'''
def _validate_model_output(raw: str, symbols: list[str]) -> None:
    parsed = _extract_json_object(raw)
    rows = parsed["decisions"]
    if len(rows) != len(symbols):
        raise ValueError(f"Expected {len(symbols)} decisions, received {len(rows)}")
    by_symbol = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("A decision entry is not an object")
        symbol = str(item.get("symbol", "")).upper().strip()
        if symbol not in symbols or symbol in by_symbol:
            raise ValueError(f"Unknown or duplicate symbol: {symbol!r}")
        action = str(item.get("action", "")).lower().strip()
        confidence = _number(item.get("confidence"), -1)
        notional = _number(item.get("notional"), -1)
        if action not in {"buy", "sell", "hold"}:
            raise ValueError(f"{symbol}: invalid action")
        if not 0 <= confidence <= 1:
            raise ValueError(f"{symbol}: invalid confidence")
        if notional < 0:
            raise ValueError(f"{symbol}: invalid notional")
        if action == "buy" and notional <= 0:
            raise ValueError(f"{symbol}: BUY requires positive notional")
        if action == "hold" and notional != 0:
            raise ValueError(f"{symbol}: HOLD must have zero notional")
        by_symbol[symbol] = item
    if set(by_symbol) != set(symbols):
        raise ValueError("Response is missing portfolio symbols")
    if all(
        str(item.get("action", "")).lower().strip() == "hold"
        and _number(item.get("confidence"), -1) == 0.5
        and _number(item.get("notional"), -1) == 0
        and str(item.get("reason", "")).strip().lower() == "mixed signals"
        for item in rows
    ):
        raise ValueError("Detected repeated example-style HOLD response")


def _run_decision_with_fallback(
    cfg: dict, task_description: str, symbols: list[str], api_key: str
) -> tuple[str, str, list[str]]:
    llm_cfg = cfg["llm"]
    candidates = [llm_cfg.get("model"), *llm_cfg.get("fallback_models", [])]
    models = list(dict.fromkeys(
        model.strip() for model in candidates
        if isinstance(model, str) and model.strip()
    ))
    if not models:
        raise RuntimeError("No LLM models are configured")
    errors = []
    for model_name in models:
        try:
            llm = LLM(
                model=model_name, api_key=api_key,
                temperature=float(llm_cfg.get("temperature", 0.2)),
                max_tokens=int(llm_cfg.get("max_tokens", 1800)),
            )
            agent = Agent(
                role="Compact Portfolio Decision-Maker",
                goal="Return an evidence-based JSON portfolio decision set.",
                backstory=(
                    "You are conservative and evidence-based. You operate on an "
                    "Alpaca PAPER account. Missing or contradictory data means HOLD."
                ),
                llm=llm, verbose=False, max_iter=1, allow_delegation=False,
            )
            task = Task(
                description=task_description,
                expected_output="Only valid JSON with one decision per portfolio symbol.",
                agent=agent,
            )
            crew = Crew(
                agents=[agent], tasks=[task],
                process=Process.sequential, verbose=False,
            )
            raw = str(crew.kickoff())
            _validate_model_output(raw, symbols)
            if errors:
                print(
                    f"  Model fallback succeeded: {model_name} "
                    f"after {len(errors)} failed attempt(s)."
                )
            return raw, model_name, errors
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:220]}"
            errors.append(f"{model_name} -> {message}")
            print(f"  [LLM fallback] {model_name} failed: {message}")
    raise RuntimeError(
        "All configured LLM models failed validation/access; no orders submitted. "
        + " | ".join(errors)
    )
'''

NEW_DESCRIPTION = r'''    description = (
        "Make one conservative decision pass for an Alpaca PAPER account. "
        "Use only the supplied market snapshot, account details, positions, and risk limits. "
        "Never invent prices, signals, or missing data. Interpret trend='up' and MACD > 0 "
        "as bullish evidence, and trend='down' and MACD < 0 as bearish evidence. The snapshot "
        "does not provide a MACD signal line or crossover event, so do not claim that a "
        "crossover occurred. Use RSI, rsi_signal, Bollinger position, and relative volume as "
        "additional evidence where available. Consider BUY only when trend and MACD evidence "
        "are bullish, RSI is not overbought (prefer 35-68), other available indicators do not "
        "strongly contradict the setup, and account cash and position headroom allow the trade. "
        "Consider SELL only to reduce an existing long position when available evidence is "
        "bearish. Never short, use leverage, or propose options. If signals are weak, "
        "contradictory, or missing, choose HOLD; do not trade just to create activity. "
        "Confidence must reflect the strength of the actual evidence. Do not default every "
        "symbol to confidence 0.5 or copy a sample decision. Buy confidence must meet the "
        "configured threshold. Orders must respect the supplied position limit and cash reserve. "
        "Limit each reason to 8 words. Return only a valid JSON object with one key, 'decisions'. "
        "Its value must be an array containing exactly one object for every supplied symbol. "
        "Each object must contain symbol, action, notional, confidence, and reason. Symbol must "
        "match the supplied symbol; action must be buy, sell, or hold; confidence must be between "
        "0 and 1; notional must be non-negative; HOLD notional must be zero; BUY notional must be "
        "positive. Do not include markdown or text outside the JSON object.\n\n"
        "DATA:\n" + json.dumps(context, separators=(",", ":"), ensure_ascii=True)
    )
    return description'''

MODEL_BLOCK_PATTERN = re.compile(
    r'''        llm = LLM\(
.*?        raw_decision = str\(crew\.kickoff\(\)\)
''', re.S
)

def backup(path: Path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, target)
    print(f"Backup: {target}")

def update_yaml(path: Path):
    if not path.exists():
        print(f"Skip missing config: {path}")
        return
    original = path.read_text(encoding="utf-8")
    pattern = re.compile(r'(?ms)^llm:\n(?:^[ \t].*\n|^\n)*')
    replacement = '''llm:
  provider: "groq"
  model: "groq/openai/gpt-oss-120b"
  fallback_models:
    - "groq/qwen/qwen3.8-27b"
    - "groq/openai/gpt-oss-20b"
  temperature: 0.2
  max_tokens: 1800

'''
    changed, count = pattern.subn(lambda _: replacement, original, count=1)
    if count:
        backup(path)
        path.write_text(changed, encoding="utf-8")
        print(f"Updated model priority in {path}")
    else:
        print(f"Could not locate llm: block in {path}; left unchanged")

def main():
    if not CREW.exists():
        print("Run this patcher from your project root (the folder containing crew/ and config/).")
        sys.exit(2)
    source = CREW.read_text(encoding="utf-8")
    changed = source

    start = changed.find("def _decision_task_description(")
    end = changed.find(chr(10) + "def run_one_cycle(", start)
    if start < 0 or end < 0:
        print("Could not locate decision prompt function; no code changes made.")
        sys.exit(3)
    section = changed[start:end]
    section2, n = re.subn(
        r'(?ms)^    description = \(.*?^    return description$',
        lambda _: NEW_DESCRIPTION, section, count=1,
    )
    if n != 1:
        print("Could not safely replace decision prompt; no code changes made.")
        sys.exit(4)
    changed = changed[:start] + section2 + changed[end:]

    changed2, n = MODEL_BLOCK_PATTERN.subn(
        lambda _: '''        task_description = _decision_task_description(
            cfg, snapshot, account, positions, trading_halted
        )
        raw_decision, selected_model, model_attempt_errors = (
            _run_decision_with_fallback(
                cfg=cfg, task_description=task_description,
                symbols=symbols, api_key=api_key,
            )
        )
''',
        changed, count=1,
    )
    if n != 1:
        print("Could not safely replace single-model block; no code changes made.")
        sys.exit(5)
    changed = changed2

    if "def _run_decision_with_fallback(" not in changed:
        marker = chr(10) + "def run_one_cycle("
        if marker not in changed:
            print("Could not find insertion point for fallback helpers.")
            sys.exit(6)
        changed = changed.replace(marker, chr(10) + FALLBACK_FUNCTIONS + marker, 1)

    changed = changed.replace(
        '''                "safety_before_order": safety,
                "pre_order_validation_notes": validation_notes,''',
        '''                "safety_before_order": safety,
                "selected_model": selected_model,
                "model_attempt_errors": model_attempt_errors,
                "pre_order_validation_notes": validation_notes,''',
        1,
    )
    changed = changed.replace(
        '''f"  LLM calls: 1 | model: {llm_cfg['model']} | Alpaca PAPER only\\n"''',
        '''f"  LLM attempts: {len(model_attempt_errors) + 1} | "
            f"selected model: {selected_model} | Alpaca PAPER only\\n"''',
        1,
    )

    backup(CREW)
    CREW.write_text(changed, encoding="utf-8")
    print("Updated crew/trading_crew.py")
    update_yaml(ROOT / "config" / "aman.yaml")
    update_yaml(ROOT / "config" / "config.example.yaml")
    print("\\nPatch complete. Review the diff, then run:")
    print("  python -m py_compile .\\\\crew\\\\trading_crew.py")
    print("  python .\\\\scripts\\\\run_session.py --config .\\\\config\\\\aman.yaml --once")
    print("\\nThe patch does not change Alpaca paper mode or risk limits.")

if __name__ == "__main__":
    main()
