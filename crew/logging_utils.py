"""
logging_utils.py
-----------------
Implements the brief's exact logging requirement (section 8):

    "Instruct coding agent that output of each agent is saved in an md
    file before, say, a model-run, folder in the format:
    agentNameDDMMYYYY-HH:MM.md"

Also implements the "improve-as-we-go" feedback loop: load the most
recent previous DECISION-MAKER log so today's run can read yesterday's
reasoning and results before committing to today's strategy.
"""

import os
import re
from datetime import datetime
from pathlib import Path


def run_folder(base_dir: str, run_label: str | None = None) -> Path:
    """
    One folder per run (crew.kickoff() cycle), nested under base_dir.
    run_label defaults to today's date so all cycles in one trading
    session land in the same day's folder — makes the "compare today vs
    yesterday" step trivial.
    """
    label = run_label or datetime.now().strftime("%d%m%Y")
    folder = Path(base_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_agent_log(base_dir: str, agent_name: str, content: str) -> Path:
    """
    Save one agent's output as agentNameDDMMYYYY-HH:MM.md, exactly the
    naming the brief specifies. Colons aren't filesystem-safe on Windows,
    so we substitute them with a hyphen there but keep the brief's literal
    format on Linux/Mac (adjust FORCE_WINDOWS_SAFE below if you're on
    Windows and hit filename errors).
    """
    FORCE_WINDOWS_SAFE = False

    now = datetime.now()
    date_part = now.strftime("%d%m%Y")
    time_part = now.strftime("%H-%M" if FORCE_WINDOWS_SAFE else "%H:%M")

    safe_agent_name = re.sub(r"[^A-Za-z0-9_-]", "", agent_name.replace(" ", ""))
    filename = f"{safe_agent_name}{date_part}-{time_part}.md"

    folder = run_folder(base_dir)
    path = folder / filename
    path.write_text(content, encoding="utf-8")
    return path


def load_previous_decision_report(base_dir: str) -> str | None:
    """
    Find the most recent PortfolioDecisionMaker log (any previous day,
    most recent cycle) and return its content, for feeding into today's
    decision-maker task as "yesterday's results" context. Returns None if
    no prior run exists yet (first-ever run of the session).
    """
    base = Path(base_dir)
    if not base.exists():
        return None

    candidates = sorted(base.glob("*/PortfolioDecisionMaker*.md"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None

    latest = candidates[-1]
    return latest.read_text(encoding="utf-8")
