import re
from datetime import datetime
from pathlib import Path


def run_folder(base_dir: str, run_label: str | None = None) -> Path:
    label = run_label or datetime.now().strftime("%d%m%Y")
    folder = Path(base_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_agent_log(base_dir: str, agent_name: str, content: str) -> Path:
    now = datetime.now()
    date_part = now.strftime("%d%m%Y")
    time_part = now.strftime("%H-%M-%S-%f")

    safe_agent_name = re.sub(
        r"[^A-Za-z0-9_-]", "", agent_name.replace(" ", "")
    ) or "UnknownAgent"

    folder = run_folder(base_dir, date_part)
    path = folder / f"{safe_agent_name}{date_part}-{time_part}.md"
    path.write_text(str(content), encoding="utf-8")
    return path


def load_previous_decision_report(base_dir: str) -> str | None:
    base = Path(base_dir)
    if not base.exists():
        return None

    candidates = sorted(
        base.glob("*/PortfolioDecisionMaker*.md"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None

    return candidates[-1].read_text(encoding="utf-8")
