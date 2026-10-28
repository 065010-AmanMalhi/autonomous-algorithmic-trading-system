from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import sys

ROOT = Path.cwd()
CREW = ROOT / "crew" / "trading_crew.py"
PATCHER = ROOT / "apply_model_upgrade.py"


def main() -> int:
    if not CREW.is_file() or not PATCHER.is_file():
        print("Run this from the project root, after extracting the corrected bundle.")
        print("Expected both crew/trading_crew.py and apply_model_upgrade.py.")
        return 2

    # Confirm the corrected patcher uses callable regex replacements, preventing
    # re.sub from interpreting \n in generated Python source as real line breaks.
    patcher_source = PATCHER.read_text(encoding="utf-8")
    if "lambda _: NEW_DESCRIPTION" not in patcher_source:
        print("The corrected apply_model_upgrade.py is not installed. Extract the bundle again.")
        return 3

    backups = sorted(
        (ROOT / "crew").glob("trading_crew.py.*.bak"),
        key=lambda p: p.name,
        reverse=True,
    )
    valid_backup = None
    valid_source = None
    for candidate in backups:
        try:
            source = candidate.read_text(encoding="utf-8")
            compile(source, str(candidate), "exec")
            valid_backup, valid_source = candidate, source
            break
        except (OSError, SyntaxError, UnicodeError):
            continue

    if valid_backup is None or valid_source is None:
        print("No syntax-valid trading_crew.py backup was found; no files were changed.")
        print("Keep the current file and share the backup filenames/error output for recovery.")
        return 4

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    broken_copy = CREW.with_name(f"trading_crew.py.syntax-error-{stamp}.bak")
    shutil.copy2(CREW, broken_copy)
    CREW.write_text(valid_source, encoding="utf-8", newline="\n")
    print(f"Preserved broken file: {broken_copy}")
    print(f"Restored syntax-valid source from: {valid_backup}")

    result = subprocess.run([sys.executable, str(PATCHER)], cwd=ROOT)
    if result.returncode != 0:
        print("Corrected patcher exited with an error. The restored source backup remains available.")
        return result.returncode

    try:
        compile(CREW.read_text(encoding="utf-8"), str(CREW), "exec")
    except (OSError, SyntaxError) as exc:
        print(f"Post-patch syntax check failed: {exc}")
        return 5

    print("Recovery and patch completed; trading_crew.py passes Python syntax validation.")
    print("Next, review `git diff` and test one cycle before restarting any scheduler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
