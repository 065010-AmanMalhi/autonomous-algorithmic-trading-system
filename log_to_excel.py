import sys
import json
from datetime import datetime
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
OUT = BASE / "metrics_data" / "trading_logs.xlsx"
OUT.parent.mkdir(parents=True, exist_ok=True)

payload = json.load(sys.stdin)

if OUT.exists():
    wb = load_workbook(OUT)
    ws = wb["Logs"] if "Logs" in wb.sheetnames else wb.create_sheet("Logs")
else:
    wb = Workbook()
    ws = wb.active
    ws.title = "Logs"

if ws.max_row == 1 and ws["A1"].value is None:
    headers = ["Timestamp", "Account", "Source", "Log Level", "Message"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(wrap_text=True)

for item in payload.get("entries", []):
    message = str(item.get("message", ""))
    upper = message.upper()

    if "TRACEBACK" in upper or "ERROR" in upper or "EXCEPTION" in upper:
        level = "ERROR"
    elif "WARNING" in upper or "WARN" in upper:
        level = "WARNING"
    else:
        level = "INFO"

    ws.append([
        datetime.now().astimezone().isoformat(timespec="seconds"),
        payload.get("account", "Unknown"),
        payload.get("source", "PowerShell"),
        level,
        message
    ])

ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

for col, width in {"A": 28, "B": 14, "C": 30, "D": 12, "E": 100}.items():
    ws.column_dimensions[col].width = width

for row in ws.iter_rows(min_row=2):
    row[4].alignment = Alignment(wrap_text=True, vertical="top")

wb.save(OUT)
print(f"Excel updated: {OUT}")
