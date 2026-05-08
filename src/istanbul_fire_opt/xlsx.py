"""Small XLSX reader used when openpyxl is not installed.

The project data files are simple rectangular sheets. This parser intentionally
handles only the subset required for those files and returns pandas DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import pandas as pd


NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def read_xlsx(path: str | Path, sheet_index: int = 0) -> pd.DataFrame:
    """Read one worksheet from an XLSX file into a DataFrame.

    If pandas can use openpyxl, it is preferred. Otherwise, a lightweight XML
    parser reads the first sheet.
    """

    path = Path(path)
    try:
        return pd.read_excel(path, sheet_name=sheet_index)
    except ImportError:
        return _read_xlsx_without_openpyxl(path, sheet_index)


def _read_xlsx_without_openpyxl(path: Path, sheet_index: int) -> pd.DataFrame:
    with ZipFile(path) as archive:
        shared = _read_shared_strings(archive)
        sheet_path = _sheet_path(archive, sheet_index)
        rows = _read_sheet_rows(archive, sheet_path, shared)
    rows = _trim_empty_rows(rows)
    if not rows:
        return pd.DataFrame()
    header_index = next((idx for idx, row in enumerate(rows) if any(str(v).strip() for v in row)), 0)
    header = _dedupe_columns([str(v).strip() or f"column_{i}" for i, v in enumerate(rows[header_index])])
    data = rows[header_index + 1 :]
    width = len(header)
    normalized = [(row + [""] * width)[:width] for row in data]
    return pd.DataFrame(normalized, columns=header)


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", NS):
        text = "".join(node.text or "" for node in item.findall(".//a:t", NS))
        strings.append(text)
    return strings


def _sheet_path(archive: ZipFile, sheet_index: int) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    sheets = workbook.findall("a:sheets/a:sheet", NS)
    if sheet_index >= len(sheets):
        raise IndexError(f"sheet_index {sheet_index} is out of range")
    rel_id = sheets[sheet_index].attrib[f"{{{NS['r']}}}id"]
    rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", NS)}
    target = rel_targets[rel_id]
    if target.startswith("/"):
        return target.lstrip("/")
    return f"xl/{target}".replace("xl//", "xl/")


def _read_sheet_rows(archive: ZipFile, sheet_path: str, shared: list[str]) -> list[list[object]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[object]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: list[object] = []
        for cell in row.findall("a:c", NS):
            idx = _column_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = _cell_value(cell, shared)
        rows.append(values)
    return rows


def _cell_value(cell: ET.Element, shared: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", NS)
    if value is None:
        inline = cell.find("a:is/a:t", NS)
        return inline.text if inline is not None else ""
    raw = value.text or ""
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type == "b":
        return raw == "1"
    try:
        numeric = float(raw)
        return int(numeric) if numeric.is_integer() else numeric
    except ValueError:
        return raw


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if not letters:
        return 0
    total = 0
    for char in letters.group(0):
        total = total * 26 + ord(char) - ord("A") + 1
    return total - 1


def _trim_empty_rows(rows: list[list[object]]) -> list[list[object]]:
    trimmed = []
    for row in rows:
        while row and row[-1] == "":
            row.pop()
        if any(str(value).strip() for value in row):
            trimmed.append(row)
    return trimmed


def _dedupe_columns(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count}")
    return result
