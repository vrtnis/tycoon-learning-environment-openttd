from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class CellFormula:
    formula: str
    cached: int | float | str = 0


@dataclass(frozen=True)
class Sheet:
    name: str
    rows: list[list[Any]]
    widths: dict[int, int] | None = None
    freeze_top_row: bool = True


def write_xlsx(path: str | Path, sheets: list[Sheet]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types(len(sheets)))
        archive.writestr("_rels/.rels", _root_rels())
        archive.writestr("docProps/core.xml", _core_props())
        archive.writestr("docProps/app.xml", _app_props(sheets))
        archive.writestr("xl/workbook.xml", _workbook_xml(sheets))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(sheets))
        archive.writestr("xl/styles.xml", _styles_xml())
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(sheet))
    return output


def read_xlsx(path: str | Path) -> dict[str, dict[str, Any]]:
    workbook_path = Path(path)
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _read_shared_strings(archive)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = _read_workbook_rels(archive)
        result: dict[str, dict[str, Any]] = {}
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            name = str(sheet.attrib["name"])
            rel_id = sheet.attrib[f"{{{NS_REL}}}id"]
            target = rels[rel_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            result[name] = _read_sheet(archive, target, shared_strings)
        return result


def cell_name(row: int, col: int) -> str:
    letters = ""
    value = col
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{row}"


def _content_types(sheet_count: int) -> str:
    overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  {overrides}
</Types>"""


def _root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def _core_props() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>OpenTTD-LE</dc:creator>
  <cp:lastModifiedBy>OpenTTD-LE</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


def _app_props(sheets: list[Sheet]) -> str:
    titles = "".join(f"<vt:lpstr>{_xml(sheet.name)}</vt:lpstr>" for sheet in sheets)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenTTD-LE</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="{len(sheets)}" baseType="lpstr">{titles}</vt:vector></TitlesOfParts>
</Properties>"""


def _workbook_xml(sheets: list[Sheet]) -> str:
    sheet_xml = "\n".join(
        f'<sheet name="{_xml(sheet.name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet in enumerate(sheets, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{NS_MAIN}" xmlns:r="{NS_REL}">
  <sheets>
    {sheet_xml}
  </sheets>
</workbook>"""


def _workbook_rels(sheets: list[Sheet]) -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    style_id = len(sheets) + 1
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_PACKAGE_REL}">
  {rels}
  <Relationship Id="rId{style_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="{NS_MAIN}">
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
    <font><b/><sz val="14"/><color rgb="FF111827"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E5F"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"/><right style="thin"/><top style="thin"/><bottom style="thin"/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="4">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="4" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _worksheet_xml(sheet: Sheet) -> str:
    row_xml = []
    for row_index, row in enumerate(sheet.rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            if value is None:
                continue
            style = "2" if row_index == 1 else "1" if row_index == 3 else "0"
            cells.append(_cell_xml(row_index, col_index, value, style))
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    widths = sheet.widths or {}
    max_col = max((len(row) for row in sheet.rows), default=1)
    col_xml = "".join(
        f'<col min="{index}" max="{index}" width="{widths.get(index, 18)}" customWidth="1"/>'
        for index in range(1, max_col + 1)
    )
    pane = (
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="3" topLeftCell="A4" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        if sheet.freeze_top_row
        else '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
    )
    dimension = f"A1:{cell_name(max(1, len(sheet.rows)), max(1, max_col))}"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS_MAIN}" xmlns:r="{NS_REL}">
  <dimension ref="{dimension}"/>
  {pane}
  <cols>{col_xml}</cols>
  <sheetData>
    {''.join(row_xml)}
  </sheetData>
</worksheet>"""


def _cell_xml(row: int, col: int, value: Any, style: str) -> str:
    ref = cell_name(row, col)
    if isinstance(value, CellFormula):
        cached = _xml(str(value.cached))
        formula = _xml(value.formula[1:] if value.formula.startswith("=") else value.formula)
        return f'<c r="{ref}" s="{style}"><f>{formula}</f><v>{cached}</v></c>'
    if isinstance(value, bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
    text = _xml(str(value))
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{text}</t></is></c>'


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(f"{{{NS_MAIN}}}si"):
        parts = [node.text or "" for node in item.findall(f".//{{{NS_MAIN}}}t")]
        strings.append("".join(parts))
    return strings


def _read_workbook_rels(archive: zipfile.ZipFile) -> dict[str, str]:
    root = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rels = {}
    for rel in root.findall(f"{{{NS_PACKAGE_REL}}}Relationship"):
        rels[rel.attrib["Id"]] = rel.attrib["Target"]
    return rels


def _read_sheet(
    archive: zipfile.ZipFile,
    target: str,
    shared_strings: list[str],
) -> dict[str, Any]:
    root = ElementTree.fromstring(archive.read(target))
    cells: dict[str, Any] = {}
    rows: list[list[Any]] = []
    max_row = 0
    max_col = 0
    for cell in root.findall(f".//{{{NS_MAIN}}}c"):
        ref = cell.attrib["r"]
        row_index, col_index = _split_cell(ref)
        max_row = max(max_row, row_index)
        max_col = max(max_col, col_index)
        cells[ref] = _read_cell_value(cell, shared_strings)
    for row_index in range(1, max_row + 1):
        row = []
        for col_index in range(1, max_col + 1):
            row.append(cells.get(cell_name(row_index, col_index)))
        rows.append(row)
    return {"cells": cells, "rows": rows}


def _read_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        text_node = cell.find(f".//{{{NS_MAIN}}}t")
        return text_node.text if text_node is not None else ""
    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if value_node is None:
        return None
    value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(value)]
    if cell_type == "b":
        return value == "1"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _split_cell(ref: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {ref}")
    letters, row = match.groups()
    col = 0
    for char in letters:
        col = col * 26 + ord(char) - 64
    return int(row), col


def _xml(value: str) -> str:
    return html.escape(value, quote=True)
