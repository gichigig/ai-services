"""
Academic PDF Compiler Module for Bluv AI.
Converts markdown responses, revision guides, code, and academic solutions
into professionally typeset, multi-page PDF documents using ReportLab.
"""

import os
import re
import tempfile
import html
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


def to_snake_case(text: str) -> str:
    """
    Sanitizes arbitrary text into clean snake_case suitable for filesystems and download links.
    Example: 'Operating Systems: Concurrency & Semaphores (Part 1)' -> 'operating_systems_concurrency_semaphores_part_1.pdf'
    """
    if not text:
        return "bluv_ai_study_notes.pdf"
    
    # Strip non-alphanumeric characters (keep spaces and underscores)
    cleaned = re.sub(r'[^a-zA-Z0-9\s_-]', '', text)
    # Convert spaces and dashes to underscores
    cleaned = re.sub(r'[\s-]+', '_', cleaned)
    # Collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned).strip('_').lower()
    
    # Limit length to 60 chars
    if len(cleaned) > 60:
        cleaned = cleaned[:60].rstrip('_')
        
    if not cleaned:
        cleaned = "bluv_ai_study_notes"
        
    return f"{cleaned}.pdf"


class TopicNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that renders 'Page X of Y', a running header displaying
    the question topic, and an official academic footer.
    """
    def __init__(self, *args, **kwargs):
        self.topic_title = kwargs.pop("topic_title", "ACADEMIC STUDY NOTES")
        super(TopicNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(TopicNumberedCanvas, self).showPage()
        super(TopicNumberedCanvas, self).save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        
        # Running Header on page 2 and later
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#5C33CF"))
            topic_header = (self.topic_title[:55] + "...") if len(self.topic_title) > 55 else self.topic_title
            self.drawString(54, 11 * inch - 36, topic_header.upper())
            
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#797693"))
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "CAMPUSCONNECT • BLUV AI")
            
            self.setStrokeColor(colors.HexColor("#EFEFF8"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Running Footer on all pages
        self.setStrokeColor(colors.HexColor("#EFEFF8"))
        self.setLineWidth(0.75)
        self.line(54, 46, 8.5 * inch - 54, 46)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#797693"))
        self.drawString(54, 32, "CampusConnect Academic Companion • Confidential Study Material")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, page_text)
        
        self.restoreState()


def format_inline_markdown(text: str) -> str:
    """
    Safely converts inline markdown (**bold**, *italic*, `code`) into ReportLab XML formatting.
    """
    # Escape raw XML entities first
    text = html.escape(text)
    
    # Inline code: `code` -> <font face="Courier" color="#5C33CF">code</font>
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" color="#5C33CF"><b>\1</b></font>', text)
    
    # Bold: **text** or __text__ -> <b>text</b>
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__([^_]+)__', r'<b>\1</b>', text)
    
    # Italic: *text* or _text_ -> <i>text</i>
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'<i>\1</i>', text)
    
    # Replace newlines inside paragraph with break
    text = text.replace('\n', '<br/>')
    
    return text


def build_pdf_story(title: str, markdown_content: str, styles: Any) -> List[Any]:
    """
    Parses full markdown document into a list of ReportLab Flowables.
    Handles headers, bullet lists, numbered lists, blockquotes, code blocks, tables, and paragraphs.
    """
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E1B39'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#5C33CF'),
        spaceAfter=2
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#797693'),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1E1B39'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#5C33CF'),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1E1B39'),
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569')
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1E1B39')
    )

    # 1. Document Header Block
    story.append(Paragraph("CAMPUSCONNECT ACADEMIC COMPANION", subtitle_style))
    story.append(Paragraph(html.escape(title), title_style))
    created_at = datetime.now().strftime("%B %d, %Y • %I:%M %p")
    story.append(Paragraph(f"Compiled by Bluv AI on {created_at}", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#5C33CF"), spaceAfter=12, spaceBefore=2))

    # 2. Parse Markdown Lines
    lines = markdown_content.splitlines()
    in_code_block = False
    code_buffer: List[str] = []
    in_table = False
    table_buffer: List[List[str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle Code Fences
        if stripped.startswith("```"):
            if in_code_block:
                # Flush code block
                code_text = html.escape("\n".join(code_buffer))
                p_code = Paragraph(f"<font face='Courier'>{code_text.replace(chr(10), '<br/>')}</font>", code_style)
                code_table = Table([[p_code]], colWidths=[7.0 * inch])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#E2E8F0')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(Spacer(1, 4))
                story.append(code_table)
                story.append(Spacer(1, 6))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # Handle Tables
        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
            # Check if separator row like |---|---|
            if re.match(r'^\|?[\s:-]+\|[\s|:-]+$', stripped):
                i += 1
                continue
            
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if cells:
                table_buffer.append(cells)
                in_table = True
            i += 1
            continue
        elif in_table:
            # End of table, flush table buffer
            if table_buffer:
                max_cols = max(len(row) for row in table_buffer)
                col_width = (7.0 * inch) / max_cols if max_cols > 0 else 7.0 * inch
                table_data = []
                for r_idx, row in enumerate(table_buffer):
                    formatted_row = []
                    # Pad cells if necessary
                    padded = row + [""] * (max_cols - len(row))
                    for c_idx, cell in enumerate(padded):
                        is_header = (r_idx == 0)
                        p_style = ParagraphStyle(
                            f"Cell_{r_idx}_{c_idx}",
                            parent=body_style,
                            fontName="Helvetica-Bold" if is_header else "Helvetica",
                            fontSize=8.5,
                            leading=11,
                            textColor=colors.HexColor("#5C33CF") if is_header else colors.HexColor("#1E1B39")
                        )
                        formatted_row.append(Paragraph(format_inline_markdown(cell), p_style))
                    table_data.append(formatted_row)
                
                t = Table(table_data, colWidths=[col_width] * max_cols)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0EEFC')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#EFEFF8')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            table_buffer = []
            in_table = False

        # Empty line
        if not stripped:
            i += 1
            continue

        # Horizontal rules
        if stripped in ("---", "***", "___"):
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#EFEFF8"), spaceAfter=8, spaceBefore=8))
            i += 1
            continue

        # Headings
        if stripped.startswith("### "):
            story.append(Paragraph(format_inline_markdown(stripped[4:]), h3_style))
            i += 1
            continue
        elif stripped.startswith("## "):
            story.append(Paragraph(format_inline_markdown(stripped[3:]), h2_style))
            i += 1
            continue
        elif stripped.startswith("# "):
            story.append(Paragraph(format_inline_markdown(stripped[2:]), h1_style))
            i += 1
            continue

        # Blockquote (> text)
        if stripped.startswith(">"):
            quote_text = stripped.lstrip(">").strip()
            p_quote = Paragraph(format_inline_markdown(quote_text), callout_style)
            quote_table = Table([[p_quote]], colWidths=[7.0 * inch])
            quote_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0EEFC')),
                ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor('#5C33CF')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(quote_table)
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Unordered Lists (*, -)
        if re.match(r'^[\*\-]\s+', stripped):
            item_text = re.sub(r'^[\*\-]\s+', '', stripped)
            formatted = f"•&nbsp;&nbsp;{format_inline_markdown(item_text)}"
            story.append(Paragraph(formatted, bullet_style))
            i += 1
            continue

        # Ordered Lists (1., 2.)
        num_match = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if num_match:
            num = num_match.group(1)
            item_text = num_match.group(2)
            formatted = f"<b>{num}.</b>&nbsp;&nbsp;{format_inline_markdown(item_text)}"
            story.append(Paragraph(formatted, bullet_style))
            i += 1
            continue

        # Standard Paragraph
        story.append(Paragraph(format_inline_markdown(stripped), body_style))
        i += 1

    # Flush any remaining table
    if table_buffer:
        max_cols = max(len(row) for row in table_buffer)
        col_width = (7.0 * inch) / max_cols if max_cols > 0 else 7.0 * inch
        table_data = []
        for r_idx, row in enumerate(table_buffer):
            formatted_row = []
            padded = row + [""] * (max_cols - len(row))
            for c_idx, cell in enumerate(padded):
                is_header = (r_idx == 0)
                p_style = ParagraphStyle(
                    f"CellEnd_{r_idx}_{c_idx}",
                    parent=body_style,
                    fontName="Helvetica-Bold" if is_header else "Helvetica",
                    fontSize=8.5,
                    leading=11,
                    textColor=colors.HexColor("#5C33CF") if is_header else colors.HexColor("#1E1B39")
                )
                formatted_row.append(Paragraph(format_inline_markdown(cell), p_style))
            table_data.append(formatted_row)
        
        t = Table(table_data, colWidths=[col_width] * max_cols)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0EEFC')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#EFEFF8')),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    return story


def compile_pdf_document(title: str, markdown_content: str, output_path: str):
    """
    Builds and writes a PDF file from title and markdown content to output_path.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    story = build_pdf_story(title, markdown_content, styles)
    
    # Custom Canvas factory with topic title
    def canvas_maker(*args, **kwargs):
        return TopicNumberedCanvas(*args, topic_title=title, **kwargs)

    doc.build(story, canvasmaker=canvas_maker)


def compile_markdown_to_pdf_and_upload(title: str, markdown_content: str, backend_url: str = "https://school.ishinadwelly.com") -> Optional[Dict[str, str]]:
    """
    Compiles markdown to PDF with sanitized snake_case filename, uploads it to backend,
    and returns a dict with filename and download URL.
    Returns: {"filename": "operating_systems_summary.pdf", "url": "https://school.ishinadwelly.com/api/v1/files/..."}
    """
    safe_filename = to_snake_case(title)
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="bluv_pdf_")
    local_pdf_path = os.path.join(temp_dir, safe_filename)
    
    try:
        compile_pdf_document(title, markdown_content, local_pdf_path)
        
        if not os.path.exists(local_pdf_path) or os.path.getsize(local_pdf_path) == 0:
            print("PDF generation failed: Output file is empty or missing")
            return None
            
        # Upload to Spring Boot backend
        upload_url = f"{backend_url}/api/v1/upload"
        with open(local_pdf_path, "rb") as f:
            files = {"file": (safe_filename, f, "application/pdf")}
            response = requests.post(upload_url, files=files, timeout=30)
            
        if response.status_code == 200:
            res_data = response.json()
            relative_url = res_data.get("url") or res_data.get("fileUrl")
            if relative_url:
                final_url = f"{backend_url}{relative_url}" if relative_url.startswith("/") else relative_url
                return {
                    "filename": safe_filename,
                    "url": final_url,
                    "title": title
                }
        else:
            print(f"Backend PDF upload failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error compiling or uploading PDF: {e}")
    finally:
        # Cleanup
        try:
            if os.path.exists(local_pdf_path):
                os.remove(local_pdf_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass
            
    return None
