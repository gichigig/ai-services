"""
Academic PDF Generator Module
Uses ReportLab to generate university-grade assignment documents with headers, styling, callouts, and page numbering.
"""

import os
import re
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that automatically adds 'Page X of Y' and header/footer rules.
    """
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        
        # Header (pages after first)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(54, 11 * inch - 36, "BRUV ACADEMIC AUTOMATION SUITE")
            self.setFont("Helvetica", 8)
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "STUDENT SUBMISSION REPORT")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(54, 46, 8.5 * inch - 54, 46)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 32, "Confidential — Academic Work Submission")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 32, page_text)
        
        self.restoreState()


class AcademicPDFGenerator:
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Sanitizes filename against path traversal."""
        cleaned = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.basename(name))
        return cleaned or "assignment_submission.pdf"

    @classmethod
    def generate(cls, data: Dict[str, Any], output_dir: str = "output") -> str:
        """
        Builds and saves the PDF file from the structured assignment data.
        Returns the absolute filepath of the generated PDF.
        """
        os.makedirs(output_dir, exist_ok=True)
        safe_name = cls.sanitize_filename(f"{data.get('unit_code', 'UNIT')}_{data.get('assignment_title', 'Assignment')}.pdf")
        pdf_path = os.path.abspath(os.path.join(output_dir, safe_name))

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#0284c7'),
            spaceAfter=12
        )
        
        heading_style = ParagraphStyle(
            'SectionHead',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=colors.HexColor('#1e3a8a'),
            spaceBefore=14,
            spaceAfter=6
        )

        q_head_style = ParagraphStyle(
            'QuestionHead',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=15,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14.5,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8
        )
        
        code_style = ParagraphStyle(
            'CodeText',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor('#0f172a')
        )

        story = []

        # Document Header
        story.append(Paragraph(f"{data.get('unit_code', 'UNIT 101')}: {data.get('unit_name', 'Course Unit')}", subtitle_style))
        story.append(Paragraph(f"{data.get('assignment_title', 'Course Assignment')}", title_style))
        story.append(Spacer(1, 8))

        # Student & Submission Metadata Table
        meta_data = [
            [
                Paragraph("<b>Student Name:</b>", body_style),
                Paragraph(data.get('student_name', 'Student'), body_style),
                Paragraph("<b>Admission No:</b>", body_style),
                Paragraph(data.get('admission_no', 'ADM/2026/001'), body_style),
            ],
            [
                Paragraph("<b>Date of Submission:</b>", body_style),
                Paragraph(data.get('date', 'August 2026'), body_style),
                Paragraph("<b>Status:</b>", body_style),
                Paragraph("<font color='#059669'><b>Original Academic Solution</b></font>", body_style),
            ]
        ]
        meta_table = Table(meta_data, colWidths=[110, 150, 110, 134])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=12))

        # Render Content Sections
        for section in data.get('sections', []):
            sec_type = section.get('type')
            
            if sec_type == 'heading':
                story.append(Paragraph(section.get('title', ''), heading_style))
                story.append(Paragraph(section.get('content', '').replace('\n', '<br/>'), body_style))
                story.append(Spacer(1, 8))
                
            elif sec_type == 'question_block':
                q_elements = []
                q_num = section.get('q_number', 'Question')
                q_text = section.get('question', '')
                q_elements.append(Paragraph(f"<b>{q_num}: {q_text}</b>", q_head_style))
                q_elements.append(Spacer(1, 4))
                
                # Question Box
                q_table = Table([[q_elements]], colWidths=[504])
                q_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eff6ff')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bfdbfe')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(q_table)
                story.append(Spacer(1, 6))

                # Optional Code snippet
                if section.get('code'):
                    code_snippet = section.get('code', '')
                    code_flowable = Preformatted(code_snippet, code_style)
                    code_table = Table([[code_flowable]], colWidths=[504])
                    code_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
                        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#334155')),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    story.append(code_table)
                    story.append(Spacer(1, 6))

                # Solution text
                sol_text = section.get('solution', '')
                # Format bold markdown and linebreaks
                formatted_sol = sol_text.replace('\n', '<br/>')
                formatted_sol = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_sol)
                story.append(Paragraph(formatted_sol, body_style))
                story.append(Spacer(1, 10))

        doc.build(story, canvasmaker=NumberedCanvas)
        return pdf_path
