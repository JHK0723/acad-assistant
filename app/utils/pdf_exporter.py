"""
PDF Export Utility for Academic Assistant
Generates well-formatted PDF reports from parsing and summarization results
"""
import io
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, grey
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import structlog

from app.models import NotesParseResponse, SummaryResponse, KeywordExtraction, ConceptExtraction, StudyQuestion

logger = structlog.get_logger()


class PDFExporter:
    """Handles PDF generation for academic content analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for better PDF formatting"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            textColor=HexColor('#2C3E50'),
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=HexColor('#34495E'),
            borderWidth=1,
            borderColor=HexColor('#BDC3C7'),
            borderPadding=5,
            leftIndent=0,
            rightIndent=0
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=HexColor('#2980B9')
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceBefore=3,
            spaceAfter=3,
            bulletIndent=10
        ))
        
        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=grey,
            spaceBefore=5,
            spaceAfter=5
        ))
    
    def export_parse_results(self, result: NotesParseResponse, original_filename: str = None) -> bytes:
        """Export parsing and summarization results to PDF"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the PDF content
        story = []
        
        # Title
        title = "Academic Content Analysis Report"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Metadata section
        story.append(self._create_metadata_section(result, original_filename))
        story.append(Spacer(1, 20))
        
        # Summary section (main content)
        if result.parsed_content:
            story.append(Paragraph("Summary", self.styles['SectionHeader']))
            story.append(self._format_summary_content(result.parsed_content))
            story.append(Spacer(1, 15))
        
        # Keywords section
        if result.keywords:
            story.append(Paragraph("Key Terms", self.styles['SectionHeader']))
            story.append(self._create_keywords_table(result.keywords))
            story.append(Spacer(1, 15))
        
        # Concepts section
        if result.concepts:
            story.append(Paragraph("Main Concepts", self.styles['SectionHeader']))
            story.append(self._create_concepts_section(result.concepts))
            story.append(Spacer(1, 15))
        
        # Study questions section
        if result.study_questions:
            story.append(Paragraph("Study Questions", self.styles['SectionHeader']))
            story.append(self._create_questions_section(result.study_questions))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def export_summary_results(self, result: SummaryResponse, original_filename: str = None) -> bytes:
        """Export summarization-only results to PDF"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Content Summary Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata_data = [
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Original File", original_filename or "Direct Text Input"],
            ["Processing Time", f"{result.processing_time:.2f} seconds"],
            ["Original Length", f"{result.original_length} words"],
            ["Summary Length", f"{result.summary_length} words"],
            ["Compression Ratio", f"{result.compression_ratio:.2f}"],
            ["Agent Used", result.agent_used]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#BDC3C7'))
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Summary content
        story.append(Paragraph("Summary", self.styles['SectionHeader']))
        story.append(self._format_summary_content(result.summary))
        
        # Key points if available
        if result.key_points:
            story.append(Spacer(1, 15))
            story.append(Paragraph("Key Points", self.styles['SectionHeader']))
            for point in result.key_points:
                story.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    
    def _create_metadata_section(self, result: NotesParseResponse, original_filename: str = None):
        """Create metadata table for parsing results"""
        metadata_data = [
            ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Original File", original_filename or "Direct Text Input"],
            ["Processing Time", f"{result.processing_time:.2f} seconds"],
            ["Agent Used", result.agent_used],
            ["Keywords Found", str(len(result.keywords))],
            ["Concepts Found", str(len(result.concepts))],
            ["Questions Found", str(len(result.study_questions))]
        ]
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#ECF0F1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#BDC3C7'))
        ]))
        
        return metadata_table
    
    def _format_summary_content(self, content: str):
        """Format summary content with bullet points and proper spacing"""
        elements = []
        
        # Split content into paragraphs
        paragraphs = content.split('\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if it's a bullet point
            if paragraph.startswith('•') or paragraph.startswith('-') or paragraph.startswith('*'):
                # Format as bullet point
                clean_text = re.sub(r'^[•\-\*]\s*', '', paragraph)
                elements.append(Paragraph(f"• {clean_text}", self.styles['BulletPoint']))
            elif paragraph.startswith('#'):
                # Format as header
                clean_text = re.sub(r'^#+\s*', '', paragraph)
                elements.append(Paragraph(clean_text, self.styles['SubsectionHeader']))
            else:
                # Regular paragraph
                elements.append(Paragraph(paragraph, self.styles['Normal']))
        
        return KeepTogether(elements)
    
    def _create_keywords_table(self, keywords: List[KeywordExtraction]):
        """Create a formatted table for keywords"""
        
        # Sort keywords by importance
        sorted_keywords = sorted(keywords, key=lambda x: x.importance_score, reverse=True)
        
        # Create table data
        table_data = [["Keyword", "Importance Score", "Context"]]
        
        for kw in sorted_keywords:
            importance_str = f"{kw.importance_score:.2f}"
            context_str = (kw.context[:50] + "...") if kw.context and len(kw.context) > 50 else (kw.context or "N/A")
            table_data.append([kw.keyword, importance_str, context_str])
        
        table = Table(table_data, colWidths=[2*inch, 1*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#BDC3C7')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        return table
    
    def _create_concepts_section(self, concepts: List[ConceptExtraction]):
        """Create formatted section for concepts"""
        elements = []
        
        # Sort concepts by importance
        sorted_concepts = sorted(concepts, key=lambda x: x.importance_score, reverse=True)
        
        for concept in sorted_concepts:
            # Concept name as subheader
            elements.append(Paragraph(f"{concept.concept}", self.styles['SubsectionHeader']))
            
            # Definition if available
            if concept.definition:
                elements.append(Paragraph(f"Definition: {concept.definition}", self.styles['Normal']))
            
            # Related terms if available
            if concept.related_terms:
                terms_str = ", ".join(concept.related_terms)
                elements.append(Paragraph(f"Related Terms: {terms_str}", self.styles['Normal']))
            
            # Importance score
            elements.append(Paragraph(f"Importance: {concept.importance_score:.2f}", self.styles['Metadata']))
            elements.append(Spacer(1, 10))
        
        return KeepTogether(elements)
    
    def _create_questions_section(self, questions: List[StudyQuestion]):
        """Create formatted section for study questions"""
        elements = []
        
        for i, question in enumerate(questions, 1):
            # Question number and text
            elements.append(Paragraph(f"Q{i}. {question.question}", self.styles['Normal']))
            
            # Question metadata
            metadata_text = f"Difficulty: {question.difficulty_level.title()} | Type: {question.question_type.replace('_', ' ').title()}"
            elements.append(Paragraph(metadata_text, self.styles['Metadata']))
            
            # Suggested answer if available
            if question.suggested_answer:
                elements.append(Paragraph(f"Suggested Answer: {question.suggested_answer}", self.styles['Normal']))
            
            elements.append(Spacer(1, 8))
        
        return KeepTogether(elements)
    
    def generate_filename(self, base_name: str = "academic_analysis", file_type: str = "parse") -> str:
        """Generate a proper filename for the PDF"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = re.sub(r'[^\w\-_]', '_', base_name)
        return f"{clean_name}_{file_type}_{timestamp}.pdf"


# Global PDF exporter instance
pdf_exporter = PDFExporter()
