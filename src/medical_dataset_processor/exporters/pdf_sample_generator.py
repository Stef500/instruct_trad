"""
PDF sample generator for creating review documents from consolidated datasets.
"""
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

from ..models.core import ConsolidatedDataset, ProcessedSample


class PDFSampleGenerator:
    """Generates PDF samples from consolidated datasets for review purposes."""
    
    def __init__(self, page_size=A4, margin=0.75*inch):
        """
        Initialize the PDF generator.
        
        Args:
            page_size: Page size for the PDF (default: A4)
            margin: Page margins in inches (default: 0.75 inch)
        """
        self.page_size = page_size
        self.margin = margin
        self.styles = self._create_styles()
    
    def _create_styles(self) -> dict:
        """Create custom paragraph styles for the PDF."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        custom_styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=16,
                spaceAfter=20,
                alignment=TA_CENTER
            ),
            'heading': ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=12,
                spaceBefore=12,
                spaceAfter=6,
                textColor='#2E4057'
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=8,
                alignment=TA_JUSTIFY
            ),
            'metadata': ParagraphStyle(
                'CustomMetadata',
                parent=styles['Normal'],
                fontSize=8,
                textColor='#666666',
                spaceAfter=4
            ),
            'sample_separator': ParagraphStyle(
                'SampleSeparator',
                parent=styles['Normal'],
                fontSize=10,
                spaceBefore=15,
                spaceAfter=10,
                alignment=TA_CENTER,
                textColor='#888888'
            )
        }
        
        return custom_styles
    
    def generate_sample(
        self, 
        dataset: ConsolidatedDataset, 
        sample_size: int = 100, 
        output_path: str = "sample_review.pdf",
        random_seed: Optional[int] = None
    ) -> None:
        """
        Generate a PDF sample from the consolidated dataset.
        
        Args:
            dataset: The consolidated dataset to sample from
            sample_size: Number of samples to include (default: 100)
            output_path: Path for the output PDF file
            random_seed: Optional seed for reproducible random selection
        
        Raises:
            ValueError: If dataset is empty or sample_size is invalid
            IOError: If unable to write to output_path
        """
        if not dataset.samples:
            raise ValueError("Dataset cannot be empty")
        
        if sample_size <= 0:
            raise ValueError("Sample size must be positive")
        
        if sample_size > len(dataset.samples):
            sample_size = len(dataset.samples)
        
        # Set random seed for reproducible results
        if random_seed is not None:
            random.seed(random_seed)
        
        # Select random samples
        selected_samples = self._select_random_samples(dataset.samples, sample_size)
        
        # Create PDF
        self._create_pdf(selected_samples, dataset, output_path)
    
    def _select_random_samples(self, samples: List[ProcessedSample], count: int) -> List[ProcessedSample]:
        """
        Select random samples from the dataset.
        
        Args:
            samples: List of all processed samples
            count: Number of samples to select
            
        Returns:
            List of randomly selected samples
        """
        return random.sample(samples, count)
    
    def _create_pdf(
        self, 
        samples: List[ProcessedSample], 
        dataset: ConsolidatedDataset, 
        output_path: str
    ) -> None:
        """
        Create the PDF document with the selected samples.
        
        Args:
            samples: Selected samples to include in PDF
            dataset: Original consolidated dataset for metadata
            output_path: Path for the output PDF file
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.page_size,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )
        
        # Build content
        story = []
        
        # Add title page
        story.extend(self._create_title_page(dataset, len(samples)))
        story.append(PageBreak())
        
        # Add samples
        for i, sample in enumerate(samples, 1):
            story.extend(self._format_sample_for_pdf(sample, i))
            
            # Add page break every 3 samples (except for the last one)
            if i % 3 == 0 and i < len(samples):
                story.append(PageBreak())
        
        # Build PDF
        try:
            doc.build(story)
        except Exception as e:
            raise IOError(f"Failed to create PDF: {str(e)}")
    
    def _create_title_page(self, dataset: ConsolidatedDataset, sample_count: int) -> List:
        """
        Create the title page content.
        
        Args:
            dataset: The consolidated dataset
            sample_count: Number of samples in the PDF
            
        Returns:
            List of PDF elements for the title page
        """
        elements = []
        
        # Title
        elements.append(Paragraph("Medical Dataset Processing Review", self.styles['title']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Summary information
        translation_count = len(dataset.get_translation_samples())
        generation_count = len(dataset.get_generation_samples())
        dataset_counts = dataset.get_sample_count_by_dataset()
        
        summary_text = f"""
        <b>Dataset Summary:</b><br/>
        • Total samples in dataset: {len(dataset.samples)}<br/>
        • Samples in this review: {sample_count}<br/>
        • Translated samples: {translation_count}<br/>
        • Generated samples: {generation_count}<br/>
        • Creation date: {dataset.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <br/>
        <b>Source Datasets:</b><br/>
        """
        
        for dataset_name, count in dataset_counts.items():
            summary_text += f"• {dataset_name}: {count} samples<br/>"
        
        elements.append(Paragraph(summary_text, self.styles['body']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Generation info
        generation_info = f"""
        <b>Document Information:</b><br/>
        • Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        • Purpose: Quality review of processed medical dataset samples<br/>
        • Note: This document contains a random sample for review purposes
        """
        
        elements.append(Paragraph(generation_info, self.styles['body']))
        
        return elements
    
    def _format_sample_for_pdf(self, sample: ProcessedSample, sample_number: int) -> List:
        """
        Format a single sample for PDF display.
        
        Args:
            sample: The processed sample to format
            sample_number: Sequential number of this sample in the PDF
            
        Returns:
            List of PDF elements representing the sample
        """
        elements = []
        
        # Sample header
        header_text = f"Sample #{sample_number} - {sample.processing_type.title()}"
        elements.append(Paragraph(header_text, self.styles['sample_separator']))
        
        # Metadata
        metadata_text = f"""
        <b>Source:</b> {sample.original_sample.source_dataset} | 
        <b>ID:</b> {sample.original_sample.id} | 
        <b>Type:</b> {sample.processing_type}
        """
        if sample.quality_score is not None:
            metadata_text += f" | <b>Quality Score:</b> {sample.quality_score:.2f}"
        
        elements.append(Paragraph(metadata_text, self.styles['metadata']))
        elements.append(Spacer(1, 6))
        
        # Original text
        elements.append(Paragraph("<b>Original Text:</b>", self.styles['heading']))
        original_text = self._escape_html(sample.original_sample.original_text)
        elements.append(Paragraph(original_text, self.styles['body']))
        elements.append(Spacer(1, 8))
        
        # Processed content
        if sample.processing_type == "translation":
            elements.append(Paragraph("<b>Translation:</b>", self.styles['heading']))
        else:
            elements.append(Paragraph("<b>Generated Content:</b>", self.styles['heading']))
        
        processed_text = self._escape_html(sample.processed_content)
        elements.append(Paragraph(processed_text, self.styles['body']))
        
        # Additional metadata if available
        if sample.metadata:
            elements.append(Spacer(1, 6))
            metadata_items = []
            for key, value in sample.metadata.items():
                if key not in ['processing_timestamp']:  # Skip timestamp as it's shown elsewhere
                    metadata_items.append(f"<b>{key.replace('_', ' ').title()}:</b> {value}")
            
            if metadata_items:
                metadata_text = " | ".join(metadata_items)
                elements.append(Paragraph(metadata_text, self.styles['metadata']))
        
        elements.append(Spacer(1, 15))
        
        return elements
    
    def _escape_html(self, text: str) -> str:
        """
        Escape HTML characters in text for safe PDF rendering.
        
        Args:
            text: Text to escape
            
        Returns:
            HTML-escaped text
        """
        if not text:
            return ""
        
        # Basic HTML escaping
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        # Truncate very long text to prevent PDF issues
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"
        
        return text