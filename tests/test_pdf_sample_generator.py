"""
Tests for the PDF sample generator.
"""
import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.medical_dataset_processor.models.core import (
    Sample, ProcessedSample, ConsolidatedDataset, ProcessingType
)
from src.medical_dataset_processor.exporters.pdf_sample_generator import PDFSampleGenerator


class TestPDFSampleGenerator:
    """Test cases for PDFSampleGenerator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        samples = []
        
        # Create translated samples
        for i in range(5):
            sample = Sample(
                id=f"trans_{i}",
                content={"question": f"Question {i}", "answer": f"Answer {i}"},
                source_dataset="MedQA",
                original_text=f"What is the treatment for condition {i}?"
            )
            
            processed_sample = ProcessedSample(
                original_sample=sample,
                processed_content=f"Quel est le traitement pour la condition {i}?",
                processing_type=ProcessingType.TRANSLATION.value,
                metadata={
                    "translation_service": "deepl",
                    "target_language": "FR",
                    "processing_timestamp": datetime.now()
                },
                quality_score=0.9
            )
            samples.append(processed_sample)
        
        # Create generated samples
        for i in range(5):
            sample = Sample(
                id=f"gen_{i}",
                content={"prompt": f"Prompt {i}"},
                source_dataset="PubMedQA",
                original_text=f"The symptoms of disease {i} include"
            )
            
            processed_sample = ProcessedSample(
                original_sample=sample,
                processed_content=f"The symptoms of disease {i} include fever, headache, and fatigue. Treatment typically involves rest and medication.",
                processing_type=ProcessingType.GENERATION.value,
                metadata={
                    "model": "gpt-4o-mini",
                    "prompt_length": 50,
                    "processing_timestamp": datetime.now()
                },
                quality_score=0.85
            )
            samples.append(processed_sample)
        
        return ConsolidatedDataset(
            samples=samples,
            metadata={"total_datasets": 2, "processing_date": "2024-01-15"},
            creation_timestamp=datetime(2024, 1, 15, 10, 30, 0)
        )
    
    @pytest.fixture
    def pdf_generator(self):
        """Create a PDFSampleGenerator instance."""
        return PDFSampleGenerator()
    
    def test_init_default_parameters(self):
        """Test PDFSampleGenerator initialization with default parameters."""
        generator = PDFSampleGenerator()
        
        assert generator.page_size is not None
        assert generator.margin > 0
        assert generator.styles is not None
        assert 'title' in generator.styles
        assert 'heading' in generator.styles
        assert 'body' in generator.styles
        assert 'metadata' in generator.styles
    
    def test_init_custom_parameters(self):
        """Test PDFSampleGenerator initialization with custom parameters."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        
        custom_margin = 1.0 * inch
        generator = PDFSampleGenerator(page_size=letter, margin=custom_margin)
        
        assert generator.page_size == letter
        assert generator.margin == custom_margin
    
    def test_select_random_samples(self, pdf_generator, sample_data):
        """Test random sample selection."""
        samples = sample_data.samples
        
        # Test selecting fewer samples than available
        selected = pdf_generator._select_random_samples(samples, 3)
        assert len(selected) == 3
        assert all(isinstance(s, ProcessedSample) for s in selected)
        assert all(s in samples for s in selected)
        
        # Test selecting all samples
        selected_all = pdf_generator._select_random_samples(samples, len(samples))
        assert len(selected_all) == len(samples)
        # Check that all original samples are represented (compare by ID)
        selected_ids = {s.original_sample.id for s in selected_all}
        original_ids = {s.original_sample.id for s in samples}
        assert selected_ids == original_ids
    
    def test_select_random_samples_reproducible(self, pdf_generator, sample_data):
        """Test that random selection is reproducible with seed."""
        samples = sample_data.samples
        
        # Test with same seed produces same results
        import random
        random.seed(42)
        selected1 = pdf_generator._select_random_samples(samples, 3)
        
        random.seed(42)
        selected2 = pdf_generator._select_random_samples(samples, 3)
        
        assert selected1 == selected2
    
    def test_escape_html(self, pdf_generator):
        """Test HTML escaping functionality."""
        # Test basic HTML characters
        text = "<script>alert('test')</script>"
        escaped = pdf_generator._escape_html(text)
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "<script>" not in escaped
        
        # Test ampersand escaping
        text = "Tom & Jerry"
        escaped = pdf_generator._escape_html(text)
        assert "&amp;" in escaped
        assert "&" not in escaped.replace("&amp;", "")
        
        # Test quotes
        text = 'He said "Hello" and she said \'Hi\''
        escaped = pdf_generator._escape_html(text)
        assert "&quot;" in escaped
        assert "&#x27;" in escaped
        
        # Test empty string
        assert pdf_generator._escape_html("") == ""
        assert pdf_generator._escape_html(None) == ""
    
    def test_escape_html_truncation(self, pdf_generator):
        """Test that very long text gets truncated."""
        long_text = "A" * 3000  # Longer than max_length (2000)
        escaped = pdf_generator._escape_html(long_text)
        
        assert len(escaped) < len(long_text)
        assert "... [truncated]" in escaped
    
    def test_format_sample_for_pdf_translation(self, pdf_generator, sample_data):
        """Test formatting a translation sample for PDF."""
        translation_sample = next(s for s in sample_data.samples if s.processing_type == "translation")
        
        elements = pdf_generator._format_sample_for_pdf(translation_sample, 1)
        
        assert len(elements) > 0
        # Should contain various paragraph elements
        assert any("Sample #1" in str(elem) for elem in elements)
        assert any("Translation:" in str(elem) for elem in elements)
        assert any("Original Text:" in str(elem) for elem in elements)
    
    def test_format_sample_for_pdf_generation(self, pdf_generator, sample_data):
        """Test formatting a generation sample for PDF."""
        generation_sample = next(s for s in sample_data.samples if s.processing_type == "generation")
        
        elements = pdf_generator._format_sample_for_pdf(generation_sample, 2)
        
        assert len(elements) > 0
        assert any("Sample #2" in str(elem) for elem in elements)
        assert any("Generated Content:" in str(elem) for elem in elements)
        assert any("Original Text:" in str(elem) for elem in elements)
    
    def test_create_title_page(self, pdf_generator, sample_data):
        """Test title page creation."""
        elements = pdf_generator._create_title_page(sample_data, 5)
        
        assert len(elements) > 0
        # Should contain title and summary information
        title_found = any("Medical Dataset Processing Review" in str(elem) for elem in elements)
        assert title_found
        
        # Should contain dataset statistics
        stats_found = any("Total samples in dataset: 10" in str(elem) for elem in elements)
        assert stats_found
    
    def test_generate_sample_basic(self, pdf_generator, sample_data):
        """Test basic PDF generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_sample.pdf")
            
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=3,
                output_path=output_path,
                random_seed=42
            )
            
            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
    
    def test_generate_sample_full_dataset(self, pdf_generator, sample_data):
        """Test PDF generation with all samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_full.pdf")
            
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=len(sample_data.samples),
                output_path=output_path
            )
            
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
    
    def test_generate_sample_creates_directory(self, pdf_generator, sample_data):
        """Test that PDF generation creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "test.pdf")
            
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=2,
                output_path=nested_path
            )
            
            assert os.path.exists(nested_path)
    
    def test_generate_sample_empty_dataset(self, pdf_generator):
        """Test error handling for empty dataset."""
        # Create a minimal sample to satisfy ConsolidatedDataset validation
        sample = Sample(
            id="test_id",
            content={"text": "test"},
            source_dataset="TestDataset",
            original_text="Test text"
        )
        processed_sample = ProcessedSample(
            original_sample=sample,
            processed_content="Processed text",
            processing_type=ProcessingType.TRANSLATION.value
        )
        
        # Create dataset and then manually clear samples to test empty condition
        dataset = ConsolidatedDataset(samples=[processed_sample])
        dataset.samples = []  # Manually clear to test empty condition
        
        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            pdf_generator.generate_sample(dataset, 5)
    
    def test_generate_sample_invalid_sample_size(self, pdf_generator, sample_data):
        """Test error handling for invalid sample size."""
        with pytest.raises(ValueError, match="Sample size must be positive"):
            pdf_generator.generate_sample(sample_data, 0)
        
        with pytest.raises(ValueError, match="Sample size must be positive"):
            pdf_generator.generate_sample(sample_data, -1)
    
    def test_generate_sample_size_larger_than_dataset(self, pdf_generator, sample_data):
        """Test handling when requested sample size is larger than dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_large.pdf")
            
            # Request more samples than available
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=1000,  # Much larger than the 10 samples in dataset
                output_path=output_path
            )
            
            # Should still create PDF with all available samples
            assert os.path.exists(output_path)
    
    def test_generate_sample_reproducible_with_seed(self, pdf_generator, sample_data):
        """Test that PDF generation is reproducible with random seed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path1 = os.path.join(temp_dir, "test1.pdf")
            output_path2 = os.path.join(temp_dir, "test2.pdf")
            
            # Generate two PDFs with same seed
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=5,
                output_path=output_path1,
                random_seed=123
            )
            
            pdf_generator.generate_sample(
                dataset=sample_data,
                sample_size=5,
                output_path=output_path2,
                random_seed=123
            )
            
            # Files should exist and have same size (indicating same content selection)
            assert os.path.exists(output_path1)
            assert os.path.exists(output_path2)
            assert os.path.getsize(output_path1) == os.path.getsize(output_path2)
    
    @patch('src.medical_dataset_processor.exporters.pdf_sample_generator.SimpleDocTemplate')
    def test_create_pdf_io_error(self, mock_doc, pdf_generator, sample_data):
        """Test error handling when PDF creation fails."""
        # Mock the document to raise an exception during build
        mock_instance = MagicMock()
        mock_instance.build.side_effect = Exception("PDF creation failed")
        mock_doc.return_value = mock_instance
        
        selected_samples = sample_data.samples[:3]
        
        with pytest.raises(IOError, match="Failed to create PDF"):
            pdf_generator._create_pdf(selected_samples, sample_data, "test.pdf")
    
    def test_sample_with_no_quality_score(self, pdf_generator):
        """Test formatting sample without quality score."""
        sample = Sample(
            id="test_id",
            content={"text": "test content"},
            source_dataset="TestDataset",
            original_text="Original test text"
        )
        
        processed_sample = ProcessedSample(
            original_sample=sample,
            processed_content="Processed test text",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={"service": "test"},
            quality_score=None  # No quality score
        )
        
        elements = pdf_generator._format_sample_for_pdf(processed_sample, 1)
        
        # Should still format correctly without quality score
        assert len(elements) > 0
        # Quality score should not appear in the output
        quality_text = "".join(str(elem) for elem in elements)
        assert "Quality Score:" not in quality_text
    
    def test_sample_with_empty_metadata(self, pdf_generator):
        """Test formatting sample with empty metadata."""
        sample = Sample(
            id="test_id",
            content={"text": "test content"},
            source_dataset="TestDataset",
            original_text="Original test text"
        )
        
        processed_sample = ProcessedSample(
            original_sample=sample,
            processed_content="Processed test text",
            processing_type=ProcessingType.GENERATION.value,
            metadata={}  # Empty metadata
        )
        
        elements = pdf_generator._format_sample_for_pdf(processed_sample, 1)
        
        # Should still format correctly with empty metadata
        assert len(elements) > 0