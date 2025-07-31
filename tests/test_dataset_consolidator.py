"""
Tests for the DatasetConsolidator class.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.medical_dataset_processor.models.core import (
    Sample, TranslatedSample, GeneratedSample, ProcessedSample, 
    ConsolidatedDataset, ProcessingType
)
from src.medical_dataset_processor.processors.dataset_consolidator import DatasetConsolidator


class TestDatasetConsolidator:
    """Test cases for DatasetConsolidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.consolidator = DatasetConsolidator()
        
        # Create sample data
        self.sample1 = Sample(
            id="sample_1",
            content={"question": "What is fever?", "answer": "High body temperature"},
            source_dataset="medqa",
            original_text="What is fever? High body temperature"
        )
        
        self.sample2 = Sample(
            id="sample_2", 
            content={"question": "What causes headache?", "answer": "Various factors"},
            source_dataset="pubmedqa",
            original_text="What causes headache? Various factors"
        )
        
        self.sample3 = Sample(
            id="sample_3",
            content={"question": "How to treat cold?", "answer": "Rest and fluids"},
            source_dataset="healthsearchqa", 
            original_text="How to treat cold? Rest and fluids"
        )
        
        # Create translated samples
        self.translated_sample1 = TranslatedSample(
            sample=self.sample1,
            translated_text="Qu'est-ce que la fièvre? Température corporelle élevée",
            translation_metadata={"api_version": "1.0", "confidence": 0.95},
            processing_timestamp=datetime.now() - timedelta(minutes=5)
        )
        
        self.translated_sample2 = TranslatedSample(
            sample=self.sample2,
            translated_text="Qu'est-ce qui cause les maux de tête? Divers facteurs",
            translation_metadata={"api_version": "1.0", "confidence": 0.92},
            processing_timestamp=datetime.now() - timedelta(minutes=3)
        )
        
        # Create generated samples
        self.generated_sample1 = GeneratedSample(
            sample=self.sample3,
            prompt="How to treat cold?",
            generated_text="How to treat cold? Rest, drink plenty of fluids, and get adequate sleep. Over-the-counter medications can help with symptoms.",
            generation_metadata={"model": "gpt-4o-mini", "temperature": 0.7, "tokens": 150},
            processing_timestamp=datetime.now() - timedelta(minutes=2)
        )
    
    def test_consolidate_success(self):
        """Test successful consolidation of samples."""
        translated_samples = [self.translated_sample1, self.translated_sample2]
        generated_samples = [self.generated_sample1]
        
        result = self.consolidator.consolidate(translated_samples, generated_samples)
        
        assert isinstance(result, ConsolidatedDataset)
        assert len(result.samples) == 3
        
        # Check that we have the right number of each type
        translation_samples = result.get_translation_samples()
        generation_samples = result.get_generation_samples()
        
        assert len(translation_samples) == 2
        assert len(generation_samples) == 1
        
        # Verify metadata is properly added
        for sample in result.samples:
            assert 'processing_timestamp' in sample.metadata
            assert 'consolidation_timestamp' in sample.metadata
            assert 'processor_version' in sample.metadata
            assert sample.metadata['processor_version'] == '1.0.0'
    
    def test_consolidate_with_additional_metadata(self):
        """Test consolidation with additional metadata."""
        translated_samples = [self.translated_sample1]
        generated_samples = [self.generated_sample1]
        additional_metadata = {"experiment_id": "exp_001", "researcher": "Dr. Smith"}
        
        result = self.consolidator.consolidate(
            translated_samples, 
            generated_samples, 
            additional_metadata
        )
        
        assert result.metadata["experiment_id"] == "exp_001"
        assert result.metadata["researcher"] == "Dr. Smith"
    
    def test_consolidate_only_translated(self):
        """Test consolidation with only translated samples."""
        translated_samples = [self.translated_sample1, self.translated_sample2]
        generated_samples = []
        
        result = self.consolidator.consolidate(translated_samples, generated_samples)
        
        assert len(result.samples) == 2
        assert len(result.get_translation_samples()) == 2
        assert len(result.get_generation_samples()) == 0
    
    def test_consolidate_only_generated(self):
        """Test consolidation with only generated samples."""
        translated_samples = []
        generated_samples = [self.generated_sample1]
        
        result = self.consolidator.consolidate(translated_samples, generated_samples)
        
        assert len(result.samples) == 1
        assert len(result.get_translation_samples()) == 0
        assert len(result.get_generation_samples()) == 1
    
    def test_consolidate_empty_inputs_raises_error(self):
        """Test that empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="At least one translated or generated sample must be provided"):
            self.consolidator.consolidate([], [])
    
    def test_consolidate_invalid_translated_sample_type(self):
        """Test that invalid translated sample types raise ValueError."""
        invalid_samples = ["not_a_sample"]
        
        with pytest.raises(ValueError, match="All translated samples must be TranslatedSample instances"):
            self.consolidator.consolidate(invalid_samples, [])
    
    def test_consolidate_invalid_generated_sample_type(self):
        """Test that invalid generated sample types raise ValueError."""
        invalid_samples = ["not_a_sample"]
        
        with pytest.raises(ValueError, match="All generated samples must be GeneratedSample instances"):
            self.consolidator.consolidate([], invalid_samples)
    
    def test_consolidate_duplicate_sample_ids(self):
        """Test that duplicate sample IDs raise ValueError."""
        # Create samples with duplicate IDs
        duplicate_sample = Sample(
            id="sample_1",  # Same ID as sample1
            content={"question": "Duplicate", "answer": "Duplicate"},
            source_dataset="test",
            original_text="Duplicate"
        )
        
        translated_duplicate = TranslatedSample(
            sample=duplicate_sample,
            translated_text="Duplicate translated",
            translation_metadata={}
        )
        
        with pytest.raises(ValueError, match="Duplicate sample IDs found in input samples"):
            self.consolidator.consolidate([self.translated_sample1, translated_duplicate], [])
    
    def test_convert_translated_sample(self):
        """Test conversion of translated sample to processed sample."""
        processed = self.consolidator._convert_translated_sample(self.translated_sample1)
        
        assert isinstance(processed, ProcessedSample)
        assert processed.original_sample == self.sample1
        assert processed.processed_content == self.translated_sample1.translated_text
        assert processed.processing_type == ProcessingType.TRANSLATION.value
        assert 'api_version' in processed.metadata
        assert processed.metadata['api_version'] == "1.0"
        assert 'confidence' in processed.metadata
        assert processed.metadata['confidence'] == 0.95
    
    def test_convert_generated_sample(self):
        """Test conversion of generated sample to processed sample."""
        processed = self.consolidator._convert_generated_sample(self.generated_sample1)
        
        assert isinstance(processed, ProcessedSample)
        assert processed.original_sample == self.sample3
        assert processed.processed_content == self.generated_sample1.generated_text
        assert processed.processing_type == ProcessingType.GENERATION.value
        assert 'model' in processed.metadata
        assert processed.metadata['model'] == "gpt-4o-mini"
        assert 'prompt' in processed.metadata
        assert processed.metadata['prompt'] == "How to treat cold?"
    
    def test_create_sample_metadata(self):
        """Test creation of sample metadata."""
        timestamp = datetime.now()
        original_metadata = {"test_key": "test_value"}
        
        metadata = self.consolidator._create_sample_metadata(
            processing_type=ProcessingType.TRANSLATION.value,
            processing_timestamp=timestamp,
            original_metadata=original_metadata,
            prompt="test prompt"
        )
        
        assert metadata['processing_type'] == ProcessingType.TRANSLATION.value
        assert metadata['processing_timestamp'] == timestamp.isoformat()
        assert 'consolidation_timestamp' in metadata
        assert metadata['processor_version'] == '1.0.0'
        assert metadata['test_key'] == "test_value"
        assert metadata['prompt'] == "test prompt"
    
    def test_create_dataset_metadata(self):
        """Test creation of dataset metadata."""
        # Set some stats first
        self.consolidator.processing_stats = {
            'translated_count': 2,
            'generated_count': 1,
            'total_processed': 3,
            'validation_errors': 0
        }
        
        additional_metadata = {"experiment": "test_exp"}
        metadata = self.consolidator._create_dataset_metadata(additional_metadata)
        
        assert 'consolidation_timestamp' in metadata
        assert metadata['processor_version'] == '1.0.0'
        assert metadata['total_samples'] == 3
        assert metadata['translated_samples'] == 2
        assert metadata['generated_samples'] == 1
        assert metadata['validation_errors'] == 0
        assert metadata['experiment'] == "test_exp"
    
    def test_validate_consolidated_dataset_success(self):
        """Test successful validation of consolidated dataset."""
        processed_sample = ProcessedSample(
            original_sample=self.sample1,
            processed_content="Processed content",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'consolidation_timestamp': datetime.now().isoformat()
            }
        )
        
        dataset = ConsolidatedDataset(
            samples=[processed_sample],
            metadata={}
        )
        
        # Should not raise any exception
        self.consolidator._validate_consolidated_dataset(dataset)
    
    def test_validate_consolidated_dataset_empty_samples(self):
        """Test validation fails with empty samples."""
        # Create a mock dataset that bypasses the ConsolidatedDataset validation
        class MockDataset:
            def __init__(self):
                self.samples = []
        
        mock_dataset = MockDataset()
        
        with pytest.raises(ValueError, match="Consolidated dataset must contain at least one sample"):
            self.consolidator._validate_consolidated_dataset(mock_dataset)
    
    def test_validate_consolidated_dataset_duplicate_ids(self):
        """Test validation fails with duplicate sample IDs."""
        processed_sample1 = ProcessedSample(
            original_sample=self.sample1,
            processed_content="Content 1",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'consolidation_timestamp': datetime.now().isoformat()
            }
        )
        
        # Create another sample with same ID
        duplicate_sample = Sample(
            id="sample_1",  # Same ID
            content={"test": "test"},
            source_dataset="test",
            original_text="test"
        )
        
        processed_sample2 = ProcessedSample(
            original_sample=duplicate_sample,
            processed_content="Content 2",
            processing_type=ProcessingType.GENERATION.value,
            metadata={
                'processing_timestamp': datetime.now().isoformat(),
                'consolidation_timestamp': datetime.now().isoformat()
            }
        )
        
        dataset = ConsolidatedDataset(
            samples=[processed_sample1, processed_sample2],
            metadata={}
        )
        
        with pytest.raises(ValueError, match="Duplicate sample ID found: sample_1"):
            self.consolidator._validate_consolidated_dataset(dataset)
    
    def test_validate_consolidated_dataset_missing_metadata(self):
        """Test validation fails with missing required metadata."""
        processed_sample = ProcessedSample(
            original_sample=self.sample1,
            processed_content="Processed content",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={}  # Missing required metadata
        )
        
        dataset = ConsolidatedDataset(
            samples=[processed_sample],
            metadata={}
        )
        
        with pytest.raises(ValueError, match="missing processing_timestamp in metadata"):
            self.consolidator._validate_consolidated_dataset(dataset)
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        # Set some stats
        self.consolidator.processing_stats = {
            'translated_count': 5,
            'generated_count': 3,
            'total_processed': 8,
            'validation_errors': 1
        }
        
        stats = self.consolidator.get_processing_stats()
        
        assert stats['translated_count'] == 5
        assert stats['generated_count'] == 3
        assert stats['total_processed'] == 8
        assert stats['validation_errors'] == 1
        
        # Ensure it's a copy
        stats['translated_count'] = 10
        assert self.consolidator.processing_stats['translated_count'] == 5
    
    def test_reset_stats(self):
        """Test resetting processing statistics."""
        # Set some stats
        self.consolidator.processing_stats = {
            'translated_count': 5,
            'generated_count': 3,
            'total_processed': 8,
            'validation_errors': 1
        }
        
        self.consolidator.reset_stats()
        
        stats = self.consolidator.get_processing_stats()
        assert stats['translated_count'] == 0
        assert stats['generated_count'] == 0
        assert stats['total_processed'] == 0
        assert stats['validation_errors'] == 0
    
    @patch('src.medical_dataset_processor.processors.dataset_consolidator.logger')
    @patch.object(DatasetConsolidator, '_convert_translated_sample')
    def test_consolidate_with_processing_errors(self, mock_convert, mock_logger):
        """Test consolidation handles processing errors gracefully."""
        # Mock the conversion method to raise an exception
        mock_convert.side_effect = Exception("Conversion error")
        
        # This should handle the error and continue
        result = self.consolidator.consolidate([self.translated_sample1], [self.generated_sample1])
        
        # Should have logged an error
        mock_logger.error.assert_called()
        
        # Should still process the valid generated sample
        assert len(result.samples) == 1
        assert result.samples[0].processing_type == ProcessingType.GENERATION.value
        
        # Should track the validation error
        stats = self.consolidator.get_processing_stats()
        assert stats['validation_errors'] == 1
    
    def test_integration_full_consolidation_workflow(self):
        """Test the complete consolidation workflow."""
        translated_samples = [self.translated_sample1, self.translated_sample2]
        generated_samples = [self.generated_sample1]
        additional_metadata = {"batch_id": "batch_001"}
        
        result = self.consolidator.consolidate(
            translated_samples, 
            generated_samples, 
            additional_metadata
        )
        
        # Verify the consolidated dataset structure
        assert isinstance(result, ConsolidatedDataset)
        assert len(result.samples) == 3
        
        # Verify sample counts by type
        assert len(result.get_translation_samples()) == 2
        assert len(result.get_generation_samples()) == 1
        
        # Verify sample counts by dataset
        counts = result.get_sample_count_by_dataset()
        assert counts["medqa"] == 1
        assert counts["pubmedqa"] == 1
        assert counts["healthsearchqa"] == 1
        
        # Verify metadata
        assert result.metadata["batch_id"] == "batch_001"
        assert result.metadata["total_samples"] == 3
        assert result.metadata["translated_samples"] == 2
        assert result.metadata["generated_samples"] == 1
        
        # Verify all samples have proper metadata
        for sample in result.samples:
            assert 'processing_timestamp' in sample.metadata
            assert 'consolidation_timestamp' in sample.metadata
            assert 'processor_version' in sample.metadata
            
        # Verify processing stats
        stats = self.consolidator.get_processing_stats()
        assert stats['translated_count'] == 2
        assert stats['generated_count'] == 1
        assert stats['total_processed'] == 3
        assert stats['validation_errors'] == 0