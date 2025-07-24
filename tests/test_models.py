"""
Unit tests for data models.
"""
import pytest
from datetime import datetime
from medical_dataset_processor.models import (
    DatasetConfig,
    Sample,
    TranslatedSample,
    GeneratedSample,
    ProcessedSample,
    ConsolidatedDataset,
    SourceType,
    ProcessingType,
)


class TestDatasetConfig:
    """Test cases for DatasetConfig validation."""
    
    def test_valid_dataset_config(self):
        """Test creating a valid DatasetConfig."""
        config = DatasetConfig(
            name="test_dataset",
            source_type="huggingface",
            source_path="test/path",
            format="json",
            text_fields=["question", "answer"]
        )
        assert config.name == "test_dataset"
        assert config.source_type == "huggingface"
        assert config.source_path == "test/path"
        assert config.format == "json"
        assert config.text_fields == ["question", "answer"]
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Dataset name cannot be empty"):
            DatasetConfig(
                name="",
                source_type="huggingface",
                source_path="test/path",
                format="json",
                text_fields=["question"]
            )
    
    def test_invalid_source_type_raises_error(self):
        """Test that invalid source_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_type"):
            DatasetConfig(
                name="test",
                source_type="invalid_type",
                source_path="test/path",
                format="json",
                text_fields=["question"]
            )
    
    def test_empty_source_path_raises_error(self):
        """Test that empty source_path raises ValueError."""
        with pytest.raises(ValueError, match="Source path cannot be empty"):
            DatasetConfig(
                name="test",
                source_type="local",
                source_path="",
                format="json",
                text_fields=["question"]
            )
    
    def test_empty_format_raises_error(self):
        """Test that empty format raises ValueError."""
        with pytest.raises(ValueError, match="Format cannot be empty"):
            DatasetConfig(
                name="test",
                source_type="local",
                source_path="test/path",
                format="",
                text_fields=["question"]
            )
    
    def test_empty_text_fields_raises_error(self):
        """Test that empty text_fields raises ValueError."""
        with pytest.raises(ValueError, match="At least one text field must be specified"):
            DatasetConfig(
                name="test",
                source_type="local",
                source_path="test/path",
                format="json",
                text_fields=[]
            )


class TestSample:
    """Test cases for Sample validation."""
    
    def test_valid_sample(self):
        """Test creating a valid Sample."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?", "answer": "Artificial Intelligence"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        assert sample.id == "sample_1"
        assert sample.content["question"] == "What is AI?"
        assert sample.source_dataset == "test_dataset"
        assert sample.original_text == "What is AI?"
    
    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="Sample ID cannot be empty"):
            Sample(
                id="",
                content={"question": "What is AI?"},
                source_dataset="test_dataset",
                original_text="What is AI?"
            )
    
    def test_empty_source_dataset_raises_error(self):
        """Test that empty source_dataset raises ValueError."""
        with pytest.raises(ValueError, match="Source dataset cannot be empty"):
            Sample(
                id="sample_1",
                content={"question": "What is AI?"},
                source_dataset="",
                original_text="What is AI?"
            )
    
    def test_empty_original_text_raises_error(self):
        """Test that empty original_text raises ValueError."""
        with pytest.raises(ValueError, match="Original text cannot be empty"):
            Sample(
                id="sample_1",
                content={"question": "What is AI?"},
                source_dataset="test_dataset",
                original_text=""
            )


class TestTranslatedSample:
    """Test cases for TranslatedSample validation."""
    
    def test_valid_translated_sample(self):
        """Test creating a valid TranslatedSample."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        translated = TranslatedSample(
            sample=sample,
            translated_text="Qu'est-ce que l'IA?",
            translation_metadata={"api": "deepl", "confidence": 0.95}
        )
        assert translated.sample == sample
        assert translated.translated_text == "Qu'est-ce que l'IA?"
        assert translated.translation_metadata["api"] == "deepl"
        assert isinstance(translated.processing_timestamp, datetime)
    
    def test_invalid_sample_raises_error(self):
        """Test that invalid sample raises ValueError."""
        with pytest.raises(ValueError, match="Sample must be a Sample instance"):
            TranslatedSample(
                sample="not_a_sample",
                translated_text="Qu'est-ce que l'IA?"
            )
    
    def test_empty_translated_text_raises_error(self):
        """Test that empty translated_text raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Translated text cannot be empty"):
            TranslatedSample(
                sample=sample,
                translated_text=""
            )


class TestGeneratedSample:
    """Test cases for GeneratedSample validation."""
    
    def test_valid_generated_sample(self):
        """Test creating a valid GeneratedSample."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        generated = GeneratedSample(
            sample=sample,
            prompt="What is AI?",
            generated_text="Artificial Intelligence is a field of computer science...",
            generation_metadata={"model": "gpt-4o-mini", "temperature": 0.7}
        )
        assert generated.sample == sample
        assert generated.prompt == "What is AI?"
        assert "Artificial Intelligence" in generated.generated_text
        assert generated.generation_metadata["model"] == "gpt-4o-mini"
        assert isinstance(generated.processing_timestamp, datetime)
    
    def test_invalid_sample_raises_error(self):
        """Test that invalid sample raises ValueError."""
        with pytest.raises(ValueError, match="Sample must be a Sample instance"):
            GeneratedSample(
                sample="not_a_sample",
                prompt="What is AI?",
                generated_text="AI is..."
            )
    
    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            GeneratedSample(
                sample=sample,
                prompt="",
                generated_text="AI is..."
            )
    
    def test_empty_generated_text_raises_error(self):
        """Test that empty generated_text raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Generated text cannot be empty"):
            GeneratedSample(
                sample=sample,
                prompt="What is AI?",
                generated_text=""
            )


class TestProcessedSample:
    """Test cases for ProcessedSample validation."""
    
    def test_valid_processed_sample_translation(self):
        """Test creating a valid ProcessedSample for translation."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        processed = ProcessedSample(
            original_sample=sample,
            processed_content="Qu'est-ce que l'IA?",
            processing_type="translation",
            metadata={"api": "deepl"},
            quality_score=0.95
        )
        assert processed.original_sample == sample
        assert processed.processed_content == "Qu'est-ce que l'IA?"
        assert processed.processing_type == "translation"
        assert processed.quality_score == 0.95
    
    def test_valid_processed_sample_generation(self):
        """Test creating a valid ProcessedSample for generation."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        processed = ProcessedSample(
            original_sample=sample,
            processed_content="AI is a field of computer science...",
            processing_type="generation",
            metadata={"model": "gpt-4o-mini"}
        )
        assert processed.processing_type == "generation"
        assert processed.quality_score is None
    
    def test_invalid_sample_raises_error(self):
        """Test that invalid sample raises ValueError."""
        with pytest.raises(ValueError, match="Original sample must be a Sample instance"):
            ProcessedSample(
                original_sample="not_a_sample",
                processed_content="Some content",
                processing_type="translation"
            )
    
    def test_empty_processed_content_raises_error(self):
        """Test that empty processed_content raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Processed content cannot be empty"):
            ProcessedSample(
                original_sample=sample,
                processed_content="",
                processing_type="translation"
            )
    
    def test_invalid_processing_type_raises_error(self):
        """Test that invalid processing_type raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Invalid processing_type"):
            ProcessedSample(
                original_sample=sample,
                processed_content="Some content",
                processing_type="invalid_type"
            )
    
    def test_invalid_quality_score_raises_error(self):
        """Test that invalid quality_score raises ValueError."""
        sample = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        with pytest.raises(ValueError, match="Quality score must be between 0 and 1"):
            ProcessedSample(
                original_sample=sample,
                processed_content="Some content",
                processing_type="translation",
                quality_score=1.5
            )


class TestConsolidatedDataset:
    """Test cases for ConsolidatedDataset validation."""
    
    def test_valid_consolidated_dataset(self):
        """Test creating a valid ConsolidatedDataset."""
        sample1 = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        sample2 = Sample(
            id="sample_2",
            content={"question": "What is ML?"},
            source_dataset="test_dataset",
            original_text="What is ML?"
        )
        
        processed1 = ProcessedSample(
            original_sample=sample1,
            processed_content="Qu'est-ce que l'IA?",
            processing_type="translation"
        )
        processed2 = ProcessedSample(
            original_sample=sample2,
            processed_content="Machine Learning is...",
            processing_type="generation"
        )
        
        dataset = ConsolidatedDataset(
            samples=[processed1, processed2],
            metadata={"total_samples": 2}
        )
        
        assert len(dataset.samples) == 2
        assert dataset.metadata["total_samples"] == 2
        assert isinstance(dataset.creation_timestamp, datetime)
    
    def test_empty_samples_raises_error(self):
        """Test that empty samples list raises ValueError."""
        with pytest.raises(ValueError, match="Consolidated dataset must contain at least one sample"):
            ConsolidatedDataset(samples=[])
    
    def test_invalid_sample_type_raises_error(self):
        """Test that invalid sample type raises ValueError."""
        with pytest.raises(ValueError, match="All samples must be ProcessedSample instances"):
            ConsolidatedDataset(samples=["not_a_processed_sample"])
    
    def test_get_translation_samples(self):
        """Test filtering translation samples."""
        sample1 = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        sample2 = Sample(
            id="sample_2",
            content={"question": "What is ML?"},
            source_dataset="test_dataset",
            original_text="What is ML?"
        )
        
        processed1 = ProcessedSample(
            original_sample=sample1,
            processed_content="Qu'est-ce que l'IA?",
            processing_type="translation"
        )
        processed2 = ProcessedSample(
            original_sample=sample2,
            processed_content="Machine Learning is...",
            processing_type="generation"
        )
        
        dataset = ConsolidatedDataset(samples=[processed1, processed2])
        translation_samples = dataset.get_translation_samples()
        
        assert len(translation_samples) == 1
        assert translation_samples[0].processing_type == "translation"
    
    def test_get_generation_samples(self):
        """Test filtering generation samples."""
        sample1 = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="test_dataset",
            original_text="What is AI?"
        )
        sample2 = Sample(
            id="sample_2",
            content={"question": "What is ML?"},
            source_dataset="test_dataset",
            original_text="What is ML?"
        )
        
        processed1 = ProcessedSample(
            original_sample=sample1,
            processed_content="Qu'est-ce que l'IA?",
            processing_type="translation"
        )
        processed2 = ProcessedSample(
            original_sample=sample2,
            processed_content="Machine Learning is...",
            processing_type="generation"
        )
        
        dataset = ConsolidatedDataset(samples=[processed1, processed2])
        generation_samples = dataset.get_generation_samples()
        
        assert len(generation_samples) == 1
        assert generation_samples[0].processing_type == "generation"
    
    def test_get_sample_count_by_dataset(self):
        """Test counting samples by dataset."""
        sample1 = Sample(
            id="sample_1",
            content={"question": "What is AI?"},
            source_dataset="dataset_a",
            original_text="What is AI?"
        )
        sample2 = Sample(
            id="sample_2",
            content={"question": "What is ML?"},
            source_dataset="dataset_b",
            original_text="What is ML?"
        )
        sample3 = Sample(
            id="sample_3",
            content={"question": "What is DL?"},
            source_dataset="dataset_a",
            original_text="What is DL?"
        )
        
        processed1 = ProcessedSample(
            original_sample=sample1,
            processed_content="Qu'est-ce que l'IA?",
            processing_type="translation"
        )
        processed2 = ProcessedSample(
            original_sample=sample2,
            processed_content="Machine Learning is...",
            processing_type="generation"
        )
        processed3 = ProcessedSample(
            original_sample=sample3,
            processed_content="Deep Learning is...",
            processing_type="generation"
        )
        
        dataset = ConsolidatedDataset(samples=[processed1, processed2, processed3])
        counts = dataset.get_sample_count_by_dataset()
        
        assert counts["dataset_a"] == 2
        assert counts["dataset_b"] == 1