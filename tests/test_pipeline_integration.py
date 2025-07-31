"""
Integration tests for the medical dataset processing pipeline.
"""
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.medical_dataset_processor.pipeline import (
    MedicalDatasetProcessor, 
    PipelineConfig,
    create_default_config,
    create_config_from_dict
)
from src.medical_dataset_processor.models.core import (
    Sample, 
    TranslatedSample, 
    GeneratedSample,
    ConsolidatedDataset
)


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        assert config.datasets_yaml_path == "datasets.yaml"
        assert config.translation_count == 50
        assert config.generation_count == 50
        assert config.target_language == "FR"
        assert config.output_dir == "output"
    
    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "datasets_yaml_path": "test_datasets.yaml",
            "translation_count": 25,
            "generation_count": 25,
            "target_language": "ES",
            "output_dir": "test_output"
        }
        
        config = create_config_from_dict(config_dict)
        
        assert config.datasets_yaml_path == "test_datasets.yaml"
        assert config.translation_count == 25
        assert config.generation_count == 25
        assert config.target_language == "ES"
        assert config.output_dir == "test_output"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid translation count
        with pytest.raises(ValueError, match="translation_count must be positive"):
            PipelineConfig(translation_count=0)
        
        # Test invalid generation count
        with pytest.raises(ValueError, match="generation_count must be positive"):
            PipelineConfig(generation_count=-1)
        
        # Test empty datasets path
        with pytest.raises(ValueError, match="datasets_yaml_path cannot be empty"):
            PipelineConfig(datasets_yaml_path="")
    
    def test_api_keys_from_environment(self):
        """Test API keys are loaded from environment variables."""
        with patch.dict(os.environ, {
            'DEEPL_API_KEY': 'test_deepl_key',
            'OPENAI_API_KEY': 'test_openai_key'
        }):
            config = PipelineConfig()
            assert config.deepl_api_key == 'test_deepl_key'
            assert config.openai_api_key == 'test_openai_key'


class TestMedicalDatasetProcessor:
    """Test the main medical dataset processor."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_datasets_yaml(self, temp_dir):
        """Create a sample datasets.yaml file."""
        datasets_config = {
            "test_dataset": {
                "name": "test_dataset",
                "source_type": "local",
                "source_path": str(temp_dir / "test_data.json"),
                "format": "json",
                "text_fields": ["question", "answer"]
            }
        }
        
        yaml_path = temp_dir / "datasets.yaml"
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(datasets_config, f)
        
        # Create test data file
        test_data = [
            {
                "question": f"What is the treatment for condition {i}?",
                "answer": f"The treatment for condition {i} involves medication and therapy."
            }
            for i in range(150)  # Enough samples for testing
        ]
        
        with open(temp_dir / "test_data.json", 'w') as f:
            json.dump(test_data, f)
        
        return str(yaml_path)
    
    @pytest.fixture
    def test_config(self, temp_dir, sample_datasets_yaml):
        """Create a test configuration."""
        return PipelineConfig(
            datasets_yaml_path=sample_datasets_yaml,
            deepl_api_key="test_deepl_key",
            openai_api_key="test_openai_key",
            translation_count=10,
            generation_count=10,
            output_dir=str(temp_dir / "output"),
            random_seed=42
        )
    
    def test_processor_initialization(self, test_config):
        """Test processor initialization."""
        processor = MedicalDatasetProcessor(test_config)
        
        assert processor.config == test_config
        assert processor.dataset_loader is not None
        assert processor.sample_selector is not None
        assert processor.consolidator is not None
        assert processor.jsonl_exporter is not None
        assert processor.pdf_generator is not None
        assert processor.translation_processor is None  # Created when needed
        assert processor.generation_processor is None  # Created when needed
    
    def test_load_all_datasets(self, test_config):
        """Test loading all datasets."""
        processor = MedicalDatasetProcessor(test_config)
        
        samples = processor._load_all_datasets()
        
        assert len(samples) == 150
        assert all(isinstance(sample, Sample) for sample in samples)
        assert processor.processing_stats["datasets_loaded"] == 1
    
    def test_select_samples(self, test_config):
        """Test sample selection."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Create test samples
        samples = [
            Sample(
                id=f"test_{i}",
                content={"question": f"Question {i}", "answer": f"Answer {i}"},
                source_dataset="test_dataset",
                original_text=f"Question {i} Answer {i}"
            )
            for i in range(50)
        ]
        
        translation_samples, generation_samples = processor._select_samples(samples)
        
        assert len(translation_samples) == 10
        assert len(generation_samples) == 10
        assert processor.processing_stats["samples_selected"] == 20
        
        # Verify no overlap
        translation_ids = {s.id for s in translation_samples}
        generation_ids = {s.id for s in generation_samples}
        assert len(translation_ids.intersection(generation_ids)) == 0
    
    def test_process_translations(self, test_config):
        """Test translation processing."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Mock the translation processor after initialization
        mock_processor = Mock()
        mock_translated_samples = [
            Mock(spec=TranslatedSample) for _ in range(5)
        ]
        mock_processor.translate_samples.return_value = mock_translated_samples
        processor.translation_processor = mock_processor
        
        # Create test samples
        samples = [
            Sample(
                id=f"test_{i}",
                content={"question": f"Question {i}"},
                source_dataset="test_dataset",
                original_text=f"Question {i}"
            )
            for i in range(5)
        ]
        
        result = processor._process_translations(samples)
        
        assert len(result) == 5
        assert processor.processing_stats["samples_translated"] == 5
        mock_processor.translate_samples.assert_called_once_with(samples)
    
    def test_process_generations(self, test_config):
        """Test generation processing."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Mock the generation processor after initialization
        mock_processor = Mock()
        mock_generated_samples = [
            Mock(spec=GeneratedSample) for _ in range(5)
        ]
        mock_processor.generate_from_prompts.return_value = mock_generated_samples
        processor.generation_processor = mock_processor
        
        # Create test samples
        samples = [
            Sample(
                id=f"test_{i}",
                content={"question": f"Question {i}"},
                source_dataset="test_dataset",
                original_text=f"Question {i}"
            )
            for i in range(5)
        ]
        
        result = processor._process_generations(samples)
        
        assert len(result) == 5
        assert processor.processing_stats["samples_generated"] == 5
        mock_processor.generate_from_prompts.assert_called_once_with(samples)
    
    def test_consolidate_results(self, test_config):
        """Test result consolidation."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Create mock samples
        translated_samples = [Mock(spec=TranslatedSample) for _ in range(3)]
        generated_samples = [Mock(spec=GeneratedSample) for _ in range(3)]
        
        # Mock the consolidator
        mock_dataset = Mock(spec=ConsolidatedDataset)
        mock_dataset.samples = [Mock() for _ in range(6)]
        processor.consolidator.consolidate = Mock(return_value=mock_dataset)
        
        result = processor._consolidate_results(translated_samples, generated_samples)
        
        assert result == mock_dataset
        processor.consolidator.consolidate.assert_called_once()
        
        # Check that additional metadata was passed
        call_args = processor.consolidator.consolidate.call_args
        assert call_args[0][0] == translated_samples
        assert call_args[0][1] == generated_samples
        # Check the third positional argument (additional_metadata)
        assert len(call_args[0]) == 3
        additional_metadata = call_args[0][2]
        assert "pipeline_config" in additional_metadata
    
    def test_export_results(self, test_config, temp_dir):
        """Test result export."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Create mock dataset
        mock_dataset = Mock(spec=ConsolidatedDataset)
        mock_dataset.samples = [Mock() for _ in range(10)]
        
        # Mock exporters
        processor.jsonl_exporter.export = Mock()
        processor.pdf_generator.generate_sample = Mock()
        
        processor._export_results(mock_dataset)
        
        # Verify exports were called
        processor.jsonl_exporter.export.assert_called_once()
        processor.pdf_generator.generate_sample.assert_called_once()
        
        # Check output directory was created
        assert (temp_dir / "output").exists()
        
        # Verify export stats
        assert processor.processing_stats["samples_exported"] == 10
    
    def test_validation_configuration(self, test_config):
        """Test configuration validation."""
        processor = MedicalDatasetProcessor(test_config)
        
        validation_results = processor.validate_configuration()
        
        assert validation_results["valid"] is True
        assert len(validation_results["errors"]) == 0
    
    def test_validation_missing_api_keys(self, temp_dir, sample_datasets_yaml):
        """Test validation with missing API keys."""
        config = PipelineConfig(
            datasets_yaml_path=sample_datasets_yaml,
            output_dir=str(temp_dir / "output")
        )
        
        processor = MedicalDatasetProcessor(config)
        validation_results = processor.validate_configuration()
        
        assert validation_results["valid"] is False
        assert any("DeepL API key" in error for error in validation_results["errors"])
        assert any("OpenAI API key" in error for error in validation_results["errors"])
    
    def test_validation_missing_datasets_file(self, test_config):
        """Test validation with missing datasets file."""
        test_config.datasets_yaml_path = "nonexistent.yaml"
        
        processor = MedicalDatasetProcessor(test_config)
        validation_results = processor.validate_configuration()
        
        assert validation_results["valid"] is False
        assert any("Dataset configuration file not found" in error for error in validation_results["errors"])
    
    def test_get_processing_stats(self, test_config):
        """Test getting processing statistics."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Set some stats
        processor.processing_stats["samples_translated"] = 10
        processor.processing_stats["samples_generated"] = 10
        
        stats = processor.get_processing_stats()
        
        assert stats["samples_translated"] == 10
        assert stats["samples_generated"] == 10
        assert "start_time" in stats
        assert "end_time" in stats
    
    def test_full_pipeline_integration(self, test_config):
        """Test the complete pipeline integration."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Mock the processors after initialization
        mock_trans_processor = Mock()
        mock_gen_processor = Mock()
        
        # Create mock results
        mock_translated_samples = [Mock(spec=TranslatedSample) for _ in range(10)]
        mock_generated_samples = [Mock(spec=GeneratedSample) for _ in range(10)]
        mock_trans_processor.translate_samples.return_value = mock_translated_samples
        mock_gen_processor.generate_from_prompts.return_value = mock_generated_samples
        
        # Set the mocked processors
        processor.translation_processor = mock_trans_processor
        processor.generation_processor = mock_gen_processor
        
        # Mock consolidator and exporters
        mock_dataset = Mock(spec=ConsolidatedDataset)
        mock_dataset.samples = [Mock() for _ in range(20)]
        processor.consolidator.consolidate = Mock(return_value=mock_dataset)
        processor.jsonl_exporter.export = Mock()
        processor.pdf_generator.generate_sample = Mock()
        
        # Run the pipeline
        result = processor.process_datasets()
        
        # Verify the result
        assert result == mock_dataset
        
        # Verify all steps were executed
        assert processor.processing_stats["datasets_loaded"] == 1
        assert processor.processing_stats["samples_selected"] == 20
        assert processor.processing_stats["samples_translated"] == 10
        assert processor.processing_stats["samples_generated"] == 10
        assert processor.processing_stats["samples_exported"] == 20
        assert processor.processing_stats["start_time"] is not None
        assert processor.processing_stats["end_time"] is not None
        
        # Verify components were called
        mock_trans_processor.translate_samples.assert_called_once()
        mock_gen_processor.generate_from_prompts.assert_called_once()
        processor.consolidator.consolidate.assert_called_once()
        processor.jsonl_exporter.export.assert_called_once()
        processor.pdf_generator.generate_sample.assert_called_once()
    
    def test_pipeline_error_handling(self, test_config):
        """Test pipeline error handling."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Mock a failure in dataset loading
        processor.dataset_loader.load_config = Mock(side_effect=Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            processor.process_datasets()
        
        # Verify error was recorded
        assert len(processor.processing_stats["errors"]) > 0
        assert processor.processing_stats["end_time"] is not None
    
    def test_empty_samples_handling(self, test_config):
        """Test handling of empty sample lists."""
        processor = MedicalDatasetProcessor(test_config)
        
        # Test empty translation samples
        result = processor._process_translations([])
        assert result == []
        
        # Test empty generation samples
        result = processor._process_generations([])
        assert result == []
    
    def test_missing_api_key_error(self, test_config):
        """Test error when API keys are missing during processing."""
        # Remove API keys
        test_config.deepl_api_key = None
        test_config.openai_api_key = None
        
        processor = MedicalDatasetProcessor(test_config)
        
        # Create test samples
        samples = [Mock(spec=Sample)]
        
        # Test translation without API key
        with pytest.raises(ValueError, match="DeepL API key is required"):
            processor._process_translations(samples)
        
        # Test generation without API key
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            processor._process_generations(samples)


class TestPipelineEndToEnd:
    """End-to-end pipeline tests with minimal mocking."""
    
    @pytest.fixture
    def minimal_test_setup(self):
        """Create minimal test setup with real files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test dataset file
            test_data = [
                {
                    "question": "What is hypertension?",
                    "answer": "Hypertension is high blood pressure."
                },
                {
                    "question": "What causes diabetes?",
                    "answer": "Diabetes is caused by insulin resistance."
                }
            ]
            
            data_file = temp_path / "test_data.json"
            with open(data_file, 'w') as f:
                json.dump(test_data, f)
            
            # Create datasets config
            datasets_config = {
                "test_dataset": {
                    "name": "test_dataset",
                    "source_type": "local",
                    "source_path": str(data_file),
                    "format": "json",
                    "text_fields": ["question", "answer"]
                }
            }
            
            yaml_file = temp_path / "datasets.yaml"
            with open(yaml_file, 'w') as f:
                import yaml
                yaml.dump(datasets_config, f)
            
            yield {
                "temp_dir": temp_path,
                "datasets_yaml": str(yaml_file),
                "data_file": str(data_file)
            }
    
    def test_dataset_loading_only(self, minimal_test_setup):
        """Test just the dataset loading part of the pipeline."""
        config = PipelineConfig(
            datasets_yaml_path=minimal_test_setup["datasets_yaml"],
            deepl_api_key="test_key",
            openai_api_key="test_key",
            translation_count=1,
            generation_count=1,
            output_dir=str(minimal_test_setup["temp_dir"] / "output")
        )
        
        processor = MedicalDatasetProcessor(config)
        samples = processor._load_all_datasets()
        
        assert len(samples) == 2
        assert all(isinstance(sample, Sample) for sample in samples)
        assert samples[0].original_text == "What is hypertension? Hypertension is high blood pressure."
        assert samples[1].original_text == "What causes diabetes? Diabetes is caused by insulin resistance."