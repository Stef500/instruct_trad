"""
Tests for selective Ollama usage (translation only, generation only, or both).
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from medical_dataset_processor.pipeline import PipelineConfig, MedicalDatasetProcessor
from medical_dataset_processor.processors.ollama_processor import OllamaProcessor
from medical_dataset_processor.processors.translation_processor import TranslationProcessor
from medical_dataset_processor.processors.generation_processor import GenerationProcessor


class TestOllamaSelectiveUsage:
    """Test selective usage of Ollama for translation and/or generation."""
    
    def test_ollama_translation_only_config(self):
        """Test configuration with Ollama for translation only."""
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=True,
            use_ollama_for_generation=False,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            openai_api_key="test-openai-key",
            deepl_api_key=None,  # Not needed when using Ollama for translation
            translation_count=10,
            generation_count=10
        )
        
        assert config.use_ollama_for_translation is True
        assert config.use_ollama_for_generation is False
        assert config.use_ollama is False
    
    def test_ollama_generation_only_config(self):
        """Test configuration with Ollama for generation only."""
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=False,
            use_ollama_for_generation=True,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            deepl_api_key="test-deepl-key",
            openai_api_key=None,  # Not needed when using Ollama for generation
            translation_count=10,
            generation_count=10
        )
        
        assert config.use_ollama_for_translation is False
        assert config.use_ollama_for_generation is True
        assert config.use_ollama is False
    
    def test_ollama_both_config(self):
        """Test configuration with Ollama for both translation and generation."""
        config = PipelineConfig(
            use_ollama=True,
            use_ollama_for_translation=False,  # Should be ignored when use_ollama=True
            use_ollama_for_generation=False,   # Should be ignored when use_ollama=True
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            translation_count=10,
            generation_count=10
        )
        
        # When use_ollama=True, both translation and generation should use Ollama
        assert config.use_ollama is True
    
    def test_ollama_translation_only_processing(self):
        """Test processing with Ollama for translation only."""
        # Create configuration
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=True,
            use_ollama_for_generation=False,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            openai_api_key="test-openai-key",
            translation_count=5,
            generation_count=5
        )
        
        processor = MedicalDatasetProcessor(config)
        
        # Test that the configuration is correctly set
        assert processor.config.use_ollama_for_translation is True
        assert processor.config.use_ollama_for_generation is False
        assert processor.config.use_ollama is False
        
        # Test that validation passes
        validation = processor.validate_configuration()
        assert validation["valid"] is True
    
    def test_ollama_generation_only_processing(self):
        """Test processing with Ollama for generation only."""
        # Create configuration
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=False,
            use_ollama_for_generation=True,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            deepl_api_key="test-deepl-key",
            translation_count=5,
            generation_count=5
        )
        
        processor = MedicalDatasetProcessor(config)
        
        # Test that the configuration is correctly set
        assert processor.config.use_ollama_for_translation is False
        assert processor.config.use_ollama_for_generation is True
        assert processor.config.use_ollama is False
        
        # Test that validation passes
        validation = processor.validate_configuration()
        assert validation["valid"] is True
    
    def test_config_validation_ollama_translation_only(self):
        """Test that configuration validation works for Ollama translation only."""
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=True,
            use_ollama_for_generation=False,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            openai_api_key="test-openai-key",  # Required for generation
            translation_count=10,
            generation_count=10
        )
        
        processor = MedicalDatasetProcessor(config)
        validation = processor.validate_configuration()
        
        # Should be valid since we have OpenAI key for generation
        assert validation["valid"] is True
    
    def test_config_validation_ollama_generation_only(self):
        """Test that configuration validation works for Ollama generation only."""
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=False,
            use_ollama_for_generation=True,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            deepl_api_key="test-deepl-key",  # Required for translation
            translation_count=10,
            generation_count=10
        )
        
        processor = MedicalDatasetProcessor(config)
        validation = processor.validate_configuration()
        
        # Should be valid since we have DeepL key for translation
        assert validation["valid"] is True
    
    def test_config_validation_missing_keys(self):
        """Test that configuration validation fails when required keys are missing."""
        # Test missing OpenAI key when using Ollama for translation only
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=True,
            use_ollama_for_generation=False,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            openai_api_key=None,  # Missing - should cause validation error
            translation_count=10,
            generation_count=10
        )
        
        processor = MedicalDatasetProcessor(config)
        validation = processor.validate_configuration()
        
        # Should be invalid since OpenAI key is required for generation
        assert validation["valid"] is False
        assert any("OpenAI" in error for error in validation["errors"])
        
        # Test missing DeepL key when using Ollama for generation only
        config = PipelineConfig(
            use_ollama=False,
            use_ollama_for_translation=False,
            use_ollama_for_generation=True,
            ollama_model_name="croissantlm",
            ollama_base_url="http://localhost:11434",
            deepl_api_key=None,  # Missing - should cause validation error
            translation_count=10,
            generation_count=10
        )
        
        processor = MedicalDatasetProcessor(config)
        validation = processor.validate_configuration()
        
        # Should be invalid since DeepL key is required for translation
        assert validation["valid"] is False
        assert any("DeepL" in error for error in validation["errors"])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
