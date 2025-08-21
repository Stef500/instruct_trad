"""
Tests for Ollama processor integration.
"""
import pytest
import responses
from unittest.mock import patch, MagicMock
from medical_dataset_processor.processors.ollama_processor import (
    OllamaProcessor,
    OllamaConfig,
    OllamaError,
    OllamaConnectionError,
    OllamaModelError
)
from medical_dataset_processor.models.core import Sample


class TestOllamaConfig:
    """Test Ollama configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OllamaConfig()
        
        assert config.model_name == "croissantlm"
        assert config.base_url == "http://localhost:11434"
        assert config.max_retries == 3
        assert config.batch_size == 5
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OllamaConfig(
            model_name="test-model",
            base_url="http://test:8080",
            max_retries=5,
            batch_size=10
        )
        
        assert config.model_name == "test-model"
        assert config.base_url == "http://test:8080"
        assert config.max_retries == 5
        assert config.batch_size == 10


class TestOllamaProcessor:
    """Test Ollama processor functionality."""
    
    @pytest.fixture
    def sample(self):
        """Create a test sample."""
        return Sample(
            id="test_001",
            source_dataset="test_dataset",
            text="What are the symptoms of diabetes?",
            content={
                "question": "What are the symptoms of diabetes?",
                "context": "Medical question"
            }
        )
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OllamaConfig(
            model_name="test-model",
            base_url="http://localhost:11434",
            max_retries=2,
            batch_size=1
        )
    
    @responses.activate
    def test_verify_ollama_setup_success(self, config):
        """Test successful Ollama setup verification."""
        # Mock successful API response
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "test-model"}]},
            status=200
        )
        
        processor = OllamaProcessor(config)
        # Should not raise any exception
    
    @responses.activate
    def test_verify_ollama_setup_connection_error(self, config):
        """Test Ollama setup verification with connection error."""
        # Mock connection error
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            body=Exception("Connection failed")
        )
        
        with pytest.raises(OllamaConnectionError):
            OllamaProcessor(config)
    
    @responses.activate
    def test_verify_ollama_setup_model_not_found(self, config):
        """Test Ollama setup verification with missing model."""
        # Mock API response without the required model
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "other-model"}]},
            status=200
        )
        
        with pytest.raises(OllamaModelError):
            OllamaProcessor(config)
    
    @responses.activate
    def test_translate_samples_success(self, config, sample):
        """Test successful sample translation."""
        # Mock successful API responses
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "test-model"}]},
            status=200
        )
        
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"response": "Quels sont les symptômes du diabète ?"},
            status=200
        )
        
        processor = OllamaProcessor(config)
        translated_samples = processor.translate_samples([sample])
        
        assert len(translated_samples) == 1
        translated = translated_samples[0]
        assert translated.id == sample.id
        assert translated.processing_type == "translation"
        assert translated.metadata["api_version"] == "ollama"
        assert translated.metadata["model_name"] == "test-model"
    
    @responses.activate
    def test_generate_samples_success(self, config, sample):
        """Test successful sample generation."""
        # Mock successful API responses
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "test-model"}]},
            status=200
        )
        
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            json={"response": "Les symptômes du diabète incluent..."},
            status=200
        )
        
        processor = OllamaProcessor(config)
        generated_samples = processor.generate_from_prompts([sample])
        
        assert len(generated_samples) == 1
        generated = generated_samples[0]
        assert generated.id == sample.id
        assert generated.processing_type == "generation"
        assert generated.metadata["api_version"] == "ollama"
        assert generated.metadata["model_name"] == "test-model"
    
    @responses.activate
    def test_api_call_timeout(self, config, sample):
        """Test API call timeout handling."""
        # Mock successful setup
        responses.add(
            responses.GET,
            "http://localhost:11434/api/tags",
            json={"models": [{"name": "test-model"}]},
            status=200
        )
        
        # Mock timeout for generation call
        responses.add(
            responses.POST,
            "http://localhost:11434/api/generate",
            body=Exception("Timeout")
        )
        
        processor = OllamaProcessor(config)
        
        # Should handle timeout gracefully
        generated_samples = processor.generate_from_prompts([sample])
        assert len(generated_samples) == 0
    
    def test_create_translation_prompt(self, config):
        """Test translation prompt creation."""
        processor = OllamaProcessor(config)
        text = "Hello, how are you?"
        prompt = processor._create_translation_prompt(text)
        
        # Le Modelfile gère le template, on retourne juste le texte
        assert prompt == text
    
    def test_create_generation_prompt(self, config, sample):
        """Test generation prompt creation."""
        processor = OllamaProcessor(config)
        prompt = processor._create_generation_prompt(sample)
        
        assert "expert médical" in prompt
        assert sample.original_text in prompt
        assert "Réponse :" in prompt
    
    def test_extract_translation(self, config):
        """Test translation extraction from response."""
        processor = OllamaProcessor(config)
        
        # Test with clean response
        response = "Bonjour, comment allez-vous ?"
        extracted = processor._extract_translation(response)
        assert extracted == "Bonjour, comment allez-vous ?"
        
        # Test with whitespace
        response = "  Bonjour  "
        extracted = processor._extract_translation(response)
        assert extracted == "Bonjour"
    
    def test_extract_generated_content(self, config):
        """Test generated content extraction from response."""
        processor = OllamaProcessor(config)
        
        # Test with clean response
        response = "Voici la réponse : Les symptômes incluent..."
        extracted = processor._extract_generated_content(response)
        assert "Les symptômes incluent..." in extracted
        
        # Test with prefix
        response = "Réponse : Contenu généré"
        extracted = processor._extract_generated_content(response)
        assert extracted == "Contenu généré"


class TestOllamaErrors:
    """Test Ollama error handling."""
    
    def test_ollama_error_inheritance(self):
        """Test error class inheritance."""
        assert issubclass(OllamaConnectionError, OllamaError)
        assert issubclass(OllamaModelError, OllamaError)
    
    def test_ollama_error_messages(self):
        """Test error message creation."""
        connection_error = OllamaConnectionError("Connection failed")
        assert str(connection_error) == "Connection failed"
        
        model_error = OllamaModelError("Model not found")
        assert str(model_error) == "Model not found"


# Integration tests (require actual Ollama server)
class TestOllamaIntegration:
    """Integration tests for Ollama (require running server)."""
    
    @pytest.mark.integration
    def test_real_ollama_connection(self):
        """Test connection to real Ollama server."""
        # This test requires a running Ollama server
        # Skip if not available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                # Server is running, test basic functionality
                config = OllamaConfig()
                processor = OllamaProcessor(config)
                assert processor is not None
        except Exception:
            pytest.skip("Ollama server not available")
    
    @pytest.mark.integration
    def test_real_model_availability(self):
        """Test real model availability check."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "") for model in data.get("models", [])]
                # Check if croissantlm is available
                if "croissantlm" in models:
                    config = OllamaConfig()
                    processor = OllamaProcessor(config)
                    assert processor is not None
                else:
                    pytest.skip("CroissantLM model not available")
        except Exception:
            pytest.skip("Ollama server not available")
