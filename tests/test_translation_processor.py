"""
Unit tests for TranslationProcessor with DeepL API mocks.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime

import deepl

from src.medical_dataset_processor.models.core import Sample, TranslatedSample
from src.medical_dataset_processor.processors.translation_processor import (
    TranslationProcessor,
    TranslationConfig,
    TranslationError,
    RateLimitError
)


@pytest.fixture
def translation_config():
    """Create a test translation configuration."""
    return TranslationConfig(
        api_key="test_api_key",
        target_language="FR",
        max_retries=3,
        base_delay=0.1,  # Shorter delays for testing
        max_delay=1.0,
        batch_size=10
    )


@pytest.fixture
def sample_data():
    """Create test sample data."""
    return [
        Sample(
            id="sample_1",
            content={"question": "What is diabetes?", "answer": "A metabolic disorder"},
            source_dataset="test_dataset",
            original_text="What is diabetes?"
        ),
        Sample(
            id="sample_2",
            content={"question": "What causes hypertension?", "answer": "Various factors"},
            source_dataset="test_dataset",
            original_text="What causes hypertension?"
        )
    ]


@pytest.fixture
def mock_deepl_translator():
    """Create a mock DeepL translator."""
    translator = Mock(spec=deepl.Translator)
    
    # Mock successful translation result
    mock_result = Mock()
    mock_result.text = "Qu'est-ce que le diabète?"
    mock_result.detected_source_lang = "EN"
    
    translator.translate_text.return_value = mock_result
    
    # Mock usage info
    mock_usage = Mock()
    mock_usage.character.count = 1000
    mock_usage.character.limit = 500000
    translator.get_usage.return_value = mock_usage
    
    # Mock target languages
    mock_lang = Mock()
    mock_lang.code = "FR"
    translator.get_target_languages.return_value = [mock_lang]
    
    return translator


class TestTranslationConfig:
    """Test TranslationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TranslationConfig(api_key="test_key")
        
        assert config.api_key == "test_key"
        assert config.target_language == "FR"
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.batch_size == 10
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TranslationConfig(
            api_key="custom_key",
            target_language="ES",
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            batch_size=20
        )
        
        assert config.api_key == "custom_key"
        assert config.target_language == "ES"
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.batch_size == 20


class TestTranslationProcessor:
    """Test TranslationProcessor class."""
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_initialization_success(self, mock_translator_class, translation_config):
        """Test successful processor initialization."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        processor = TranslationProcessor(translation_config)
        
        assert processor.config == translation_config
        mock_translator_class.assert_called_once_with(translation_config.api_key)
        mock_translator.get_usage.assert_called_once()
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_initialization_auth_error(self, mock_translator_class, translation_config):
        """Test initialization with authentication error."""
        mock_translator_class.side_effect = deepl.AuthorizationException("Invalid API key")
        
        with pytest.raises(TranslationError, match="Invalid DeepL API key"):
            TranslationProcessor(translation_config)
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_initialization_general_error(self, mock_translator_class, translation_config):
        """Test initialization with general error."""
        mock_translator_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(TranslationError, match="Failed to initialize DeepL translator"):
            TranslationProcessor(translation_config)
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_translate_samples_success(self, mock_translator_class, translation_config, sample_data):
        """Test successful translation of samples."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock translation results
        mock_result1 = Mock()
        mock_result1.text = "Qu'est-ce que le diabète?"
        mock_result1.detected_source_lang = "EN"
        
        mock_result2 = Mock()
        mock_result2.text = "Qu'est-ce qui cause l'hypertension?"
        mock_result2.detected_source_lang = "EN"
        
        mock_translator.translate_text.side_effect = [mock_result1, mock_result2]
        
        processor = TranslationProcessor(translation_config)
        results = processor.translate_samples(sample_data)
        
        assert len(results) == 2
        assert all(isinstance(result, TranslatedSample) for result in results)
        assert results[0].translated_text == "Qu'est-ce que le diabète?"
        assert results[1].translated_text == "Qu'est-ce qui cause l'hypertension?"
        
        # Check metadata
        assert results[0].translation_metadata["source_language"] == "EN"
        assert results[0].translation_metadata["target_language"] == "FR"
        assert results[0].translation_metadata["api_version"] == "deepl"
        assert results[0].translation_metadata["attempt"] == 1
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_translate_samples_empty_list(self, mock_translator_class, translation_config):
        """Test translation with empty sample list."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        processor = TranslationProcessor(translation_config)
        results = processor.translate_samples([])
        
        assert results == []
        mock_translator.translate_text.assert_not_called()
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_translate_single_sample_quota_exceeded(self, mock_translator_class, translation_config, sample_data):
        """Test handling of quota exceeded error."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        mock_translator.translate_text.side_effect = deepl.QuotaExceededException("Quota exceeded")
        
        processor = TranslationProcessor(translation_config)
        
        with pytest.raises(TranslationError, match="DeepL quota exceeded"):
            processor._translate_single_sample(sample_data[0])
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    @patch('time.sleep')
    def test_translate_single_sample_rate_limit_retry(self, mock_sleep, mock_translator_class, translation_config, sample_data):
        """Test retry logic for rate limit errors."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # First two calls raise rate limit, third succeeds
        mock_result = Mock()
        mock_result.text = "Qu'est-ce que le diabète?"
        mock_result.detected_source_lang = "EN"
        
        mock_translator.translate_text.side_effect = [
            deepl.TooManyRequestsException("Rate limit"),
            deepl.TooManyRequestsException("Rate limit"),
            mock_result
        ]
        
        processor = TranslationProcessor(translation_config)
        result = processor._translate_single_sample(sample_data[0])
        
        assert isinstance(result, TranslatedSample)
        assert result.translated_text == "Qu'est-ce que le diabète?"
        assert result.translation_metadata["attempt"] == 3
        
        # Verify sleep was called for backoff
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry: base_delay * 2^0
        mock_sleep.assert_any_call(0.2)  # Second retry: base_delay * 2^1
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    @patch('time.sleep')
    def test_translate_single_sample_max_retries_exceeded(self, mock_sleep, mock_translator_class, translation_config, sample_data):
        """Test failure after max retries exceeded."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        mock_translator.translate_text.side_effect = deepl.TooManyRequestsException("Rate limit")
        
        processor = TranslationProcessor(translation_config)
        
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            processor._translate_single_sample(sample_data[0])
        
        # Verify all retries were attempted
        assert mock_translator.translate_text.call_count == translation_config.max_retries
        assert mock_sleep.call_count == translation_config.max_retries - 1
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_translate_single_sample_auth_error(self, mock_translator_class, translation_config, sample_data):
        """Test handling of authorization errors."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        mock_translator.translate_text.side_effect = deepl.AuthorizationException("Invalid key")
        
        processor = TranslationProcessor(translation_config)
        
        with pytest.raises(TranslationError, match="Authorization failed"):
            processor._translate_single_sample(sample_data[0])
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    @patch('time.sleep')
    def test_translate_single_sample_general_deepl_error(self, mock_sleep, mock_translator_class, translation_config, sample_data):
        """Test handling of general DeepL API errors with retry."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock a general DeepL error that should be retried
        mock_translator.translate_text.side_effect = deepl.DeepLException("API error")
        
        processor = TranslationProcessor(translation_config)
        
        with pytest.raises(TranslationError, match="DeepL API error"):
            processor._translate_single_sample(sample_data[0])
        
        # Verify retries were attempted
        assert mock_translator.translate_text.call_count == translation_config.max_retries
        assert mock_sleep.call_count == translation_config.max_retries - 1
    
    def test_calculate_backoff_delay(self, translation_config):
        """Test exponential backoff delay calculation."""
        processor = TranslationProcessor.__new__(TranslationProcessor)
        processor.config = translation_config
        
        # Test exponential backoff
        assert processor._calculate_backoff_delay(0) == 0.1  # base_delay * 2^0
        assert processor._calculate_backoff_delay(1) == 0.2  # base_delay * 2^1
        assert processor._calculate_backoff_delay(2) == 0.4  # base_delay * 2^2
        
        # Test max delay cap
        processor.config.base_delay = 10.0
        assert processor._calculate_backoff_delay(10) == 1.0  # Should be capped at max_delay
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_get_usage_info_success(self, mock_translator_class, translation_config):
        """Test successful usage info retrieval."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        mock_usage = Mock()
        mock_usage.character.count = 1000
        mock_usage.character.limit = 500000
        mock_translator.get_usage.return_value = mock_usage
        
        processor = TranslationProcessor(translation_config)
        usage_info = processor.get_usage_info()
        
        assert usage_info is not None
        assert usage_info["character_count"] == 1000
        assert usage_info["character_limit"] == 500000
        assert usage_info["character_usage_percent"] == 0.2  # 1000/500000 * 100
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_get_usage_info_error(self, mock_translator_class, translation_config):
        """Test usage info retrieval with error."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock successful initialization but failed usage retrieval
        mock_usage = Mock()
        mock_usage.character.count = 1000
        mock_usage.character.limit = 500000
        mock_translator.get_usage.side_effect = [mock_usage, Exception("API error")]
        
        processor = TranslationProcessor(translation_config)
        usage_info = processor.get_usage_info()
        
        assert usage_info is None
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_validate_target_language_success(self, mock_translator_class, translation_config):
        """Test successful language validation."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        mock_lang = Mock()
        mock_lang.code = "FR"
        mock_translator.get_target_languages.return_value = [mock_lang]
        
        processor = TranslationProcessor(translation_config)
        
        assert processor.validate_target_language("FR") is True
        assert processor.validate_target_language("fr") is True  # Case insensitive
        assert processor.validate_target_language("ES") is False
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_validate_target_language_error(self, mock_translator_class, translation_config):
        """Test language validation with API error."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        mock_translator.get_target_languages.side_effect = Exception("API error")
        
        processor = TranslationProcessor(translation_config)
        
        assert processor.validate_target_language("FR") is False
    
    @patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
    def test_translate_samples_partial_failure(self, mock_translator_class, translation_config, sample_data):
        """Test translation with some samples failing."""
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # First sample succeeds, second fails
        mock_result = Mock()
        mock_result.text = "Qu'est-ce que le diabète?"
        mock_result.detected_source_lang = "EN"
        
        mock_translator.translate_text.side_effect = [
            mock_result,
            deepl.QuotaExceededException("Quota exceeded")
        ]
        
        processor = TranslationProcessor(translation_config)
        results = processor.translate_samples(sample_data)
        
        # Should return only successful translations
        assert len(results) == 1
        assert results[0].translated_text == "Qu'est-ce que le diabète?"


class TestTranslationErrors:
    """Test custom exception classes."""
    
    def test_translation_error(self):
        """Test TranslationError exception."""
        error = TranslationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_rate_limit_error(self):
        """Test RateLimitError exception."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, TranslationError)
        assert isinstance(error, Exception)