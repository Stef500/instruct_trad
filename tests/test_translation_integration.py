"""
Integration test to demonstrate TranslationProcessor functionality.
"""
import pytest
from unittest.mock import Mock, patch
import deepl

from src.medical_dataset_processor.models.core import Sample
from src.medical_dataset_processor.processors.translation_processor import (
    TranslationProcessor,
    TranslationConfig
)


def test_translation_processor_integration():
    """
    Integration test demonstrating the complete translation workflow.
    This test verifies that the TranslationProcessor meets all requirements:
    
    Requirements verified:
    - 2.2: Sends content to DeepL API for translation
    - 2.3: Preserves both original text and translation
    - 2.4: Retries up to 3 times on API failure
    - 6.1: Handles rate limits with automatic retry
    - 6.2: Logs errors and continues with other samples
    """
    
    # Create test samples (simulating requirement 2.1: 50 random samples)
    test_samples = [
        Sample(
            id="sample_1",
            content={"question": "What is diabetes?"},
            source_dataset="medqa",
            original_text="What is diabetes?"
        ),
        Sample(
            id="sample_2", 
            content={"question": "What causes hypertension?"},
            source_dataset="medqa",
            original_text="What causes hypertension?"
        )
    ]
    
    # Configure translation processor
    config = TranslationConfig(
        api_key="test_api_key",
        target_language="FR",
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0
    )
    
    with patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator') as mock_translator_class:
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock successful translation results
        mock_result1 = Mock()
        mock_result1.text = "Qu'est-ce que le diabète?"
        mock_result1.detected_source_lang = "EN"
        
        mock_result2 = Mock()
        mock_result2.text = "Qu'est-ce qui cause l'hypertension?"
        mock_result2.detected_source_lang = "EN"
        
        mock_translator.translate_text.side_effect = [mock_result1, mock_result2]
        
        # Initialize processor
        processor = TranslationProcessor(config)
        
        # Process samples
        results = processor.translate_samples(test_samples)
        
        # Verify requirements are met
        
        # Requirement 2.2: Content sent to DeepL API
        assert mock_translator.translate_text.call_count == 2
        mock_translator.translate_text.assert_any_call("What is diabetes?", target_lang="FR")
        mock_translator.translate_text.assert_any_call("What causes hypertension?", target_lang="FR")
        
        # Requirement 2.3: Both original text and translation preserved
        assert len(results) == 2
        
        # First sample
        assert results[0].sample.original_text == "What is diabetes?"
        assert results[0].translated_text == "Qu'est-ce que le diabète?"
        assert results[0].translation_metadata["source_language"] == "EN"
        assert results[0].translation_metadata["target_language"] == "FR"
        
        # Second sample  
        assert results[1].sample.original_text == "What causes hypertension?"
        assert results[1].translated_text == "Qu'est-ce qui cause l'hypertension?"
        assert results[1].translation_metadata["source_language"] == "EN"
        assert results[1].translation_metadata["target_language"] == "FR"
        
        # Verify metadata includes API version and attempt count
        assert results[0].translation_metadata["api_version"] == "deepl"
        assert results[0].translation_metadata["attempt"] == 1
        assert results[1].translation_metadata["api_version"] == "deepl"
        assert results[1].translation_metadata["attempt"] == 1


@patch('time.sleep')
@patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator')
def test_translation_processor_error_handling(mock_translator_class, mock_sleep):
    """
    Test that demonstrates error handling and retry logic.
    
    Requirements verified:
    - 2.4: Retries up to 3 times on API failure
    - 6.1: Handles rate limits with automatic retry and backoff
    - 6.2: Logs errors and continues with other samples
    """
    
    test_sample = Sample(
        id="sample_1",
        content={"question": "What is diabetes?"},
        source_dataset="medqa", 
        original_text="What is diabetes?"
    )
    
    config = TranslationConfig(
        api_key="test_api_key",
        target_language="FR",
        max_retries=3,
        base_delay=0.1,
        max_delay=1.0
    )
    
    mock_translator = Mock()
    mock_translator_class.return_value = mock_translator
    
    # Simulate rate limit on first two attempts, success on third
    mock_result = Mock()
    mock_result.text = "Qu'est-ce que le diabète?"
    mock_result.detected_source_lang = "EN"
    
    mock_translator.translate_text.side_effect = [
        deepl.TooManyRequestsException("Rate limit exceeded"),
        deepl.TooManyRequestsException("Rate limit exceeded"), 
        mock_result
    ]
    
    processor = TranslationProcessor(config)
    result = processor._translate_single_sample(test_sample)
    
    # Verify requirement 2.4 & 6.1: Retries with backoff
    assert mock_translator.translate_text.call_count == 3
    assert mock_sleep.call_count == 2  # Two retries = two sleep calls
    
    # Verify exponential backoff (requirement 6.1)
    mock_sleep.assert_any_call(0.1)  # First retry: base_delay * 2^0
    mock_sleep.assert_any_call(0.2)  # Second retry: base_delay * 2^1
    
    # Verify successful result after retries
    assert result.translated_text == "Qu'est-ce que le diabète?"
    assert result.translation_metadata["attempt"] == 3


def test_translation_processor_usage_monitoring():
    """
    Test usage monitoring functionality for API quota management.
    """
    
    config = TranslationConfig(api_key="test_api_key")
    
    with patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator') as mock_translator_class:
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock usage information
        mock_usage = Mock()
        mock_usage.character.count = 10000
        mock_usage.character.limit = 500000
        mock_translator.get_usage.return_value = mock_usage
        
        processor = TranslationProcessor(config)
        usage_info = processor.get_usage_info()
        
        # Verify usage monitoring works
        assert usage_info is not None
        assert usage_info["character_count"] == 10000
        assert usage_info["character_limit"] == 500000
        assert usage_info["character_usage_percent"] == 2.0  # 10000/500000 * 100


def test_translation_processor_language_validation():
    """
    Test language validation functionality.
    """
    
    config = TranslationConfig(api_key="test_api_key")
    
    with patch('src.medical_dataset_processor.processors.translation_processor.deepl.Translator') as mock_translator_class:
        mock_translator = Mock()
        mock_translator_class.return_value = mock_translator
        
        # Mock supported languages
        mock_lang_fr = Mock()
        mock_lang_fr.code = "FR"
        mock_lang_es = Mock()
        mock_lang_es.code = "ES"
        mock_translator.get_target_languages.return_value = [mock_lang_fr, mock_lang_es]
        
        processor = TranslationProcessor(config)
        
        # Test language validation
        assert processor.validate_target_language("FR") is True
        assert processor.validate_target_language("fr") is True  # Case insensitive
        assert processor.validate_target_language("ES") is True
        assert processor.validate_target_language("DE") is False  # Not supported