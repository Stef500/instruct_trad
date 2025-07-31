"""
Translation processor using DeepL API with error handling and rate limiting.
"""
import time
import logging
from typing import List, Optional
from dataclasses import dataclass
import deepl

from ..models.core import Sample, TranslatedSample


@dataclass
class TranslationConfig:
    """Configuration for translation processing."""
    api_key: str
    target_language: str = "FR"
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    batch_size: int = 10


class TranslationError(Exception):
    """Custom exception for translation errors."""
    pass


class RateLimitError(TranslationError):
    """Exception raised when rate limit is exceeded."""
    pass


class TranslationProcessor:
    """
    Processes samples for translation using DeepL API with robust error handling.
    """
    
    def __init__(self, config: TranslationConfig):
        """
        Initialize the translation processor.
        
        Args:
            config: Translation configuration including API key and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize DeepL translator
        try:
            self.translator = deepl.Translator(config.api_key)
            # Test the connection
            self.translator.get_usage()
        except deepl.AuthorizationException:
            raise TranslationError("Invalid DeepL API key")
        except Exception as e:
            raise TranslationError(f"Failed to initialize DeepL translator: {str(e)}")
    
    def translate_samples(self, samples: List[Sample]) -> List[TranslatedSample]:
        """
        Translate a list of samples using DeepL API.
        
        Args:
            samples: List of samples to translate
            
        Returns:
            List of translated samples
            
        Raises:
            TranslationError: If translation fails after all retries
        """
        if not samples:
            return []
        
        translated_samples = []
        failed_samples = []
        
        self.logger.info(f"Starting translation of {len(samples)} samples")
        
        for i, sample in enumerate(samples):
            try:
                translated_sample = self._translate_single_sample(sample)
                translated_samples.append(translated_sample)
                self.logger.debug(f"Successfully translated sample {i+1}/{len(samples)}: {sample.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to translate sample {sample.id}: {str(e)}")
                failed_samples.append((sample, str(e)))
        
        self.logger.info(f"Translation completed: {len(translated_samples)} successful, {len(failed_samples)} failed")
        
        if failed_samples:
            self.logger.warning(f"Failed samples: {[s.id for s, _ in failed_samples]}")
        
        return translated_samples
    
    def _translate_single_sample(self, sample: Sample) -> TranslatedSample:
        """
        Translate a single sample with retry logic.
        
        Args:
            sample: Sample to translate
            
        Returns:
            TranslatedSample with translation result
            
        Raises:
            TranslationError: If translation fails after all retries
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Perform the translation
                result = self.translator.translate_text(
                    sample.original_text,
                    target_lang=self.config.target_language
                )
                
                # Create metadata
                metadata = {
                    "source_language": result.detected_source_lang,
                    "target_language": self.config.target_language,
                    "api_version": "deepl",
                    "attempt": attempt + 1
                }
                
                return TranslatedSample(
                    sample=sample,
                    translated_text=result.text,
                    translation_metadata=metadata
                )
                
            except deepl.QuotaExceededException as e:
                self.logger.error(f"DeepL quota exceeded for sample {sample.id}")
                raise TranslationError(f"DeepL quota exceeded: {str(e)}")
                
            except deepl.TooManyRequestsException as e:
                self.logger.warning(f"Rate limit hit for sample {sample.id}, attempt {attempt + 1}")
                last_exception = RateLimitError(f"Rate limit exceeded: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                
            except deepl.AuthorizationException as e:
                self.logger.error(f"Authorization error for sample {sample.id}")
                raise TranslationError(f"Authorization failed: {str(e)}")
                
            except deepl.DeepLException as e:
                self.logger.warning(f"DeepL API error for sample {sample.id}, attempt {attempt + 1}: {str(e)}")
                last_exception = TranslationError(f"DeepL API error: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                
            except Exception as e:
                self.logger.warning(f"Unexpected error for sample {sample.id}, attempt {attempt + 1}: {str(e)}")
                last_exception = TranslationError(f"Unexpected error: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise TranslationError(f"Translation failed after {self.config.max_retries} attempts")
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)
    
    def get_usage_info(self) -> Optional[dict]:
        """
        Get current DeepL API usage information.
        
        Returns:
            Dictionary with usage information or None if unavailable
        """
        try:
            usage = self.translator.get_usage()
            return {
                "character_count": usage.character.count,
                "character_limit": usage.character.limit,
                "character_usage_percent": (usage.character.count / usage.character.limit * 100) if usage.character.limit > 0 else 0
            }
        except Exception as e:
            self.logger.warning(f"Could not retrieve usage info: {str(e)}")
            return None
    
    def validate_target_language(self, language_code: str) -> bool:
        """
        Validate if the target language is supported by DeepL.
        
        Args:
            language_code: Language code to validate
            
        Returns:
            True if language is supported, False otherwise
        """
        try:
            target_languages = self.translator.get_target_languages()
            supported_codes = [lang.code for lang in target_languages]
            return language_code.upper() in supported_codes
        except Exception as e:
            self.logger.warning(f"Could not validate language code: {str(e)}")
            return False