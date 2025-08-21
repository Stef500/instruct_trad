"""
Ollama processor using local CroissantLM model for translation and generation.
"""
import time
import logging
import subprocess
import json
import requests
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import signal

from ..models.core import Sample, TranslatedSample, GeneratedSample


@dataclass
class OllamaConfig:
    """Configuration for Ollama processing."""
    model_name: str = "croissantlm"
    base_url: str = "http://localhost:11434"
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    batch_size: int = 5  # Plus petit que les APIs cloud
    timeout: int = 60
    temperature: float = 0.0  # Déterministe pour la traduction
    max_tokens: int = 1000


class OllamaError(Exception):
    """Custom exception for Ollama errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Exception raised when cannot connect to Ollama."""
    pass


class OllamaModelError(OllamaError):
    """Exception raised when model is not available."""
    pass


class OllamaProcessor:
    """
    Processes samples using local Ollama with CroissantLM model.
    """
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize the Ollama processor.
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._interrupted = False
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Verify Ollama installation and model availability
        self._verify_ollama_setup()
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        self.logger.info("Interruption signal received. Finishing current request and exiting gracefully...")
        self._interrupted = True
    
    def _verify_ollama_setup(self):
        """Verify that Ollama is installed and the model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise OllamaConnectionError("Cannot connect to Ollama server")
            
            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            # Check for exact match or with :latest suffix
            if self.config.model_name not in model_names and f"{self.config.model_name}:latest" not in model_names:
                self.logger.warning(f"Model '{self.config.model_name}' not found. Available models: {model_names}")
                self.logger.info("You can create the model using: ollama create croissantlm Modelfile")
                raise OllamaModelError(f"Model '{self.config.model_name}' not available")
            
            self.logger.info(f"Ollama setup verified. Model '{self.config.model_name}' is available.")
            
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError("Cannot connect to Ollama server. Make sure Ollama is running.")
        except Exception as e:
            raise OllamaError(f"Failed to verify Ollama setup: {str(e)}")
    
    def translate_samples(self, samples: List[Sample]) -> List[TranslatedSample]:
        """
        Translate samples using CroissantLM via Ollama.
        
        Args:
            samples: List of samples to translate
            
        Returns:
            List of translated samples
        """
        if not samples:
            return []
        
        translated_samples = []
        failed_samples = []
        
        self.logger.info(f"Starting translation of {len(samples)} samples using Ollama")
        
        for i, sample in enumerate(samples):
            if self._interrupted:
                self.logger.info("Translation interrupted by user")
                break
                
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

    def translate_text(self, text: str) -> str:
        """
        Translate arbitrary text to French using the CroissantLM translation prompt.

        Chooses chunked translation for long inputs.
        """
        if not text:
            return ""
        if len(text) > 500:
            return self._translate_long_text(text)
        return self._translate_text_direct(text)
    
    def generate_from_prompts(self, samples: List[Sample]) -> List[GeneratedSample]:
        """
        Generate content from sample prompts using CroissantLM via Ollama.
        
        Args:
            samples: List of samples to generate content for
            
        Returns:
            List of generated samples
        """
        if not samples:
            return []
        
        generated_samples = []
        failed_samples = []
        
        self.logger.info(f"Starting generation for {len(samples)} samples using Ollama")
        
        for i, sample in enumerate(samples):
            if self._interrupted:
                self.logger.info("Generation interrupted by user")
                break
                
            try:
                generated_sample = self._generate_single_sample(sample)
                generated_samples.append(generated_sample)
                self.logger.debug(f"Successfully generated content for sample {i+1}/{len(samples)}: {sample.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate content for sample {sample.id}: {str(e)}")
                failed_samples.append((sample, str(e)))
        
        self.logger.info(f"Generation completed: {len(generated_samples)} successful, {len(failed_samples)} failed")
        
        if failed_samples:
            self.logger.warning(f"Failed samples: {[s.id for s, _ in failed_samples]}")
        
        return generated_samples
    
    def _translate_single_sample(self, sample: Sample) -> TranslatedSample:
        """
        Translate a single sample with retry logic.
        
        Args:
            sample: Sample to translate
            
        Returns:
            Translated sample
            
        Raises:
            OllamaError: If translation fails after all retries
        """
        text_to_translate = sample.original_text
        
        # For long texts, try to translate in parts
        if len(text_to_translate) > 500:
            self.logger.info(f"Long text detected ({len(text_to_translate)} chars), attempting chunked translation")
            translated_text = self._translate_long_text(text_to_translate)
        else:
            translated_text = self._translate_text_direct(text_to_translate)
        
        return TranslatedSample(
            sample=sample,
            translated_text=translated_text,
            translation_metadata={
                "processor_version": "1.0.0",
                "source_language": "EN",
                "target_language": "FR",
                "api_version": "ollama",
                "model_name": self.config.model_name,
                "attempt": 1,
                "quality_score": None
            }
        )
    
    def _translate_text_direct(self, text: str) -> str:
        """Translate text directly with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                # Create translation prompt
                prompt = self._create_translation_prompt(text)
                
                # Call Ollama API
                response = self._call_ollama_api(prompt, temperature=0.0)
                
                # Extract translation from response
                translated_text = self._extract_translation(response)
                
                return translated_text
                
            except Exception as e:
                self.logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = min(self.config.base_delay * (2 ** attempt), self.config.max_delay)
                    time.sleep(delay)
                else:
                    raise OllamaError(f"Translation failed after {self.config.max_retries} attempts: {str(e)}")
    
    def _translate_long_text(self, text: str) -> str:
        """Translate long text by splitting into sentences and translating each part."""
        import re
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        translated_parts = []
        
        for sentence in sentences:
            if len(sentence) > 50:  # Only translate sentences longer than 50 chars
                try:
                    translated_sentence = self._translate_text_direct(sentence)
                    translated_parts.append(translated_sentence)
                except Exception as e:
                    self.logger.warning(f"Failed to translate sentence: {sentence[:50]}... Error: {str(e)}")
                    translated_parts.append(sentence)  # Keep original if translation fails
            else:
                translated_parts.append(sentence)  # Keep short sentences as-is
        
        return ' '.join(translated_parts)
    
    def _generate_single_sample(self, sample: Sample) -> GeneratedSample:
        """
        Generate content for a single sample with retry logic.
        
        Args:
            sample: Sample to generate content for
            
        Returns:
            Generated sample
            
        Raises:
            OllamaError: If generation fails after all retries
        """
        # Create generation prompt based on sample content
        prompt = self._create_generation_prompt(sample)
        
        for attempt in range(self.config.max_retries):
            try:
                # Call Ollama API with higher temperature for generation
                response = self._call_ollama_api(prompt, temperature=0.7)
                
                # Extract generated content from response
                generated_content = self._extract_generated_content(response)
                
                return GeneratedSample(
                    sample=sample,
                    prompt=prompt,
                    generated_text=generated_content,
                    generation_metadata={
                        "processor_version": "1.0.0",
                        "api_version": "ollama",
                        "model_name": self.config.model_name,
                        "attempt": attempt + 1,
                        "quality_score": None
                    }
                )
                
            except Exception as e:
                self.logger.warning(f"Generation attempt {attempt + 1} failed for sample {sample.id}: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = min(self.config.base_delay * (2 ** attempt), self.config.max_delay)
                    time.sleep(delay)
                else:
                    raise OllamaError(f"Generation failed after {self.config.max_retries} attempts: {str(e)}")
    
    def _create_translation_prompt(self, text: str) -> str:
        """Create a translation prompt for the given text."""
        # Format the prompt according to the Modelfile template
        return f"""<|im_start|>system
Traducteur médical anglais-français. Traduisez uniquement le texte donné, sans ajouter d'explications.

RÈGLES :
- Traduisez mot à mot
- Utilisez la terminologie médicale française
- Pas d'ajouts ni d'explications
- Réponse courte et directe

Exemples :
- "moles" → "grains de beauté"
- "symptoms" → "symptômes"
- "What causes moles to suddenly appear?" → "Qu'est-ce qui provoque l'apparition soudaine de grains de beauté ?"

Traduisez :
<|im_end|>
<|im_start|>user
{text}
<|im_end|>
<|im_start|>assistant
"""
    
    def _create_generation_prompt(self, sample: Sample) -> str:
        """Create a generation prompt based on the sample content."""
        # Extract relevant information from the sample
        if hasattr(sample.content, 'question'):
            question = sample.content.question
        else:
            question = sample.original_text
        
        return f"""Expert médical. Répondez de manière concise et factuelle à cette question médicale :

{question}

Réponse :"""
    
    def _call_ollama_api(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Call the Ollama API with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation (0.0 for deterministic)
            
        Returns:
            The model's response
            
        Raises:
            OllamaError: If the API call fails
        """
        try:
            # Use the generate API with the formatted prompt
            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": self.config.max_tokens,
                    "stop": ["<|im_end|>"]  # Utilise le stop token du Modelfile
                }
            }
            
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code != 200:
                raise OllamaError(f"Ollama API returned status code {response.status_code}")
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.Timeout:
            raise OllamaError("Ollama API request timed out")
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Ollama API request failed: {str(e)}")
        except Exception as e:
            raise OllamaError(f"Unexpected error calling Ollama API: {str(e)}")
    
    def _extract_translation(self, response: str) -> str:
        """Extract the translation from the model response."""
        # Le Modelfile gère déjà le template, on retourne directement la réponse
        return response.strip()
    
    def _extract_generated_content(self, response: str) -> str:
        """Extract the generated content from the model response."""
        # Clean up the response and extract the generated content
        response = response.strip()
        
        # Remove any potential prefixes or suffixes
        if "Réponse :" in response:
            response = response.split("Réponse :")[-1].strip()
        
        return response
