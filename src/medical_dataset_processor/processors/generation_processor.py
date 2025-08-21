"""
Generation processor using OpenAI GPT-4o-mini API with error handling and rate limiting.
"""
import time
import logging
import signal
from typing import List, Optional
from dataclasses import dataclass
import openai

from ..models.core import Sample, GeneratedSample


@dataclass
class GenerationConfig:
    """Configuration for generation processing."""
    api_key: str
    model: str = "gpt-4o-mini"
    max_retries: int = 2  # Réduit de 3 à 2
    base_delay: float = 0.5
    max_delay: float = 5.0  # Réduit de 10.0 à 5.0
    batch_size: int = 10
    prompt_length: int = 100
    max_tokens: int = 500
    temperature: float = 0.7
    timeout: int = 30
    request_timeout: int = 25  # Réduit de 60 à 25 secondes
    enable_streaming: bool = True  # Nouveau: activation du streaming


class GenerationError(Exception):
    """Custom exception for generation errors."""
    pass


class GenerationRateLimitError(GenerationError):
    """Exception raised when rate limit is exceeded."""
    pass


class GenerationInterruptedError(GenerationError):
    """Exception raised when generation is interrupted by user."""
    pass


class GenerationProcessor:
    """
    Processes samples for content generation using OpenAI API with robust error handling.
    """
    
    def __init__(self, config: GenerationConfig):
        """
        Initialize the generation processor.
        
        Args:
            config: Generation configuration including API key and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._interrupted = False
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize OpenAI client
        try:
            self.client = openai.OpenAI(api_key=config.api_key)
            # Test the connection by making a simple request
            self._test_connection()
        except openai.AuthenticationError:
            raise GenerationError("Invalid OpenAI API key")
        except Exception as e:
            raise GenerationError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        self.logger.info("Interruption signal received. Finishing current request and exiting gracefully...")
        self._interrupted = True
    
    def _test_connection(self):
        """Test the OpenAI API connection."""
        try:
            self.logger.debug("Testing OpenAI API connection...")
            # Make a minimal request to test the connection
            self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                timeout=self.config.request_timeout
            )
            self.logger.debug("OpenAI API connection test successful")
        except Exception as e:
            raise GenerationError(f"Failed to connect to OpenAI API: {str(e)}")
    
    def generate_from_prompts(self, samples: List[Sample]) -> List[GeneratedSample]:
        """
        Generate content from sample prompts using OpenAI API.
        
        Args:
            samples: List of samples to generate content from
            
        Returns:
            List of generated samples
            
        Raises:
            GenerationError: If generation fails after all retries
            GenerationInterruptedError: If generation is interrupted by user
        """
        if not samples:
            return []
        
        generated_samples = []
        failed_samples = []
        
        self.logger.info(f"Starting generation for {len(samples)} samples")
        
        for i, sample in enumerate(samples):
            # Check for interruption
            if self._interrupted:
                self.logger.warning(f"Generation interrupted by user. Processed {len(generated_samples)}/{len(samples)} samples")
                # Save partial results before raising exception
                self._last_generated_samples = generated_samples
                raise GenerationInterruptedError("Generation interrupted by user")
            
            try:
                self.logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.id}")
                generated_sample = self._generate_single_sample(sample)
                generated_samples.append(generated_sample)
                self.logger.debug(f"Successfully generated content for sample {i+1}/{len(samples)}: {sample.id}")
                
            except GenerationInterruptedError:
                # Save partial results before re-raising
                self._last_generated_samples = generated_samples
                raise
            except Exception as e:
                self.logger.error(f"Failed to generate content for sample {sample.id}: {str(e)}")
                failed_samples.append((sample, str(e)))
        
        self.logger.info(f"Generation completed: {len(generated_samples)} successful, {len(failed_samples)} failed")
        
        if failed_samples:
            self.logger.warning(f"Failed samples: {[s.id for s, _ in failed_samples]}")
        
        return generated_samples
    
    def _generate_single_sample(self, sample: Sample) -> GeneratedSample:
        """
        Generate content for a single sample with retry logic.
        
        Args:
            sample: Sample to generate content from
            
        Returns:
            GeneratedSample with generation result
            
        Raises:
            GenerationError: If generation fails after all retries
            GenerationInterruptedError: If generation is interrupted by user
        """
        # Extract prompt from the sample
        prompt = self._extract_prompt(sample, self.config.prompt_length)
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            # Check for interruption before each attempt
            if self._interrupted:
                raise GenerationInterruptedError("Generation interrupted by user")
            
            try:
                self.logger.info(f"Sending request to OpenAI API for sample {sample.id} (attempt {attempt + 1}/{self.config.max_retries})")
                
                # Create the generation request with timeout and optional streaming
                if self.config.enable_streaming:
                    response = self._generate_with_streaming(prompt)
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a medical content generator. Continue the given text in a coherent and medically accurate manner."
                            },
                            {
                                "role": "user",
                                "content": f"Continue this medical text: {prompt}"
                            }
                        ],
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        timeout=self.config.request_timeout
                    )
                
                generated_text = response.choices[0].message.content.strip()
                
                # Create metadata
                metadata = {
                    "model": self.config.model,
                    "prompt_length": len(prompt),
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "attempt": attempt + 1,
                    "streaming": self.config.enable_streaming,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
                
                self.logger.info(f"Successfully generated content for sample {sample.id} (attempt {attempt + 1})")
                
                return GeneratedSample(
                    sample=sample,
                    prompt=prompt,
                    generated_text=generated_text,
                    generation_metadata=metadata
                )
                
            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit hit for sample {sample.id}, attempt {attempt + 1}")
                last_exception = GenerationRateLimitError(f"Rate limit exceeded: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry for sample {sample.id}...")
                    time.sleep(delay)
                
            except openai.APITimeoutError as e:
                self.logger.warning(f"API timeout for sample {sample.id}, attempt {attempt + 1}")
                last_exception = GenerationError(f"API timeout: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry for sample {sample.id}...")
                    time.sleep(delay)
                
            except openai.AuthenticationError as e:
                self.logger.error(f"Authentication error for sample {sample.id}")
                raise GenerationError(f"Authentication failed: {str(e)}")
                
            except openai.BadRequestError as e:
                self.logger.error(f"Bad request error for sample {sample.id}: {str(e)}")
                raise GenerationError(f"Bad request: {str(e)}")
                
            except openai.APIError as e:
                self.logger.warning(f"OpenAI API error for sample {sample.id}, attempt {attempt + 1}: {str(e)}")
                last_exception = GenerationError(f"OpenAI API error: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry for sample {sample.id}...")
                    time.sleep(delay)
                
            except Exception as e:
                self.logger.warning(f"Unexpected error for sample {sample.id}, attempt {attempt + 1}: {str(e)}")
                last_exception = GenerationError(f"Unexpected error: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry for sample {sample.id}...")
                    time.sleep(delay)
        
        # All retries exhausted
        if last_exception:
            raise last_exception
        else:
            raise GenerationError(f"Generation failed after {self.config.max_retries} attempts")
    
    def _generate_with_streaming(self, prompt: str):
        """
        Generate content with streaming for immediate feedback.
        
        Args:
            prompt: The prompt to generate from
            
        Returns:
            OpenAI response object
        """
        self.logger.debug("Using streaming mode for generation...")
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical content generator. Continue the given text in a coherent and medically accurate manner."
                },
                {
                    "role": "user",
                    "content": f"Continue this medical text: {prompt}"
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.request_timeout,
            stream=True
        )
        
        # Collect streaming response
        collected_chunks = []
        collected_content = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                collected_content += content
                collected_chunks.append(chunk)
        
        # Create a mock response object that matches the non-streaming format
        if collected_chunks and collected_content:
            # Create a simple response object with the collected content
            from types import SimpleNamespace
            
            # Create a mock choice object
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = collected_content
            
            # Create a mock response object
            mock_response = SimpleNamespace()
            mock_response.choices = [choice]
            mock_response.usage = None  # Usage not available in streaming
            
            return mock_response
        else:
            raise GenerationError("No content received from streaming response")
    
    def _extract_prompt(self, sample: Sample, prompt_length: int = 100) -> str:
        """
        Extract a prompt from the sample's original text.
        
        Args:
            sample: Sample to extract prompt from
            prompt_length: Maximum length of the prompt in characters
            
        Returns:
            Extracted prompt string
        """
        original_text = sample.original_text.strip()
        
        if len(original_text) <= prompt_length:
            return original_text
        
        # Find a good breaking point (end of sentence or word)
        truncated = original_text[:prompt_length]
        
        # Try to break at sentence end
        sentence_endings = ['. ', '! ', '? ']
        best_break = -1
        
        for ending in sentence_endings:
            pos = truncated.rfind(ending)
            if pos > best_break:
                best_break = pos + len(ending) - 1
        
        # If no sentence break found, break at word boundary
        if best_break == -1:
            space_pos = truncated.rfind(' ')
            if space_pos > prompt_length // 2:  # Only if we don't lose too much
                best_break = space_pos
            else:
                best_break = prompt_length
        
        return original_text[:best_break].strip()
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        import random
        
        # Exponential backoff with jitter to prevent thundering herd
        delay = self.config.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
        total_delay = delay + jitter
        
        return min(total_delay, self.config.max_delay)
    
    def validate_model(self, model_name: str) -> bool:
        """
        Validate if the model is available.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Try to make a minimal request with the model
            self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            self.logger.warning(f"Could not validate model {model_name}: {str(e)}")
            return False