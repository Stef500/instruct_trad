"""
Unit tests for the GenerationProcessor.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import openai

from src.medical_dataset_processor.models.core import Sample, GeneratedSample
from src.medical_dataset_processor.processors.generation_processor import (
    GenerationProcessor,
    GenerationConfig,
    GenerationError,
    GenerationRateLimitError
)


@pytest.fixture
def sample_config():
    """Create a sample generation configuration."""
    return GenerationConfig(
        api_key="test-api-key",
        model="gpt-4o-mini",
        max_retries=3,
        base_delay=0.1,  # Shorter delay for tests
        max_delay=1.0,
        prompt_length=50,
        max_tokens=100,
        temperature=0.7
    )


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return Sample(
        id="test-sample-1",
        content={"question": "What is hypertension?", "answer": "High blood pressure"},
        source_dataset="test_dataset",
        original_text="What is hypertension? Hypertension is a condition where blood pressure is consistently elevated above normal levels."
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a generated continuation of the medical text."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 15
    mock_response.usage.total_tokens = 35
    return mock_response


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.batch_size == 10
        assert config.prompt_length == 100
        assert config.max_tokens == 500
        assert config.temperature == 0.7
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            api_key="custom-key",
            model="gpt-4",
            max_retries=5,
            prompt_length=200,
            temperature=0.5
        )
        
        assert config.api_key == "custom-key"
        assert config.model == "gpt-4"
        assert config.max_retries == 5
        assert config.prompt_length == 200
        assert config.temperature == 0.5


class TestGenerationProcessor:
    """Test GenerationProcessor class."""
    
    @patch('openai.OpenAI')
    def test_initialization_success(self, mock_openai_class, sample_config):
        """Test successful processor initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock successful test connection
        mock_client.chat.completions.create.return_value = Mock()
        
        processor = GenerationProcessor(sample_config)
        
        assert processor.config == sample_config
        assert processor.client == mock_client
        mock_openai_class.assert_called_once_with(api_key="test-api-key")
    
    @patch('openai.OpenAI')
    def test_initialization_auth_error(self, mock_openai_class, sample_config):
        """Test initialization with authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_openai_class.side_effect = openai.AuthenticationError(
            "Invalid API key", 
            response=mock_response, 
            body={"error": "Invalid API key"}
        )
        
        with pytest.raises(GenerationError, match="Invalid OpenAI API key"):
            GenerationProcessor(sample_config)
    
    @patch('openai.OpenAI')
    def test_initialization_connection_error(self, mock_openai_class, sample_config):
        """Test initialization with connection error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        
        with pytest.raises(GenerationError, match="Failed to connect to OpenAI API"):
            GenerationProcessor(sample_config)
    
    @patch('openai.OpenAI')
    def test_generate_from_prompts_empty_list(self, mock_openai_class, sample_config):
        """Test generation with empty sample list."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = Mock()
        
        processor = GenerationProcessor(sample_config)
        result = processor.generate_from_prompts([])
        
        assert result == []
    
    @patch('openai.OpenAI')
    def test_generate_from_prompts_success(self, mock_openai_class, sample_config, sample_data, mock_openai_response):
        """Test successful generation from prompts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection and actual generation
        mock_client.chat.completions.create.side_effect = [Mock(), mock_openai_response]
        
        processor = GenerationProcessor(sample_config)
        result = processor.generate_from_prompts([sample_data])
        
        assert len(result) == 1
        assert isinstance(result[0], GeneratedSample)
        assert result[0].sample == sample_data
        assert result[0].generated_text == "This is a generated continuation of the medical text."
        assert "model" in result[0].generation_metadata
        assert result[0].generation_metadata["model"] == "gpt-4o-mini"
    
    @patch('openai.OpenAI')
    def test_generate_single_sample_rate_limit_retry(self, mock_openai_class, sample_config, sample_data, mock_openai_response):
        """Test rate limit handling with successful retry."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection, rate limit error, then success
        mock_response = Mock()
        mock_response.status_code = 429
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            openai.RateLimitError("Rate limit exceeded", response=mock_response, body={"error": "Rate limit exceeded"}),  # First attempt
            mock_openai_response  # Second attempt success
        ]
        
        processor = GenerationProcessor(sample_config)
        result = processor._generate_single_sample(sample_data)
        
        assert isinstance(result, GeneratedSample)
        assert result.generation_metadata["attempt"] == 2
    
    @patch('openai.OpenAI')
    def test_generate_single_sample_max_retries_exceeded(self, mock_openai_class, sample_config, sample_data):
        """Test generation failure after max retries."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection and repeated rate limit errors
        mock_response = Mock()
        mock_response.status_code = 429
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            openai.RateLimitError("Rate limit exceeded", response=mock_response, body={"error": "Rate limit exceeded"}),
            openai.RateLimitError("Rate limit exceeded", response=mock_response, body={"error": "Rate limit exceeded"}),
            openai.RateLimitError("Rate limit exceeded", response=mock_response, body={"error": "Rate limit exceeded"})
        ]
        
        processor = GenerationProcessor(sample_config)
        
        with pytest.raises(GenerationRateLimitError):
            processor._generate_single_sample(sample_data)
    
    @patch('openai.OpenAI')
    def test_generate_single_sample_auth_error(self, mock_openai_class, sample_config, sample_data):
        """Test generation with authentication error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection and auth error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            openai.AuthenticationError("Invalid API key", response=mock_response, body={"error": "Invalid API key"})
        ]
        
        processor = GenerationProcessor(sample_config)
        
        with pytest.raises(GenerationError, match="Authentication failed"):
            processor._generate_single_sample(sample_data)
    
    @patch('openai.OpenAI')
    def test_generate_single_sample_bad_request(self, mock_openai_class, sample_config, sample_data):
        """Test generation with bad request error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection and bad request error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            openai.BadRequestError("Invalid request", response=mock_response, body={"error": "Invalid request"})
        ]
        
        processor = GenerationProcessor(sample_config)
        
        with pytest.raises(GenerationError, match="Bad request"):
            processor._generate_single_sample(sample_data)
    
    def test_extract_prompt_short_text(self, sample_config):
        """Test prompt extraction from short text."""
        sample = Sample(
            id="test",
            content={},
            source_dataset="test",
            original_text="Short text."
        )
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = Mock()
            
            processor = GenerationProcessor(sample_config)
            prompt = processor._extract_prompt(sample, 100)
            
            assert prompt == "Short text."
    
    def test_extract_prompt_long_text_sentence_break(self, sample_config):
        """Test prompt extraction with sentence break."""
        sample = Sample(
            id="test",
            content={},
            source_dataset="test",
            original_text="This is the first sentence. This is the second sentence. This is the third sentence."
        )
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = Mock()
            
            processor = GenerationProcessor(sample_config)
            prompt = processor._extract_prompt(sample, 50)
            
            # Should break at the end of first sentence
            assert prompt == "This is the first sentence."
    
    def test_extract_prompt_long_text_word_break(self, sample_config):
        """Test prompt extraction with word break."""
        sample = Sample(
            id="test",
            content={},
            source_dataset="test",
            original_text="This is a very long text without proper sentence endings that should be broken at word boundaries"
        )
        
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = Mock()
            
            processor = GenerationProcessor(sample_config)
            prompt = processor._extract_prompt(sample, 50)
            
            # Should break at a word boundary
            assert len(prompt) <= 50
            assert not prompt.endswith(" ")  # Should not end with space
            assert " " in prompt  # Should contain at least one word
    
    def test_calculate_backoff_delay(self, sample_config):
        """Test exponential backoff delay calculation."""
        with patch('openai.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = Mock()
            
            processor = GenerationProcessor(sample_config)
            
            # Test exponential backoff
            assert processor._calculate_backoff_delay(0) == 0.1  # base_delay
            assert processor._calculate_backoff_delay(1) == 0.2  # base_delay * 2
            assert processor._calculate_backoff_delay(2) == 0.4  # base_delay * 4
            
            # Test max delay cap
            assert processor._calculate_backoff_delay(10) == 1.0  # max_delay
    
    @patch('openai.OpenAI')
    def test_validate_model_success(self, mock_openai_class, sample_config):
        """Test successful model validation."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection and model validation
        mock_client.chat.completions.create.return_value = Mock()
        
        processor = GenerationProcessor(sample_config)
        result = processor.validate_model("gpt-4o-mini")
        
        assert result is True
    
    @patch('openai.OpenAI')
    def test_validate_model_failure(self, mock_openai_class, sample_config):
        """Test model validation failure."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection success, model validation failure
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            Exception("Model not found")  # Model validation
        ]
        
        processor = GenerationProcessor(sample_config)
        result = processor.validate_model("invalid-model")
        
        assert result is False


class TestGenerationExceptions:
    """Test custom exception classes."""
    
    def test_generation_error(self):
        """Test GenerationError exception."""
        error = GenerationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_generation_rate_limit_error(self):
        """Test GenerationRateLimitError exception."""
        error = GenerationRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, GenerationError)
        assert isinstance(error, Exception)


@pytest.mark.integration
class TestGenerationProcessorIntegration:
    """Integration tests for GenerationProcessor."""
    
    @patch('openai.OpenAI')
    def test_full_generation_workflow(self, mock_openai_class, sample_config):
        """Test complete generation workflow with multiple samples."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Create multiple mock responses
        mock_responses = []
        for i in range(3):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = f"Generated content {i+1}"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 20
            mock_response.usage.completion_tokens = 15
            mock_response.usage.total_tokens = 35
            mock_responses.append(mock_response)
        
        # Mock test connection and generation calls
        mock_client.chat.completions.create.side_effect = [Mock()] + mock_responses
        
        # Create test samples
        samples = []
        for i in range(3):
            sample = Sample(
                id=f"test-sample-{i+1}",
                content={"question": f"Question {i+1}"},
                source_dataset="test_dataset",
                original_text=f"This is test content for sample {i+1}. It contains medical information."
            )
            samples.append(sample)
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts(samples)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, GeneratedSample)
            assert result.sample.id == f"test-sample-{i+1}"
            assert result.generated_text == f"Generated content {i+1}"
            assert "model" in result.generation_metadata
            assert "usage" in result.generation_metadata