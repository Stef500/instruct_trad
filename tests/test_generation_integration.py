"""
Integration tests for GenerationProcessor with other components.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import openai

from src.medical_dataset_processor.models.core import Sample, GeneratedSample, ProcessedSample, ProcessingType
from src.medical_dataset_processor.processors.generation_processor import (
    GenerationProcessor,
    GenerationConfig,
    GenerationError
)


@pytest.fixture
def sample_config():
    """Create a sample generation configuration."""
    return GenerationConfig(
        api_key="test-api-key",
        model="gpt-4o-mini",
        max_retries=2,
        base_delay=0.1,
        prompt_length=100,
        max_tokens=200,
        temperature=0.7
    )


@pytest.fixture
def medical_samples():
    """Create medical sample data for testing."""
    return [
        Sample(
            id="medqa-001",
            content={
                "question": "What is the most common cause of hypertension?",
                "answer": "Essential hypertension"
            },
            source_dataset="MedQA",
            original_text="What is the most common cause of hypertension? Essential hypertension, also known as primary hypertension, accounts for approximately 90-95% of all hypertension cases. This condition develops gradually over many years and has no identifiable cause."
        ),
        Sample(
            id="pubmed-002",
            content={
                "title": "Diabetes Management",
                "abstract": "Type 2 diabetes management strategies"
            },
            source_dataset="PubMedQA",
            original_text="Type 2 diabetes management requires a comprehensive approach including lifestyle modifications, medication management, and regular monitoring. The primary goals include maintaining optimal blood glucose levels, preventing complications, and improving quality of life."
        ),
        Sample(
            id="health-003",
            content={
                "query": "Heart disease symptoms",
                "response": "Common cardiovascular symptoms"
            },
            source_dataset="HealthSearchQA",
            original_text="Heart disease symptoms can vary depending on the specific condition, but common signs include chest pain or discomfort, shortness of breath, fatigue, irregular heartbeat, and swelling in the legs, ankles, or feet."
        )
    ]


class TestGenerationProcessorIntegration:
    """Integration tests for GenerationProcessor with real-world scenarios."""
    
    @patch('openai.OpenAI')
    def test_generate_medical_content_workflow(self, mock_openai_class, sample_config, medical_samples):
        """Test complete workflow with medical samples."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Create realistic mock responses for medical content
        mock_responses = [
            self._create_mock_response("The underlying mechanisms involve complex interactions between genetic predisposition, environmental factors, and lifestyle choices that contribute to vascular resistance."),
            self._create_mock_response("Patients should work closely with healthcare providers to develop individualized treatment plans that may include metformin, insulin therapy, and dietary counseling."),
            self._create_mock_response("Early detection and prompt medical attention are crucial for preventing serious complications such as heart attack, stroke, or heart failure.")
        ]
        
        # Mock test connection and generation calls
        mock_client.chat.completions.create.side_effect = [Mock()] + mock_responses
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts(medical_samples)
        
        # Verify results
        assert len(results) == 3
        
        for i, result in enumerate(results):
            assert isinstance(result, GeneratedSample)
            assert result.sample == medical_samples[i]
            assert len(result.prompt) > 0
            assert len(result.generated_text) > 0
            assert result.generation_metadata["model"] == "gpt-4o-mini"
            assert "usage" in result.generation_metadata
            assert isinstance(result.processing_timestamp, datetime)
    
    @patch('openai.OpenAI')
    def test_prompt_extraction_quality(self, mock_openai_class, sample_config):
        """Test that prompt extraction maintains medical context."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = Mock()
        
        processor = GenerationProcessor(sample_config)
        
        # Test with medical text that has clear sentence boundaries
        medical_sample = Sample(
            id="test-prompt",
            content={},
            source_dataset="test",
            original_text="Hypertension is a chronic medical condition. It affects millions of people worldwide. The condition is characterized by persistently elevated blood pressure readings above 140/90 mmHg. Treatment typically involves lifestyle modifications and medication."
        )
        
        prompt = processor._extract_prompt(medical_sample, 80)
        
        # Should break at sentence boundary and maintain medical context
        assert prompt == "Hypertension is a chronic medical condition."
        assert len(prompt) <= 80
    
    @patch('openai.OpenAI')
    def test_generated_sample_to_processed_sample_conversion(self, mock_openai_class, sample_config, medical_samples):
        """Test conversion of GeneratedSample to ProcessedSample."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = self._create_mock_response("Generated medical content continuation.")
        mock_client.chat.completions.create.side_effect = [Mock(), mock_response]
        
        processor = GenerationProcessor(sample_config)
        generated_samples = processor.generate_from_prompts([medical_samples[0]])
        
        # Convert to ProcessedSample (simulating what would happen in the consolidator)
        generated_sample = generated_samples[0]
        processed_sample = ProcessedSample(
            original_sample=generated_sample.sample,
            processed_content=f"{generated_sample.prompt} {generated_sample.generated_text}",
            processing_type=ProcessingType.GENERATION.value,
            metadata={
                "generation_metadata": generated_sample.generation_metadata,
                "prompt_length": len(generated_sample.prompt),
                "generated_length": len(generated_sample.generated_text)
            }
        )
        
        # Verify conversion
        assert isinstance(processed_sample, ProcessedSample)
        assert processed_sample.processing_type == ProcessingType.GENERATION.value
        assert processed_sample.original_sample == medical_samples[0]
        assert "generation_metadata" in processed_sample.metadata
    
    @patch('openai.OpenAI')
    def test_error_handling_with_partial_success(self, mock_openai_class, sample_config, medical_samples):
        """Test handling of mixed success/failure scenarios."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock test connection, success, failure, success
        mock_response = Mock()
        mock_response.status_code = 429
        mock_client.chat.completions.create.side_effect = [
            Mock(),  # Test connection
            self._create_mock_response("First sample generated successfully."),
            openai.RateLimitError("Rate limit", response=mock_response, body={"error": "Rate limit"}),  # All retries fail
            openai.RateLimitError("Rate limit", response=mock_response, body={"error": "Rate limit"}),
            openai.RateLimitError("Rate limit", response=mock_response, body={"error": "Rate limit"}),
            self._create_mock_response("Third sample generated successfully.")
        ]
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts(medical_samples)
        
        # Should have 2 successful results (first and third samples)
        assert len(results) == 2
        assert results[0].sample == medical_samples[0]
        assert results[1].sample == medical_samples[2]
    
    @patch('openai.OpenAI')
    def test_metadata_completeness(self, mock_openai_class, sample_config, medical_samples):
        """Test that all required metadata is included in generated samples."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = self._create_mock_response("Generated content with full metadata.")
        mock_client.chat.completions.create.side_effect = [Mock(), mock_response]
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts([medical_samples[0]])
        
        result = results[0]
        metadata = result.generation_metadata
        
        # Verify all expected metadata fields
        required_fields = ["model", "prompt_length", "max_tokens", "temperature", "attempt", "usage"]
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # Verify usage metadata structure
        usage = metadata["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        
        # Verify values are reasonable
        assert metadata["model"] == "gpt-4o-mini"
        assert metadata["prompt_length"] > 0
        assert metadata["max_tokens"] == 200
        assert metadata["temperature"] == 0.7
        assert metadata["attempt"] == 1
    
    def _create_mock_response(self, content: str):
        """Helper to create a mock OpenAI response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = len(content.split())
        mock_response.usage.total_tokens = 25 + len(content.split())
        return mock_response


class TestGenerationProcessorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('openai.OpenAI')
    def test_very_short_original_text(self, mock_openai_class, sample_config):
        """Test generation with very short original text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated continuation."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        
        mock_client.chat.completions.create.side_effect = [Mock(), mock_response]
        
        short_sample = Sample(
            id="short",
            content={},
            source_dataset="test",
            original_text="Pain."
        )
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts([short_sample])
        
        assert len(results) == 1
        assert results[0].prompt == "Pain."
        assert results[0].generated_text == "Generated continuation."
    
    @patch('openai.OpenAI')
    def test_very_long_original_text(self, mock_openai_class, sample_config):
        """Test generation with very long original text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated continuation."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 53
        
        mock_client.chat.completions.create.side_effect = [Mock(), mock_response]
        
        # Create a very long medical text
        long_text = "Cardiovascular disease represents a complex pathophysiological condition. " * 20
        long_sample = Sample(
            id="long",
            content={},
            source_dataset="test",
            original_text=long_text
        )
        
        processor = GenerationProcessor(sample_config)
        results = processor.generate_from_prompts([long_sample])
        
        assert len(results) == 1
        # Prompt should be truncated to reasonable length
        assert len(results[0].prompt) <= sample_config.prompt_length
        # Should end at a sentence boundary
        assert results[0].prompt.endswith(".")