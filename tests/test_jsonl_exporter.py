"""
Unit tests for JSONLExporter.
"""
import json
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open

from src.medical_dataset_processor.models.core import (
    Sample, ProcessedSample, ConsolidatedDataset, ProcessingType
)
from src.medical_dataset_processor.exporters.jsonl_exporter import JSONLExporter


class TestJSONLExporter:
    """Test cases for JSONLExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = JSONLExporter()
        
        # Create test samples
        self.sample1 = Sample(
            id="test_1",
            content={"question": "What is fever?", "answer": "High body temperature"},
            source_dataset="MedQA",
            original_text="What is fever? High body temperature"
        )
        
        self.sample2 = Sample(
            id="test_2", 
            content={"question": "What causes headache?", "answer": "Various factors"},
            source_dataset="PubMedQA",
            original_text="What causes headache? Various factors"
        )
        
        # Create processed samples
        self.processed_sample1 = ProcessedSample(
            original_sample=self.sample1,
            processed_content="Qu'est-ce que la fièvre? Température corporelle élevée",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={"api_used": "deepl", "confidence": 0.95},
            quality_score=0.9
        )
        
        self.processed_sample2 = ProcessedSample(
            original_sample=self.sample2,
            processed_content="Headaches can be caused by stress, dehydration, or medical conditions.",
            processing_type=ProcessingType.GENERATION.value,
            metadata={"api_used": "openai", "model": "gpt-4o-mini"},
            quality_score=0.85
        )
        
        # Create consolidated dataset
        self.dataset = ConsolidatedDataset(
            samples=[self.processed_sample1, self.processed_sample2],
            metadata={"version": "1.0", "source": "test"}
        )
    
    def test_init(self):
        """Test JSONLExporter initialization."""
        exporter = JSONLExporter()
        assert exporter.exported_count == 0
        assert exporter.validation_errors == []
    
    def test_export_success(self):
        """Test successful export to JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.exporter.export(self.dataset, tmp_path)
            
            # Verify file was created and has content
            assert Path(tmp_path).exists()
            
            # Read and verify content
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Should have 2 sample lines + 1 metadata line
            assert len(lines) == 3
            
            # Verify each line is valid JSON
            for line in lines:
                data = json.loads(line.strip())
                assert isinstance(data, dict)
            
            # Verify first sample
            sample1_data = json.loads(lines[0].strip())
            assert sample1_data["id"] == "test_1"
            assert sample1_data["source_dataset"] == "MedQA"
            assert sample1_data["processing_type"] == "translation"
            assert sample1_data["record_type"] == "processed_sample"
            
            # Verify metadata line
            metadata_data = json.loads(lines[-1].strip())
            assert metadata_data["record_type"] == "dataset_metadata"
            assert metadata_data["total_samples"] == 2
            assert metadata_data["translation_samples"] == 1
            assert metadata_data["generation_samples"] == 1
            
            # Verify export stats
            assert self.exporter.exported_count == 2
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_export_invalid_dataset(self):
        """Test export with invalid dataset."""
        with pytest.raises(ValueError, match="Dataset must be a ConsolidatedDataset instance"):
            self.exporter.export("invalid", "output.jsonl")
    
    def test_export_empty_output_path(self):
        """Test export with empty output path."""
        with pytest.raises(ValueError, match="Output path cannot be empty"):
            self.exporter.export(self.dataset, "")
    
    def test_export_creates_directory(self):
        """Test that export creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "subdir" / "output.jsonl"
            
            self.exporter.export(self.dataset, str(output_path))
            
            assert output_path.exists()
            assert output_path.parent.exists()
    
    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_export_io_error(self, mock_file):
        """Test export with IO error."""
        with pytest.raises(IOError, match="Permission denied"):
            self.exporter.export(self.dataset, "output.jsonl")
    
    def test_convert_sample_to_jsonl(self):
        """Test conversion of ProcessedSample to JSONL format."""
        with patch('src.medical_dataset_processor.exporters.jsonl_exporter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
            
            jsonl_data = self.exporter._convert_sample_to_jsonl(self.processed_sample1)
            
            # Verify required fields
            assert jsonl_data["id"] == "test_1"
            assert jsonl_data["source_dataset"] == "MedQA"
            assert jsonl_data["original_text"] == "What is fever? High body temperature"
            assert jsonl_data["processed_content"] == "Qu'est-ce que la fièvre? Température corporelle élevée"
            assert jsonl_data["processing_type"] == "translation"
            assert jsonl_data["record_type"] == "processed_sample"
            
            # Verify metadata
            assert jsonl_data["metadata"]["api_used"] == "deepl"
            assert jsonl_data["metadata"]["confidence"] == 0.95
            assert jsonl_data["metadata"]["quality_score"] == 0.9
            assert jsonl_data["metadata"]["export_timestamp"] == "2024-01-01T12:00:00"
            
            # Verify original content is preserved
            assert jsonl_data["original_content"] == self.sample1.content
    
    def test_create_metadata_line(self):
        """Test creation of dataset metadata line."""
        with patch('src.medical_dataset_processor.exporters.jsonl_exporter.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
            
            self.exporter.exported_count = 2
            metadata_line = self.exporter._create_metadata_line(self.dataset)
            
            assert metadata_line["record_type"] == "dataset_metadata"
            assert metadata_line["total_samples"] == 2
            assert metadata_line["translation_samples"] == 1
            assert metadata_line["generation_samples"] == 1
            assert metadata_line["samples_by_dataset"] == {"MedQA": 1, "PubMedQA": 1}
            assert metadata_line["export_timestamp"] == "2024-01-01T12:00:00"
            assert metadata_line["dataset_metadata"] == {"version": "1.0", "source": "test"}
            assert metadata_line["export_stats"]["exported_count"] == 2
    
    def test_validate_jsonl_format_valid_sample(self):
        """Test JSONL format validation with valid sample data."""
        valid_data = {
            "id": "test_1",
            "source_dataset": "MedQA",
            "original_text": "What is fever?",
            "processed_content": "Qu'est-ce que la fièvre?",
            "processing_type": "translation",
            "record_type": "processed_sample",
            "metadata": {}
        }
        
        assert self.exporter._validate_jsonl_format(valid_data) is True
        assert len(self.exporter.validation_errors) == 0
    
    def test_validate_jsonl_format_valid_metadata(self):
        """Test JSONL format validation with valid metadata."""
        valid_metadata = {
            "record_type": "dataset_metadata",
            "total_samples": 10,
            "creation_timestamp": "2024-01-01T12:00:00",
            "export_timestamp": "2024-01-01T12:00:00"
        }
        
        assert self.exporter._validate_jsonl_format(valid_metadata) is True
        assert len(self.exporter.validation_errors) == 0
    
    def test_validate_jsonl_format_invalid_data_type(self):
        """Test JSONL format validation with invalid data type."""
        assert self.exporter._validate_jsonl_format("not a dict") is False
        assert "Data must be a dictionary" in self.exporter.validation_errors
    
    def test_validate_jsonl_format_missing_required_fields(self):
        """Test JSONL format validation with missing required fields."""
        invalid_data = {
            "id": "test_1",
            "record_type": "processed_sample"
            # Missing required fields
        }
        
        assert self.exporter._validate_jsonl_format(invalid_data) is False
        assert any("Missing required field" in error for error in self.exporter.validation_errors)
    
    def test_validate_jsonl_format_empty_string_fields(self):
        """Test JSONL format validation with empty string fields."""
        invalid_data = {
            "id": "",  # Empty string
            "source_dataset": "MedQA",
            "original_text": "What is fever?",
            "processed_content": "Qu'est-ce que la fièvre?",
            "processing_type": "translation",
            "record_type": "processed_sample"
        }
        
        assert self.exporter._validate_jsonl_format(invalid_data) is False
        assert any("cannot be empty" in error for error in self.exporter.validation_errors)
    
    def test_validate_jsonl_format_invalid_processing_type(self):
        """Test JSONL format validation with invalid processing type."""
        invalid_data = {
            "id": "test_1",
            "source_dataset": "MedQA",
            "original_text": "What is fever?",
            "processed_content": "Qu'est-ce que la fièvre?",
            "processing_type": "invalid_type",
            "record_type": "processed_sample"
        }
        
        assert self.exporter._validate_jsonl_format(invalid_data) is False
        assert any("Invalid processing_type" in error for error in self.exporter.validation_errors)
    
    def test_validate_jsonl_format_unknown_record_type(self):
        """Test JSONL format validation with unknown record type."""
        invalid_data = {
            "record_type": "unknown_type"
        }
        
        assert self.exporter._validate_jsonl_format(invalid_data) is False
        assert any("Unknown record_type" in error for error in self.exporter.validation_errors)
    
    def test_validate_jsonl_format_non_serializable(self):
        """Test JSONL format validation with non-JSON-serializable data."""
        invalid_data = {
            "id": "test_1",
            "source_dataset": "MedQA",
            "original_text": "What is fever?",
            "processed_content": "Qu'est-ce que la fièvre?",
            "processing_type": "translation",
            "record_type": "processed_sample",
            "non_serializable": datetime.now()  # datetime objects are not JSON serializable by default
        }
        
        assert self.exporter._validate_jsonl_format(invalid_data) is False
        assert any("not JSON serializable" in error for error in self.exporter.validation_errors)
    
    def test_validate_jsonl_file_success(self):
        """Test validation of existing JSONL file."""
        # Create a valid JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            # Write valid JSONL content
            sample_line = {
                "id": "test_1",
                "source_dataset": "MedQA",
                "original_text": "What is fever?",
                "processed_content": "Qu'est-ce que la fièvre?",
                "processing_type": "translation",
                "record_type": "processed_sample",
                "metadata": {}
            }
            metadata_line = {
                "record_type": "dataset_metadata",
                "total_samples": 1,
                "creation_timestamp": "2024-01-01T12:00:00",
                "export_timestamp": "2024-01-01T12:00:00"
            }
            
            tmp_file.write(json.dumps(sample_line) + '\n')
            tmp_file.write(json.dumps(metadata_line) + '\n')
            tmp_path = tmp_file.name
        
        try:
            result = self.exporter.validate_jsonl_file(tmp_path)
            
            assert result["valid"] is True
            assert result["line_count"] == 2
            assert result["sample_count"] == 1
            assert result["metadata_count"] == 1
            assert result["file_path"] == tmp_path
            assert len(result["errors"]) == 0
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_jsonl_file_not_exists(self):
        """Test validation of non-existent file."""
        result = self.exporter.validate_jsonl_file("nonexistent.jsonl")
        
        assert result["valid"] is False
        assert "File does not exist" in result["error"]
        assert result["line_count"] == 0
        assert result["sample_count"] == 0
        assert result["metadata_count"] == 0
    
    def test_validate_jsonl_file_invalid_json(self):
        """Test validation of file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_file.write('{"valid": "json"}\n')
            tmp_file.write('invalid json line\n')  # Invalid JSON
            tmp_path = tmp_file.name
        
        try:
            result = self.exporter.validate_jsonl_file(tmp_path)
            
            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert any("Invalid JSON" in error for error in result["errors"])
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_jsonl_file_empty_lines(self):
        """Test validation of file with empty lines (should be skipped)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            sample_line = {
                "id": "test_1",
                "source_dataset": "MedQA",
                "original_text": "What is fever?",
                "processed_content": "Qu'est-ce que la fièvre?",
                "processing_type": "translation",
                "record_type": "processed_sample",
                "metadata": {}
            }
            
            tmp_file.write(json.dumps(sample_line) + '\n')
            tmp_file.write('\n')  # Empty line
            tmp_file.write('   \n')  # Whitespace only line
            tmp_path = tmp_file.name
        
        try:
            result = self.exporter.validate_jsonl_file(tmp_path)
            
            # Empty lines should be skipped, so only 1 sample should be counted
            assert result["sample_count"] == 1
            assert result["line_count"] == 3  # Total lines including empty ones
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_validate_jsonl_file_io_error(self):
        """Test validation with IO error."""
        # Create a file that exists but mock the open to fail
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with patch("builtins.open", side_effect=IOError("Permission denied")):
                result = self.exporter.validate_jsonl_file(tmp_path)
            
            assert result["valid"] is False
            assert "Cannot read file" in result["error"]
            assert result["line_count"] == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_get_export_stats(self):
        """Test getting export statistics."""
        # Set some test values
        self.exporter.exported_count = 5
        self.exporter.validation_errors = ["error1", "error2"]
        
        stats = self.exporter.get_export_stats()
        
        assert stats["exported_count"] == 5
        assert stats["validation_errors"] == 2
        assert stats["errors"] == ["error1", "error2"]
        
        # Verify it returns a copy of errors (not reference)
        stats["errors"].append("error3")
        assert len(self.exporter.validation_errors) == 2
    
    def test_export_with_validation_errors(self):
        """Test export behavior when some samples fail validation."""
        # Create a sample with invalid processing_type that will fail JSONL validation
        # We need to bypass the ProcessedSample validation by creating it with valid data
        # then modifying it to have invalid data for JSONL validation
        valid_sample = ProcessedSample(
            original_sample=self.sample1,
            processed_content="Valid content",
            processing_type=ProcessingType.TRANSLATION.value,
            metadata={}
        )
        
        # Mock the _convert_sample_to_jsonl method to return invalid data for one sample
        original_convert = self.exporter._convert_sample_to_jsonl
        
        def mock_convert(sample):
            if sample == valid_sample:
                # Return invalid data that will fail JSONL validation
                return {
                    "id": "",  # Empty ID will fail validation
                    "source_dataset": "MedQA",
                    "original_text": "What is fever?",
                    "processed_content": "Valid content",
                    "processing_type": "translation",
                    "record_type": "processed_sample"
                }
            else:
                return original_convert(sample)
        
        dataset_with_invalid = ConsolidatedDataset(
            samples=[self.processed_sample1, valid_sample],
            metadata={}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with patch.object(self.exporter, '_convert_sample_to_jsonl', side_effect=mock_convert):
                self.exporter.export(dataset_with_invalid, tmp_path)
            
            # Should export only the valid sample (the first one)
            assert self.exporter.exported_count == 1
            assert len(self.exporter.validation_errors) > 0
            
            # Verify file content
            with open(tmp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Should have 1 valid sample + 1 metadata line
            assert len(lines) == 2
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)