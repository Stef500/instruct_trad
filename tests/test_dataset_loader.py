"""
Unit tests for DatasetLoader.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import yaml
from datasets import Dataset

from src.medical_dataset_processor.loaders.dataset_loader import DatasetLoader
from src.medical_dataset_processor.models.core import DatasetConfig, Sample


class TestDatasetLoader(unittest.TestCase):
    """Test cases for DatasetLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DatasetLoader()
        
        # Sample configuration data
        self.sample_config_data = {
            "test_dataset": {
                "name": "test_dataset",
                "source_type": "local",
                "source_path": "/path/to/test.json",
                "format": "json",
                "text_fields": ["question", "answer"]
            },
            "hf_dataset": {
                "name": "hf_dataset",
                "source_type": "huggingface",
                "source_path": "test/dataset",
                "format": "json",
                "text_fields": ["text"]
            }
        }
        
        # Sample dataset data
        self.sample_dataset_data = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "id": 1
            },
            {
                "question": "What is 2+2?",
                "answer": "4",
                "id": 2
            }
        ]
    
    def test_init(self):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader()
        self.assertEqual(loader.loaded_configs, {})
        self.assertEqual(loader.loaded_datasets, {})
    
    def test_load_config_success(self):
        """Test successful configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.sample_config_data, f)
            temp_path = f.name
        
        try:
            configs = self.loader.load_config(temp_path)
            
            self.assertEqual(len(configs), 2)
            self.assertIn("test_dataset", configs)
            self.assertIn("hf_dataset", configs)
            
            test_config = configs["test_dataset"]
            self.assertEqual(test_config.name, "test_dataset")
            self.assertEqual(test_config.source_type, "local")
            self.assertEqual(test_config.source_path, "/path/to/test.json")
            self.assertEqual(test_config.format, "json")
            self.assertEqual(test_config.text_fields, ["question", "answer"])
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_config("/non/existent/file.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test configuration loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                self.loader.load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_invalid_dataset_config(self):
        """Test configuration loading with invalid dataset configuration."""
        invalid_config = {
            "invalid_dataset": {
                "name": "test",
                # Missing required fields
                "source_type": "invalid_type"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.loader.load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_config_empty_file(self):
        """Test configuration loading with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            configs = self.loader.load_config(temp_path)
            self.assertEqual(configs, {})
        finally:
            os.unlink(temp_path)
    
    @patch('src.medical_dataset_processor.loaders.dataset_loader.load_dataset')
    def test_fetch_dataset_huggingface_success(self, mock_load_dataset):
        """Test successful Hugging Face dataset fetching."""
        # Mock dataset
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        config = DatasetConfig(
            name="test_hf",
            source_type="huggingface",
            source_path="test/dataset",
            format="json",
            text_fields=["text"]
        )
        
        result = self.loader.fetch_dataset("test_hf", config)
        
        self.assertEqual(result, mock_dataset)
        mock_load_dataset.assert_called_once_with("test/dataset", split="train")
    
    @patch('src.medical_dataset_processor.loaders.dataset_loader.load_dataset')
    def test_fetch_dataset_medqa(self, mock_load_dataset):
        """Test MedQA dataset fetching."""
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        config = DatasetConfig(
            name="medqa",
            source_type="huggingface",
            source_path="bigbio/med_qa",
            format="json",
            text_fields=["question", "answer"]
        )
        
        result = self.loader.fetch_dataset("medqa", config)
        
        self.assertEqual(result, mock_dataset)
        mock_load_dataset.assert_called_once_with("bigbio/med_qa", split="train")
    
    @patch('src.medical_dataset_processor.loaders.dataset_loader.load_dataset')
    def test_fetch_dataset_mmlu(self, mock_load_dataset):
        """Test MMLU clinical dataset fetching."""
        mock_dataset = Mock(spec=Dataset)
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset
        
        config = DatasetConfig(
            name="mmlu_clinical",
            source_type="huggingface",
            source_path="cais/mmlu",
            format="json",
            text_fields=["question", "choices", "answer"]
        )
        
        result = self.loader.fetch_dataset("mmlu_clinical", config)
        
        self.assertEqual(result, mock_dataset)
        mock_load_dataset.assert_called_once_with("cais/mmlu", "clinical_knowledge", split="test")
    
    def test_fetch_dataset_local_json_success(self):
        """Test successful local JSON dataset fetching."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_dataset_data, f)
            temp_path = f.name
        
        try:
            config = DatasetConfig(
                name="test_local",
                source_type="local",
                source_path=temp_path,
                format="json",
                text_fields=["question", "answer"]
            )
            
            result = self.loader.fetch_dataset("test_local", config)
            
            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(temp_path)
    
    def test_fetch_dataset_local_jsonl_success(self):
        """Test successful local JSONL dataset fetching."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in self.sample_dataset_data:
                f.write(json.dumps(item) + '\n')
            temp_path = f.name
        
        try:
            config = DatasetConfig(
                name="test_local_jsonl",
                source_type="local",
                source_path=temp_path,
                format="jsonl",
                text_fields=["question", "answer"]
            )
            
            result = self.loader.fetch_dataset("test_local_jsonl", config)
            
            self.assertIsInstance(result, Dataset)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(temp_path)
    
    def test_fetch_dataset_local_file_not_found(self):
        """Test local dataset fetching with non-existent file."""
        config = DatasetConfig(
            name="test_missing",
            source_type="local",
            source_path="/non/existent/file.json",
            format="json",
            text_fields=["text"]
        )
        
        with self.assertRaises(Exception):
            self.loader.fetch_dataset("test_missing", config)
    
    @patch('urllib.request.urlretrieve')
    @patch('os.remove')
    def test_fetch_dataset_url_success(self, mock_remove, mock_urlretrieve):
        """Test successful URL dataset fetching."""
        # Create a temporary file to simulate downloaded content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_dataset_data, f)
            temp_path = f.name
        
        # Mock urlretrieve to use our temp file
        mock_urlretrieve.side_effect = lambda url, filename: None
        
        try:
            config = DatasetConfig(
                name="test_url",
                source_type="url",
                source_path="http://example.com/dataset.json",
                format="json",
                text_fields=["question", "answer"]
            )
            
            # Patch the temp file path
            with patch('src.medical_dataset_processor.loaders.dataset_loader.DatasetLoader._load_local_dataset') as mock_load_local:
                mock_dataset = Mock(spec=Dataset)
                mock_dataset.__len__ = Mock(return_value=2)
                mock_load_local.return_value = mock_dataset
                
                result = self.loader.fetch_dataset("test_url", config)
                
                self.assertEqual(result, mock_dataset)
                mock_urlretrieve.assert_called_once()
                mock_remove.assert_called_once()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_fetch_dataset_unsupported_source_type(self):
        """Test dataset fetching with unsupported source type."""
        # Create a config with valid source type first, then modify it
        config = DatasetConfig(
            name="test_unsupported",
            source_type="local",
            source_path="path",
            format="json",
            text_fields=["text"]
        )
        # Manually set unsupported source type to bypass validation
        config.source_type = "unsupported"
        
        with self.assertRaises(Exception):
            self.loader.fetch_dataset("test_unsupported", config)
    
    def test_fetch_dataset_no_config(self):
        """Test dataset fetching without configuration."""
        with self.assertRaises(ValueError):
            self.loader.fetch_dataset("non_existent")
    
    def test_fetch_dataset_caching(self):
        """Test dataset caching functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_dataset_data, f)
            temp_path = f.name
        
        try:
            config = DatasetConfig(
                name="test_cache",
                source_type="local",
                source_path=temp_path,
                format="json",
                text_fields=["question", "answer"]
            )
            
            # First fetch
            result1 = self.loader.fetch_dataset("test_cache", config)
            
            # Second fetch should use cache
            result2 = self.loader.fetch_dataset("test_cache", config)
            
            self.assertEqual(result1, result2)
        finally:
            os.unlink(temp_path)
    
    def test_convert_to_samples(self):
        """Test conversion of dataset to samples."""
        dataset = Dataset.from_list(self.sample_dataset_data)
        config = DatasetConfig(
            name="test_convert",
            source_type="local",
            source_path="test.json",
            format="json",
            text_fields=["question", "answer"]
        )
        
        samples = self.loader.convert_to_samples(dataset, config)
        
        self.assertEqual(len(samples), 2)
        
        sample1 = samples[0]
        self.assertEqual(sample1.id, "test_convert_0")
        self.assertEqual(sample1.source_dataset, "test_convert")
        self.assertEqual(sample1.original_text, "What is the capital of France? Paris")
        self.assertEqual(sample1.content["question"], "What is the capital of France?")
        self.assertEqual(sample1.content["answer"], "Paris")
    
    def test_convert_to_samples_missing_fields(self):
        """Test conversion with missing text fields."""
        dataset_data = [
            {"question": "Test question", "other_field": "value"},
            {"answer": "Test answer", "other_field": "value"}
        ]
        dataset = Dataset.from_list(dataset_data)
        config = DatasetConfig(
            name="test_missing",
            source_type="local",
            source_path="test.json",
            format="json",
            text_fields=["question", "answer"]
        )
        
        samples = self.loader.convert_to_samples(dataset, config)
        
        # Should create samples for items that have at least one text field
        # First item has "question", second item has "answer" but missing "question"
        # The second item will be skipped due to missing required fields
        self.assertEqual(len(samples), 1)  # Only first item should be converted
        self.assertEqual(samples[0].original_text, "Test question")
    
    def test_get_available_datasets(self):
        """Test getting available dataset names."""
        self.loader.loaded_configs = {
            "dataset1": Mock(),
            "dataset2": Mock()
        }
        
        available = self.loader.get_available_datasets()
        
        self.assertEqual(set(available), {"dataset1", "dataset2"})
    
    def test_clear_cache(self):
        """Test cache clearing."""
        self.loader.loaded_datasets = {"test": Mock()}
        
        self.loader.clear_cache()
        
        self.assertEqual(self.loader.loaded_datasets, {})


if __name__ == '__main__':
    unittest.main()