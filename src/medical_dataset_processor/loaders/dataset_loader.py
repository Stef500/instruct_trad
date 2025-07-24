"""
Dataset loader for medical datasets with support for multiple source types.
"""
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset, load_dataset

from ..models.core import DatasetConfig, Sample


logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and manages medical datasets from various sources."""
    
    def __init__(self):
        """Initialize the dataset loader."""
        self.loaded_configs: Dict[str, DatasetConfig] = {}
        self.loaded_datasets: Dict[str, Dataset] = {}
    
    def load_config(self, yaml_path: str) -> Dict[str, DatasetConfig]:
        """
        Load dataset configurations from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Dictionary mapping dataset names to their configurations
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If the configuration is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}")
        
        if not config_data:
            logger.warning(f"Empty configuration file: {yaml_path}")
            return {}
        
        configs = {}
        for dataset_name, dataset_config in config_data.items():
            try:
                config = DatasetConfig(
                    name=dataset_config.get('name', dataset_name),
                    source_type=dataset_config['source_type'],
                    source_path=dataset_config['source_path'],
                    format=dataset_config['format'],
                    text_fields=dataset_config['text_fields']
                )
                configs[dataset_name] = config
                logger.info(f"Loaded configuration for dataset: {dataset_name}")
            except (KeyError, ValueError) as e:
                logger.error(f"Invalid configuration for dataset {dataset_name}: {e}")
                raise ValueError(f"Invalid configuration for dataset {dataset_name}: {e}")
        
        self.loaded_configs.update(configs)
        return configs
    
    def fetch_dataset(self, dataset_name: str, config: Optional[DatasetConfig] = None) -> Dataset:
        """
        Fetch a dataset based on its configuration.
        
        Args:
            dataset_name: Name of the dataset to fetch
            config: Optional dataset configuration. If not provided, uses loaded config.
            
        Returns:
            The loaded dataset
            
        Raises:
            ValueError: If dataset configuration is not found or invalid
            Exception: If dataset loading fails
        """
        if config is None:
            if dataset_name not in self.loaded_configs:
                raise ValueError(f"No configuration found for dataset: {dataset_name}")
            config = self.loaded_configs[dataset_name]
        
        # Check if dataset is already loaded
        cache_key = f"{dataset_name}_{config.source_type}_{config.source_path}"
        if cache_key in self.loaded_datasets:
            logger.info(f"Using cached dataset: {dataset_name}")
            return self.loaded_datasets[cache_key]
        
        try:
            if config.source_type == "huggingface":
                dataset = self._load_huggingface_dataset(config)
            elif config.source_type == "local":
                dataset = self._load_local_dataset(config)
            elif config.source_type == "url":
                dataset = self._load_url_dataset(config)
            else:
                raise ValueError(f"Unsupported source type: {config.source_type}")
            
            # Cache the loaded dataset
            self.loaded_datasets[cache_key] = dataset
            logger.info(f"Successfully loaded dataset: {dataset_name} ({len(dataset)} samples)")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise Exception(f"Failed to load dataset {dataset_name}: {e}")
    
    def _load_huggingface_dataset(self, config: DatasetConfig) -> Dataset:
        """Load a dataset from Hugging Face Hub."""
        try:
            # Handle specific dataset configurations
            if "medqa" in config.source_path.lower():
                # MedQA dataset
                dataset = load_dataset(config.source_path, split="train")
            elif "pubmedqa" in config.source_path.lower():
                # PubMedQA dataset
                dataset = load_dataset(config.source_path, split="train")
            elif "mmlu" in config.source_path.lower():
                # MMLU clinical subjects
                dataset = load_dataset(config.source_path, "clinical_knowledge", split="test")
            else:
                # Generic Hugging Face dataset
                dataset = load_dataset(config.source_path, split="train")
            
            return dataset
        except Exception as e:
            raise Exception(f"Failed to load Hugging Face dataset {config.source_path}: {e}")
    
    def _load_local_dataset(self, config: DatasetConfig) -> Dataset:
        """Load a dataset from local file."""
        file_path = Path(config.source_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {config.source_path}")
        
        try:
            if config.format.lower() == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    dataset = Dataset.from_list(data)
                else:
                    dataset = Dataset.from_dict(data)
            elif config.format.lower() == "jsonl":
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                dataset = Dataset.from_list(data)
            elif config.format.lower() == "csv":
                dataset = Dataset.from_csv(str(file_path))
            elif config.format.lower() == "parquet":
                dataset = Dataset.from_parquet(str(file_path))
            else:
                raise ValueError(f"Unsupported local file format: {config.format}")
            
            return dataset
        except Exception as e:
            raise Exception(f"Failed to load local dataset {config.source_path}: {e}")
    
    def _load_url_dataset(self, config: DatasetConfig) -> Dataset:
        """Load a dataset from URL."""
        try:
            # Download the file to a temporary location
            temp_file = f"/tmp/{config.name}_{config.format}"
            urllib.request.urlretrieve(config.source_path, temp_file)
            
            # Create a temporary config for local loading
            temp_config = DatasetConfig(
                name=config.name,
                source_type="local",
                source_path=temp_file,
                format=config.format,
                text_fields=config.text_fields
            )
            
            dataset = self._load_local_dataset(temp_config)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            return dataset
        except Exception as e:
            raise Exception(f"Failed to load dataset from URL {config.source_path}: {e}")
    
    def convert_to_samples(self, dataset: Dataset, config: DatasetConfig) -> List[Sample]:
        """
        Convert a dataset to a list of Sample objects.
        
        Args:
            dataset: The dataset to convert
            config: Configuration specifying text fields
            
        Returns:
            List of Sample objects
        """
        samples = []
        
        for i, item in enumerate(dataset):
            # Extract text from specified fields
            text_parts = []
            for field in config.text_fields:
                if field in item and item[field]:
                    text_parts.append(str(item[field]))
            
            if not text_parts:
                logger.warning(f"No text found in fields {config.text_fields} for item {i}")
                continue
            
            original_text = " ".join(text_parts)
            
            sample = Sample(
                id=f"{config.name}_{i}",
                content=item,
                source_dataset=config.name,
                original_text=original_text
            )
            samples.append(sample)
        
        logger.info(f"Converted {len(samples)} items to samples for dataset {config.name}")
        return samples
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.loaded_configs.keys())
    
    def clear_cache(self):
        """Clear the dataset cache."""
        self.loaded_datasets.clear()
        logger.info("Dataset cache cleared")