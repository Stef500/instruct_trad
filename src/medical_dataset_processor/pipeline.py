"""
Main pipeline orchestrator for medical dataset processing.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .loaders.dataset_loader import DatasetLoader
from .processors.sample_selector import SampleSelector
from .processors.translation_processor import TranslationProcessor, TranslationConfig
from .processors.generation_processor import GenerationProcessor, GenerationConfig
from .processors.dataset_consolidator import DatasetConsolidator
from .exporters.jsonl_exporter import JSONLExporter
from .exporters.pdf_sample_generator import PDFSampleGenerator
from .models.core import ConsolidatedDataset, Sample
from .utils.logging import ProcessingLogger


@dataclass
class PipelineConfig:
    """Configuration for the medical dataset processing pipeline."""
    
    # Dataset configuration
    datasets_yaml_path: str = "datasets.yaml"
    
    # API configurations
    deepl_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Processing parameters
    translation_count: int = 50
    generation_count: int = 50
    target_language: str = "FR"
    
    # Output configuration
    output_dir: str = "output"
    jsonl_filename: str = "consolidated_dataset.jsonl"
    pdf_filename: str = "sample_review.pdf"
    pdf_sample_size: int = 100
    
    # Processing options
    random_seed: Optional[int] = None
    max_retries: int = 3
    batch_size: int = 10
    
    # Advanced options
    enable_parallel_processing: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.translation_count <= 0:
            raise ValueError("translation_count must be positive")
        
        if self.generation_count <= 0:
            raise ValueError("generation_count must be positive")
        
        if not self.datasets_yaml_path:
            raise ValueError("datasets_yaml_path cannot be empty")
        
        # Try to get API keys from environment if not provided
        if not self.deepl_api_key:
            self.deepl_api_key = os.getenv("DEEPL_API_KEY")
        
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


class MedicalDatasetProcessor:
    """
    Main orchestrator for the medical dataset processing pipeline.
    
    This class coordinates the entire pipeline:
    1. Load datasets from configuration
    2. Select samples for translation and generation
    3. Process samples through translation and generation APIs
    4. Consolidate results into a unified dataset
    5. Export to JSONL and generate PDF sample
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the medical dataset processor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.dataset_loader = DatasetLoader()
        self.sample_selector = SampleSelector(random_seed=config.random_seed)
        self.consolidator = DatasetConsolidator()
        self.jsonl_exporter = JSONLExporter()
        self.pdf_generator = PDFSampleGenerator()
        
        # Initialize processors (will be created when needed)
        self.translation_processor: Optional[TranslationProcessor] = None
        self.generation_processor: Optional[GenerationProcessor] = None
        
        # Processing state
        self.processing_stats = {
            "start_time": None,
            "end_time": None,
            "datasets_loaded": 0,
            "samples_selected": 0,
            "samples_translated": 0,
            "samples_generated": 0,
            "samples_exported": 0,
            "errors": []
        }
    
    def process_datasets(self) -> ConsolidatedDataset:
        """
        Execute the complete processing pipeline.
        
        Returns:
            ConsolidatedDataset: The final consolidated dataset
            
        Raises:
            Exception: If any critical step fails
        """
        self.processing_stats["start_time"] = datetime.now()
        self.logger.info("Starting medical dataset processing pipeline")
        
        try:
            # Step 1: Load datasets
            self.logger.info("Step 1: Loading datasets")
            all_samples = self._load_all_datasets()
            
            # Step 2: Select samples
            self.logger.info("Step 2: Selecting samples for processing")
            translation_samples, generation_samples = self._select_samples(all_samples)
            
            # Step 3: Process samples
            self.logger.info("Step 3: Processing samples")
            translated_samples = self._process_translations(translation_samples)
            generated_samples = self._process_generations(generation_samples)
            
            # Step 4: Consolidate results
            self.logger.info("Step 4: Consolidating results")
            consolidated_dataset = self._consolidate_results(translated_samples, generated_samples)
            
            # Step 5: Export results
            self.logger.info("Step 5: Exporting results")
            self._export_results(consolidated_dataset)
            
            self.processing_stats["end_time"] = datetime.now()
            self.logger.info("Pipeline completed successfully")
            
            return consolidated_dataset
            
        except Exception as e:
            self.processing_stats["end_time"] = datetime.now()
            self.processing_stats["errors"].append(str(e))
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_all_datasets(self) -> List[Sample]:
        """
        Load all datasets from configuration and convert to samples.
        
        Returns:
            List of all samples from all datasets
        """
        # Load dataset configurations
        configs = self.dataset_loader.load_config(self.config.datasets_yaml_path)
        self.logger.info(f"Loaded {len(configs)} dataset configurations")
        
        all_samples = []
        successful_datasets = 0
        
        for dataset_name, dataset_config in configs.items():
            try:
                # Fetch the dataset
                dataset = self.dataset_loader.fetch_dataset(dataset_name, dataset_config)
                
                # Convert to samples
                samples = self.dataset_loader.convert_to_samples(dataset, dataset_config)
                all_samples.extend(samples)
                successful_datasets += 1
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
                
            except Exception as e:
                error_msg = f"Failed to load dataset {dataset_name}: {e}"
                self.logger.error(error_msg)
                self.processing_stats["errors"].append(error_msg)
                # Continue with other datasets
        
        self.processing_stats["datasets_loaded"] = successful_datasets
        self.logger.info(f"Total samples loaded: {len(all_samples)} from {successful_datasets} datasets")
        
        if not all_samples:
            raise ValueError("No samples were loaded from any dataset")
        
        return all_samples
    
    def _select_samples(self, all_samples: List[Sample]) -> tuple[List[Sample], List[Sample]]:
        """
        Select samples for translation and generation processing.
        
        Args:
            all_samples: All available samples
            
        Returns:
            Tuple of (translation_samples, generation_samples)
        """
        translation_samples, generation_samples = self.sample_selector.select_samples_by_dataset(
            all_samples,
            translation_count=self.config.translation_count,
            generation_count=self.config.generation_count
        )
        
        # Validate no overlap
        if not self.sample_selector.validate_no_overlap(translation_samples, generation_samples):
            raise ValueError("Overlap detected between translation and generation samples")
        
        self.processing_stats["samples_selected"] = len(translation_samples) + len(generation_samples)
        
        # Log selection statistics
        stats = self.sample_selector.get_selection_stats(all_samples, translation_samples, generation_samples)
        self.logger.info(f"Sample selection stats: {stats}")
        
        return translation_samples, generation_samples
    
    def _process_translations(self, samples: List[Sample]) -> List:
        """
        Process samples through translation API.
        
        Args:
            samples: Samples to translate
            
        Returns:
            List of translated samples
        """
        if not samples:
            return []
        
        if not self.config.deepl_api_key:
            raise ValueError("DeepL API key is required for translation processing")
        
        # Initialize translation processor if needed
        if self.translation_processor is None:
            translation_config = TranslationConfig(
                api_key=self.config.deepl_api_key,
                target_language=self.config.target_language,
                max_retries=self.config.max_retries,
                batch_size=self.config.batch_size
            )
            self.translation_processor = TranslationProcessor(translation_config)
        
        # Process translations
        translated_samples = self.translation_processor.translate_samples(samples)
        self.processing_stats["samples_translated"] = len(translated_samples)
        
        self.logger.info(f"Successfully translated {len(translated_samples)} samples")
        return translated_samples
    
    def _process_generations(self, samples: List[Sample]) -> List:
        """
        Process samples through generation API.
        
        Args:
            samples: Samples to generate content from
            
        Returns:
            List of generated samples
        """
        if not samples:
            return []
        
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key is required for generation processing")
        
        # Initialize generation processor if needed
        if self.generation_processor is None:
            generation_config = GenerationConfig(
                api_key=self.config.openai_api_key,
                model="gpt-4o-mini",
                max_retries=self.config.max_retries,
                batch_size=self.config.batch_size
            )
            self.generation_processor = GenerationProcessor(generation_config)
        
        # Process generations
        generated_samples = self.generation_processor.generate_from_prompts(samples)
        self.processing_stats["samples_generated"] = len(generated_samples)
        
        self.logger.info(f"Successfully generated content for {len(generated_samples)} samples")
        return generated_samples
    
    def _consolidate_results(self, translated_samples: List, generated_samples: List) -> ConsolidatedDataset:
        """
        Consolidate translation and generation results.
        
        Args:
            translated_samples: List of translated samples
            generated_samples: List of generated samples
            
        Returns:
            ConsolidatedDataset with all processed samples
        """
        # Create additional metadata for the dataset
        additional_metadata = {
            "pipeline_config": {
                "translation_count": self.config.translation_count,
                "generation_count": self.config.generation_count,
                "target_language": self.config.target_language,
                "random_seed": self.config.random_seed
            },
            "processing_stats": self.processing_stats.copy()
        }
        
        consolidated_dataset = self.consolidator.consolidate(
            translated_samples,
            generated_samples,
            additional_metadata
        )
        
        self.logger.info(f"Consolidated {len(consolidated_dataset.samples)} samples")
        return consolidated_dataset
    
    def _export_results(self, dataset: ConsolidatedDataset) -> None:
        """
        Export the consolidated dataset to JSONL and generate PDF sample.
        
        Args:
            dataset: The consolidated dataset to export
        """
        # Ensure output directory exists
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to JSONL
        jsonl_path = output_dir / self.config.jsonl_filename
        self.jsonl_exporter.export(dataset, str(jsonl_path))
        self.processing_stats["samples_exported"] = len(dataset.samples)
        
        # Generate PDF sample
        pdf_path = output_dir / self.config.pdf_filename
        self.pdf_generator.generate_sample(
            dataset,
            sample_size=min(self.config.pdf_sample_size, len(dataset.samples)),
            output_path=str(pdf_path),
            random_seed=self.config.random_seed
        )
        
        self.logger.info(f"Exported results to {jsonl_path} and {pdf_path}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.processing_stats.copy()
        
        # Add duration if both start and end times are available
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["duration_seconds"] = duration.total_seconds()
            stats["duration_formatted"] = str(duration)
        
        # Add component stats if available
        if self.consolidator:
            stats["consolidator_stats"] = self.consolidator.get_processing_stats()
        
        if self.jsonl_exporter:
            stats["export_stats"] = self.jsonl_exporter.get_export_stats()
        
        return stats
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the pipeline configuration and dependencies.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check dataset configuration file
        if not Path(self.config.datasets_yaml_path).exists():
            validation_results["errors"].append(f"Dataset configuration file not found: {self.config.datasets_yaml_path}")
            validation_results["valid"] = False
        
        # Check API keys
        if not self.config.deepl_api_key:
            validation_results["errors"].append("DeepL API key is required")
            validation_results["valid"] = False
        
        if not self.config.openai_api_key:
            validation_results["errors"].append("OpenAI API key is required")
            validation_results["valid"] = False
        
        # Check output directory permissions
        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Try to create a test file
            test_file = output_dir / "test_write_permissions.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            validation_results["errors"].append(f"Cannot write to output directory {self.config.output_dir}: {e}")
            validation_results["valid"] = False
        
        # Validate processing parameters
        if self.config.translation_count + self.config.generation_count > 1000:
            validation_results["warnings"].append("Large number of samples may take significant time to process")
        
        return validation_results


def create_default_config() -> PipelineConfig:
    """
    Create a default pipeline configuration.
    
    Returns:
        PipelineConfig with default values
    """
    return PipelineConfig()


def create_config_from_dict(config_dict: Dict[str, Any]) -> PipelineConfig:
    """
    Create a pipeline configuration from a dictionary.
    
    Args:
        config_dict: Dictionary with configuration values
        
    Returns:
        PipelineConfig instance
    """
    return PipelineConfig(**config_dict)