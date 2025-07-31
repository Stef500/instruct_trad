"""
Dataset consolidator for combining translated and generated samples.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from ..models.core import (
    TranslatedSample, 
    GeneratedSample, 
    ProcessedSample, 
    ConsolidatedDataset,
    ProcessingType
)


logger = logging.getLogger(__name__)


class DatasetConsolidator:
    """
    Consolidates translated and generated samples into a unified dataset.
    
    This class is responsible for:
    - Combining translated and generated samples
    - Adding metadata and timestamps
    - Validating consolidated data
    - Creating a unified dataset structure
    """
    
    def __init__(self):
        """Initialize the dataset consolidator."""
        self.processing_stats = {
            'translated_count': 0,
            'generated_count': 0,
            'total_processed': 0,
            'validation_errors': 0
        }
    
    def consolidate(
        self, 
        translated_samples: List[TranslatedSample], 
        generated_samples: List[GeneratedSample],
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ConsolidatedDataset:
        """
        Consolidate translated and generated samples into a unified dataset.
        
        Args:
            translated_samples: List of translated samples
            generated_samples: List of generated samples
            additional_metadata: Optional additional metadata for the dataset
            
        Returns:
            ConsolidatedDataset: The consolidated dataset
            
        Raises:
            ValueError: If input validation fails
        """
        logger.info(f"Starting consolidation of {len(translated_samples)} translated and {len(generated_samples)} generated samples")
        
        # Validate inputs
        self._validate_inputs(translated_samples, generated_samples)
        
        # Convert samples to ProcessedSample format
        processed_samples = []
        
        # Process translated samples
        for translated_sample in translated_samples:
            try:
                processed_sample = self._convert_translated_sample(translated_sample)
                processed_samples.append(processed_sample)
                self.processing_stats['translated_count'] += 1
            except Exception as e:
                logger.error(f"Error processing translated sample {translated_sample.sample.id}: {e}")
                self.processing_stats['validation_errors'] += 1
        
        # Process generated samples
        for generated_sample in generated_samples:
            try:
                processed_sample = self._convert_generated_sample(generated_sample)
                processed_samples.append(processed_sample)
                self.processing_stats['generated_count'] += 1
            except Exception as e:
                logger.error(f"Error processing generated sample {generated_sample.sample.id}: {e}")
                self.processing_stats['validation_errors'] += 1
        
        self.processing_stats['total_processed'] = len(processed_samples)
        
        # Create consolidated dataset metadata
        dataset_metadata = self._create_dataset_metadata(additional_metadata)
        
        # Create and validate consolidated dataset
        consolidated_dataset = ConsolidatedDataset(
            samples=processed_samples,
            metadata=dataset_metadata,
            creation_timestamp=datetime.now()
        )
        
        # Validate the consolidated dataset
        self._validate_consolidated_dataset(consolidated_dataset)
        
        logger.info(f"Successfully consolidated {len(processed_samples)} samples")
        return consolidated_dataset
    
    def _validate_inputs(
        self, 
        translated_samples: List[TranslatedSample], 
        generated_samples: List[GeneratedSample]
    ) -> None:
        """
        Validate input samples.
        
        Args:
            translated_samples: List of translated samples
            generated_samples: List of generated samples
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(translated_samples, list):
            raise ValueError("translated_samples must be a list")
        
        if not isinstance(generated_samples, list):
            raise ValueError("generated_samples must be a list")
        
        if len(translated_samples) == 0 and len(generated_samples) == 0:
            raise ValueError("At least one translated or generated sample must be provided")
        
        # Validate sample types
        for sample in translated_samples:
            if not isinstance(sample, TranslatedSample):
                raise ValueError("All translated samples must be TranslatedSample instances")
        
        for sample in generated_samples:
            if not isinstance(sample, GeneratedSample):
                raise ValueError("All generated samples must be GeneratedSample instances")
        
        # Check for duplicate sample IDs
        all_ids = []
        for sample in translated_samples:
            all_ids.append(sample.sample.id)
        for sample in generated_samples:
            all_ids.append(sample.sample.id)
        
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate sample IDs found in input samples")
    
    def _convert_translated_sample(self, translated_sample: TranslatedSample) -> ProcessedSample:
        """
        Convert a TranslatedSample to ProcessedSample format.
        
        Args:
            translated_sample: The translated sample to convert
            
        Returns:
            ProcessedSample: The converted sample
        """
        metadata = self._create_sample_metadata(
            processing_type=ProcessingType.TRANSLATION.value,
            processing_timestamp=translated_sample.processing_timestamp,
            original_metadata=translated_sample.translation_metadata
        )
        
        return ProcessedSample(
            original_sample=translated_sample.sample,
            processed_content=translated_sample.translated_text,
            processing_type=ProcessingType.TRANSLATION.value,
            metadata=metadata
        )
    
    def _convert_generated_sample(self, generated_sample: GeneratedSample) -> ProcessedSample:
        """
        Convert a GeneratedSample to ProcessedSample format.
        
        Args:
            generated_sample: The generated sample to convert
            
        Returns:
            ProcessedSample: The converted sample
        """
        metadata = self._create_sample_metadata(
            processing_type=ProcessingType.GENERATION.value,
            processing_timestamp=generated_sample.processing_timestamp,
            original_metadata=generated_sample.generation_metadata,
            prompt=generated_sample.prompt
        )
        
        return ProcessedSample(
            original_sample=generated_sample.sample,
            processed_content=generated_sample.generated_text,
            processing_type=ProcessingType.GENERATION.value,
            metadata=metadata
        )
    
    def _create_sample_metadata(
        self,
        processing_type: str,
        processing_timestamp: datetime,
        original_metadata: Dict[str, Any],
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create enriched metadata for a processed sample.
        
        Args:
            processing_type: Type of processing applied
            processing_timestamp: When the processing occurred
            original_metadata: Original metadata from the sample
            prompt: Optional prompt used for generation
            
        Returns:
            Dict[str, Any]: Enriched metadata
        """
        metadata = {
            'processing_type': processing_type,
            'processing_timestamp': processing_timestamp.isoformat(),
            'consolidation_timestamp': datetime.now().isoformat(),
            'processor_version': '1.0.0',
            **original_metadata  # Include original metadata
        }
        
        if prompt is not None:
            metadata['prompt'] = prompt
        
        return metadata
    
    def _create_dataset_metadata(self, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create metadata for the consolidated dataset.
        
        Args:
            additional_metadata: Optional additional metadata
            
        Returns:
            Dict[str, Any]: Dataset metadata
        """
        metadata = {
            'consolidation_timestamp': datetime.now().isoformat(),
            'processor_version': '1.0.0',
            'processing_stats': self.processing_stats.copy(),
            'total_samples': self.processing_stats['total_processed'],
            'translated_samples': self.processing_stats['translated_count'],
            'generated_samples': self.processing_stats['generated_count'],
            'validation_errors': self.processing_stats['validation_errors']
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _validate_consolidated_dataset(self, dataset: ConsolidatedDataset) -> None:
        """
        Validate the consolidated dataset.
        
        Args:
            dataset: The consolidated dataset to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not dataset.samples:
            raise ValueError("Consolidated dataset must contain at least one sample")
        
        # Validate sample consistency
        sample_ids = set()
        for sample in dataset.samples:
            if sample.original_sample.id in sample_ids:
                raise ValueError(f"Duplicate sample ID found: {sample.original_sample.id}")
            sample_ids.add(sample.original_sample.id)
            
            # Validate processing type
            if sample.processing_type not in [ProcessingType.TRANSLATION.value, ProcessingType.GENERATION.value]:
                raise ValueError(f"Invalid processing type: {sample.processing_type}")
            
            # Validate metadata presence
            if 'processing_timestamp' not in sample.metadata:
                raise ValueError(f"Sample {sample.original_sample.id} missing processing_timestamp in metadata")
            
            if 'consolidation_timestamp' not in sample.metadata:
                raise ValueError(f"Sample {sample.original_sample.id} missing consolidation_timestamp in metadata")
        
        logger.info(f"Validated consolidated dataset with {len(dataset.samples)} samples")
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.
        
        Returns:
            Dict[str, int]: Processing statistics
        """
        return self.processing_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'translated_count': 0,
            'generated_count': 0,
            'total_processed': 0,
            'validation_errors': 0
        }