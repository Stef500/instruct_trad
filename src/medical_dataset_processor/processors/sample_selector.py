"""
Sample selector for choosing samples for translation and generation processing.
"""
import logging
import random
from typing import List, Optional, Set

from ..models.core import Sample


logger = logging.getLogger(__name__)


class SampleSelector:
    """Selects samples for translation and generation processing."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the sample selector.
        
        Args:
            random_seed: Optional seed for reproducible random selection
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
    
    def select_for_translation(self, samples: List[Sample], count: int = 50) -> List[Sample]:
        """
        Select samples for translation processing.
        
        Args:
            samples: List of available samples
            count: Number of samples to select (default: 50)
            
        Returns:
            List of selected samples for translation
            
        Raises:
            ValueError: If not enough samples available or invalid count
        """
        if count <= 0:
            raise ValueError("Count must be positive")
        
        if len(samples) < count:
            raise ValueError(f"Not enough samples available. Requested: {count}, Available: {len(samples)}")
        
        # Create a copy to avoid modifying the original list
        available_samples = samples.copy()
        
        # Set seed if specified for reproducible results
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # Randomly select samples
        selected = random.sample(available_samples, count)
        
        logger.info(f"Selected {len(selected)} samples for translation from {len(samples)} available")
        return selected
    
    def select_for_generation(
        self, 
        samples: List[Sample], 
        count: int = 50, 
        exclude: Optional[List[Sample]] = None
    ) -> List[Sample]:
        """
        Select samples for generation processing, excluding already selected samples.
        
        Args:
            samples: List of available samples
            count: Number of samples to select (default: 50)
            exclude: List of samples to exclude from selection
            
        Returns:
            List of selected samples for generation
            
        Raises:
            ValueError: If not enough samples available after exclusions or invalid count
        """
        if count <= 0:
            raise ValueError("Count must be positive")
        
        # Create set of excluded sample IDs for efficient lookup
        excluded_ids: Set[str] = set()
        if exclude:
            excluded_ids = {sample.id for sample in exclude}
        
        # Filter out excluded samples
        available_samples = [
            sample for sample in samples 
            if sample.id not in excluded_ids
        ]
        
        if len(available_samples) < count:
            raise ValueError(
                f"Not enough samples available after exclusions. "
                f"Requested: {count}, Available: {len(available_samples)}, "
                f"Excluded: {len(excluded_ids)}"
            )
        
        # Set seed if specified for reproducible results
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # Randomly select from available samples
        selected = random.sample(available_samples, count)
        
        logger.info(
            f"Selected {len(selected)} samples for generation from {len(available_samples)} available "
            f"(excluded {len(excluded_ids)} samples)"
        )
        return selected
    
    def select_samples_by_dataset(
        self, 
        samples: List[Sample], 
        translation_count: int = 50, 
        generation_count: int = 50
    ) -> tuple[List[Sample], List[Sample]]:
        """
        Select samples for both translation and generation, ensuring no overlap.
        
        Args:
            samples: List of available samples
            translation_count: Number of samples for translation (default: 50)
            generation_count: Number of samples for generation (default: 50)
            
        Returns:
            Tuple of (translation_samples, generation_samples)
            
        Raises:
            ValueError: If not enough samples available for both operations
        """
        total_needed = translation_count + generation_count
        
        if len(samples) < total_needed:
            raise ValueError(
                f"Not enough samples for both operations. "
                f"Needed: {total_needed}, Available: {len(samples)}"
            )
        
        # First select samples for translation
        translation_samples = self.select_for_translation(samples, translation_count)
        
        # Then select different samples for generation
        generation_samples = self.select_for_generation(
            samples, generation_count, exclude=translation_samples
        )
        
        logger.info(
            f"Selected {len(translation_samples)} samples for translation and "
            f"{len(generation_samples)} samples for generation from {len(samples)} total samples"
        )
        
        return translation_samples, generation_samples
    
    def validate_no_overlap(
        self, 
        translation_samples: List[Sample], 
        generation_samples: List[Sample]
    ) -> bool:
        """
        Validate that there's no overlap between translation and generation samples.
        
        Args:
            translation_samples: Samples selected for translation
            generation_samples: Samples selected for generation
            
        Returns:
            True if no overlap exists, False otherwise
        """
        translation_ids = {sample.id for sample in translation_samples}
        generation_ids = {sample.id for sample in generation_samples}
        
        overlap = translation_ids.intersection(generation_ids)
        
        if overlap:
            logger.error(f"Found overlap between translation and generation samples: {overlap}")
            return False
        
        logger.debug("No overlap found between translation and generation samples")
        return True
    
    def get_selection_stats(
        self, 
        samples: List[Sample], 
        translation_samples: List[Sample], 
        generation_samples: List[Sample]
    ) -> dict:
        """
        Get statistics about the sample selection.
        
        Args:
            samples: Original list of samples
            translation_samples: Selected translation samples
            generation_samples: Selected generation samples
            
        Returns:
            Dictionary with selection statistics
        """
        total_samples = len(samples)
        translation_count = len(translation_samples)
        generation_count = len(generation_samples)
        
        # Count samples by dataset
        dataset_counts = {}
        for sample in samples:
            dataset = sample.source_dataset
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        translation_dataset_counts = {}
        for sample in translation_samples:
            dataset = sample.source_dataset
            translation_dataset_counts[dataset] = translation_dataset_counts.get(dataset, 0) + 1
        
        generation_dataset_counts = {}
        for sample in generation_samples:
            dataset = sample.source_dataset
            generation_dataset_counts[dataset] = generation_dataset_counts.get(dataset, 0) + 1
        
        stats = {
            "total_samples": total_samples,
            "translation_samples": translation_count,
            "generation_samples": generation_count,
            "unused_samples": total_samples - translation_count - generation_count,
            "dataset_distribution": dataset_counts,
            "translation_dataset_distribution": translation_dataset_counts,
            "generation_dataset_distribution": generation_dataset_counts,
            "has_overlap": not self.validate_no_overlap(translation_samples, generation_samples)
        }
        
        return stats