"""
Core data models for the medical dataset processor.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class SourceType(Enum):
    """Supported dataset source types."""
    HUGGINGFACE = "huggingface"
    URL = "url"
    LOCAL = "local"


class ProcessingType(Enum):
    """Types of processing applied to samples."""
    TRANSLATION = "translation"
    GENERATION = "generation"


@dataclass
class DatasetConfig:
    """Configuration for a dataset source."""
    name: str
    source_type: str
    source_path: str
    format: str
    text_fields: List[str]
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        
        if self.source_type not in [e.value for e in SourceType]:
            raise ValueError(f"Invalid source_type: {self.source_type}. Must be one of: {[e.value for e in SourceType]}")
        
        if not self.source_path:
            raise ValueError("Source path cannot be empty")
        
        if not self.format:
            raise ValueError("Format cannot be empty")
        
        if not self.text_fields or len(self.text_fields) == 0:
            raise ValueError("At least one text field must be specified")


@dataclass
class Sample:
    """A single data sample from a dataset."""
    id: str
    content: Dict[str, Any]
    source_dataset: str
    original_text: str
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.id:
            raise ValueError("Sample ID cannot be empty")
        
        if not self.source_dataset:
            raise ValueError("Source dataset cannot be empty")
        
        if not self.original_text:
            raise ValueError("Original text cannot be empty")


@dataclass
class TranslatedSample:
    """A sample that has been translated."""
    sample: Sample
    translated_text: str
    translation_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not isinstance(self.sample, Sample):
            raise ValueError("Sample must be a Sample instance")
        
        if not self.translated_text:
            raise ValueError("Translated text cannot be empty")


@dataclass
class GeneratedSample:
    """A sample with generated content."""
    sample: Sample
    prompt: str
    generated_text: str
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not isinstance(self.sample, Sample):
            raise ValueError("Sample must be a Sample instance")
        
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        
        if not self.generated_text:
            raise ValueError("Generated text cannot be empty")


@dataclass
class ProcessedSample:
    """A unified representation of processed samples."""
    original_sample: Sample
    processed_content: str
    processing_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not isinstance(self.original_sample, Sample):
            raise ValueError("Original sample must be a Sample instance")
        
        if not self.processed_content:
            raise ValueError("Processed content cannot be empty")
        
        if self.processing_type not in [e.value for e in ProcessingType]:
            raise ValueError(f"Invalid processing_type: {self.processing_type}. Must be one of: {[e.value for e in ProcessingType]}")
        
        if self.quality_score is not None and (self.quality_score < 0 or self.quality_score > 1):
            raise ValueError("Quality score must be between 0 and 1")


@dataclass
class ConsolidatedDataset:
    """A consolidated dataset containing all processed samples."""
    samples: List[ProcessedSample]
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the consolidated dataset."""
        if not self.samples:
            raise ValueError("Consolidated dataset must contain at least one sample")
        
        for sample in self.samples:
            if not isinstance(sample, ProcessedSample):
                raise ValueError("All samples must be ProcessedSample instances")
    
    def get_translation_samples(self) -> List[ProcessedSample]:
        """Get all translated samples."""
        return [s for s in self.samples if s.processing_type == ProcessingType.TRANSLATION.value]
    
    def get_generation_samples(self) -> List[ProcessedSample]:
        """Get all generated samples."""
        return [s for s in self.samples if s.processing_type == ProcessingType.GENERATION.value]
    
    def get_sample_count_by_dataset(self) -> Dict[str, int]:
        """Get count of samples by source dataset."""
        counts = {}
        for sample in self.samples:
            dataset = sample.original_sample.source_dataset
            counts[dataset] = counts.get(dataset, 0) + 1
        return counts