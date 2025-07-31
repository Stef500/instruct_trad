"""
Logging and error handling utilities for the medical dataset processor.
"""
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from ..models.core import Sample, ProcessedSample


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class APIError:
    """Represents an API error."""
    api_name: str
    error_type: str
    error_message: str
    sample_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "api_name": self.api_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "severity": self.severity.value
        }


@dataclass
class ProcessingStats:
    """Statistics about processing operations."""
    total_samples: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    api_calls_made: int = 0
    total_processing_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_samples": self.total_samples,
            "successful_translations": self.successful_translations,
            "failed_translations": self.failed_translations,
            "successful_generations": self.successful_generations,
            "failed_generations": self.failed_generations,
            "api_calls_made": self.api_calls_made,
            "total_processing_time": self.total_processing_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    errors: List[APIError] = field(default_factory=list)
    stats: Optional[ProcessingStats] = None
    generation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "errors": [error.to_dict() for error in self.errors],
            "stats": self.stats.to_dict() if self.stats else None,
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "summary": self.get_summary()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the errors."""
        if not self.errors:
            return {"total_errors": 0}
        
        error_by_api = {}
        error_by_severity = {}
        
        for error in self.errors:
            # Count by API
            error_by_api[error.api_name] = error_by_api.get(error.api_name, 0) + 1
            # Count by severity
            error_by_severity[error.severity.value] = error_by_severity.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "errors_by_api": error_by_api,
            "errors_by_severity": error_by_severity,
            "most_problematic_api": max(error_by_api.items(), key=lambda x: x[1])[0] if error_by_api else None
        }


@dataclass
class ProcessingState:
    """State information for processing recovery."""
    processed_samples: List[str] = field(default_factory=list)  # Sample IDs
    failed_samples: List[str] = field(default_factory=list)  # Sample IDs
    current_dataset: Optional[str] = None
    processing_stage: str = "not_started"  # not_started, loading, translating, generating, consolidating, exporting
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    checkpoint_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "processed_samples": self.processed_samples,
            "failed_samples": self.failed_samples,
            "current_dataset": self.current_dataset,
            "processing_stage": self.processing_stage,
            "stats": self.stats.to_dict(),
            "checkpoint_timestamp": self.checkpoint_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create ProcessingState from dictionary."""
        stats_data = data.get("stats", {})
        stats = ProcessingStats(**stats_data)
        if stats_data.get("start_time"):
            stats.start_time = datetime.fromisoformat(stats_data["start_time"])
        if stats_data.get("end_time"):
            stats.end_time = datetime.fromisoformat(stats_data["end_time"])
        
        return cls(
            processed_samples=data.get("processed_samples", []),
            failed_samples=data.get("failed_samples", []),
            current_dataset=data.get("current_dataset"),
            processing_stage=data.get("processing_stage", "not_started"),
            stats=stats,
            checkpoint_timestamp=datetime.fromisoformat(data.get("checkpoint_timestamp", datetime.now().isoformat()))
        )


class ProcessingLogger:
    """Logger for processing operations with error tracking and state management."""
    
    def __init__(self, log_dir: str = "logs", checkpoint_dir: str = "checkpoints"):
        """Initialize the processing logger.
        
        Args:
            log_dir: Directory for log files
            checkpoint_dir: Directory for checkpoint files
        """
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories if they don't exist
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize tracking
        self.errors: List[APIError] = []
        self.stats = ProcessingStats()
        self.state = ProcessingState()
        
        # Logger instance
        self.logger = logging.getLogger("medical_dataset_processor")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
    
    def log_api_error(self, api_name: str, error: Exception, sample_id: str, retry_count: int = 0) -> None:
        """Log an API error.
        
        Args:
            api_name: Name of the API that failed
            error: The exception that occurred
            sample_id: ID of the sample being processed
            retry_count: Number of retries attempted
        """
        # Determine severity based on error type and retry count
        severity = self._determine_error_severity(error, retry_count)
        
        api_error = APIError(
            api_name=api_name,
            error_type=type(error).__name__,
            error_message=str(error),
            sample_id=sample_id,
            retry_count=retry_count,
            severity=severity
        )
        
        self.errors.append(api_error)
        
        # Log to file and console
        self.logger.error(
            f"API Error - {api_name}: {type(error).__name__} for sample {sample_id} "
            f"(retry {retry_count}): {str(error)}"
        )
        
        # Update stats
        if api_name.lower() in ["deepl", "translation"]:
            self.stats.failed_translations += 1
        elif api_name.lower() in ["openai", "generation"]:
            self.stats.failed_generations += 1
    
    def log_processing_stats(self, stats: ProcessingStats) -> None:
        """Log processing statistics.
        
        Args:
            stats: Processing statistics to log
        """
        self.stats = stats
        
        self.logger.info(f"Processing Stats: {stats.to_dict()}")
        
        # Save stats to file
        stats_file = self.log_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
    
    def generate_error_report(self) -> ErrorReport:
        """Generate a comprehensive error report.
        
        Returns:
            ErrorReport containing all errors and statistics
        """
        report = ErrorReport(
            errors=self.errors.copy(),
            stats=self.stats
        )
        
        # Save report to file
        report_file = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        self.logger.info(f"Error report generated: {report_file}")
        
        return report
    
    def save_checkpoint(self, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Save current processing state as checkpoint.
        
        Args:
            additional_data: Additional data to save with the checkpoint
        """
        self.state.checkpoint_timestamp = datetime.now()
        
        checkpoint_data = {
            "state": self.state.to_dict(),
            "errors": [error.to_dict() for error in self.errors],
            "stats": self.stats.to_dict()
        }
        
        if additional_data:
            checkpoint_data["additional_data"] = additional_data
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.
        
        Returns:
            Checkpoint data if found, None otherwise
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        
        if not checkpoint_files:
            self.logger.info("No checkpoints found")
            return None
        
        # Get the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            if "state" in checkpoint_data:
                self.state = ProcessingState.from_dict(checkpoint_data["state"])
            
            # Restore errors
            if "errors" in checkpoint_data:
                self.errors = [
                    APIError(**error_data) for error_data in checkpoint_data["errors"]
                ]
            
            # Restore stats
            if "stats" in checkpoint_data:
                stats_data = checkpoint_data["stats"]
                self.stats = ProcessingStats(**stats_data)
                if stats_data.get("start_time"):
                    self.stats.start_time = datetime.fromisoformat(stats_data["start_time"])
                if stats_data.get("end_time"):
                    self.stats.end_time = datetime.fromisoformat(stats_data["end_time"])
            
            self.logger.info(f"Checkpoint loaded: {latest_checkpoint}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def update_processing_stage(self, stage: str, dataset: Optional[str] = None) -> None:
        """Update the current processing stage.
        
        Args:
            stage: Current processing stage
            dataset: Current dataset being processed
        """
        self.state.processing_stage = stage
        if dataset:
            self.state.current_dataset = dataset
        
        self.logger.info(f"Processing stage updated: {stage}" + (f" (dataset: {dataset})" if dataset else ""))
    
    def mark_sample_processed(self, sample_id: str) -> None:
        """Mark a sample as successfully processed.
        
        Args:
            sample_id: ID of the processed sample
        """
        if sample_id not in self.state.processed_samples:
            self.state.processed_samples.append(sample_id)
        
        # Remove from failed samples if it was there
        if sample_id in self.state.failed_samples:
            self.state.failed_samples.remove(sample_id)
    
    def mark_sample_failed(self, sample_id: str) -> None:
        """Mark a sample as failed.
        
        Args:
            sample_id: ID of the failed sample
        """
        if sample_id not in self.state.failed_samples:
            self.state.failed_samples.append(sample_id)
    
    def is_sample_processed(self, sample_id: str) -> bool:
        """Check if a sample has been processed.
        
        Args:
            sample_id: ID of the sample to check
            
        Returns:
            True if sample was processed, False otherwise
        """
        return sample_id in self.state.processed_samples
    
    def is_sample_failed(self, sample_id: str) -> bool:
        """Check if a sample has failed.
        
        Args:
            sample_id: ID of the sample to check
            
        Returns:
            True if sample failed, False otherwise
        """
        return sample_id in self.state.failed_samples
    
    def _determine_error_severity(self, error: Exception, retry_count: int) -> ErrorSeverity:
        """Determine the severity of an error.
        
        Args:
            error: The exception that occurred
            retry_count: Number of retries attempted
            
        Returns:
            ErrorSeverity level
        """
        error_type = type(error).__name__
        
        # Critical errors that should stop processing
        if error_type in ["AuthenticationError", "PermissionError", "ConfigurationError"]:
            return ErrorSeverity.CRITICAL
        
        # High severity for repeated failures
        if retry_count >= 3:
            return ErrorSeverity.HIGH
        
        # Medium severity for API errors
        if error_type in ["APIError", "HTTPError", "RequestException"]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for temporary issues
        return ErrorSeverity.LOW


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_file = log_path / f"medical_dataset_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("deepl").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")