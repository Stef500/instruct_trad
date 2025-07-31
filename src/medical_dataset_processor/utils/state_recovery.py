"""
State recovery utilities for resuming interrupted processing.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from .logging import ProcessingLogger, ProcessingState
from ..models.core import Sample, ProcessedSample


class StateRecoveryManager:
    """Manages state recovery for interrupted processing operations."""
    
    def __init__(self, logger: ProcessingLogger):
        """Initialize the state recovery manager.
        
        Args:
            logger: ProcessingLogger instance for state management
        """
        self.logger = logger
        self.recovery_data: Optional[Dict[str, Any]] = None
    
    def can_recover(self) -> bool:
        """Check if recovery is possible from saved state.
        
        Returns:
            True if recovery data is available, False otherwise
        """
        if self.recovery_data is None:
            self.recovery_data = self.logger.load_latest_checkpoint()
        
        return self.recovery_data is not None
    
    def get_recovery_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the recovery state.
        
        Returns:
            Dictionary with recovery information or None if no recovery possible
        """
        if not self.can_recover():
            return None
        
        state_data = self.recovery_data.get("state", {})
        stats_data = self.recovery_data.get("stats", {})
        errors_data = self.recovery_data.get("errors", [])
        
        return {
            "processing_stage": state_data.get("processing_stage", "unknown"),
            "current_dataset": state_data.get("current_dataset"),
            "processed_samples_count": len(state_data.get("processed_samples", [])),
            "failed_samples_count": len(state_data.get("failed_samples", [])),
            "total_errors": len(errors_data),
            "checkpoint_timestamp": state_data.get("checkpoint_timestamp"),
            "last_stats": {
                "total_samples": stats_data.get("total_samples", 0),
                "successful_translations": stats_data.get("successful_translations", 0),
                "successful_generations": stats_data.get("successful_generations", 0),
                "failed_translations": stats_data.get("failed_translations", 0),
                "failed_generations": stats_data.get("failed_generations", 0)
            }
        }
    
    def get_processed_sample_ids(self) -> Set[str]:
        """Get set of already processed sample IDs.
        
        Returns:
            Set of sample IDs that have been successfully processed
        """
        if not self.can_recover():
            return set()
        
        state_data = self.recovery_data.get("state", {})
        return set(state_data.get("processed_samples", []))
    
    def get_failed_sample_ids(self) -> Set[str]:
        """Get set of failed sample IDs.
        
        Returns:
            Set of sample IDs that have failed processing
        """
        if not self.can_recover():
            return set()
        
        state_data = self.recovery_data.get("state", {})
        return set(state_data.get("failed_samples", []))
    
    def should_skip_sample(self, sample_id: str) -> bool:
        """Check if a sample should be skipped during recovery.
        
        Args:
            sample_id: ID of the sample to check
            
        Returns:
            True if sample should be skipped (already processed or failed), False otherwise
        """
        processed_ids = self.get_processed_sample_ids()
        failed_ids = self.get_failed_sample_ids()
        
        return sample_id in processed_ids or sample_id in failed_ids
    
    def filter_samples_for_processing(self, samples: List[Sample]) -> List[Sample]:
        """Filter samples to exclude already processed or failed ones.
        
        Args:
            samples: List of samples to filter
            
        Returns:
            List of samples that need to be processed
        """
        if not self.can_recover():
            return samples
        
        processed_ids = self.get_processed_sample_ids()
        failed_ids = self.get_failed_sample_ids()
        skip_ids = processed_ids.union(failed_ids)
        
        filtered_samples = [sample for sample in samples if sample.id not in skip_ids]
        
        if len(filtered_samples) < len(samples):
            skipped_count = len(samples) - len(filtered_samples)
            self.logger.logger.info(
                f"Recovery: Skipping {skipped_count} samples "
                f"({len(processed_ids)} processed, {len(failed_ids)} failed)"
            )
        
        return filtered_samples
    
    def get_current_processing_stage(self) -> str:
        """Get the current processing stage from recovery data.
        
        Returns:
            Current processing stage or 'not_started' if no recovery data
        """
        if not self.can_recover():
            return "not_started"
        
        state_data = self.recovery_data.get("state", {})
        return state_data.get("processing_stage", "not_started")
    
    def get_current_dataset(self) -> Optional[str]:
        """Get the current dataset being processed from recovery data.
        
        Returns:
            Current dataset name or None if not available
        """
        if not self.can_recover():
            return None
        
        state_data = self.recovery_data.get("state", {})
        return state_data.get("current_dataset")
    
    def should_resume_from_stage(self, target_stage: str) -> bool:
        """Check if processing should resume from a specific stage.
        
        Args:
            target_stage: The stage to check for resumption
            
        Returns:
            True if processing should resume from this stage, False otherwise
        """
        if not self.can_recover():
            return False
        
        current_stage = self.get_current_processing_stage()
        
        # Define stage order
        stage_order = [
            "not_started",
            "loading",
            "selecting",
            "translating", 
            "generating",
            "consolidating",
            "exporting",
            "completed"
        ]
        
        try:
            current_index = stage_order.index(current_stage)
            target_index = stage_order.index(target_stage)
            
            # Resume if target stage is at or after current stage
            return target_index >= current_index
            
        except ValueError:
            # Unknown stage, assume we should resume
            return True
    
    def create_recovery_summary(self) -> str:
        """Create a human-readable summary of recovery state.
        
        Returns:
            Formatted string with recovery information
        """
        if not self.can_recover():
            return "No recovery data available."
        
        info = self.get_recovery_info()
        if not info:
            return "No recovery information available."
        
        summary_lines = [
            "=== Recovery Information ===",
            f"Processing Stage: {info['processing_stage']}",
            f"Current Dataset: {info['current_dataset'] or 'None'}",
            f"Processed Samples: {info['processed_samples_count']}",
            f"Failed Samples: {info['failed_samples_count']}",
            f"Total Errors: {info['total_errors']}",
            f"Checkpoint Time: {info['checkpoint_timestamp']}",
            "",
            "=== Last Statistics ===",
            f"Total Samples: {info['last_stats']['total_samples']}",
            f"Successful Translations: {info['last_stats']['successful_translations']}",
            f"Successful Generations: {info['last_stats']['successful_generations']}",
            f"Failed Translations: {info['last_stats']['failed_translations']}",
            f"Failed Generations: {info['last_stats']['failed_generations']}",
        ]
        
        return "\n".join(summary_lines)
    
    def clear_recovery_data(self) -> None:
        """Clear recovery data after successful completion."""
        self.recovery_data = None
        
        # Optionally, you could also delete checkpoint files here
        # but we'll keep them for audit purposes
        self.logger.logger.info("Recovery data cleared - processing completed successfully")


class BatchProcessor:
    """Utility for processing samples in batches with recovery support."""
    
    def __init__(self, logger: ProcessingLogger, batch_size: int = 10):
        """Initialize the batch processor.
        
        Args:
            logger: ProcessingLogger instance
            batch_size: Number of samples to process in each batch
        """
        self.logger = logger
        self.batch_size = batch_size
        self.recovery_manager = StateRecoveryManager(logger)
    
    def process_samples_with_recovery(
        self,
        samples: List[Sample],
        process_function: callable,
        stage_name: str,
        dataset_name: Optional[str] = None
    ) -> List[Any]:
        """Process samples in batches with automatic recovery support.
        
        Args:
            samples: List of samples to process
            process_function: Function to process each sample
            stage_name: Name of the processing stage
            dataset_name: Name of the dataset being processed
            
        Returns:
            List of processed results
        """
        # Update processing stage
        self.logger.update_processing_stage(stage_name, dataset_name)
        
        # Filter samples if recovering
        samples_to_process = self.recovery_manager.filter_samples_for_processing(samples)
        
        if len(samples_to_process) == 0:
            self.logger.logger.info(f"All samples already processed for stage: {stage_name}")
            return []
        
        results = []
        processed_count = 0
        
        # Process in batches
        for i in range(0, len(samples_to_process), self.batch_size):
            batch = samples_to_process[i:i + self.batch_size]
            batch_results = []
            
            for sample in batch:
                try:
                    result = process_function(sample)
                    batch_results.append(result)
                    self.logger.mark_sample_processed(sample.id)
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.log_api_error(
                        api_name=stage_name,
                        error=e,
                        sample_id=sample.id
                    )
                    self.logger.mark_sample_failed(sample.id)
            
            results.extend(batch_results)
            
            # Save checkpoint after each batch
            self.logger.save_checkpoint({
                "stage": stage_name,
                "dataset": dataset_name,
                "batch_number": i // self.batch_size + 1,
                "processed_in_batch": len(batch_results)
            })
            
            self.logger.logger.info(
                f"Processed batch {i // self.batch_size + 1}: "
                f"{len(batch_results)}/{len(batch)} samples successful"
            )
        
        self.logger.logger.info(
            f"Stage '{stage_name}' completed: {processed_count}/{len(samples_to_process)} samples processed"
        )
        
        return results