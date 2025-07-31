"""
Tests for the state recovery utilities.
"""
import tempfile
from unittest.mock import Mock, patch
import pytest

from src.medical_dataset_processor.utils.logging import ProcessingLogger, ProcessingStats
from src.medical_dataset_processor.utils.state_recovery import StateRecoveryManager, BatchProcessor
from src.medical_dataset_processor.models.core import Sample


class TestStateRecoveryManager:
    """Test cases for StateRecoveryManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
        self.recovery_manager = StateRecoveryManager(self.logger)
    
    def test_can_recover_no_checkpoint(self):
        """Test can_recover when no checkpoint exists."""
        assert not self.recovery_manager.can_recover()
    
    def test_can_recover_with_checkpoint(self):
        """Test can_recover when checkpoint exists."""
        # Create a checkpoint
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.processing_stage = "translating"
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        
        assert new_recovery_manager.can_recover()
    
    def test_get_recovery_info_no_checkpoint(self):
        """Test get_recovery_info when no checkpoint exists."""
        info = self.recovery_manager.get_recovery_info()
        assert info is None
    
    def test_get_recovery_info_with_checkpoint(self):
        """Test get_recovery_info with existing checkpoint."""
        # Set up state
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.failed_samples = ["sample_3"]
        self.logger.state.processing_stage = "generating"
        self.logger.state.current_dataset = "medqa"
        
        # Set up stats
        self.logger.stats.total_samples = 100
        self.logger.stats.successful_translations = 45
        self.logger.stats.failed_translations = 5
        
        # Add some errors
        self.logger.log_api_error("deepl", Exception("Test error"), "sample_3")
        
        # Save checkpoint
        self.logger.save_checkpoint()
        
        # Create new recovery manager and get info
        new_recovery_manager = StateRecoveryManager(self.logger)
        info = new_recovery_manager.get_recovery_info()
        
        assert info is not None
        assert info["processing_stage"] == "generating"
        assert info["current_dataset"] == "medqa"
        assert info["processed_samples_count"] == 2
        assert info["failed_samples_count"] == 1
        assert info["total_errors"] == 1
        assert info["last_stats"]["total_samples"] == 100
        assert info["last_stats"]["successful_translations"] == 45
        assert info["last_stats"]["failed_translations"] == 6  # 5 + 1 from the logged error
    
    def test_get_processed_sample_ids(self):
        """Test getting processed sample IDs."""
        # Set up state and save checkpoint
        self.logger.state.processed_samples = ["sample_1", "sample_2", "sample_3"]
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        processed_ids = new_recovery_manager.get_processed_sample_ids()
        
        assert processed_ids == {"sample_1", "sample_2", "sample_3"}
    
    def test_get_failed_sample_ids(self):
        """Test getting failed sample IDs."""
        # Set up state and save checkpoint
        self.logger.state.failed_samples = ["sample_4", "sample_5"]
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        failed_ids = new_recovery_manager.get_failed_sample_ids()
        
        assert failed_ids == {"sample_4", "sample_5"}
    
    def test_should_skip_sample(self):
        """Test should_skip_sample logic."""
        # Set up state and save checkpoint
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.failed_samples = ["sample_3"]
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        
        # Test skipping processed samples
        assert new_recovery_manager.should_skip_sample("sample_1")
        assert new_recovery_manager.should_skip_sample("sample_2")
        
        # Test skipping failed samples
        assert new_recovery_manager.should_skip_sample("sample_3")
        
        # Test not skipping new samples
        assert not new_recovery_manager.should_skip_sample("sample_4")
    
    def test_filter_samples_for_processing(self):
        """Test filtering samples for processing."""
        # Create test samples
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
            Sample("sample_3", {"text": "text3"}, "dataset1", "original3"),
            Sample("sample_4", {"text": "text4"}, "dataset1", "original4"),
        ]
        
        # Set up state (sample_1 processed, sample_3 failed)
        self.logger.state.processed_samples = ["sample_1"]
        self.logger.state.failed_samples = ["sample_3"]
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        filtered_samples = new_recovery_manager.filter_samples_for_processing(samples)
        
        # Should only return sample_2 and sample_4
        assert len(filtered_samples) == 2
        assert filtered_samples[0].id == "sample_2"
        assert filtered_samples[1].id == "sample_4"
    
    def test_filter_samples_no_recovery(self):
        """Test filtering samples when no recovery data exists."""
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
        ]
        
        # No checkpoint saved
        filtered_samples = self.recovery_manager.filter_samples_for_processing(samples)
        
        # Should return all samples
        assert len(filtered_samples) == 2
        assert filtered_samples == samples
    
    def test_get_current_processing_stage(self):
        """Test getting current processing stage."""
        # Test with no recovery data
        assert self.recovery_manager.get_current_processing_stage() == "not_started"
        
        # Set up state and save checkpoint
        self.logger.state.processing_stage = "translating"
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        assert new_recovery_manager.get_current_processing_stage() == "translating"
    
    def test_get_current_dataset(self):
        """Test getting current dataset."""
        # Test with no recovery data
        assert self.recovery_manager.get_current_dataset() is None
        
        # Set up state and save checkpoint
        self.logger.state.current_dataset = "pubmedqa"
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        assert new_recovery_manager.get_current_dataset() == "pubmedqa"
    
    def test_should_resume_from_stage(self):
        """Test should_resume_from_stage logic."""
        # Set up state at 'translating' stage
        self.logger.state.processing_stage = "translating"
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        
        # Should NOT resume from earlier stages (already completed)
        assert not new_recovery_manager.should_resume_from_stage("loading")
        # Should resume from current and later stages
        assert new_recovery_manager.should_resume_from_stage("translating")
        assert new_recovery_manager.should_resume_from_stage("generating")
        assert new_recovery_manager.should_resume_from_stage("consolidating")
        
        # Test with no recovery data
        no_recovery_manager = StateRecoveryManager(ProcessingLogger(
            log_dir=f"{self.temp_dir}/empty_logs",
            checkpoint_dir=f"{self.temp_dir}/empty_checkpoints"
        ))
        assert not no_recovery_manager.should_resume_from_stage("translating")
    
    def test_create_recovery_summary(self):
        """Test creating recovery summary."""
        # Test with no recovery data
        summary = self.recovery_manager.create_recovery_summary()
        assert "No recovery data available" in summary
        
        # Set up state and save checkpoint
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.failed_samples = ["sample_3"]
        self.logger.state.processing_stage = "generating"
        self.logger.state.current_dataset = "medqa"
        self.logger.stats.total_samples = 50
        self.logger.stats.successful_translations = 20
        self.logger.log_api_error("deepl", Exception("Test"), "sample_3")
        self.logger.save_checkpoint()
        
        # Create new recovery manager
        new_recovery_manager = StateRecoveryManager(self.logger)
        summary = new_recovery_manager.create_recovery_summary()
        
        assert "Recovery Information" in summary
        assert "Processing Stage: generating" in summary
        assert "Current Dataset: medqa" in summary
        assert "Processed Samples: 2" in summary
        assert "Failed Samples: 1" in summary
        assert "Total Errors: 1" in summary
        assert "Total Samples: 50" in summary
        assert "Successful Translations: 20" in summary
    
    def test_clear_recovery_data(self):
        """Test clearing recovery data."""
        # Set up recovery data
        self.logger.state.processed_samples = ["sample_1"]
        self.logger.save_checkpoint()
        
        recovery_manager = StateRecoveryManager(self.logger)
        assert recovery_manager.can_recover()
        
        # Clear recovery data
        recovery_manager.clear_recovery_data()
        
        # Recovery data should be cleared from manager (but checkpoint files remain)
        assert recovery_manager.recovery_data is None


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
        self.batch_processor = BatchProcessor(self.logger, batch_size=2)
    
    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        assert self.batch_processor.logger == self.logger
        assert self.batch_processor.batch_size == 2
        assert isinstance(self.batch_processor.recovery_manager, StateRecoveryManager)
    
    def test_process_samples_with_recovery_success(self):
        """Test successful batch processing."""
        # Create test samples
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
            Sample("sample_3", {"text": "text3"}, "dataset1", "original3"),
        ]
        
        # Mock processing function
        def mock_process_function(sample):
            return f"processed_{sample.id}"
        
        # Process samples
        results = self.batch_processor.process_samples_with_recovery(
            samples=samples,
            process_function=mock_process_function,
            stage_name="test_stage",
            dataset_name="test_dataset"
        )
        
        # Verify results
        assert len(results) == 3
        assert results[0] == "processed_sample_1"
        assert results[1] == "processed_sample_2"
        assert results[2] == "processed_sample_3"
        
        # Verify all samples marked as processed
        assert self.logger.is_sample_processed("sample_1")
        assert self.logger.is_sample_processed("sample_2")
        assert self.logger.is_sample_processed("sample_3")
        
        # Verify processing stage was updated
        assert self.logger.state.processing_stage == "test_stage"
        assert self.logger.state.current_dataset == "test_dataset"
    
    def test_process_samples_with_recovery_partial_failure(self):
        """Test batch processing with some failures."""
        # Create test samples
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
            Sample("sample_3", {"text": "text3"}, "dataset1", "original3"),
        ]
        
        # Mock processing function that fails on sample_2
        def mock_process_function(sample):
            if sample.id == "sample_2":
                raise Exception("Processing failed")
            return f"processed_{sample.id}"
        
        # Process samples
        results = self.batch_processor.process_samples_with_recovery(
            samples=samples,
            process_function=mock_process_function,
            stage_name="test_stage",
            dataset_name="test_dataset"
        )
        
        # Verify results (only successful ones)
        assert len(results) == 2
        assert "processed_sample_1" in results
        assert "processed_sample_3" in results
        
        # Verify sample states
        assert self.logger.is_sample_processed("sample_1")
        assert self.logger.is_sample_failed("sample_2")
        assert self.logger.is_sample_processed("sample_3")
        
        # Verify error was logged
        assert len(self.logger.errors) == 1
        assert self.logger.errors[0].sample_id == "sample_2"
    
    def test_process_samples_with_recovery_skip_processed(self):
        """Test batch processing with recovery skipping already processed samples."""
        # Create test samples
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
            Sample("sample_3", {"text": "text3"}, "dataset1", "original3"),
        ]
        
        # Mark sample_1 as already processed and sample_3 as failed
        self.logger.mark_sample_processed("sample_1")
        self.logger.mark_sample_failed("sample_3")
        self.logger.save_checkpoint()
        
        # Create new batch processor (simulating recovery)
        new_batch_processor = BatchProcessor(self.logger, batch_size=2)
        
        # Mock processing function
        def mock_process_function(sample):
            return f"processed_{sample.id}"
        
        # Process samples
        results = new_batch_processor.process_samples_with_recovery(
            samples=samples,
            process_function=mock_process_function,
            stage_name="test_stage",
            dataset_name="test_dataset"
        )
        
        # Should only process sample_2 (sample_1 already processed, sample_3 failed)
        assert len(results) == 1
        assert results[0] == "processed_sample_2"
        
        # Verify states
        assert self.logger.is_sample_processed("sample_1")  # Still processed
        assert self.logger.is_sample_processed("sample_2")  # Newly processed
        assert self.logger.is_sample_failed("sample_3")     # Still failed
    
    def test_process_samples_all_already_processed(self):
        """Test batch processing when all samples are already processed."""
        # Create test samples
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
        ]
        
        # Mark all samples as processed
        self.logger.mark_sample_processed("sample_1")
        self.logger.mark_sample_processed("sample_2")
        self.logger.save_checkpoint()
        
        # Create new batch processor
        new_batch_processor = BatchProcessor(self.logger, batch_size=2)
        
        # Mock processing function
        def mock_process_function(sample):
            return f"processed_{sample.id}"
        
        # Process samples
        results = new_batch_processor.process_samples_with_recovery(
            samples=samples,
            process_function=mock_process_function,
            stage_name="test_stage",
            dataset_name="test_dataset"
        )
        
        # Should return empty list
        assert len(results) == 0
    
    def test_batch_processing_checkpoints(self):
        """Test that checkpoints are saved after each batch."""
        # Create test samples (more than batch size)
        samples = [
            Sample("sample_1", {"text": "text1"}, "dataset1", "original1"),
            Sample("sample_2", {"text": "text2"}, "dataset1", "original2"),
            Sample("sample_3", {"text": "text3"}, "dataset1", "original3"),
            Sample("sample_4", {"text": "text4"}, "dataset1", "original4"),
            Sample("sample_5", {"text": "text5"}, "dataset1", "original5"),
        ]
        
        # Mock processing function
        def mock_process_function(sample):
            return f"processed_{sample.id}"
        
        # Process samples
        self.batch_processor.process_samples_with_recovery(
            samples=samples,
            process_function=mock_process_function,
            stage_name="test_stage",
            dataset_name="test_dataset"
        )
        
        # Verify checkpoints were created (should be multiple due to batching)
        checkpoint_files = list(self.batch_processor.logger.checkpoint_dir.glob("checkpoint_*.pkl"))
        assert len(checkpoint_files) > 0
        
        # Verify all samples were processed
        for sample in samples:
            assert self.logger.is_sample_processed(sample.id)