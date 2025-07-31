"""
Tests for the logging and error handling system.
"""
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.medical_dataset_processor.utils.logging import (
    ProcessingLogger, APIError, ProcessingStats, ErrorReport, ProcessingState, ErrorSeverity
)
from src.medical_dataset_processor.models.core import Sample


class TestAPIError:
    """Test cases for APIError class."""
    
    def test_api_error_creation(self):
        """Test creating an APIError instance."""
        error = APIError(
            api_name="deepl",
            error_type="HTTPError",
            error_message="Rate limit exceeded",
            sample_id="sample_123",
            retry_count=2,
            severity=ErrorSeverity.HIGH
        )
        
        assert error.api_name == "deepl"
        assert error.error_type == "HTTPError"
        assert error.error_message == "Rate limit exceeded"
        assert error.sample_id == "sample_123"
        assert error.retry_count == 2
        assert error.severity == ErrorSeverity.HIGH
        assert isinstance(error.timestamp, datetime)
    
    def test_api_error_to_dict(self):
        """Test converting APIError to dictionary."""
        error = APIError(
            api_name="openai",
            error_type="APIError",
            error_message="Invalid request",
            sample_id="sample_456"
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["api_name"] == "openai"
        assert error_dict["error_type"] == "APIError"
        assert error_dict["error_message"] == "Invalid request"
        assert error_dict["sample_id"] == "sample_456"
        assert error_dict["retry_count"] == 0
        assert error_dict["severity"] == ErrorSeverity.MEDIUM.value
        assert "timestamp" in error_dict


class TestProcessingStats:
    """Test cases for ProcessingStats class."""
    
    def test_processing_stats_creation(self):
        """Test creating ProcessingStats instance."""
        stats = ProcessingStats(
            total_samples=100,
            successful_translations=45,
            failed_translations=5,
            successful_generations=48,
            failed_generations=2
        )
        
        assert stats.total_samples == 100
        assert stats.successful_translations == 45
        assert stats.failed_translations == 5
        assert stats.successful_generations == 48
        assert stats.failed_generations == 2
    
    def test_processing_stats_to_dict(self):
        """Test converting ProcessingStats to dictionary."""
        start_time = datetime.now()
        stats = ProcessingStats(
            total_samples=50,
            start_time=start_time
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_samples"] == 50
        assert stats_dict["start_time"] == start_time.isoformat()
        assert stats_dict["end_time"] is None


class TestErrorReport:
    """Test cases for ErrorReport class."""
    
    def test_error_report_creation(self):
        """Test creating an ErrorReport instance."""
        errors = [
            APIError("deepl", "HTTPError", "Rate limit", "sample_1"),
            APIError("openai", "APIError", "Invalid key", "sample_2")
        ]
        stats = ProcessingStats(total_samples=100)
        
        report = ErrorReport(errors=errors, stats=stats)
        
        assert len(report.errors) == 2
        assert report.stats == stats
        assert isinstance(report.generation_timestamp, datetime)
    
    def test_error_report_summary(self):
        """Test error report summary generation."""
        errors = [
            APIError("deepl", "HTTPError", "Rate limit", "sample_1", severity=ErrorSeverity.HIGH),
            APIError("deepl", "APIError", "Network error", "sample_2", severity=ErrorSeverity.MEDIUM),
            APIError("openai", "AuthError", "Invalid key", "sample_3", severity=ErrorSeverity.CRITICAL)
        ]
        
        report = ErrorReport(errors=errors)
        summary = report.get_summary()
        
        assert summary["total_errors"] == 3
        assert summary["errors_by_api"]["deepl"] == 2
        assert summary["errors_by_api"]["openai"] == 1
        assert summary["errors_by_severity"]["high"] == 1
        assert summary["errors_by_severity"]["medium"] == 1
        assert summary["errors_by_severity"]["critical"] == 1
        assert summary["most_problematic_api"] == "deepl"
    
    def test_error_report_empty(self):
        """Test error report with no errors."""
        report = ErrorReport()
        summary = report.get_summary()
        
        assert summary["total_errors"] == 0


class TestProcessingState:
    """Test cases for ProcessingState class."""
    
    def test_processing_state_creation(self):
        """Test creating ProcessingState instance."""
        state = ProcessingState(
            processed_samples=["sample_1", "sample_2"],
            failed_samples=["sample_3"],
            current_dataset="medqa",
            processing_stage="translating"
        )
        
        assert state.processed_samples == ["sample_1", "sample_2"]
        assert state.failed_samples == ["sample_3"]
        assert state.current_dataset == "medqa"
        assert state.processing_stage == "translating"
    
    def test_processing_state_serialization(self):
        """Test ProcessingState serialization and deserialization."""
        original_state = ProcessingState(
            processed_samples=["sample_1"],
            current_dataset="pubmedqa",
            processing_stage="generating"
        )
        
        # Convert to dict and back
        state_dict = original_state.to_dict()
        restored_state = ProcessingState.from_dict(state_dict)
        
        assert restored_state.processed_samples == original_state.processed_samples
        assert restored_state.current_dataset == original_state.current_dataset
        assert restored_state.processing_stage == original_state.processing_stage


class TestProcessingLogger:
    """Test cases for ProcessingLogger class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
    
    def test_logger_initialization(self):
        """Test ProcessingLogger initialization."""
        assert self.logger.log_dir.exists()
        assert self.logger.checkpoint_dir.exists()
        assert len(self.logger.errors) == 0
        assert isinstance(self.logger.stats, ProcessingStats)
        assert isinstance(self.logger.state, ProcessingState)
    
    def test_log_api_error(self):
        """Test logging API errors."""
        error = Exception("Test error")
        
        self.logger.log_api_error("deepl", error, "sample_123", retry_count=1)
        
        assert len(self.logger.errors) == 1
        api_error = self.logger.errors[0]
        assert api_error.api_name == "deepl"
        assert api_error.error_type == "Exception"
        assert api_error.error_message == "Test error"
        assert api_error.sample_id == "sample_123"
        assert api_error.retry_count == 1
        assert self.logger.stats.failed_translations == 1
    
    def test_log_api_error_openai(self):
        """Test logging OpenAI API errors."""
        error = Exception("OpenAI error")
        
        self.logger.log_api_error("openai", error, "sample_456")
        
        assert len(self.logger.errors) == 1
        assert self.logger.stats.failed_generations == 1
    
    def test_log_processing_stats(self):
        """Test logging processing statistics."""
        stats = ProcessingStats(
            total_samples=100,
            successful_translations=50,
            successful_generations=45
        )
        
        self.logger.log_processing_stats(stats)
        
        assert self.logger.stats == stats
        
        # Check that stats file was created
        stats_files = list(self.logger.log_dir.glob("stats_*.json"))
        assert len(stats_files) == 1
    
    def test_generate_error_report(self):
        """Test generating error reports."""
        # Add some errors
        self.logger.log_api_error("deepl", Exception("Error 1"), "sample_1")
        self.logger.log_api_error("openai", Exception("Error 2"), "sample_2")
        
        report = self.logger.generate_error_report()
        
        assert len(report.errors) == 2
        assert report.stats == self.logger.stats
        
        # Check that report file was created
        report_files = list(self.logger.log_dir.glob("error_report_*.json"))
        assert len(report_files) == 1
    
    def test_save_checkpoint(self):
        """Test saving checkpoints."""
        # Set up some state
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.processing_stage = "translating"
        self.logger.log_api_error("deepl", Exception("Test"), "sample_3")
        
        additional_data = {"custom_field": "custom_value"}
        self.logger.save_checkpoint(additional_data)
        
        # Check that checkpoint file was created
        checkpoint_files = list(self.logger.checkpoint_dir.glob("checkpoint_*.pkl"))
        assert len(checkpoint_files) == 1
        
        # Verify checkpoint content
        with open(checkpoint_files[0], 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        assert "state" in checkpoint_data
        assert "errors" in checkpoint_data
        assert "stats" in checkpoint_data
        assert "additional_data" in checkpoint_data
        assert checkpoint_data["additional_data"]["custom_field"] == "custom_value"
    
    def test_load_latest_checkpoint(self):
        """Test loading the latest checkpoint."""
        # Save a checkpoint first
        self.logger.state.processed_samples = ["sample_1"]
        self.logger.state.processing_stage = "generating"
        self.logger.log_api_error("openai", Exception("Test error"), "sample_2")
        self.logger.save_checkpoint()
        
        # Create a new logger and load checkpoint
        new_logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
        
        checkpoint_data = new_logger.load_latest_checkpoint()
        
        assert checkpoint_data is not None
        assert new_logger.state.processed_samples == ["sample_1"]
        assert new_logger.state.processing_stage == "generating"
        assert len(new_logger.errors) == 1
        assert new_logger.errors[0].sample_id == "sample_2"
    
    def test_load_checkpoint_no_files(self):
        """Test loading checkpoint when no files exist."""
        empty_logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/empty_logs",
            checkpoint_dir=f"{self.temp_dir}/empty_checkpoints"
        )
        
        checkpoint_data = empty_logger.load_latest_checkpoint()
        
        assert checkpoint_data is None
    
    def test_update_processing_stage(self):
        """Test updating processing stage."""
        self.logger.update_processing_stage("translating", "medqa")
        
        assert self.logger.state.processing_stage == "translating"
        assert self.logger.state.current_dataset == "medqa"
    
    def test_mark_sample_processed(self):
        """Test marking samples as processed."""
        self.logger.mark_sample_processed("sample_123")
        
        assert "sample_123" in self.logger.state.processed_samples
        assert self.logger.is_sample_processed("sample_123")
        assert not self.logger.is_sample_failed("sample_123")
    
    def test_mark_sample_failed(self):
        """Test marking samples as failed."""
        self.logger.mark_sample_failed("sample_456")
        
        assert "sample_456" in self.logger.state.failed_samples
        assert self.logger.is_sample_failed("sample_456")
        assert not self.logger.is_sample_processed("sample_456")
    
    def test_mark_sample_processed_after_failed(self):
        """Test marking a previously failed sample as processed."""
        # First mark as failed
        self.logger.mark_sample_failed("sample_789")
        assert self.logger.is_sample_failed("sample_789")
        
        # Then mark as processed
        self.logger.mark_sample_processed("sample_789")
        assert self.logger.is_sample_processed("sample_789")
        assert not self.logger.is_sample_failed("sample_789")
    
    def test_determine_error_severity(self):
        """Test error severity determination."""
        # Create custom exception classes for testing
        class AuthenticationError(Exception):
            pass
        
        class APIError(Exception):
            pass
        
        class HTTPError(Exception):
            pass
        
        # Test critical errors
        auth_error = AuthenticationError("Authentication failed")
        severity = self.logger._determine_error_severity(auth_error, 0)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test high severity for repeated failures
        normal_error = Exception("Normal error")
        severity = self.logger._determine_error_severity(normal_error, 3)
        assert severity == ErrorSeverity.HIGH
        
        # Test medium severity for API errors
        api_error = APIError("API error")
        severity = self.logger._determine_error_severity(api_error, 1)
        assert severity == ErrorSeverity.MEDIUM
        
        # Test medium severity for HTTP errors
        http_error = HTTPError("HTTP error")
        severity = self.logger._determine_error_severity(http_error, 1)
        assert severity == ErrorSeverity.MEDIUM
        
        # Test low severity for other errors
        other_error = Exception("Other error")
        severity = self.logger._determine_error_severity(other_error, 0)
        assert severity == ErrorSeverity.LOW


class TestIntegration:
    """Integration tests for the logging system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
    
    def test_full_processing_workflow(self):
        """Test a complete processing workflow with logging."""
        # Start processing
        self.logger.update_processing_stage("loading", "medqa")
        self.logger.stats.start_time = datetime.now()
        
        # Process some samples
        self.logger.mark_sample_processed("sample_1")
        self.logger.mark_sample_processed("sample_2")
        
        # Encounter some errors
        self.logger.log_api_error("deepl", Exception("Rate limit"), "sample_3", retry_count=1)
        self.logger.mark_sample_failed("sample_3")
        
        # Continue processing
        self.logger.update_processing_stage("generating")
        self.logger.mark_sample_processed("sample_4")
        
        # Save checkpoint
        self.logger.save_checkpoint({"current_batch": 1})
        
        # Update stats
        stats = ProcessingStats(
            total_samples=4,
            successful_translations=2,
            failed_translations=1,
            successful_generations=1,
            failed_generations=0
        )
        self.logger.log_processing_stats(stats)
        
        # Generate final report
        report = self.logger.generate_error_report()
        
        # Verify the complete workflow
        assert len(self.logger.state.processed_samples) == 3
        assert len(self.logger.state.failed_samples) == 1
        assert len(self.logger.errors) == 1
        assert report.stats.total_samples == 4
        
        # Verify files were created
        assert len(list(self.logger.log_dir.glob("*.log"))) >= 1
        assert len(list(self.logger.log_dir.glob("stats_*.json"))) == 1
        assert len(list(self.logger.log_dir.glob("error_report_*.json"))) == 1
        assert len(list(self.logger.checkpoint_dir.glob("checkpoint_*.pkl"))) == 1
    
    def test_recovery_workflow(self):
        """Test recovery from a saved checkpoint."""
        # Simulate initial processing
        self.logger.state.processed_samples = ["sample_1", "sample_2"]
        self.logger.state.processing_stage = "translating"
        self.logger.state.current_dataset = "pubmedqa"
        self.logger.log_api_error("deepl", Exception("Network error"), "sample_3")
        self.logger.save_checkpoint()
        
        # Create new logger instance (simulating restart)
        recovery_logger = ProcessingLogger(
            log_dir=f"{self.temp_dir}/logs",
            checkpoint_dir=f"{self.temp_dir}/checkpoints"
        )
        
        # Load checkpoint
        checkpoint_data = recovery_logger.load_latest_checkpoint()
        
        # Verify recovery
        assert checkpoint_data is not None
        assert recovery_logger.state.processed_samples == ["sample_1", "sample_2"]
        assert recovery_logger.state.processing_stage == "translating"
        assert recovery_logger.state.current_dataset == "pubmedqa"
        assert len(recovery_logger.errors) == 1
        
        # Continue processing from where we left off
        assert recovery_logger.is_sample_processed("sample_1")
        assert recovery_logger.is_sample_processed("sample_2")
        assert not recovery_logger.is_sample_processed("sample_4")