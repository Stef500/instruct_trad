"""
Tests for the CLI interface.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from medical_dataset_processor.cli import cli, validate_api_keys, display_config_summary, display_processing_stats
from medical_dataset_processor.pipeline import PipelineConfig


class TestCLI:
    """Test cases for the CLI interface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test datasets.yaml file
        self.datasets_yaml = Path(self.temp_dir) / "datasets.yaml"
        self.datasets_yaml.write_text("""
datasets:
  test_dataset:
    name: "test_dataset"
    source_type: "local"
    source_path: "test_data.json"
    format: "json"
    text_fields: ["question", "answer"]
""")
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Medical Dataset Processor CLI" in result.output
        assert "process" in result.output
        assert "validate" in result.output
    
    def test_process_command_help(self):
        """Test process command help."""
        result = self.runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--datasets-config" in result.output
        assert "--deepl-key" in result.output
        assert "--openai-key" in result.output
        assert "--translation-count" in result.output
        assert "--generation-count" in result.output
    
    def test_validate_command_help(self):
        """Test validate command help."""
        result = self.runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--datasets-config" in result.output
    
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "Medical Dataset Processor v0.1.0" in result.output
    
    def test_validate_api_keys_missing_both(self):
        """Test API key validation with missing keys."""
        with patch('medical_dataset_processor.cli.console') as mock_console:
            result = validate_api_keys(None, None)
            assert result is False
            assert mock_console.print.call_count == 2  # Two error messages
    
    def test_validate_api_keys_missing_deepl(self):
        """Test API key validation with missing DeepL key."""
        with patch('medical_dataset_processor.cli.console') as mock_console:
            result = validate_api_keys(None, "openai-key")
            assert result is False
            assert mock_console.print.call_count == 1
    
    def test_validate_api_keys_missing_openai(self):
        """Test API key validation with missing OpenAI key."""
        with patch('medical_dataset_processor.cli.console') as mock_console:
            result = validate_api_keys("deepl-key", None)
            assert result is False
            assert mock_console.print.call_count == 1
    
    def test_validate_api_keys_success(self):
        """Test API key validation with both keys present."""
        result = validate_api_keys("deepl-key", "openai-key")
        assert result is True
    
    def test_display_config_summary(self):
        """Test configuration summary display."""
        config = PipelineConfig(
            datasets_yaml_path="test.yaml",
            translation_count=25,
            generation_count=25
        )
        
        with patch('medical_dataset_processor.cli.console') as mock_console:
            display_config_summary(config)
            mock_console.print.assert_called_once()
    
    def test_display_processing_stats(self):
        """Test processing statistics display."""
        stats = {
            "datasets_loaded": 2,
            "samples_selected": 100,
            "samples_translated": 50,
            "samples_generated": 50,
            "samples_exported": 100,
            "duration_formatted": "00:05:30",
            "errors": ["Test error"]
        }
        
        with patch('medical_dataset_processor.cli.console') as mock_console:
            display_processing_stats(stats)
            # Should print table and errors
            assert mock_console.print.call_count >= 2
    
    def test_validate_command_success(self):
        """Test validate command with valid configuration."""
        with patch('medical_dataset_processor.loaders.dataset_loader.DatasetLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_config.return_value = {
                "test_dataset": Mock(
                    source_type="local",
                    source_path="test.json",
                    text_fields=["question", "answer"]
                )
            }
            
            result = self.runner.invoke(cli, [
                "validate",
                "--datasets-config", str(self.datasets_yaml)
            ])
            
            assert result.exit_code == 0
            assert "Successfully loaded" in result.output
    
    def test_validate_command_invalid_file(self):
        """Test validate command with invalid configuration file."""
        result = self.runner.invoke(cli, [
            "validate",
            "--datasets-config", "nonexistent.yaml"
        ])
        
        assert result.exit_code == 2  # Click error for missing file
    
    def test_process_command_missing_api_keys(self):
        """Test process command with missing API keys."""
        result = self.runner.invoke(cli, [
            "process",
            "--datasets-config", str(self.datasets_yaml)
        ])
        
        assert result.exit_code == 1
        assert "API key is required" in result.output
    
    def test_process_command_dry_run(self):
        """Test process command with dry run."""
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key",
                "--dry-run"
            ])
            
            assert result.exit_code == 0
            assert "Dry run completed" in result.output
            # Should not call process_datasets in dry run
            mock_processor.process_datasets.assert_not_called()
    
    def test_process_command_validation_failure(self):
        """Test process command with validation failure."""
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": False,
                "errors": ["Test validation error"],
                "warnings": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key"
            ])
            
            assert result.exit_code == 1
            assert "Configuration validation failed" in result.output
            assert "Test validation error" in result.output
    
    @patch('medical_dataset_processor.cli.click.confirm')
    def test_process_command_user_cancellation(self, mock_confirm):
        """Test process command when user cancels."""
        mock_confirm.return_value = False
        
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key"
            ])
            
            assert result.exit_code == 0
            assert "Processing cancelled" in result.output
    
    @patch('medical_dataset_processor.cli.click.confirm')
    def test_process_command_success(self, mock_confirm):
        """Test successful process command execution."""
        mock_confirm.return_value = True
        
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            mock_processor.process_datasets.return_value = Mock()
            mock_processor.get_processing_stats.return_value = {
                "datasets_loaded": 1,
                "samples_selected": 100,
                "samples_translated": 50,
                "samples_generated": 50,
                "samples_exported": 100,
                "errors": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key",
                "--output-dir", str(self.temp_dir)
            ])
            
            assert result.exit_code == 0
            assert "Processing completed successfully" in result.output
            mock_processor.process_datasets.assert_called_once()
    
    @patch('medical_dataset_processor.cli.click.confirm')
    def test_process_command_with_stats_file(self, mock_confirm):
        """Test process command with statistics file output."""
        mock_confirm.return_value = True
        stats_file = Path(self.temp_dir) / "stats.json"
        
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            mock_processor.process_datasets.return_value = Mock()
            mock_processor.get_processing_stats.return_value = {
                "datasets_loaded": 1,
                "samples_selected": 100,
                "samples_translated": 50,
                "samples_generated": 50,
                "samples_exported": 100,
                "errors": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key",
                "--output-dir", str(self.temp_dir),
                "--stats-file", str(stats_file)
            ])
            
            assert result.exit_code == 0
            assert "Statistics saved" in result.output
            assert stats_file.exists()
            
            # Verify stats file content
            with open(stats_file) as f:
                saved_stats = json.load(f)
            assert saved_stats["datasets_loaded"] == 1
    
    @patch('medical_dataset_processor.cli.click.confirm')
    def test_process_command_processing_error(self, mock_confirm):
        """Test process command with processing error."""
        mock_confirm.return_value = True
        
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            mock_processor.process_datasets.side_effect = Exception("Processing failed")
            mock_processor.get_processing_stats.return_value = {
                "errors": ["Processing failed"]
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key"
            ])
            
            assert result.exit_code == 1
            assert "Processing failed" in result.output
    
    def test_show_stats_command(self):
        """Test show-stats command."""
        stats_file = Path(self.temp_dir) / "test_stats.json"
        test_stats = {
            "datasets_loaded": 2,
            "samples_selected": 100,
            "samples_translated": 50,
            "samples_generated": 50,
            "samples_exported": 100,
            "errors": []
        }
        
        with open(stats_file, 'w') as f:
            json.dump(test_stats, f)
        
        result = self.runner.invoke(cli, [
            "show-stats",
            str(stats_file)
        ])
        
        assert result.exit_code == 0
        assert "Processing Statistics" in result.output
    
    def test_show_stats_command_invalid_file(self):
        """Test show-stats command with invalid file."""
        result = self.runner.invoke(cli, [
            "show-stats",
            "nonexistent.json"
        ])
        
        assert result.exit_code == 2  # Click error for missing file
    
    def test_process_command_all_options(self):
        """Test process command with all options specified."""
        with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor
            mock_processor.validate_configuration.return_value = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            result = self.runner.invoke(cli, [
                "process",
                "--datasets-config", str(self.datasets_yaml),
                "--deepl-key", "test-deepl-key",
                "--openai-key", "test-openai-key",
                "--output-dir", str(self.temp_dir),
                "--translation-count", "25",
                "--generation-count", "25",
                "--target-language", "ES",
                "--pdf-sample-size", "50",
                "--max-retries", "5",
                "--batch-size", "5",
                "--random-seed", "42",
                "--jsonl-filename", "custom.jsonl",
                "--pdf-filename", "custom.pdf",
                "--verbose",
                "--dry-run"
            ])
            
            assert result.exit_code == 0
            
            # Verify configuration was created with correct values
            call_args = mock_processor_class.call_args[0][0]  # First positional argument (config)
            assert call_args.translation_count == 25
            assert call_args.generation_count == 25
            assert call_args.target_language == "ES"
            assert call_args.pdf_sample_size == 50
            assert call_args.max_retries == 5
            assert call_args.batch_size == 5
            assert call_args.random_seed == 42
            assert call_args.jsonl_filename == "custom.jsonl"
            assert call_args.pdf_filename == "custom.pdf"
    
    def test_process_command_environment_variables(self):
        """Test process command using environment variables for API keys."""
        with patch.dict(os.environ, {
            'DEEPL_API_KEY': 'env-deepl-key',
            'OPENAI_API_KEY': 'env-openai-key'
        }):
            with patch('medical_dataset_processor.cli.MedicalDatasetProcessor') as mock_processor_class:
                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor
                mock_processor.validate_configuration.return_value = {
                    "valid": True,
                    "errors": [],
                    "warnings": []
                }
                
                result = self.runner.invoke(cli, [
                    "process",
                    "--datasets-config", str(self.datasets_yaml),
                    "--dry-run"
                ])
                
                assert result.exit_code == 0
                
                # Verify API keys were picked up from environment
                call_args = mock_processor_class.call_args[0][0]  # First positional argument (config)
                assert call_args.deepl_api_key == "env-deepl-key"
                assert call_args.openai_api_key == "env-openai-key"
    
    def test_process_command_invalid_parameter_ranges(self):
        """Test process command with invalid parameter ranges."""
        # Test invalid translation count
        result = self.runner.invoke(cli, [
            "process",
            "--datasets-config", str(self.datasets_yaml),
            "--translation-count", "0"
        ])
        assert result.exit_code == 2  # Click validation error
        
        # Test invalid batch size
        result = self.runner.invoke(cli, [
            "process",
            "--datasets-config", str(self.datasets_yaml),
            "--batch-size", "0"
        ])
        assert result.exit_code == 2  # Click validation error


class TestProgressTracker:
    """Test cases for the ProgressTracker class."""
    
    def test_progress_tracker_lifecycle(self):
        """Test progress tracker start, update, and stop."""
        from medical_dataset_processor.cli import ProgressTracker
        
        tracker = ProgressTracker()
        
        # Test start
        tracker.start(total_steps=3)
        assert tracker.progress is not None
        assert tracker.task_id is not None
        
        # Test update
        tracker.update("Step 1", advance=1)
        tracker.update("Step 2", advance=1)
        
        # Test stop
        tracker.stop()
        # After stop, progress should be cleaned up
        # (We can't easily test the internal state without mocking rich)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_main_function(self):
        """Test the main CLI entry point function."""
        from medical_dataset_processor.cli import main
        
        with patch('medical_dataset_processor.cli.cli') as mock_cli:
            main()
            mock_cli.assert_called_once()
    
    def test_cli_main_function_with_exception(self):
        """Test main function handles exceptions gracefully."""
        from medical_dataset_processor.cli import main
        
        with patch('medical_dataset_processor.cli.cli') as mock_cli:
            mock_cli.side_effect = Exception("Test error")
            
            with patch('medical_dataset_processor.cli.console') as mock_console:
                with pytest.raises(SystemExit):
                    main()
                
                mock_console.print.assert_called_once()
                assert "Unexpected error" in str(mock_console.print.call_args)