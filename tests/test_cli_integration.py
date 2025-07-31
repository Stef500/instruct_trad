"""
Integration tests for CLI functionality.
"""
import subprocess
import sys
from pathlib import Path


def test_cli_module_import():
    """Test that the CLI module can be imported successfully."""
    try:
        from medical_dataset_processor.cli import main, cli
        assert callable(main)
        assert callable(cli)
    except ImportError as e:
        pytest.fail(f"Failed to import CLI module: {e}")


def test_cli_entry_point_exists():
    """Test that the CLI entry point is properly installed."""
    result = subprocess.run(
        [sys.executable, "-m", "medical_dataset_processor.cli", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Medical Dataset Processor CLI" in result.stdout


def test_installed_cli_command():
    """Test that the installed CLI command works."""
    result = subprocess.run(
        ["medical-dataset-processor", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Medical Dataset Processor CLI" in result.stdout


def test_cli_version_command():
    """Test CLI version command."""
    result = subprocess.run(
        ["medical-dataset-processor", "version"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Medical Dataset Processor v0.1.0" in result.stdout


def test_cli_validate_command():
    """Test CLI validate command with existing datasets.yaml."""
    datasets_yaml = Path("datasets.yaml")
    if not datasets_yaml.exists():
        pytest.skip("datasets.yaml not found")
    
    result = subprocess.run(
        ["medical-dataset-processor", "validate", "--datasets-config", str(datasets_yaml)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Successfully loaded" in result.stdout


def test_cli_process_dry_run():
    """Test CLI process command with dry run."""
    datasets_yaml = Path("datasets.yaml")
    if not datasets_yaml.exists():
        pytest.skip("datasets.yaml not found")
    
    result = subprocess.run([
        "medical-dataset-processor", "process",
        "--datasets-config", str(datasets_yaml),
        "--deepl-key", "test-key",
        "--openai-key", "test-key",
        "--dry-run"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Dry run completed" in result.stdout


def test_cli_process_missing_api_keys():
    """Test CLI process command fails without API keys."""
    datasets_yaml = Path("datasets.yaml")
    if not datasets_yaml.exists():
        pytest.skip("datasets.yaml not found")
    
    result = subprocess.run([
        "medical-dataset-processor", "process",
        "--datasets-config", str(datasets_yaml)
    ], capture_output=True, text=True)
    
    assert result.returncode == 1
    assert "API key is required" in result.stdout