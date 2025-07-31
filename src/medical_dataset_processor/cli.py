"""
Command Line Interface for Medical Dataset Processor.

This module provides a comprehensive CLI for processing medical datasets
with translation and content generation capabilities.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler

from .pipeline import MedicalDatasetProcessor, PipelineConfig
from .utils.logging import setup_logging


# Initialize rich console
console = Console()


class ProgressTracker:
    """Tracks and displays processing progress."""
    
    def __init__(self):
        self.progress = None
        self.task_id = None
        self.stats = {}
    
    def start(self, total_steps: int = 5):
        """Start progress tracking."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )
        self.progress.start()
        self.task_id = self.progress.add_task("Initializing...", total=total_steps)
    
    def update(self, description: str, advance: int = 1):
        """Update progress."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=description, advance=advance)
    
    def stop(self):
        """Stop progress tracking."""
        if self.progress:
            self.progress.stop()


def setup_cli_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging for CLI with rich formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def validate_api_keys(deepl_key: Optional[str], openai_key: Optional[str]) -> bool:
    """Validate that required API keys are available."""
    errors = []
    
    if not deepl_key:
        errors.append("DeepL API key is required (use --deepl-key or DEEPL_API_KEY env var)")
    
    if not openai_key:
        errors.append("OpenAI API key is required (use --openai-key or OPENAI_API_KEY env var)")
    
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        return False
    
    return True


def display_config_summary(config: PipelineConfig):
    """Display configuration summary in a nice table."""
    table = Table(title="Pipeline Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Dataset Config", config.datasets_yaml_path)
    table.add_row("Translation Count", str(config.translation_count))
    table.add_row("Generation Count", str(config.generation_count))
    table.add_row("Target Language", config.target_language)
    table.add_row("Output Directory", config.output_dir)
    table.add_row("JSONL File", config.jsonl_filename)
    table.add_row("PDF File", config.pdf_filename)
    table.add_row("PDF Sample Size", str(config.pdf_sample_size))
    table.add_row("Max Retries", str(config.max_retries))
    table.add_row("Batch Size", str(config.batch_size))
    
    console.print(table)


def display_processing_stats(stats: Dict[str, Any]):
    """Display processing statistics."""
    table = Table(title="Processing Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Basic stats
    table.add_row("Datasets Loaded", str(stats.get("datasets_loaded", 0)))
    table.add_row("Samples Selected", str(stats.get("samples_selected", 0)))
    table.add_row("Samples Translated", str(stats.get("samples_translated", 0)))
    table.add_row("Samples Generated", str(stats.get("samples_generated", 0)))
    table.add_row("Samples Exported", str(stats.get("samples_exported", 0)))
    
    # Duration
    if "duration_formatted" in stats:
        table.add_row("Processing Time", stats["duration_formatted"])
    
    # Errors
    error_count = len(stats.get("errors", []))
    error_style = "red" if error_count > 0 else "green"
    table.add_row("Errors", f"[{error_style}]{error_count}[/{error_style}]")
    
    console.print(table)
    
    # Display errors if any
    if stats.get("errors"):
        console.print("\n[red]Errors encountered:[/red]")
        for i, error in enumerate(stats["errors"], 1):
            console.print(f"  {i}. {error}")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Medical Dataset Processor CLI - Automated processing of medical datasets with translation and generation."""
    pass


@cli.command()
@click.option(
    "--datasets-config", "-d",
    default="datasets.yaml",
    help="Path to datasets configuration YAML file",
    type=click.Path(exists=True)
)
@click.option(
    "--deepl-key",
    envvar="DEEPL_API_KEY",
    help="DeepL API key for translation (or set DEEPL_API_KEY env var)"
)
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key for generation (or set OPENAI_API_KEY env var)"
)
@click.option(
    "--output-dir", "-o",
    default="output",
    help="Output directory for results",
    type=click.Path()
)
@click.option(
    "--translation-count", "-t",
    default=50,
    help="Number of samples to translate per dataset",
    type=click.IntRange(1, 1000)
)
@click.option(
    "--generation-count", "-g",
    default=50,
    help="Number of samples to generate per dataset",
    type=click.IntRange(1, 1000)
)
@click.option(
    "--target-language",
    default="FR",
    help="Target language for translation (DeepL language code)"
)
@click.option(
    "--pdf-sample-size",
    default=100,
    help="Number of samples to include in PDF review",
    type=click.IntRange(1, 500)
)
@click.option(
    "--max-retries",
    default=3,
    help="Maximum number of retries for API calls",
    type=click.IntRange(1, 10)
)
@click.option(
    "--batch-size",
    default=10,
    help="Batch size for API processing",
    type=click.IntRange(1, 50)
)
@click.option(
    "--random-seed",
    help="Random seed for reproducible sample selection",
    type=int
)
@click.option(
    "--jsonl-filename",
    default="consolidated_dataset.jsonl",
    help="Name of the output JSONL file"
)
@click.option(
    "--pdf-filename",
    default="sample_review.pdf",
    help="Name of the output PDF file"
)
@click.option(
    "--log-file",
    help="Path to log file (optional)",
    type=click.Path()
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without processing"
)
@click.option(
    "--stats-file",
    help="Save processing statistics to JSON file",
    type=click.Path()
)
def process(
    datasets_config: str,
    deepl_key: Optional[str],
    openai_key: Optional[str],
    output_dir: str,
    translation_count: int,
    generation_count: int,
    target_language: str,
    pdf_sample_size: int,
    max_retries: int,
    batch_size: int,
    random_seed: Optional[int],
    jsonl_filename: str,
    pdf_filename: str,
    log_file: Optional[str],
    verbose: bool,
    dry_run: bool,
    stats_file: Optional[str]
):
    """Process medical datasets with translation and content generation."""
    
    # Setup logging
    setup_cli_logging(verbose, log_file)
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Medical Dataset Processor[/bold blue]\n"
        "Automated processing with translation and generation",
        border_style="blue"
    ))
    
    # Validate API keys
    if not validate_api_keys(deepl_key, openai_key):
        sys.exit(1)
    
    # Create configuration
    try:
        config = PipelineConfig(
            datasets_yaml_path=datasets_config,
            deepl_api_key=deepl_key,
            openai_api_key=openai_key,
            output_dir=output_dir,
            translation_count=translation_count,
            generation_count=generation_count,
            target_language=target_language,
            pdf_sample_size=pdf_sample_size,
            max_retries=max_retries,
            batch_size=batch_size,
            random_seed=random_seed,
            jsonl_filename=jsonl_filename,
            pdf_filename=pdf_filename
        )
    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)
    
    # Display configuration
    display_config_summary(config)
    
    # Initialize processor
    processor = MedicalDatasetProcessor(config)
    
    # Validate configuration
    console.print("\n[yellow]Validating configuration...[/yellow]")
    validation_results = processor.validate_configuration()
    
    if not validation_results["valid"]:
        console.print("[red]Configuration validation failed:[/red]")
        for error in validation_results["errors"]:
            console.print(f"  • {error}")
        sys.exit(1)
    
    if validation_results["warnings"]:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in validation_results["warnings"]:
            console.print(f"  • {warning}")
    
    console.print("[green]✓ Configuration validated successfully[/green]")
    
    # Exit if dry run
    if dry_run:
        console.print("\n[blue]Dry run completed - configuration is valid[/blue]")
        return
    
    # Confirm processing
    if not click.confirm("\nProceed with processing?"):
        console.print("Processing cancelled.")
        return
    
    # Start processing with progress tracking
    progress_tracker = ProgressTracker()
    progress_tracker.start()
    
    try:
        # Process datasets
        progress_tracker.update("Loading datasets...")
        consolidated_dataset = processor.process_datasets()
        progress_tracker.update("Processing completed!", advance=5)
        
        # Get final statistics
        stats = processor.get_processing_stats()
        
        # Display results
        progress_tracker.stop()
        console.print("\n[green]✓ Processing completed successfully![/green]")
        
        # Display statistics
        display_processing_stats(stats)
        
        # Save statistics if requested
        if stats_file:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            console.print(f"\n[blue]Statistics saved to {stats_file}[/blue]")
        
        # Display output files
        output_path = Path(output_dir)
        jsonl_path = output_path / jsonl_filename
        pdf_path = output_path / pdf_filename
        
        console.print(f"\n[green]Output files:[/green]")
        console.print(f"  • JSONL dataset: {jsonl_path}")
        console.print(f"  • PDF sample: {pdf_path}")
        
    except KeyboardInterrupt:
        progress_tracker.stop()
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        progress_tracker.stop()
        console.print(f"\n[red]Processing failed:[/red] {e}")
        
        # Display partial statistics if available
        try:
            stats = processor.get_processing_stats()
            if stats.get("errors"):
                console.print("\n[red]Errors encountered:[/red]")
                for error in stats["errors"]:
                    console.print(f"  • {error}")
        except:
            pass
        
        sys.exit(1)


@cli.command()
@click.option(
    "--datasets-config", "-d",
    default="datasets.yaml",
    help="Path to datasets configuration YAML file",
    type=click.Path(exists=True)
)
def validate(datasets_config: str):
    """Validate datasets configuration file."""
    
    console.print(Panel.fit(
        "[bold blue]Configuration Validator[/bold blue]\n"
        "Validate datasets configuration",
        border_style="blue"
    ))
    
    try:
        # Create minimal config for validation
        config = PipelineConfig(
            datasets_yaml_path=datasets_config,
            deepl_api_key="dummy",  # Just for validation
            openai_api_key="dummy"
        )
        
        processor = MedicalDatasetProcessor(config)
        
        # Validate configuration (excluding API keys)
        console.print(f"[yellow]Validating {datasets_config}...[/yellow]")
        
        # Load and validate dataset configs
        from .loaders.dataset_loader import DatasetLoader
        loader = DatasetLoader()
        configs = loader.load_config(datasets_config)
        
        console.print(f"[green]✓ Successfully loaded {len(configs)} dataset configurations[/green]")
        
        # Display dataset summary
        table = Table(title="Dataset Configurations")
        table.add_column("Dataset", style="cyan")
        table.add_column("Source Type", style="green")
        table.add_column("Source Path", style="yellow")
        table.add_column("Text Fields", style="magenta")
        
        for name, config in configs.items():
            text_fields = ", ".join(config.text_fields) if config.text_fields else "N/A"
            table.add_row(name, config.source_type, config.source_path, text_fields)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("stats_file", type=click.Path(exists=True))
def show_stats(stats_file: str):
    """Display processing statistics from a saved stats file."""
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        console.print(Panel.fit(
            "[bold blue]Processing Statistics[/bold blue]\n"
            f"From: {stats_file}",
            border_style="blue"
        ))
        
        display_processing_stats(stats)
        
    except Exception as e:
        console.print(f"[red]Error reading stats file:[/red] {e}")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    console.print("[blue]Medical Dataset Processor v0.1.0[/blue]")


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()