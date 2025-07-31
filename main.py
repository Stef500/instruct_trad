#!/usr/bin/env python3
"""
Main entry point for the Medical Dataset Processor.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medical_dataset_processor.pipeline import MedicalDatasetProcessor, PipelineConfig
from medical_dataset_processor.utils.logging import setup_logging


def main():
    """Main entry point for the medical dataset processor."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = PipelineConfig(
            datasets_yaml_path="datasets.yaml",
            deepl_api_key=os.getenv("DEEPL_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            translation_count=50,
            generation_count=50,
            target_language="FR",
            output_dir="output",
            random_seed=42  # For reproducible results
        )
        
        # Create processor
        processor = MedicalDatasetProcessor(config)
        
        # Validate configuration
        logger.info("Validating configuration...")
        validation_results = processor.validate_configuration()
        
        if not validation_results["valid"]:
            logger.error("Configuration validation failed:")
            for error in validation_results["errors"]:
                logger.error(f"  - {error}")
            return 1
        
        if validation_results["warnings"]:
            logger.warning("Configuration warnings:")
            for warning in validation_results["warnings"]:
                logger.warning(f"  - {warning}")
        
        # Run the pipeline
        logger.info("Starting medical dataset processing pipeline...")
        consolidated_dataset = processor.process_datasets()
        
        # Print final statistics
        stats = processor.get_processing_stats()
        logger.info("Pipeline completed successfully!")
        logger.info(f"Processing statistics:")
        logger.info(f"  - Datasets loaded: {stats['datasets_loaded']}")
        logger.info(f"  - Samples selected: {stats['samples_selected']}")
        logger.info(f"  - Samples translated: {stats['samples_translated']}")
        logger.info(f"  - Samples generated: {stats['samples_generated']}")
        logger.info(f"  - Samples exported: {stats['samples_exported']}")
        logger.info(f"  - Total processing time: {stats.get('duration_formatted', 'N/A')}")
        
        if stats["errors"]:
            logger.warning(f"Encountered {len(stats['errors'])} errors during processing")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


