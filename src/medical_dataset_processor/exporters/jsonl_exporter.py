"""
JSONL exporter for consolidated medical datasets.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from ..models.core import ConsolidatedDataset, ProcessedSample


logger = logging.getLogger(__name__)


class JSONLExporter:
    """
    Exports consolidated datasets to JSONL format with metadata validation.
    
    JSONL (JSON Lines) format stores one JSON object per line, making it
    suitable for streaming and processing large datasets.
    """
    
    def __init__(self):
        """Initialize the JSONL exporter."""
        self.exported_count = 0
        self.validation_errors = []
    
    def export(self, dataset: ConsolidatedDataset, output_path: str) -> None:
        """
        Export the consolidated dataset to JSONL format.
        
        Args:
            dataset: The consolidated dataset to export
            output_path: Path where the JSONL file will be saved
            
        Raises:
            ValueError: If dataset is invalid or output path is invalid
            IOError: If file cannot be written
        """
        if not isinstance(dataset, ConsolidatedDataset):
            raise ValueError("Dataset must be a ConsolidatedDataset instance")
        
        if not output_path:
            raise ValueError("Output path cannot be empty")
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(dataset.samples)} samples to {output_path}")
        
        self.exported_count = 0
        self.validation_errors = []
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in dataset.samples:
                    jsonl_line = self._convert_sample_to_jsonl(sample)
                    
                    # Validate the JSONL format
                    if self._validate_jsonl_format(jsonl_line):
                        # Write as JSON line
                        json_str = json.dumps(jsonl_line, ensure_ascii=False, separators=(',', ':'))
                        f.write(json_str + '\n')
                        self.exported_count += 1
                    else:
                        logger.warning(f"Skipping invalid sample: {sample.original_sample.id}")
                
                # Add dataset metadata as the last line
                metadata_line = self._create_metadata_line(dataset)
                if self._validate_jsonl_format(metadata_line):
                    json_str = json.dumps(metadata_line, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_str + '\n')
        
        except IOError as e:
            logger.error(f"Failed to write JSONL file: {e}")
            raise
        
        logger.info(f"Successfully exported {self.exported_count} samples to {output_path}")
        
        if self.validation_errors:
            logger.warning(f"Encountered {len(self.validation_errors)} validation errors during export")
    
    def _convert_sample_to_jsonl(self, sample: ProcessedSample) -> Dict[str, Any]:
        """
        Convert a ProcessedSample to JSONL format with all metadata.
        
        Args:
            sample: The processed sample to convert
            
        Returns:
            Dictionary representing the JSONL line
        """
        jsonl_data = {
            # Core sample data
            "id": sample.original_sample.id,
            "source_dataset": sample.original_sample.source_dataset,
            "original_text": sample.original_sample.original_text,
            "processed_content": sample.processed_content,
            "processing_type": sample.processing_type,
            
            # Original sample content (for reference)
            "original_content": sample.original_sample.content,
            
            # Processing metadata
            "metadata": {
                **sample.metadata,
                "quality_score": sample.quality_score,
                "export_timestamp": datetime.now().isoformat(),
            },
            
            # Record type for identification
            "record_type": "processed_sample"
        }
        
        return jsonl_data
    
    def _create_metadata_line(self, dataset: ConsolidatedDataset) -> Dict[str, Any]:
        """
        Create a metadata line for the dataset.
        
        Args:
            dataset: The consolidated dataset
            
        Returns:
            Dictionary representing the metadata JSONL line
        """
        sample_counts = dataset.get_sample_count_by_dataset()
        translation_count = len(dataset.get_translation_samples())
        generation_count = len(dataset.get_generation_samples())
        
        metadata_line = {
            "record_type": "dataset_metadata",
            "total_samples": len(dataset.samples),
            "translation_samples": translation_count,
            "generation_samples": generation_count,
            "samples_by_dataset": sample_counts,
            "creation_timestamp": dataset.creation_timestamp.isoformat(),
            "export_timestamp": datetime.now().isoformat(),
            "dataset_metadata": dataset.metadata,
            "export_stats": {
                "exported_count": self.exported_count,
                "validation_errors": len(self.validation_errors)
            }
        }
        
        return metadata_line
    
    def _validate_jsonl_format(self, data: Dict[str, Any]) -> bool:
        """
        Validate that the data conforms to JSONL format requirements.
        
        Args:
            data: Dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check that it's a dictionary
            if not isinstance(data, dict):
                self.validation_errors.append("Data must be a dictionary")
                return False
            
            # Check for required fields based on record type
            record_type = data.get("record_type")
            
            if record_type == "processed_sample":
                required_fields = ["id", "source_dataset", "original_text", 
                                 "processed_content", "processing_type"]
                
                for field in required_fields:
                    if field not in data or data[field] is None:
                        self.validation_errors.append(f"Missing required field: {field}")
                        return False
                    
                    # Check that string fields are not empty
                    if isinstance(data[field], str) and not data[field].strip():
                        self.validation_errors.append(f"Field {field} cannot be empty")
                        return False
                
                # Validate processing_type
                if data["processing_type"] not in ["translation", "generation"]:
                    self.validation_errors.append(f"Invalid processing_type: {data['processing_type']}")
                    return False
            
            elif record_type == "dataset_metadata":
                required_fields = ["total_samples", "creation_timestamp", "export_timestamp"]
                
                for field in required_fields:
                    if field not in data or data[field] is None:
                        self.validation_errors.append(f"Missing required metadata field: {field}")
                        return False
            
            else:
                self.validation_errors.append(f"Unknown record_type: {record_type}")
                return False
            
            # Test JSON serialization
            try:
                json.dumps(data, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                self.validation_errors.append(f"Data is not JSON serializable: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Validation error: {e}")
            return False
    
    def validate_jsonl_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate an existing JSONL file.
        
        Args:
            file_path: Path to the JSONL file to validate
            
        Returns:
            Dictionary with validation results
        """
        if not Path(file_path).exists():
            return {
                "valid": False,
                "error": f"File does not exist: {file_path}",
                "line_count": 0,
                "sample_count": 0,
                "metadata_count": 0
            }
        
        line_count = 0
        sample_count = 0
        metadata_count = 0
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()
                    
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Reset validation errors for this line
                        self.validation_errors = []
                        
                        if self._validate_jsonl_format(data):
                            if data.get("record_type") == "processed_sample":
                                sample_count += 1
                            elif data.get("record_type") == "dataset_metadata":
                                metadata_count += 1
                        else:
                            errors.extend([f"Line {line_num}: {error}" for error in self.validation_errors])
                    
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
        
        except IOError as e:
            return {
                "valid": False,
                "error": f"Cannot read file: {e}",
                "line_count": 0,
                "sample_count": 0,
                "metadata_count": 0
            }
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "line_count": line_count,
            "sample_count": sample_count,
            "metadata_count": metadata_count,
            "file_path": file_path
        }
    
    def get_export_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the last export operation.
        
        Returns:
            Dictionary with export statistics
        """
        return {
            "exported_count": self.exported_count,
            "validation_errors": len(self.validation_errors),
            "errors": self.validation_errors.copy()
        }