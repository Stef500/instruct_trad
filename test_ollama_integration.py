#!/usr/bin/env python3
"""
Test script for Ollama integration with medical dataset processor.
"""
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medical_dataset_processor.utils.ollama_setup import (
    verify_ollama_setup,
    setup_croissantlm_model,
    print_setup_instructions
)
from medical_dataset_processor.processors.ollama_processor import (
    OllamaProcessor,
    OllamaConfig
)
from medical_dataset_processor.models.core import Sample


def test_ollama_setup():
    """Test Ollama setup verification."""
    print("=== Testing Ollama Setup ===")
    
    results = verify_ollama_setup()
    
    print(f"Ollama installed: {results['ollama_installed']}")
    print(f"Server running: {results['server_running']}")
    print(f"CroissantLM available: {results['croissantlm_available']}")
    
    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  • {error}")
    
    if results["warnings"]:
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"  • {warning}")
    
    return results


def test_ollama_processor():
    """Test Ollama processor with a simple sample."""
    print("\n=== Testing Ollama Processor ===")
    
    # Create a test sample
    test_sample = Sample(
        id="test_001",
        source_dataset="test_dataset",
        original_text="What are the symptoms of diabetes?",
        content={
            "question": "What are the symptoms of diabetes?",
            "context": "Medical question about diabetes symptoms"
        }
    )
    
    try:
        # Initialize Ollama processor
        config = OllamaConfig(
            model_name="croissantlm",
            base_url="http://localhost:11434",
            max_retries=2,
            batch_size=1
        )
        
        processor = OllamaProcessor(config)
        
        # Test translation
        print("Testing translation...")
        translated_samples = processor.translate_samples([test_sample])
        
        if translated_samples:
            translated = translated_samples[0]
            print(f"Original: {test_sample.original_text}")
            print(f"Translated: {translated.translated_text}")
            print(f"Processing type: translation")
            print(f"API version: {translated.translation_metadata['api_version']}")
        else:
            print("Translation failed - no samples returned")
        
        # Test generation
        print("\nTesting generation...")
        generated_samples = processor.generate_from_prompts([test_sample])
        
        if generated_samples:
            generated = generated_samples[0]
            print(f"Original: {test_sample.original_text}")
            print(f"Generated: {generated.generated_text}")
            print(f"Processing type: generation")
            print(f"API version: {generated.generation_metadata['api_version']}")
        else:
            print("Generation failed - no samples returned")
        
        return True
        
    except Exception as e:
        print(f"Error testing Ollama processor: {e}")
        return False


def main():
    """Main test function."""
    print("Medical Dataset Processor - Ollama Integration Test")
    print("=" * 50)
    
    # Test setup
    setup_results = test_ollama_setup()
    
    if not setup_results["ollama_installed"] or not setup_results["server_running"]:
        print("\n❌ Ollama setup incomplete. Please install and start Ollama first.")
        print_setup_instructions()
        return False
    
    if not setup_results["croissantlm_available"]:
        print("\n⚠️  CroissantLM model not available. Attempting to create it from Modelfile...")
        success, message = setup_croissantlm_model()
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            print_setup_instructions()
            return False
    
    # Test processor
    processor_success = test_ollama_processor()
    
    if processor_success:
        print("\n✅ All tests passed! Ollama integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
