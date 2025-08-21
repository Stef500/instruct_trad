"""
Utility functions for Ollama setup and verification.
"""
import subprocess
import sys
import requests
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ollama_installation() -> Tuple[bool, str]:
    """
    Check if Ollama is installed and accessible.
    
    Returns:
        Tuple of (is_installed, message)
    """
    try:
        # Try to run ollama --version
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Ollama is installed: {version}"
        else:
            return False, f"Ollama command failed: {result.stderr}"
            
    except FileNotFoundError:
        return False, "Ollama is not installed or not in PATH"
    except subprocess.TimeoutExpired:
        return False, "Ollama command timed out"
    except Exception as e:
        return False, f"Error checking Ollama installation: {str(e)}"


def check_ollama_server(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """
    Check if Ollama server is running.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Tuple of (is_running, message)
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            return True, "Ollama server is running"
        else:
            return False, f"Ollama server returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama server. Make sure it's running."
    except requests.exceptions.Timeout:
        return False, "Ollama server connection timed out"
    except Exception as e:
        return False, f"Error connecting to Ollama server: {str(e)}"


def get_available_models(base_url: str = "http://localhost:11434") -> List[Dict]:
    """
    Get list of available models from Ollama.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        List of model dictionaries
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        else:
            logger.error(f"Failed to get models: status code {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        return []


def check_model_availability(model_name: str, base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """
    Check if a specific model is available.
    
    Args:
        model_name: Name of the model to check
        base_url: Ollama server URL
        
    Returns:
        Tuple of (is_available, message)
    """
    models = get_available_models(base_url)
    model_names = [model.get("name", "") for model in models]
    
    # Check for exact match or with :latest suffix
    if model_name in model_names or f"{model_name}:latest" in model_names:
        return True, f"Model '{model_name}' is available"
    else:
        available = ", ".join(model_names) if model_names else "none"
        return False, f"Model '{model_name}' not found. Available models: {available}"


def create_model_from_modelfile(model_name: str, base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """
    Create a model from the Modelfile.
    
    Args:
        model_name: Name of the model to create
        base_url: Ollama server URL
        
    Returns:
        Tuple of (success, message)
    """
    try:
        logger.info(f"Creating model '{model_name}' from Modelfile...")
        
        # Get the current directory to find the Modelfile
        current_dir = Path(__file__).parent.parent.parent.parent
        modelfile_path = current_dir / "Modelfile"
        
        if not modelfile_path.exists():
            return False, f"Modelfile not found at {modelfile_path}"
        
        # Use subprocess to run ollama create (Modelfile is auto-detected)
        result = subprocess.run(
            ["ollama", "create", model_name],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout for model creation
            cwd=current_dir  # Run from the directory containing the Modelfile
        )
        
        if result.returncode == 0:
            return True, f"Successfully created model '{model_name}' from Modelfile"
        else:
            return False, f"Failed to create model '{model_name}': {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout while creating model '{model_name}'"
    except Exception as e:
        return False, f"Error creating model '{model_name}': {str(e)}"


def setup_croissantlm_model(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """
    Setup CroissantLM model for medical dataset processing using the Modelfile.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Tuple of (success, message)
    """
    model_name = "croissantlm"
    
    # Check if model is already available
    is_available, message = check_model_availability(model_name, base_url)
    if is_available:
        return True, f"Model '{model_name}' is already available"
    
    # Create model from Modelfile
    success, message = create_model_from_modelfile(model_name, base_url)
    return success, message


def verify_ollama_setup(base_url: str = "http://localhost:11434") -> Dict[str, any]:
    """
    Comprehensive verification of Ollama setup.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "ollama_installed": False,
        "server_running": False,
        "croissantlm_available": False,
        "errors": [],
        "warnings": []
    }
    
    # Check Ollama installation
    installed, message = check_ollama_installation()
    results["ollama_installed"] = installed
    if not installed:
        results["errors"].append(f"Ollama installation: {message}")
        return results
    
    # Check server
    running, message = check_ollama_server(base_url)
    results["server_running"] = running
    if not running:
        results["errors"].append(f"Ollama server: {message}")
        return results
    
    # Check CroissantLM model
    available, message = check_model_availability("croissantlm", base_url)
    results["croissantlm_available"] = available
    if not available:
        results["warnings"].append(f"CroissantLM model: {message}")
    
    return results


def print_setup_instructions():
    """Print setup instructions for Ollama."""
    print("""
=== Ollama Setup Instructions ===

1. Install Ollama:
   - Visit https://ollama.ai/
   - Download and install for your platform
   - Or use: curl -fsSL https://ollama.ai/install.sh | sh

2. Start Ollama server:
   - Run: ollama serve
   - Or start it as a service

3. Create CroissantLM model from Modelfile:
   - Run: ollama create croissantlm Modelfile
   - This will create the model with the optimized template (may take several minutes)

4. Verify installation:
   - Run: ollama list
   - Should show 'croissantlm' in the list

5. Test the model:
   - Run: ollama run croissantlm "Hello, how are you?"
   - Should respond in French

The Modelfile includes:
- Optimized template for EN→FR translation
- Temperature set to 0 for deterministic output
- Proper stop tokens for clean responses

For more information, visit: https://ollama.ai/library/croissantlm
""")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama setup verification")
    parser.add_argument("--check", action="store_true", help="Check Ollama setup")
    parser.add_argument("--setup", action="store_true", help="Setup CroissantLM model")
    parser.add_argument("--instructions", action="store_true", help="Show setup instructions")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    
    args = parser.parse_args()
    
    if args.instructions:
        print_setup_instructions()
        return
    
    if args.check:
        print("Checking Ollama setup...")
        results = verify_ollama_setup(args.url)
        
        print(f"✓ Ollama installed: {results['ollama_installed']}")
        print(f"✓ Server running: {results['server_running']}")
        print(f"✓ CroissantLM available: {results['croissantlm_available']}")
        
        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  • {error}")
        
        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"  • {warning}")
        
        if not results["errors"] and results["croissantlm_available"]:
            print("\n✓ Ollama setup is ready!")
        else:
            print("\n✗ Ollama setup needs attention.")
            print_setup_instructions()
    
    if args.setup:
        print("Setting up CroissantLM model from Modelfile...")
        success, message = setup_croissantlm_model(args.url)
        
        if success:
            print(f"✓ {message}")
        else:
            print(f"✗ {message}")
            print_setup_instructions()


if __name__ == "__main__":
    main()
