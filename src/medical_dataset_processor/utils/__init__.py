"""
Utility functions and helpers.
"""

from .logging import (
    ProcessingLogger,
    APIError,
    ProcessingStats,
    ErrorReport,
    ProcessingState,
    ErrorSeverity
)

from .state_recovery import (
    StateRecoveryManager,
    BatchProcessor
)

from .ollama_setup import (
    check_ollama_installation,
    check_ollama_server,
    get_available_models,
    check_model_availability,
    create_model_from_modelfile,
    setup_croissantlm_model,
    verify_ollama_setup,
    print_setup_instructions
)

__all__ = [
    "ProcessingLogger",
    "APIError", 
    "ProcessingStats",
    "ErrorReport",
    "ProcessingState",
    "ErrorSeverity",
    "StateRecoveryManager",
    "BatchProcessor",
    "check_ollama_installation",
    "check_ollama_server",
    "get_available_models",
    "check_model_availability",
    "create_model_from_modelfile",
    "setup_croissantlm_model",
    "verify_ollama_setup",
    "print_setup_instructions"
]