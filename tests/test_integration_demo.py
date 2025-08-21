#!/usr/bin/env python3
"""
Demo script to test the complete web interface integration.

This script demonstrates the full workflow from CLI to web interface.
"""
import os
import sys
import subprocess
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_integration():
    """Test CLI integration (fast, with timeouts)."""
    # Help command should return quickly
    result = subprocess.run(
        [sys.executable, "-m", "medical_dataset_processor.cli", "web", "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    
    # Automatic mode should not launch server; just print a notice and exit
    env = os.environ.copy()
    env['DEEPL_API_KEY'] = 'test-key'
    result = subprocess.run(
        [sys.executable, "-m", "medical_dataset_processor.cli", "web", "--mode", "automatic"],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Automatic mode selected" in result.stdout

def test_web_server_startup():
    """Smoke test for web server components and session persistence."""
    os.environ['DEEPL_API_KEY'] = 'test-key-for-demo'
    os.environ['TARGET_LANGUAGE'] = 'FR'
    
    from medical_dataset_processor.web.models import SessionManager, ProcessingMode
    from medical_dataset_processor.web.models import TranslationItem, TranslationSession
    
    session_manager = SessionManager(storage_dir="test_sessions")
    test_item = TranslationItem(id="test_1", source_text="Test source text")
    test_session = TranslationSession(
        session_id="test_session",
        mode=ProcessingMode.MANUAL,
        items=[test_item]
    )
    
    assert session_manager.save_session(test_session)
    loaded_session = session_manager.load_session("test_session")
    assert loaded_session and loaded_session.session_id == "test_session"
    session_manager.delete_session("test_session")

def test_error_handling():
    """Test error handling paths via CLI with timeouts."""
    env = os.environ.copy()
    # Forcer une clé vide pour empêcher load_dotenv de l'écraser depuis un fichier .env
    env['DEEPL_API_KEY'] = ''
    result = subprocess.run(
        [sys.executable, "-m", "medical_dataset_processor.cli", "web", "--mode", "semi_automatic"],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert result.returncode == 1
    assert "DeepL API key is required" in result.stdout
    
    env['DEEPL_API_KEY'] = 'test-key'
    result = subprocess.run(
        [sys.executable, "-m", "medical_dataset_processor.cli", "web", "--mode", "invalid_mode"],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )
    assert result.returncode != 0

def test_documentation():
    """Ensure web documentation exists and contains key sections."""
    doc_file = Path("docs/WEB_INTERFACE_USAGE.md")
    assert doc_file.exists(), "Documentation file not found"
    content = doc_file.read_text(encoding="utf-8")
    assert len(content) > 1000 and "Interface Web de Traduction" in content
    required_sections = [
        "Démarrage Rapide",
        "Modes de Traduction",
        "Interface Utilisateur",
        "Configuration Avancée",
        "Gestion des Erreurs",
    ]
    for section in required_sections:
        assert section in content, f"Missing documentation section: {section}"

def test_project_structure():
    """Ensure required files exist (fast check)."""
    required_files = [
        "src/medical_dataset_processor/cli.py",
        "src/medical_dataset_processor/web/app.py",
        "src/medical_dataset_processor/web/models.py",
        "src/medical_dataset_processor/web/translation_service.py",
        "src/medical_dataset_processor/web/static/translation.js",
        "src/medical_dataset_processor/web/static/style.css",
        "src/medical_dataset_processor/web/templates/index.html",
        "src/medical_dataset_processor/web/templates/translate.html",
        "docs/WEB_INTERFACE_USAGE.md",
        "tests/test_web_integration_e2e.py",
    ]
    for file_path in required_files:
        assert Path(file_path).exists(), f"Missing file: {file_path}"

# Remove custom main runner to avoid slow demo-style execution under pytest