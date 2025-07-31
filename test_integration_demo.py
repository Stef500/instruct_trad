#!/usr/bin/env python3
"""
Demo script to test the complete web interface integration.

This script demonstrates the full workflow from CLI to web interface.
"""
import os
import sys
import time
import requests
import subprocess
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_integration():
    """Test CLI integration."""
    print("🧪 Testing CLI Integration...")
    
    # Test help command
    result = subprocess.run([
        sys.executable, "-m", "medical_dataset_processor.cli", "web", "--help"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ CLI help command works")
    else:
        print("❌ CLI help command failed")
        return False
    
    # Test automatic mode
    os.environ['DEEPL_API_KEY'] = 'test-key'
    result = subprocess.run([
        sys.executable, "-m", "medical_dataset_processor.cli", 
        "web", "--mode", "automatic"
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and "Automatic mode selected" in result.stdout:
        print("✅ Automatic mode selection works")
    else:
        print("❌ Automatic mode selection failed")
        return False
    
    return True

def test_web_server_startup():
    """Test web server startup and basic functionality."""
    print("\n🌐 Testing Web Server Components...")
    
    # Set environment variables
    os.environ['DEEPL_API_KEY'] = 'test-key-for-demo'
    os.environ['TARGET_LANGUAGE'] = 'FR'
    
    try:
        # Test that we can import the components
        from medical_dataset_processor.web.app import FlaskTranslationApp
        from medical_dataset_processor.web.models import SessionManager, ProcessingMode
        from medical_dataset_processor.web.translation_service import WebTranslationService
        
        print("✅ Web components import successfully")
        
        # Test SessionManager
        session_manager = SessionManager(storage_dir="test_sessions")
        print("✅ SessionManager creation successful")
        
        # Test that we can create translation items
        from medical_dataset_processor.web.models import TranslationItem, TranslationSession
        
        test_item = TranslationItem(
            id="test_1",
            source_text="Test source text"
        )
        
        test_session = TranslationSession(
            session_id="test_session",
            mode=ProcessingMode.MANUAL,
            items=[test_item]
        )
        
        print("✅ Data models work correctly")
        
        # Test session persistence
        if session_manager.save_session(test_session):
            loaded_session = session_manager.load_session("test_session")
            if loaded_session and loaded_session.session_id == "test_session":
                print("✅ Session persistence works")
                
                # Clean up test session
                session_manager.delete_session("test_session")
                return True
            else:
                print("❌ Session loading failed")
        else:
            print("❌ Session saving failed")
        
    except Exception as e:
        print(f"❌ Web server component test failed: {e}")
        # Don't print full traceback for cleaner output
        print(f"   Error details: {str(e)[:100]}...")
    
    return False

def test_error_handling():
    """Test error handling and user notifications."""
    print("\n🚨 Testing Error Handling...")
    
    try:
        # Test missing API key
        if 'DEEPL_API_KEY' in os.environ:
            del os.environ['DEEPL_API_KEY']
        
        result = subprocess.run([
            sys.executable, "-m", "medical_dataset_processor.cli", 
            "web", "--mode", "semi_automatic"
        ], capture_output=True, text=True)
        
        if result.returncode == 1 and "DeepL API key is required" in result.stdout:
            print("✅ Missing API key error handling works")
        else:
            print("❌ Missing API key error handling failed")
            return False
        
        # Test invalid mode
        os.environ['DEEPL_API_KEY'] = 'test-key'
        result = subprocess.run([
            sys.executable, "-m", "medical_dataset_processor.cli", 
            "web", "--mode", "invalid_mode"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("✅ Invalid mode error handling works")
        else:
            print("❌ Invalid mode error handling failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_documentation():
    """Test that documentation exists and is accessible."""
    print("\n📚 Testing Documentation...")
    
    doc_file = Path("docs/WEB_INTERFACE_USAGE.md")
    if doc_file.exists():
        content = doc_file.read_text()
        if len(content) > 1000 and "Interface Web de Traduction" in content:
            print("✅ Documentation exists and has content")
            
            # Check for key sections
            required_sections = [
                "Démarrage Rapide",
                "Modes de Traduction", 
                "Interface Utilisateur",
                "Configuration Avancée",
                "Gestion des Erreurs"
            ]
            
            missing_sections = [section for section in required_sections if section not in content]
            if not missing_sections:
                print("✅ All required documentation sections present")
                return True
            else:
                print(f"❌ Missing documentation sections: {missing_sections}")
        else:
            print("❌ Documentation content insufficient")
    else:
        print("❌ Documentation file not found")
    
    return False

def test_project_structure():
    """Test that all required files are in place."""
    print("\n📁 Testing Project Structure...")
    
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
        "tests/test_web_integration_e2e.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ All required files present")
        return True
    else:
        print(f"❌ Missing files: {missing_files}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Medical Dataset Processor - Web Interface Integration Test")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("CLI Integration", test_cli_integration),
        ("Error Handling", test_error_handling),
        ("Documentation", test_documentation),
        ("Web Server", test_web_server_startup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All integration tests passed! Web interface is ready.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())