"""
End-to-end integration tests for the web translation interface.

These tests verify the complete workflow from CLI launch to web interface
interaction with real datasets and API calls.
"""
import os
import json
import time
import pytest
import requests
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

import yaml
from flask.testing import FlaskClient

from medical_dataset_processor.web.app import create_app
from medical_dataset_processor.web.models import ProcessingMode, SessionManager
from medical_dataset_processor.web.translation_service import WebTranslationService
from medical_dataset_processor.processors.translation_processor import TranslationProcessor, TranslationConfig
from medical_dataset_processor.models.core import Sample


class TestWebIntegrationE2E:
    """End-to-end tests for web translation interface."""
    
    @pytest.fixture
    def test_datasets_config(self, tmp_path):
        """Create a test datasets configuration file."""
        config = {
            'test_medical_dataset': {
                'source_type': 'jsonl',
                'source_path': str(tmp_path / 'test_dataset.jsonl'),
                'text_fields': ['text'],
                'metadata_fields': ['id', 'category']
            }
        }
        
        config_path = tmp_path / 'test_datasets.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create test dataset file
        test_data = [
            {
                'id': 'sample_1',
                'text': 'The patient presents with acute chest pain and shortness of breath.',
                'category': 'cardiology'
            },
            {
                'id': 'sample_2', 
                'text': 'Blood pressure is elevated at 180/100 mmHg.',
                'category': 'cardiology'
            },
            {
                'id': 'sample_3',
                'text': 'Recommend immediate cardiac evaluation and monitoring.',
                'category': 'cardiology'
            },
            {
                'id': 'sample_4',
                'text': 'Patient shows signs of respiratory distress.',
                'category': 'pulmonology'
            },
            {
                'id': 'sample_5',
                'text': 'Administer oxygen therapy and monitor vital signs.',
                'category': 'emergency'
            }
        ]
        
        dataset_path = tmp_path / 'test_dataset.jsonl'
        with open(dataset_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        return config_path
    
    @pytest.fixture
    def mock_deepl_api(self):
        """Mock DeepL API responses for testing."""
        with patch('deepl.Translator') as mock_translator:
            mock_instance = MagicMock()
            mock_translator.return_value = mock_instance
            
            # Mock translation responses
            def mock_translate(text, target_lang='FR'):
                translations = {
                    'The patient presents with acute chest pain and shortness of breath.': 
                        'Le patient présente une douleur thoracique aiguë et un essoufflement.',
                    'Blood pressure is elevated at 180/100 mmHg.':
                        'La pression artérielle est élevée à 180/100 mmHg.',
                    'Recommend immediate cardiac evaluation and monitoring.':
                        'Recommander une évaluation cardiaque immédiate et une surveillance.',
                    'Patient shows signs of respiratory distress.':
                        'Le patient montre des signes de détresse respiratoire.',
                    'Administer oxygen therapy and monitor vital signs.':
                        'Administrer une oxygénothérapie et surveiller les signes vitaux.'
                }
                
                result = MagicMock()
                result.text = translations.get(text, f"[TRADUCTION DE: {text}]")
                return result
            
            mock_instance.translate_text.side_effect = mock_translate
            
            # Mock usage info
            mock_usage = MagicMock()
            mock_usage.character = MagicMock()
            mock_usage.character.count = 1000
            mock_usage.character.limit = 500000
            mock_instance.get_usage.return_value = mock_usage
            
            yield mock_instance
    
    @pytest.fixture
    def flask_app(self, mock_deepl_api, tmp_path):
        """Create Flask app for testing."""
        # Set up environment
        os.environ['DEEPL_API_KEY'] = 'test-api-key'
        os.environ['TARGET_LANGUAGE'] = 'FR'
        os.environ['SECRET_KEY'] = 'test-secret-key'
        
        # Create session manager with temp directory
        session_manager = SessionManager(storage_dir=str(tmp_path / 'sessions'))
        
        # Create translation service
        translation_config = TranslationConfig(
            api_key='test-api-key',
            target_language='FR'
        )
        translation_processor = TranslationProcessor(translation_config)
        translation_service = WebTranslationService(translation_processor, session_manager)
        
        # Create app
        app = create_app(session_manager, translation_service)
        app.config['TESTING'] = True
        
        return app
    
    @pytest.fixture
    def client(self, flask_app):
        """Create test client."""
        return flask_app.test_client()
    
    def test_cli_web_command_integration(self, test_datasets_config, tmp_path):
        """Test CLI web command integration."""
        # Test CLI command parsing and validation
        from medical_dataset_processor.cli import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        # Test help command
        result = runner.invoke(cli, ['web', '--help'])
        assert result.exit_code == 0
        assert 'Launch web interface' in result.output
        
        # Test mode validation
        result = runner.invoke(cli, ['web', '--mode', 'invalid_mode'])
        assert result.exit_code != 0
        
        # Test missing API key
        result = runner.invoke(cli, [
            'web', 
            '--mode', 'semi_automatic',
            '--datasets-config', str(test_datasets_config)
        ])
        assert result.exit_code == 1
        assert 'DeepL API key is required' in result.output
    
    def test_web_app_startup_and_health(self, client):
        """Test web application startup and health check."""
        # Test health endpoint
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_mode_selection_flow(self, client):
        """Test complete mode selection flow."""
        # Test main page
        response = client.get('/')
        assert response.status_code == 200
        assert b'mode-selection' in response.data
        
        # Test mode selection
        response = client.post('/select-mode', data={'mode': 'semi_automatic'})
        assert response.status_code == 302  # Redirect
        
        # Test automatic mode redirect
        response = client.post('/select-mode', data={'mode': 'automatic'})
        assert response.status_code == 302
        
        # Test invalid mode
        response = client.post('/select-mode', data={'mode': 'invalid'})
        assert response.status_code == 400
    
    def test_translation_session_creation_and_management(self, client, mock_deepl_api):
        """Test translation session creation and management."""
        # Create session via API
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        assert response.status_code == 201
        
        data = response.get_json()
        assert 'session_id' in data
        assert 'progress' in data
        session_id = data['session_id']
        
        # Get session info
        response = client.get(f'/api/session/{session_id}')
        assert response.status_code == 200
        
        session_data = response.get_json()
        assert session_data['session']['mode'] == 'semi_automatic'
        assert session_data['progress']['total_items'] > 0
    
    def test_translation_interface_workflow(self, client, mock_deepl_api):
        """Test complete translation interface workflow."""
        # Create session
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        session_id = response.get_json()['session_id']
        
        # Set session in client
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # Get current item
        response = client.get('/api/current')
        assert response.status_code == 200
        
        current_data = response.get_json()
        assert 'item' in current_data
        assert 'progress' in current_data
        
        current_item = current_data['item']
        assert 'source_text' in current_item
        assert 'auto_translation' in current_item  # Semi-automatic mode
        
        # Save translation
        response = client.post('/api/save', 
                             json={'translation': 'Ma traduction personnalisée'})
        assert response.status_code == 200
        
        save_data = response.get_json()
        assert save_data['success'] is True
        
        # Navigate to next
        response = client.get('/api/navigate/next')
        assert response.status_code == 200
        
        nav_data = response.get_json()
        assert nav_data['success'] is True
        assert 'item' in nav_data
        
        # Navigate to previous
        response = client.get('/api/navigate/previous')
        assert response.status_code == 200
    
    def test_manual_mode_workflow(self, client, mock_deepl_api):
        """Test manual translation mode workflow."""
        # Create manual session
        response = client.post('/api/session/create', 
                             json={'mode': 'manual'})
        session_id = response.get_json()['session_id']
        
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # Get current item
        response = client.get('/api/current')
        current_item = response.get_json()['item']
        
        # In manual mode, should not have auto_translation
        assert current_item.get('auto_translation') is None
        assert current_item.get('target_text') == '' or current_item.get('target_text') is None
        
        # Save manual translation
        response = client.post('/api/save', 
                             json={'translation': 'Traduction manuelle complète'})
        assert response.status_code == 200
    
    def test_auto_save_functionality(self, client, mock_deepl_api):
        """Test auto-save functionality."""
        # Create session
        response = client.post('/api/session/create', 
                             json={'mode': 'manual'})
        session_id = response.get_json()['session_id']
        
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # Test auto-save
        response = client.post('/api/auto-save', 
                             json={'translation': 'Sauvegarde automatique'})
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert data['auto_saved'] is True
    
    def test_translation_validation(self, client):
        """Test translation validation."""
        # Test valid translation
        response = client.post('/api/validate', 
                             json={'translation': 'Une traduction valide'})
        assert response.status_code == 200
        
        # Test empty translation
        response = client.post('/api/validate', 
                             json={'translation': ''})
        assert response.status_code == 200
        
        validation_data = response.get_json()
        # Should contain validation results
        assert 'valid' in validation_data or 'error' in validation_data
    
    def test_session_export_functionality(self, client, mock_deepl_api):
        """Test session export functionality."""
        # Create and populate session
        response = client.post('/api/session/create', 
                             json={'mode': 'manual'})
        session_id = response.get_json()['session_id']
        
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # Add some translations
        client.post('/api/save', json={'translation': 'Première traduction'})
        client.get('/api/navigate/next')
        client.post('/api/save', json={'translation': 'Deuxième traduction'})
        
        # Export session
        response = client.get(f'/api/session/{session_id}/export')
        assert response.status_code == 200
        
        export_data = response.get_json()
        assert 'session_id' in export_data
        assert 'exported_count' in export_data
        assert 'translations' in export_data
        assert len(export_data['translations']) > 0
    
    def test_error_handling_and_recovery(self, client):
        """Test error handling and recovery mechanisms."""
        # Test invalid session access
        response = client.get('/api/current')
        assert response.status_code == 404
        
        error_data = response.get_json()
        assert 'error' in error_data
        assert 'timestamp' in error_data
        
        # Test invalid navigation
        with client.session_transaction() as sess:
            sess['translation_session_id'] = 'invalid-session-id'
        
        response = client.get('/api/navigate/next')
        assert response.status_code == 404
        
        # Test malformed requests
        response = client.post('/api/save', json={})
        assert response.status_code == 400
        
        response = client.post('/api/save', json={'invalid': 'data'})
        assert response.status_code == 400
    
    def test_usage_info_endpoint(self, client, mock_deepl_api):
        """Test translation usage information endpoint."""
        response = client.get('/api/usage')
        assert response.status_code == 200
        
        usage_data = response.get_json()
        assert 'character_count' in usage_data
        assert 'character_limit' in usage_data
    
    def test_session_cleanup(self, client, mock_deepl_api):
        """Test session cleanup functionality."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            response = client.post('/api/session/create', 
                                 json={'mode': 'manual'})
            session_ids.append(response.get_json()['session_id'])
        
        # Test cleanup
        response = client.post('/api/cleanup', 
                             json={'max_age_hours': 0})  # Clean all
        assert response.status_code == 200
        
        cleanup_data = response.get_json()
        assert 'cleaned_sessions' in cleanup_data
    
    def test_real_dataset_processing(self, client, mock_deepl_api, test_datasets_config):
        """Test processing with real dataset configuration."""
        # This test would use the actual dataset loader
        # For now, we test the integration points
        
        # Create session (uses demo data currently)
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        assert response.status_code == 201
        
        session_data = response.get_json()
        assert session_data['progress']['total_items'] > 0
        
        # Verify session contains realistic medical data
        session_id = session_data['session_id']
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        response = client.get('/api/current')
        current_item = response.get_json()['item']
        
        # Should contain medical-related text
        source_text = current_item['source_text'].lower()
        medical_terms = ['patient', 'medical', 'treatment', 'diagnosis', 'therapy']
        assert any(term in source_text for term in medical_terms)
    
    def test_concurrent_sessions(self, client, mock_deepl_api):
        """Test handling of concurrent translation sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            response = client.post('/api/session/create', 
                                 json={'mode': 'manual'})
            sessions.append(response.get_json()['session_id'])
        
        # Verify all sessions are accessible
        for session_id in sessions:
            response = client.get(f'/api/session/{session_id}')
            assert response.status_code == 200
            
            session_data = response.get_json()
            assert session_data['session']['session_id'] == session_id
    
    def test_translation_interface_templates(self, client, mock_deepl_api):
        """Test translation interface template rendering."""
        # Test semi-automatic mode template
        response = client.get('/translate/semi_automatic')
        assert response.status_code == 200
        assert b'translation-interface' in response.data
        assert b'semi_automatic' in response.data
        
        # Test manual mode template
        response = client.get('/translate/manual')
        assert response.status_code == 200
        assert b'translation-interface' in response.data
        assert b'manual' in response.data
        
        # Test invalid mode
        response = client.get('/translate/invalid_mode')
        assert response.status_code == 302  # Redirect to index
    
    def test_api_error_responses(self, client):
        """Test API error response format consistency."""
        # Test 404 error
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        
        error_data = response.get_json()
        assert 'error' in error_data
        assert 'message' in error_data
        assert 'timestamp' in error_data
        
        # Test 400 error
        response = client.post('/api/save', json={'invalid': 'data'})
        assert response.status_code == 400
        
        error_data = response.get_json()
        assert 'error' in error_data
    
    @pytest.mark.slow
    def test_full_translation_workflow_e2e(self, client, mock_deepl_api):
        """Complete end-to-end translation workflow test."""
        # 1. Create session
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        session_id = response.get_json()['session_id']
        
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # 2. Get initial progress
        response = client.get('/api/current')
        initial_progress = response.get_json()['progress']
        total_items = initial_progress['total_items']
        
        # 3. Process all items
        for i in range(total_items):
            # Get current item
            response = client.get('/api/current')
            assert response.status_code == 200
            
            current_item = response.get_json()['item']
            
            # Modify the auto-translation slightly
            auto_translation = current_item.get('auto_translation', '')
            modified_translation = f"[RÉVISÉ] {auto_translation}"
            
            # Save translation
            response = client.post('/api/save', 
                                 json={'translation': modified_translation})
            assert response.status_code == 200
            
            # Navigate to next (except for last item)
            if i < total_items - 1:
                response = client.get('/api/navigate/next')
                assert response.status_code == 200
        
        # 4. Export final results
        response = client.get(f'/api/session/{session_id}/export')
        assert response.status_code == 200
        
        export_data = response.get_json()
        assert export_data['exported_count'] == total_items
        
        # Verify all translations were modified
        for translation in export_data['translations']:
            assert translation['translated_text'].startswith('[RÉVISÉ]')
    
    def test_performance_and_responsiveness(self, client, mock_deepl_api):
        """Test performance and responsiveness of the interface."""
        import time
        
        # Create session
        start_time = time.time()
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        session_creation_time = time.time() - start_time
        
        # Session creation should be fast
        assert session_creation_time < 2.0  # Less than 2 seconds
        assert response.status_code == 201
        
        session_id = response.get_json()['session_id']
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        # Test API response times
        endpoints_to_test = [
            '/api/current',
            '/api/health',
            f'/api/session/{session_id}'
        ]
        
        for endpoint in endpoints_to_test:
            start_time = time.time()
            response = client.get(endpoint)
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 1.0  # Less than 1 second
        
        # Test save operation performance
        start_time = time.time()
        response = client.post('/api/save', 
                             json={'translation': 'Test translation'})
        save_time = time.time() - start_time
        
        assert response.status_code == 200
        assert save_time < 1.0  # Less than 1 second


class TestWebIntegrationWithRealAPIs:
    """Integration tests with real APIs (requires valid API keys)."""
    
    @pytest.mark.skipif(
        not os.environ.get('DEEPL_API_KEY') or 
        os.environ.get('DEEPL_API_KEY').startswith('test'),
        reason="Requires valid DEEPL_API_KEY environment variable"
    )
    def test_real_deepl_integration(self):
        """Test integration with real DeepL API."""
        from medical_dataset_processor.processors.translation_processor import TranslationProcessor, TranslationConfig
        
        config = TranslationConfig(
            api_key=os.environ['DEEPL_API_KEY'],
            target_language='FR'
        )
        
        processor = TranslationProcessor(config)
        
        # Test translation
        test_text = "The patient presents with acute chest pain."
        result = processor.translate_text(test_text)
        
        assert result.translated_text
        assert result.translated_text != test_text
        assert len(result.translated_text) > 0
        
        # Test usage info
        usage_info = processor.get_usage_info()
        assert usage_info is not None
        assert 'character_count' in usage_info
        assert 'character_limit' in usage_info


if __name__ == '__main__':
    # Run specific test categories
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-k', 'not slow'  # Skip slow tests by default
    ])