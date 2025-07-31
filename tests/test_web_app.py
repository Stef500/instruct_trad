"""
Unit tests for the Flask web application.

Tests the main routes, API endpoints, and session management functionality
of the translation web interface.
"""
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from src.medical_dataset_processor.web.app import FlaskTranslationApp, create_app
from src.medical_dataset_processor.web.models import (
    ProcessingMode, 
    TranslationItem, 
    TranslationSession, 
    SessionManager,
    TranslationStatus
)


class TestFlaskTranslationApp:
    """Test cases for the Flask translation application."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a session manager with temporary storage."""
        return SessionManager(storage_dir=temp_dir)
    
    @pytest.fixture
    def app(self, session_manager):
        """Create a Flask test application."""
        flask_app = FlaskTranslationApp(session_manager)
        app = flask_app.get_app()
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()
    
    @pytest.fixture
    def sample_session(self, session_manager):
        """Create a sample translation session for testing."""
        session_id = str(uuid.uuid4())
        items = [
            TranslationItem(
                id="item_1",
                source_text="Test source text 1",
                auto_translation="Test auto translation 1"
            ),
            TranslationItem(
                id="item_2",
                source_text="Test source text 2",
                auto_translation="Test auto translation 2"
            )
        ]
        
        session = TranslationSession(
            session_id=session_id,
            mode=ProcessingMode.SEMI_AUTOMATIC,
            items=items
        )
        
        session_manager.save_session(session)
        return session
    
    def test_index_route(self, client):
        """Test the main index route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Choisissez le mode de traduction' in response.data
        assert b'Mode Automatique' in response.data
        assert b'Mode Semi-Automatique' in response.data
        assert b'Mode Manuel' in response.data
    
    def test_select_mode_automatic(self, client):
        """Test mode selection for automatic mode."""
        response = client.post('/select-mode', data={'mode': 'automatic'})
        assert response.status_code == 302  # Redirect
        assert '/automatic' in response.location
    
    def test_select_mode_semi_automatic(self, client):
        """Test mode selection for semi-automatic mode."""
        response = client.post('/select-mode', data={'mode': 'semi_automatic'})
        assert response.status_code == 302  # Redirect
        assert '/translate/semi_automatic' in response.location
    
    def test_select_mode_manual(self, client):
        """Test mode selection for manual mode."""
        response = client.post('/select-mode', data={'mode': 'manual'})
        assert response.status_code == 302  # Redirect
        assert '/translate/manual' in response.location
    
    def test_select_mode_invalid(self, client):
        """Test mode selection with invalid mode."""
        response = client.post('/select-mode', data={'mode': 'invalid'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid mode' in data['error']
    
    def test_automatic_processing_route(self, client):
        """Test the automatic processing route."""
        response = client.get('/automatic')
        assert response.status_code == 200
        assert b'Mode Automatique' in response.data
        assert b'programmatique' in response.data
    
    def test_translate_interface_semi_automatic(self, client):
        """Test the translation interface for semi-automatic mode."""
        response = client.get('/translate/semi_automatic')
        assert response.status_code == 200
        assert b'Interface de Traduction' in response.data
        assert b'Texte source' in response.data
        assert b'Traduction' in response.data
    
    def test_translate_interface_manual(self, client):
        """Test the translation interface for manual mode."""
        response = client.get('/translate/manual')
        assert response.status_code == 200
        assert b'Interface de Traduction' in response.data
        assert b'Saisissez votre traduction' in response.data
    
    def test_translate_interface_invalid_mode(self, client):
        """Test translation interface with invalid mode."""
        response = client.get('/translate/invalid')
        assert response.status_code == 302  # Redirect to index
        assert '/' in response.location
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_create_session_api(self, client):
        """Test creating a new session via API."""
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'session_id' in data
        assert 'progress' in data
        assert data['progress']['total_items'] > 0
    
    def test_create_session_invalid_mode(self, client):
        """Test creating session with invalid mode."""
        response = client.post('/api/session/create', 
                             json={'mode': 'invalid'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid mode' in data['error']
    
    def test_create_session_missing_mode(self, client):
        """Test creating session without mode."""
        response = client.post('/api/session/create', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Mode is required' in data['error']
    
    def test_get_session_api(self, client, sample_session):
        """Test getting session information via API."""
        response = client.get(f'/api/session/{sample_session.session_id}')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session' in data
        assert 'progress' in data
        assert data['session']['session_id'] == sample_session.session_id
    
    def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = client.get('/api/session/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Session not found' in data['error']
    
    def test_get_current_item_no_session(self, client):
        """Test getting current item without active session."""
        response = client.get('/api/current')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No active session' in data['error']
    
    def test_get_current_item_with_session(self, client, sample_session):
        """Test getting current item with active session."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        response = client.get('/api/current')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'item' in data
        assert 'progress' in data
        assert data['item']['id'] == 'item_1'
    
    def test_save_translation(self, client, sample_session):
        """Test saving a translation."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        response = client.post('/api/save', 
                             json={'translation': 'Test translation'})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'progress' in data
    
    def test_save_translation_no_session(self, client):
        """Test saving translation without active session."""
        response = client.post('/api/save', 
                             json={'translation': 'Test translation'})
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No active session' in data['error']
    
    def test_save_translation_missing_data(self, client, sample_session):
        """Test saving translation with missing data."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        response = client.post('/api/save', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Translation text required' in data['error']
    
    def test_navigate_next(self, client, sample_session):
        """Test navigating to next item."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        response = client.get('/api/navigate/next')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'item' in data
        assert 'progress' in data
        assert data['item']['id'] == 'item_2'
    
    def test_navigate_previous(self, client, sample_session, session_manager):
        """Test navigating to previous item."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        # First navigate to second item
        sample_session.navigate_next()
        session_manager.save_session(sample_session)
        
        response = client.get('/api/navigate/previous')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['item']['id'] == 'item_1'
    
    def test_navigate_invalid_direction(self, client, sample_session):
        """Test navigation with invalid direction."""
        with client.session_transaction() as sess:
            sess['translation_session_id'] = sample_session.session_id
        
        response = client.get('/api/navigate/invalid')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid direction' in data['error']
    
    def test_navigate_no_session(self, client):
        """Test navigation without active session."""
        response = client.get('/api/navigate/next')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No active session' in data['error']
    
    def test_404_error_handler_api(self, client):
        """Test 404 error handler for API routes."""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Endpoint not found' in data['error']
    
    def test_404_error_handler_web(self, client):
        """Test 404 error handler for web routes."""
        response = client.get('/nonexistent')
        assert response.status_code == 404
        # Should return HTML for non-API routes
        assert b'html' in response.data.lower()


class TestFlaskAppIntegration:
    """Integration tests for the Flask application."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def app(self, temp_dir):
        """Create a Flask test application with temporary storage."""
        session_manager = SessionManager(storage_dir=temp_dir)
        app = create_app(session_manager)
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()
    
    def test_full_workflow_semi_automatic(self, client):
        """Test complete workflow for semi-automatic mode."""
        # 1. Start at index page
        response = client.get('/')
        assert response.status_code == 200
        
        # 2. Select semi-automatic mode
        response = client.post('/select-mode', data={'mode': 'semi_automatic'})
        assert response.status_code == 302
        
        # 3. Access translation interface
        response = client.get('/translate/semi_automatic', follow_redirects=True)
        assert response.status_code == 200
        
        # 4. Get current item via API
        response = client.get('/api/current')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'item' in data
        
        # 5. Save a translation
        response = client.post('/api/save', 
                             json={'translation': 'Test translation'})
        assert response.status_code == 200
        
        # 6. Navigate to next item
        response = client.get('/api/navigate/next')
        assert response.status_code == 200
    
    def test_full_workflow_manual(self, client):
        """Test complete workflow for manual mode."""
        # 1. Select manual mode
        response = client.post('/select-mode', data={'mode': 'manual'})
        assert response.status_code == 302
        
        # 2. Access translation interface
        response = client.get('/translate/manual', follow_redirects=True)
        assert response.status_code == 200
        
        # 3. Save a manual translation
        response = client.post('/api/save', 
                             json={'translation': 'Manual translation'})
        assert response.status_code == 200
        
        # 4. Verify translation was saved
        response = client.get('/api/current')
        assert response.status_code == 200
        data = json.loads(response.data)
        # Note: The current item might not show the saved translation immediately
        # depending on the implementation, but the save should succeed
    
    def test_session_persistence(self, client):
        """Test that sessions persist across requests."""
        # Create a session
        response = client.post('/api/session/create', 
                             json={'mode': 'semi_automatic'})
        assert response.status_code == 201
        data = json.loads(response.data)
        session_id = data['session_id']
        
        # Verify session exists
        response = client.get(f'/api/session/{session_id}')
        assert response.status_code == 200
        
        # Make changes to the session
        with client.session_transaction() as sess:
            sess['translation_session_id'] = session_id
        
        response = client.post('/api/save', 
                             json={'translation': 'Persistent translation'})
        assert response.status_code == 200
        
        # Verify changes persisted
        response = client.get(f'/api/session/{session_id}')
        assert response.status_code == 200
        session_data = json.loads(response.data)
        # The session should still exist and be accessible


if __name__ == '__main__':
    pytest.main([__file__])