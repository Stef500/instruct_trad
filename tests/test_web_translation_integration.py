"""
Integration tests for the web translation service with the existing translation system.
"""
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.medical_dataset_processor.models.core import Sample, TranslatedSample
from src.medical_dataset_processor.processors.translation_processor import (
    TranslationProcessor, 
    TranslationConfig, 
    TranslationError
)
from src.medical_dataset_processor.web.models import (
    ProcessingMode, 
    TranslationItem, 
    TranslationSession, 
    SessionManager,
    TranslationStatus
)
from src.medical_dataset_processor.web.translation_service import WebTranslationService


class TestWebTranslationServiceIntegration:
    """Test integration between WebTranslationService and existing translation system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test sessions."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_translation_processor(self):
        """Create a mock translation processor."""
        processor = Mock(spec=TranslationProcessor)
        
        # Mock successful translation
        def mock_translate_samples(samples):
            translated_samples = []
            for sample in samples:
                translated_sample = TranslatedSample(
                    sample=sample,
                    translated_text=f"Translated: {sample.original_text}",
                    translation_metadata={
                        'source_language': 'EN',
                        'target_language': 'FR',
                        'api_version': 'deepl'
                    }
                )
                translated_samples.append(translated_sample)
            return translated_samples
        
        processor.translate_samples.side_effect = mock_translate_samples
        processor.get_usage_info.return_value = {
            'character_count': 1000,
            'character_limit': 500000,
            'character_usage_percent': 0.2
        }
        
        return processor
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a session manager with temporary storage."""
        return SessionManager(storage_dir=temp_dir)
    
    @pytest.fixture
    def web_translation_service(self, mock_translation_processor, session_manager):
        """Create a web translation service with mocked dependencies."""
        return WebTranslationService(mock_translation_processor, session_manager)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            Sample(
                id="test_1",
                content={"text": "The patient has chest pain."},
                source_dataset="test_dataset",
                original_text="The patient has chest pain."
            ),
            Sample(
                id="test_2",
                content={"text": "Blood pressure is high."},
                source_dataset="test_dataset",
                original_text="Blood pressure is high."
            )
        ]
    
    def test_create_automatic_session(self, web_translation_service, sample_data):
        """Test creating a session in automatic mode."""
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.AUTOMATIC,
            target_language="FR"
        )
        
        assert session.mode == ProcessingMode.AUTOMATIC
        assert len(session.items) == 2
        assert session.total_items == 2
        
        # In automatic mode, no auto-translation should be pre-filled
        for item in session.items:
            assert item.auto_translation is None
            assert item.target_text is None
    
    def test_create_semi_automatic_session(self, web_translation_service, sample_data, mock_translation_processor):
        """Test creating a session in semi-automatic mode."""
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.SEMI_AUTOMATIC,
            target_language="FR"
        )
        
        assert session.mode == ProcessingMode.SEMI_AUTOMATIC
        assert len(session.items) == 2
        
        # In semi-automatic mode, auto-translation should be pre-filled
        for item in session.items:
            assert item.auto_translation is not None
            assert item.auto_translation.startswith("Translated:")
            assert item.target_text == item.auto_translation  # Pre-filled for editing
        
        # Verify translation processor was called
        assert mock_translation_processor.translate_samples.call_count == 2
    
    def test_create_manual_session(self, web_translation_service, sample_data):
        """Test creating a session in manual mode."""
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        assert session.mode == ProcessingMode.MANUAL
        assert len(session.items) == 2
        
        # In manual mode, no auto-translation should be pre-filled
        for item in session.items:
            assert item.auto_translation is None
            assert item.target_text is None
    
    def test_get_automatic_translation(self, web_translation_service, mock_translation_processor):
        """Test getting automatic translation for a single text."""
        text = "Hello world"
        translation = web_translation_service.get_automatic_translation(text, "FR")
        
        assert translation == "Translated: Hello world"
        mock_translation_processor.translate_samples.assert_called_once()
    
    def test_get_automatic_translation_error(self, web_translation_service, mock_translation_processor):
        """Test handling translation errors."""
        mock_translation_processor.translate_samples.side_effect = Exception("API Error")
        
        with pytest.raises(TranslationError, match="Translation failed"):
            web_translation_service.get_automatic_translation("Hello", "FR")
    
    def test_process_automatic_mode(self, web_translation_service, sample_data, mock_translation_processor):
        """Test processing samples in automatic mode."""
        translated_samples = web_translation_service.process_automatic_mode(
            samples=sample_data,
            target_language="FR"
        )
        
        assert len(translated_samples) == 2
        assert all(isinstance(sample, TranslatedSample) for sample in translated_samples)
        assert all(sample.translated_text.startswith("Translated:") for sample in translated_samples)
        
        mock_translation_processor.translate_samples.assert_called_once_with(sample_data)
    
    def test_validate_translation_valid(self, web_translation_service):
        """Test validation of valid translation."""
        result = web_translation_service.validate_translation("This is a valid translation.")
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_translation_empty(self, web_translation_service):
        """Test validation of empty translation."""
        result = web_translation_service.validate_translation("")
        
        assert result['valid'] is False
        assert "Translation cannot be empty" in result['errors']
    
    def test_validate_translation_too_long(self, web_translation_service):
        """Test validation of overly long translation."""
        long_text = "a" * 10001
        result = web_translation_service.validate_translation(long_text)
        
        assert result['valid'] is False
        assert "Translation is too long (max 10,000 characters)" in result['errors']
    
    def test_validate_translation_warnings(self, web_translation_service):
        """Test validation warnings."""
        result = web_translation_service.validate_translation("UPPERCASE TEXT")
        
        assert result['valid'] is True
        assert "all uppercase" in result['warnings'][0]
    
    def test_update_session_translation(self, web_translation_service, sample_data, session_manager):
        """Test updating a translation in a session."""
        # Create session
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        # Update translation
        success = web_translation_service.update_session_translation(
            session_id=session.session_id,
            item_index=0,
            translation="Updated translation"
        )
        
        assert success is True
        
        # Verify update
        updated_session = session_manager.load_session(session.session_id)
        assert updated_session.items[0].target_text == "Updated translation"
        assert updated_session.items[0].status == TranslationStatus.COMPLETED
    
    def test_update_session_translation_invalid_session(self, web_translation_service):
        """Test updating translation with invalid session ID."""
        success = web_translation_service.update_session_translation(
            session_id="invalid_session",
            item_index=0,
            translation="Test"
        )
        
        assert success is False
    
    def test_update_session_translation_invalid_index(self, web_translation_service, sample_data):
        """Test updating translation with invalid item index."""
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        success = web_translation_service.update_session_translation(
            session_id=session.session_id,
            item_index=999,  # Invalid index
            translation="Test"
        )
        
        assert success is False
    
    def test_get_session_progress(self, web_translation_service, sample_data):
        """Test getting session progress information."""
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        progress = web_translation_service.get_session_progress(session.session_id)
        
        assert progress is not None
        assert progress['current_item'] == 1
        assert progress['total_items'] == 2
        assert progress['completed_items'] == 0
        assert progress['percentage'] == 0.0
        assert progress['mode'] == ProcessingMode.MANUAL.value
        assert progress['can_navigate_previous'] is False
        assert progress['can_navigate_next'] is True
        assert progress['is_complete'] is False
    
    def test_export_session_results(self, web_translation_service, sample_data, session_manager):
        """Test exporting completed translations from a session."""
        # Create session and complete some translations
        session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        # Complete first translation
        web_translation_service.update_session_translation(
            session_id=session.session_id,
            item_index=0,
            translation="Première traduction"
        )
        
        # Export results
        translated_samples = web_translation_service.export_session_results(session.session_id)
        
        assert translated_samples is not None
        assert len(translated_samples) == 1  # Only one completed
        
        sample = translated_samples[0]
        assert isinstance(sample, TranslatedSample)
        assert sample.translated_text == "Première traduction"
        assert sample.translation_metadata['web_session_id'] == session.session_id
        assert sample.translation_metadata['processing_mode'] == ProcessingMode.MANUAL.value
    
    def test_export_session_results_invalid_session(self, web_translation_service):
        """Test exporting results from invalid session."""
        result = web_translation_service.export_session_results("invalid_session")
        assert result is None
    
    def test_cleanup_expired_sessions(self, web_translation_service, session_manager):
        """Test cleaning up expired sessions."""
        # Create a session
        session = TranslationSession(
            session_id="test_session",
            mode=ProcessingMode.MANUAL,
            items=[
                TranslationItem(
                    id="item_1",
                    source_text="Test text"
                )
            ]
        )
        session_manager.save_session(session)
        
        # Mock cleanup method
        with patch.object(session_manager, 'cleanup_old_sessions', return_value=1):
            cleaned_count = web_translation_service.cleanup_expired_sessions(24)
            assert cleaned_count == 1
    
    def test_get_translation_usage_info(self, web_translation_service, mock_translation_processor):
        """Test getting translation API usage information."""
        usage_info = web_translation_service.get_translation_usage_info()
        
        assert usage_info is not None
        assert 'character_count' in usage_info
        assert 'character_limit' in usage_info
        assert 'character_usage_percent' in usage_info
        
        mock_translation_processor.get_usage_info.assert_called_once()
    
    def test_semi_automatic_mode_translation_error_handling(self, session_manager, sample_data):
        """Test handling translation errors in semi-automatic mode."""
        # Create a processor that fails
        failing_processor = Mock(spec=TranslationProcessor)
        failing_processor.translate_samples.side_effect = TranslationError("API Error")
        
        service = WebTranslationService(failing_processor, session_manager)
        
        # Should still create session but with error metadata
        session = service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.SEMI_AUTOMATIC,
            target_language="FR"
        )
        
        assert session.mode == ProcessingMode.SEMI_AUTOMATIC
        assert len(session.items) == 2
        
        # Items should have error metadata instead of auto-translation
        for item in session.items:
            assert item.auto_translation is None
            assert 'auto_translation_error' in item.metadata
    
    def test_session_persistence_integration(self, web_translation_service, sample_data, session_manager):
        """Test that sessions are properly persisted and can be reloaded."""
        # Create session
        original_session = web_translation_service.create_translation_session(
            samples=sample_data,
            mode=ProcessingMode.SEMI_AUTOMATIC,
            target_language="FR"
        )
        
        # Update a translation
        web_translation_service.update_session_translation(
            session_id=original_session.session_id,
            item_index=0,
            translation="Persisted translation"
        )
        
        # Reload session
        reloaded_session = session_manager.load_session(original_session.session_id)
        
        assert reloaded_session is not None
        assert reloaded_session.session_id == original_session.session_id
        assert reloaded_session.items[0].target_text == "Persisted translation"
        assert reloaded_session.items[0].status == TranslationStatus.COMPLETED


class TestWebTranslationServiceErrorHandling:
    """Test error handling scenarios in the web translation service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test sessions."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a session manager with temporary storage."""
        return SessionManager(storage_dir=temp_dir)
    
    def test_translation_processor_initialization_error(self, session_manager):
        """Test handling of translation processor initialization errors."""
        failing_processor = Mock(spec=TranslationProcessor)
        failing_processor.translate_samples.side_effect = Exception("Initialization failed")
        
        service = WebTranslationService(failing_processor, session_manager)
        
        with pytest.raises(TranslationError):
            service.get_automatic_translation("Test text", "FR")
    
    def test_session_manager_save_error(self, session_manager):
        """Test handling of session manager save errors."""
        # Mock session manager to fail on save
        session_manager.save_session = Mock(return_value=False)
        
        processor = Mock(spec=TranslationProcessor)
        processor.translate_samples.return_value = []
        
        service = WebTranslationService(processor, session_manager)
        
        sample = Sample(
            id="test",
            content={"text": "Test"},
            source_dataset="test",
            original_text="Test"
        )
        
        # Should still create session object even if save fails
        session = service.create_translation_session(
            samples=[sample],
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        assert session is not None
        assert len(session.items) == 1
    
    def test_invalid_translation_validation(self, session_manager):
        """Test handling of invalid translations during update."""
        processor = Mock(spec=TranslationProcessor)
        service = WebTranslationService(processor, session_manager)
        
        # Create a session
        sample = Sample(
            id="test",
            content={"text": "Test"},
            source_dataset="test",
            original_text="Test"
        )
        
        session = service.create_translation_session(
            samples=[sample],
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        # Try to update with invalid (empty) translation
        success = service.update_session_translation(
            session_id=session.session_id,
            item_index=0,
            translation=""  # Invalid empty translation
        )
        
        assert success is False


class TestWebTranslationServicePerformance:
    """Test performance aspects of the web translation service."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test sessions."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create a session manager with temporary storage."""
        return SessionManager(storage_dir=temp_dir)
    
    def test_large_dataset_handling(self, session_manager):
        """Test handling of large datasets."""
        # Create a large number of samples
        large_sample_set = []
        for i in range(100):
            sample = Sample(
                id=f"sample_{i}",
                content={"text": f"Sample text {i}"},
                source_dataset="large_dataset",
                original_text=f"Sample text {i}"
            )
            large_sample_set.append(sample)
        
        # Mock processor for fast execution
        processor = Mock(spec=TranslationProcessor)
        processor.translate_samples.return_value = []
        
        service = WebTranslationService(processor, session_manager)
        
        # Should handle large dataset without issues
        session = service.create_translation_session(
            samples=large_sample_set,
            mode=ProcessingMode.MANUAL,
            target_language="FR"
        )
        
        assert len(session.items) == 100
        assert session.total_items == 100
    
    def test_concurrent_session_access(self, session_manager):
        """Test concurrent access to sessions."""
        processor = Mock(spec=TranslationProcessor)
        service = WebTranslationService(processor, session_manager)
        
        # Create multiple sessions
        sessions = []
        for i in range(5):
            sample = Sample(
                id=f"test_{i}",
                content={"text": f"Test {i}"},
                source_dataset="test",
                original_text=f"Test {i}"
            )
            
            session = service.create_translation_session(
                samples=[sample],
                mode=ProcessingMode.MANUAL,
                target_language="FR"
            )
            sessions.append(session)
        
        # All sessions should be accessible
        for session in sessions:
            progress = service.get_session_progress(session.session_id)
            assert progress is not None
            assert progress['session_id'] == session.session_id