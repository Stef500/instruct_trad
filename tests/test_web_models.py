"""
Unit tests for web interface data models.
"""
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.medical_dataset_processor.web.models import (
    ProcessingMode,
    TranslationStatus,
    TranslationItem,
    TranslationSession,
    SessionManager
)


class TestProcessingMode:
    """Test ProcessingMode enum."""
    
    def test_processing_mode_values(self):
        """Test that ProcessingMode has correct values."""
        assert ProcessingMode.AUTOMATIC.value == "automatic"
        assert ProcessingMode.SEMI_AUTOMATIC.value == "semi_automatic"
        assert ProcessingMode.MANUAL.value == "manual"


class TestTranslationStatus:
    """Test TranslationStatus enum."""
    
    def test_translation_status_values(self):
        """Test that TranslationStatus has correct values."""
        assert TranslationStatus.PENDING.value == "pending"
        assert TranslationStatus.IN_PROGRESS.value == "in_progress"
        assert TranslationStatus.COMPLETED.value == "completed"
        assert TranslationStatus.SKIPPED.value == "skipped"


class TestTranslationItem:
    """Test TranslationItem dataclass."""
    
    def test_translation_item_creation(self):
        """Test creating a TranslationItem."""
        item = TranslationItem(
            id="test-1",
            source_text="Hello world"
        )
        
        assert item.id == "test-1"
        assert item.source_text == "Hello world"
        assert item.target_text is None
        assert item.auto_translation is None
        assert item.status == TranslationStatus.PENDING
        assert isinstance(item.metadata, dict)
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)
    
    def test_translation_item_validation(self):
        """Test TranslationItem validation."""
        # Test empty ID
        with pytest.raises(ValueError, match="Translation item ID cannot be empty"):
            TranslationItem(id="", source_text="Hello")
        
        # Test empty source text
        with pytest.raises(ValueError, match="Source text cannot be empty"):
            TranslationItem(id="test-1", source_text="")
        
        # Test invalid status
        with pytest.raises(ValueError, match="Status must be a TranslationStatus enum value"):
            TranslationItem(id="test-1", source_text="Hello", status="invalid")
    
    def test_update_translation(self):
        """Test updating translation."""
        item = TranslationItem(id="test-1", source_text="Hello")
        original_updated_at = item.updated_at
        
        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        item.update_translation("Bonjour")
        
        assert item.target_text == "Bonjour"
        assert item.status == TranslationStatus.COMPLETED
        assert item.updated_at > original_updated_at
    
    def test_set_auto_translation(self):
        """Test setting auto translation."""
        item = TranslationItem(id="test-1", source_text="Hello")
        original_updated_at = item.updated_at
        
        import time
        time.sleep(0.001)
        
        item.set_auto_translation("Bonjour (auto)")
        
        assert item.auto_translation == "Bonjour (auto)"
        assert item.updated_at > original_updated_at
    
    def test_to_dict(self):
        """Test converting TranslationItem to dictionary."""
        item = TranslationItem(
            id="test-1",
            source_text="Hello",
            target_text="Bonjour",
            metadata={"key": "value"}
        )
        
        data = item.to_dict()
        
        assert data['id'] == "test-1"
        assert data['source_text'] == "Hello"
        assert data['target_text'] == "Bonjour"
        assert data['status'] == "pending"
        assert data['metadata'] == {"key": "value"}
        assert isinstance(data['created_at'], str)
        assert isinstance(data['updated_at'], str)
    
    def test_from_dict(self):
        """Test creating TranslationItem from dictionary."""
        data = {
            'id': "test-1",
            'source_text': "Hello",
            'target_text': "Bonjour",
            'auto_translation': None,
            'status': "completed",
            'metadata': {"key": "value"},
            'created_at': "2024-01-01T12:00:00",
            'updated_at': "2024-01-01T12:05:00"
        }
        
        item = TranslationItem.from_dict(data)
        
        assert item.id == "test-1"
        assert item.source_text == "Hello"
        assert item.target_text == "Bonjour"
        assert item.status == TranslationStatus.COMPLETED
        assert item.metadata == {"key": "value"}
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)


class TestTranslationSession:
    """Test TranslationSession dataclass."""
    
    def create_sample_items(self) -> list[TranslationItem]:
        """Create sample translation items for testing."""
        return [
            TranslationItem(id="item-1", source_text="Hello"),
            TranslationItem(id="item-2", source_text="World"),
            TranslationItem(id="item-3", source_text="Test")
        ]
    
    def test_translation_session_creation(self):
        """Test creating a TranslationSession."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.SEMI_AUTOMATIC,
            items=items
        )
        
        assert session.session_id == "session-1"
        assert session.mode == ProcessingMode.SEMI_AUTOMATIC
        assert len(session.items) == 3
        assert session.current_index == 0
        assert session.total_items == 3
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert session.auto_save_enabled is True
    
    def test_translation_session_validation(self):
        """Test TranslationSession validation."""
        items = self.create_sample_items()
        
        # Test empty session ID
        with pytest.raises(ValueError, match="Session ID cannot be empty"):
            TranslationSession(session_id="", mode=ProcessingMode.MANUAL, items=items)
        
        # Test invalid mode
        with pytest.raises(ValueError, match="Mode must be a ProcessingMode enum value"):
            TranslationSession(session_id="test", mode="invalid", items=items)
        
        # Test empty items
        with pytest.raises(ValueError, match="Session must contain at least one translation item"):
            TranslationSession(session_id="test", mode=ProcessingMode.MANUAL, items=[])
    
    def test_get_current_item(self):
        """Test getting current item."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.MANUAL,
            items=items
        )
        
        current_item = session.get_current_item()
        assert current_item is not None
        assert current_item.id == "item-1"
    
    def test_navigation(self):
        """Test navigation methods."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.MANUAL,
            items=items
        )
        
        # Test navigate_next
        assert session.navigate_next() is True
        assert session.current_index == 1
        assert session.get_current_item().id == "item-2"
        
        # Test navigate_previous
        assert session.navigate_previous() is True
        assert session.current_index == 0
        assert session.get_current_item().id == "item-1"
        
        # Test navigate_to
        assert session.navigate_to(2) is True
        assert session.current_index == 2
        assert session.get_current_item().id == "item-3"
        
        # Test invalid navigation
        assert session.navigate_to(5) is False
        assert session.current_index == 2  # Should remain unchanged
        
        assert session.navigate_next() is False  # Already at last item
        assert session.current_index == 2
    
    def test_update_current_translation(self):
        """Test updating current translation."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.MANUAL,
            items=items
        )
        
        result = session.update_current_translation("Bonjour")
        
        assert result is True
        current_item = session.get_current_item()
        assert current_item.target_text == "Bonjour"
        assert current_item.status == TranslationStatus.COMPLETED
    
    def test_get_progress_info(self):
        """Test getting progress information."""
        items = self.create_sample_items()
        items[0].status = TranslationStatus.COMPLETED
        items[1].status = TranslationStatus.COMPLETED
        
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.MANUAL,
            items=items
        )
        
        progress = session.get_progress_info()
        
        assert progress['current_item'] == 1
        assert progress['total_items'] == 3
        assert progress['completed_items'] == 2
        assert progress['percentage'] == pytest.approx(66.67, rel=1e-2)
        assert progress['session_id'] == "session-1"
        assert progress['mode'] == "manual"
    
    def test_is_complete(self):
        """Test checking if session is complete."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.MANUAL,
            items=items
        )
        
        assert session.is_complete() is False
        
        # Mark all items as completed
        for item in items:
            item.status = TranslationStatus.COMPLETED
        
        assert session.is_complete() is True
    
    def test_to_dict(self):
        """Test converting TranslationSession to dictionary."""
        items = self.create_sample_items()
        session = TranslationSession(
            session_id="session-1",
            mode=ProcessingMode.SEMI_AUTOMATIC,
            items=items
        )
        
        data = session.to_dict()
        
        assert data['session_id'] == "session-1"
        assert data['mode'] == "semi_automatic"
        assert len(data['items']) == 3
        assert data['current_index'] == 0
        assert data['total_items'] == 3
        assert isinstance(data['created_at'], str)
        assert isinstance(data['last_activity'], str)
        assert data['auto_save_enabled'] is True
    
    def test_from_dict(self):
        """Test creating TranslationSession from dictionary."""
        data = {
            'session_id': "session-1",
            'mode': "manual",
            'items': [
                {
                    'id': "item-1",
                    'source_text': "Hello",
                    'target_text': None,
                    'auto_translation': None,
                    'status': "pending",
                    'metadata': {},
                    'created_at': "2024-01-01T12:00:00",
                    'updated_at': "2024-01-01T12:00:00"
                }
            ],
            'current_index': 0,
            'total_items': 1,
            'created_at': "2024-01-01T12:00:00",
            'last_activity': "2024-01-01T12:00:00",
            'auto_save_enabled': True
        }
        
        session = TranslationSession.from_dict(data)
        
        assert session.session_id == "session-1"
        assert session.mode == ProcessingMode.MANUAL
        assert len(session.items) == 1
        assert isinstance(session.items[0], TranslationItem)
        assert session.current_index == 0
        assert session.total_items == 1
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)


class TestSessionManager:
    """Test SessionManager class."""
    
    def create_sample_session(self) -> TranslationSession:
        """Create a sample session for testing."""
        items = [
            TranslationItem(id="item-1", source_text="Hello"),
            TranslationItem(id="item-2", source_text="World")
        ]
        return TranslationSession(
            session_id="test-session",
            mode=ProcessingMode.SEMI_AUTOMATIC,
            items=items
        )
    
    def test_session_manager_init(self):
        """Test SessionManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            assert manager.storage_dir == Path(temp_dir)
            assert manager.storage_dir.exists()
    
    def test_save_and_load_session(self):
        """Test saving and loading sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            session = self.create_sample_session()
            
            # Test save
            result = manager.save_session(session)
            assert result is True
            
            # Check file exists
            session_file = Path(temp_dir) / "test-session.json"
            assert session_file.exists()
            
            # Test load
            loaded_session = manager.load_session("test-session")
            assert loaded_session is not None
            assert loaded_session.session_id == session.session_id
            assert loaded_session.mode == session.mode
            assert len(loaded_session.items) == len(session.items)
    
    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            
            loaded_session = manager.load_session("nonexistent")
            assert loaded_session is None
    
    def test_delete_session(self):
        """Test deleting a session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            session = self.create_sample_session()
            
            # Save session first
            manager.save_session(session)
            session_file = Path(temp_dir) / "test-session.json"
            assert session_file.exists()
            
            # Delete session
            result = manager.delete_session("test-session")
            assert result is True
            assert not session_file.exists()
            
            # Try to delete non-existent session
            result = manager.delete_session("nonexistent")
            assert result is False
    
    def test_list_sessions(self):
        """Test listing sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            
            # Initially empty
            sessions = manager.list_sessions()
            assert sessions == []
            
            # Save some sessions
            session1 = self.create_sample_session()
            session2 = TranslationSession(
                session_id="session-2",
                mode=ProcessingMode.MANUAL,
                items=[TranslationItem(id="item-1", source_text="Test")]
            )
            
            manager.save_session(session1)
            manager.save_session(session2)
            
            sessions = manager.list_sessions()
            assert len(sessions) == 2
            assert "test-session" in sessions
            assert "session-2" in sessions
    
    def test_auto_save_session(self):
        """Test auto-saving sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            session = self.create_sample_session()
            
            # Test auto-save enabled
            result = manager.auto_save_session(session)
            assert result is True
            
            session_file = Path(temp_dir) / "test-session.json"
            assert session_file.exists()
            
            # Test auto-save disabled
            session.auto_save_enabled = False
            session_file.unlink()  # Remove file
            
            result = manager.auto_save_session(session)
            assert result is True  # Should return True but not save
            assert not session_file.exists()
    
    def test_cleanup_old_sessions(self):
        """Test cleaning up old sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            
            # Create some sessions
            session1 = self.create_sample_session()
            session2 = TranslationSession(
                session_id="session-2",
                mode=ProcessingMode.MANUAL,
                items=[TranslationItem(id="item-1", source_text="Test")]
            )
            
            manager.save_session(session1)
            manager.save_session(session2)
            
            # Mock file modification times to simulate old files
            session1_file = Path(temp_dir) / "test-session.json"
            session2_file = Path(temp_dir) / "session-2.json"
            
            # Make session1 file appear old (25 hours ago)
            old_time = (datetime.now() - timedelta(hours=25)).timestamp()
            import os
            os.utime(session1_file, (old_time, old_time))
            
            # Clean up sessions older than 24 hours
            cleaned_count = manager.cleanup_old_sessions(max_age_hours=24)
            
            assert cleaned_count == 1
            assert not session1_file.exists()  # Should be deleted
            assert session2_file.exists()  # Should remain
    
    @patch('builtins.print')
    def test_error_handling(self, mock_print):
        """Test error handling in SessionManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(storage_dir=temp_dir)
            
            # Test save error by creating invalid session data
            session = self.create_sample_session()
            
            # Mock json.dump to raise an exception
            with patch('json.dump', side_effect=Exception("Test error")):
                result = manager.save_session(session)
                assert result is False
                mock_print.assert_called_with("Error saving session test-session: Test error")
            
            # Test load error with corrupted JSON
            session_file = Path(temp_dir) / "corrupted.json"
            session_file.write_text("invalid json content")
            
            result = manager.load_session("corrupted")
            assert result is None
            mock_print.assert_called_with("Error loading session corrupted: Expecting value: line 1 column 1 (char 0)")


if __name__ == "__main__":
    pytest.main([__file__])