"""
Web interface data models for the medical dataset processor.

This module contains the data models specific to the web interface,
including processing modes, translation items, and session management.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path


class ProcessingMode(Enum):
    """Processing modes for the web interface."""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"


class TranslationStatus(Enum):
    """Status of a translation item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class TranslationItem:
    """A single item to be translated in the web interface."""
    id: str
    source_text: str
    target_text: Optional[str] = None
    auto_translation: Optional[str] = None  # For semi-automatic mode
    status: TranslationStatus = TranslationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.id:
            raise ValueError("Translation item ID cannot be empty")
        
        if not self.source_text:
            raise ValueError("Source text cannot be empty")
        
        if not isinstance(self.status, TranslationStatus):
            raise ValueError("Status must be a TranslationStatus enum value")
    
    def update_translation(self, translation: str) -> None:
        """Update the translation and mark as updated."""
        self.target_text = translation
        self.updated_at = datetime.now()
        if translation:
            self.status = TranslationStatus.COMPLETED
    
    def set_auto_translation(self, auto_translation: str) -> None:
        """Set the automatic translation for semi-automatic mode."""
        self.auto_translation = auto_translation
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranslationItem':
        """Create instance from dictionary (JSON deserialization)."""
        # Convert string dates back to datetime objects
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert status string back to enum
        if isinstance(data['status'], str):
            data['status'] = TranslationStatus(data['status'])
        
        return cls(**data)


@dataclass
class TranslationSession:
    """A translation session containing multiple items."""
    session_id: str
    mode: ProcessingMode
    items: List[TranslationItem]
    current_index: int = 0
    total_items: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    auto_save_enabled: bool = True
    
    def __post_init__(self):
        """Validate and initialize session data."""
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        
        if not isinstance(self.mode, ProcessingMode):
            raise ValueError("Mode must be a ProcessingMode enum value")
        
        if not self.items:
            raise ValueError("Session must contain at least one translation item")
        
        # Set total_items if not provided
        if self.total_items == 0:
            self.total_items = len(self.items)
        
        # Validate current_index
        if self.current_index < 0 or self.current_index >= len(self.items):
            self.current_index = 0
    
    def get_current_item(self) -> Optional[TranslationItem]:
        """Get the current translation item."""
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None
    
    def navigate_to(self, index: int) -> bool:
        """Navigate to a specific item index."""
        if 0 <= index < len(self.items):
            self.current_index = index
            self.last_activity = datetime.now()
            return True
        return False
    
    def navigate_next(self) -> bool:
        """Navigate to the next item."""
        return self.navigate_to(self.current_index + 1)
    
    def navigate_previous(self) -> bool:
        """Navigate to the previous item."""
        return self.navigate_to(self.current_index - 1)
    
    def update_current_translation(self, translation: str) -> bool:
        """Update the translation for the current item."""
        current_item = self.get_current_item()
        if current_item:
            current_item.update_translation(translation)
            self.last_activity = datetime.now()
            return True
        return False
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get progress information for the session."""
        completed_items = sum(1 for item in self.items if item.status == TranslationStatus.COMPLETED)
        return {
            'current_item': self.current_index + 1,
            'total_items': self.total_items,
            'completed_items': completed_items,
            'percentage': (completed_items / self.total_items) * 100 if self.total_items > 0 else 0,
            'session_id': self.session_id,
            'mode': self.mode.value
        }
    
    def is_complete(self) -> bool:
        """Check if all items in the session are completed."""
        return all(item.status == TranslationStatus.COMPLETED for item in self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'mode': self.mode.value,
            'items': [item.to_dict() for item in self.items],
            'current_index': self.current_index,
            'total_items': self.total_items,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'auto_save_enabled': self.auto_save_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranslationSession':
        """Create instance from dictionary (JSON deserialization)."""
        # Convert string dates back to datetime objects
        if isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data['last_activity'], str):
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        
        # Convert mode string back to enum
        if isinstance(data['mode'], str):
            data['mode'] = ProcessingMode(data['mode'])
        
        # Convert items dictionaries back to TranslationItem objects
        if data['items'] and isinstance(data['items'][0], dict):
            data['items'] = [TranslationItem.from_dict(item_data) for item_data in data['items']]
        
        return cls(**data)


class SessionManager:
    """Manages translation sessions with JSON file persistence."""
    
    def __init__(self, storage_dir: str = "data/sessions"):
        """Initialize the session manager with a storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.storage_dir / f"{session_id}.json"
    
    def save_session(self, session: TranslationSession) -> bool:
        """Save a session to JSON file."""
        try:
            file_path = self._get_session_file_path(session.session_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving session {session.session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[TranslationSession]:
        """Load a session from JSON file."""
        try:
            file_path = self._get_session_file_path(session_id)
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return TranslationSession.from_dict(data)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        try:
            file_path = self._get_session_file_path(session_id)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        try:
            return [f.stem for f in self.storage_dir.glob("*.json")]
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    def auto_save_session(self, session: TranslationSession) -> bool:
        """Auto-save a session if auto-save is enabled."""
        if session.auto_save_enabled:
            return self.save_session(session)
        return True
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than max_age_hours."""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for session_file in self.storage_dir.glob("*.json"):
                if session_file.stat().st_mtime < cutoff_time:
                    session_file.unlink()
                    cleaned_count += 1
            
            return cleaned_count
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0