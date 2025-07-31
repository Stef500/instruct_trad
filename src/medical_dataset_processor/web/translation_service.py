"""
Web translation service that integrates the existing TranslationProcessor
with the web interface for the three translation modes.
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.core import Sample, TranslatedSample
from ..processors.translation_processor import TranslationProcessor, TranslationConfig, TranslationError
from .models import (
    ProcessingMode, 
    TranslationItem, 
    TranslationSession, 
    SessionManager,
    TranslationStatus
)


class WebTranslationService:
    """Service that adapts the existing TranslationProcessor for web interface use."""
    
    def __init__(self, translation_processor: TranslationProcessor, session_manager: SessionManager):
        """
        Initialize the web translation service.
        
        Args:
            translation_processor: The existing translation processor
            session_manager: Session manager for persistence
        """
        self.translation_processor = translation_processor
        self.session_manager = session_manager
        self.logger = logging.getLogger(__name__)
    
    def create_translation_session(
        self, 
        samples: List[Sample], 
        mode: ProcessingMode,
        target_language: str = "FR"
    ) -> TranslationSession:
        """
        Create a new translation session from dataset samples.
        
        Args:
            samples: List of samples to translate
            mode: Processing mode (automatic, semi-automatic, manual)
            target_language: Target language code
            
        Returns:
            TranslationSession with prepared translation items
        """
        session_id = str(uuid.uuid4())
        
        # Prepare translation items based on mode
        translation_items = []
        
        for i, sample in enumerate(samples):
            item_id = f"{session_id}_{i}"
            
            # Create base translation item
            translation_item = TranslationItem(
                id=item_id,
                source_text=sample.original_text,
                metadata={
                    'sample_id': sample.id,
                    'source_dataset': sample.source_dataset,
                    'target_language': target_language,
                    'original_content': sample.content
                }
            )
            
            # For semi-automatic mode, get automatic translation
            if mode == ProcessingMode.SEMI_AUTOMATIC:
                try:
                    auto_translation = self.get_automatic_translation(
                        sample.original_text, 
                        target_language
                    )
                    translation_item.set_auto_translation(auto_translation)
                    translation_item.target_text = auto_translation  # Pre-fill for editing
                except TranslationError as e:
                    self.logger.warning(f"Failed to get auto translation for item {item_id}: {e}")
                    translation_item.metadata['auto_translation_error'] = str(e)
            
            translation_items.append(translation_item)
        
        # Create session
        session = TranslationSession(
            session_id=session_id,
            mode=mode,
            items=translation_items,
            total_items=len(translation_items)
        )
        
        # Save session
        self.session_manager.save_session(session)
        
        self.logger.info(f"Created translation session {session_id} with {len(translation_items)} items in {mode.value} mode")
        
        return session
    
    def get_automatic_translation(self, text: str, target_language: str) -> str:
        """
        Get automatic translation for a single text using the existing processor.
        
        Args:
            text: Text to translate
            target_language: Target language code
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        # Create a temporary sample for translation
        temp_sample = Sample(
            id=f"temp_{uuid.uuid4()}",
            content={'text': text},
            source_dataset="web_interface",
            original_text=text
        )
        
        try:
            # Use existing translation processor
            translated_samples = self.translation_processor.translate_samples([temp_sample])
            
            if translated_samples and len(translated_samples) > 0:
                return translated_samples[0].translated_text
            else:
                raise TranslationError("No translation returned from processor")
                
        except Exception as e:
            self.logger.error(f"Translation failed for text: {text[:50]}... Error: {e}")
            raise TranslationError(f"Translation failed: {str(e)}")
    
    def process_automatic_mode(self, samples: List[Sample], target_language: str = "FR") -> List[TranslatedSample]:
        """
        Process samples in automatic mode (no web interface).
        
        Args:
            samples: List of samples to translate
            target_language: Target language code
            
        Returns:
            List of translated samples
        """
        self.logger.info(f"Processing {len(samples)} samples in automatic mode")
        
        try:
            translated_samples = self.translation_processor.translate_samples(samples)
            self.logger.info(f"Automatic processing completed: {len(translated_samples)} samples translated")
            return translated_samples
        except Exception as e:
            self.logger.error(f"Automatic processing failed: {e}")
            raise TranslationError(f"Automatic processing failed: {str(e)}")
    
    def validate_translation(self, translation: str) -> Dict[str, Any]:
        """
        Validate a translation text.
        
        Args:
            translation: Translation text to validate
            
        Returns:
            Validation result with status and messages
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Basic validation checks
        if not translation or not translation.strip():
            validation_result['valid'] = False
            validation_result['errors'].append("Translation cannot be empty")
        
        if len(translation) > 10000:  # Reasonable length limit
            validation_result['valid'] = False
            validation_result['errors'].append("Translation is too long (max 10,000 characters)")
        
        # Check for potential issues
        if translation == translation.upper():
            validation_result['warnings'].append("Translation appears to be all uppercase")
        
        if len(translation.split()) < 2:
            validation_result['warnings'].append("Translation appears to be very short")
        
        return validation_result
    
    def update_session_translation(
        self, 
        session_id: str, 
        item_index: int, 
        translation: str
    ) -> bool:
        """
        Update a translation in a session.
        
        Args:
            session_id: Session identifier
            item_index: Index of the item to update
            translation: New translation text
            
        Returns:
            True if update was successful
        """
        try:
            session = self.session_manager.load_session(session_id)
            if not session:
                self.logger.error(f"Session {session_id} not found")
                return False
            
            if item_index < 0 or item_index >= len(session.items):
                self.logger.error(f"Invalid item index {item_index} for session {session_id}")
                return False
            
            # Validate translation
            validation = self.validate_translation(translation)
            if not validation['valid']:
                self.logger.error(f"Translation validation failed: {validation['errors']}")
                return False
            
            # Update the translation
            session.items[item_index].update_translation(translation)
            session.last_activity = datetime.now()
            
            # Save session
            success = self.session_manager.save_session(session)
            
            if success:
                self.logger.debug(f"Updated translation for item {item_index} in session {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update translation: {e}")
            return False
    
    def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress information for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Progress information or None if session not found
        """
        try:
            session = self.session_manager.load_session(session_id)
            if not session:
                return None
            
            progress = session.get_progress_info()
            
            # Add additional web-specific progress info
            progress.update({
                'mode': session.mode.value,
                'can_navigate_previous': session.current_index > 0,
                'can_navigate_next': session.current_index < len(session.items) - 1,
                'is_complete': session.is_complete(),
                'last_activity': session.last_activity.isoformat()
            })
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Failed to get session progress: {e}")
            return None
    
    def export_session_results(self, session_id: str) -> Optional[List[TranslatedSample]]:
        """
        Export completed translations from a session as TranslatedSample objects.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of TranslatedSample objects or None if session not found
        """
        try:
            session = self.session_manager.load_session(session_id)
            if not session:
                return None
            
            translated_samples = []
            
            for item in session.items:
                if item.status == TranslationStatus.COMPLETED and item.target_text:
                    # Reconstruct original sample from metadata
                    original_sample = Sample(
                        id=item.metadata.get('sample_id', item.id),
                        content=item.metadata.get('original_content', {}),
                        source_dataset=item.metadata.get('source_dataset', 'web_interface'),
                        original_text=item.source_text
                    )
                    
                    # Create translated sample
                    translated_sample = TranslatedSample(
                        sample=original_sample,
                        translated_text=item.target_text,
                        translation_metadata={
                            'web_session_id': session_id,
                            'processing_mode': session.mode.value,
                            'target_language': item.metadata.get('target_language', 'FR'),
                            'translation_timestamp': item.updated_at.isoformat(),
                            'auto_translation_used': item.auto_translation is not None,
                            'manual_edits': item.target_text != item.auto_translation if item.auto_translation else True
                        },
                        processing_timestamp=item.updated_at
                    )
                    
                    translated_samples.append(translated_sample)
            
            self.logger.info(f"Exported {len(translated_samples)} translated samples from session {session_id}")
            return translated_samples
            
        except Exception as e:
            self.logger.error(f"Failed to export session results: {e}")
            return None
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired sessions.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            cleaned_count = self.session_manager.cleanup_old_sessions(max_age_hours)
            self.logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_translation_usage_info(self) -> Optional[Dict[str, Any]]:
        """
        Get translation API usage information.
        
        Returns:
            Usage information from the translation processor
        """
        try:
            return self.translation_processor.get_usage_info()
        except Exception as e:
            self.logger.error(f"Failed to get usage info: {e}")
            return None