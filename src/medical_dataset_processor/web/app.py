"""
Flask web application for the medical dataset processor translation interface.

This module provides a web interface for translating medical datasets with
three modes: automatic, semi-automatic, and manual translation.
"""
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

from ..models.core import Sample
from ..processors.translation_processor import TranslationProcessor, TranslationConfig, TranslationError
from .models import (
    ProcessingMode, 
    TranslationItem, 
    TranslationSession, 
    SessionManager,
    TranslationStatus
)
from .translation_service import WebTranslationService


class FlaskTranslationApp:
    """Flask application for translation interface."""
    
    def __init__(
        self, 
        session_manager: Optional[SessionManager] = None,
        translation_service: Optional[WebTranslationService] = None
    ):
        """Initialize the Flask application."""
        self.session_manager = session_manager or SessionManager()
        
        # Initialize translation service if not provided
        if translation_service is None:
            # Create default translation processor
            api_key = os.environ.get('DEEPL_API_KEY')
            if not api_key:
                raise ValueError("DEEPL_API_KEY environment variable is required")
            
            translation_config = TranslationConfig(
                api_key=api_key,
                target_language=os.environ.get('TARGET_LANGUAGE', 'FR'),
                max_retries=int(os.environ.get('TRANSLATION_MAX_RETRIES', '3')),
                base_delay=float(os.environ.get('TRANSLATION_BASE_DELAY', '1.0')),
                max_delay=float(os.environ.get('TRANSLATION_MAX_DELAY', '60.0'))
            )
            
            translation_processor = TranslationProcessor(translation_config)
            self.translation_service = WebTranslationService(translation_processor, self.session_manager)
        else:
            self.translation_service = translation_service
        
        self.app = self._create_app()
    
    def _create_app(self) -> Flask:
        """Create and configure the Flask application."""
        app = Flask(__name__)
        
        # Configuration
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        app.config['SESSION_TYPE'] = 'filesystem'
        app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
        
        # Enable CORS for API routes
        CORS(app, resources={r"/api/*": {"origins": "*"}})
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: Flask) -> None:
        """Register all application routes."""
        
        @app.route('/')
        def index():
            """Main page with mode selection."""
            return render_template('index.html')
        
        @app.route('/select-mode', methods=['POST'])
        def select_mode():
            """Handle mode selection from the main page."""
            mode = request.form.get('mode')
            
            if mode not in [m.value for m in ProcessingMode]:
                return jsonify({'error': 'Invalid mode selected'}), 400
            
            # Store selected mode in session
            session['selected_mode'] = mode
            session.permanent = True
            
            if mode == ProcessingMode.AUTOMATIC.value:
                # For automatic mode, redirect to a processing page or start processing
                return redirect(url_for('automatic_processing'))
            else:
                # For semi-automatic and manual modes, redirect to translation interface
                return redirect(url_for('translate_interface', mode=mode))
        
        @app.route('/automatic')
        def automatic_processing():
            """Handle automatic processing mode."""
            return render_template('automatic.html', 
                                 message="Automatic processing mode selected. Processing will run without web interface.")
        
        @app.route('/translate/<mode>')
        def translate_interface(mode):
            """Translation interface for semi-automatic and manual modes."""
            if mode not in [ProcessingMode.SEMI_AUTOMATIC.value, ProcessingMode.MANUAL.value]:
                return redirect(url_for('index'))
            
            # Get or create translation session
            session_id = session.get('translation_session_id')
            translation_session = None
            
            if session_id:
                translation_session = self.session_manager.load_session(session_id)
            
            if not translation_session:
                # Create a session using the translation service with demo data
                demo_samples = self._create_demo_samples()
                processing_mode = ProcessingMode(mode)
                
                try:
                    translation_session = self.translation_service.create_translation_session(
                        samples=demo_samples,
                        mode=processing_mode,
                        target_language=os.environ.get('TARGET_LANGUAGE', 'FR')
                    )
                    session['translation_session_id'] = translation_session.session_id
                except TranslationError as e:
                    return render_template('500.html', error=f"Failed to create translation session: {e}"), 500
            
            current_item = translation_session.get_current_item()
            progress_info = translation_session.get_progress_info()
            
            return render_template('translate.html',
                                 mode=mode,
                                 current_item=current_item,
                                 progress=progress_info,
                                 session_id=translation_session.session_id)
        
        # API Routes
        @app.route('/api/current')
        def get_current_item():
            """Get the current translation item."""
            session_id = session.get('translation_session_id')
            if not session_id:
                return jsonify({'error': 'No active session'}), 404
            
            translation_session = self.session_manager.load_session(session_id)
            if not translation_session:
                return jsonify({'error': 'Session not found'}), 404
            
            current_item = translation_session.get_current_item()
            if not current_item:
                return jsonify({'error': 'No current item'}), 404
            
            return jsonify({
                'item': current_item.to_dict(),
                'progress': translation_session.get_progress_info()
            })
        
        @app.route('/api/save', methods=['POST'])
        def save_translation():
            """Save a translation for the current item."""
            session_id = session.get('translation_session_id')
            if not session_id:
                return jsonify({'error': 'No active session'}), 404
            
            translation_session = self.session_manager.load_session(session_id)
            if not translation_session:
                return jsonify({'error': 'Session not found'}), 404
            
            data = request.get_json()
            if not data or 'translation' not in data:
                return jsonify({'error': 'Translation text required'}), 400
            
            translation = data['translation']
            
            # Use translation service for validation and saving
            success = self.translation_service.update_session_translation(
                session_id=session_id,
                item_index=translation_session.current_index,
                translation=translation
            )
            
            if success:
                # Get updated progress info
                progress_info = self.translation_service.get_session_progress(session_id)
                return jsonify({
                    'success': True,
                    'progress': progress_info
                })
            else:
                return jsonify({'error': 'Failed to save translation'}), 500
        
        @app.route('/api/navigate/<direction>')
        def navigate(direction):
            """Navigate to previous or next item."""
            session_id = session.get('translation_session_id')
            if not session_id:
                return jsonify({'error': 'No active session'}), 404
            
            translation_session = self.session_manager.load_session(session_id)
            if not translation_session:
                return jsonify({'error': 'Session not found'}), 404
            
            success = False
            if direction == 'next':
                success = translation_session.navigate_next()
            elif direction == 'previous':
                success = translation_session.navigate_previous()
            else:
                return jsonify({'error': 'Invalid direction'}), 400
            
            if success:
                self.session_manager.save_session(translation_session)
                current_item = translation_session.get_current_item()
                return jsonify({
                    'success': True,
                    'item': current_item.to_dict() if current_item else None,
                    'progress': translation_session.get_progress_info()
                })
            else:
                return jsonify({'error': 'Navigation failed'}), 400
        
        @app.route('/api/session/create', methods=['POST'])
        def create_session():
            """Create a new translation session."""
            data = request.get_json()
            if not data or 'mode' not in data:
                return jsonify({'error': 'Mode is required'}), 400
            
            mode = data['mode']
            if mode not in [m.value for m in ProcessingMode]:
                return jsonify({'error': 'Invalid mode'}), 400
            
            # Create session using translation service with demo data
            demo_samples = self._create_demo_samples()
            processing_mode = ProcessingMode(mode)
            
            try:
                translation_session = self.translation_service.create_translation_session(
                    samples=demo_samples,
                    mode=processing_mode,
                    target_language=os.environ.get('TARGET_LANGUAGE', 'FR')
                )
                
                # Store session ID in Flask session
                session['translation_session_id'] = translation_session.session_id
                session.permanent = True
                
                return jsonify({
                    'session_id': translation_session.session_id,
                    'progress': translation_session.get_progress_info()
                }), 201
                
            except TranslationError as e:
                return jsonify({'error': f'Failed to create session: {str(e)}'}), 500
        
        @app.route('/api/session/<session_id>')
        def get_session(session_id):
            """Get session information."""
            translation_session = self.session_manager.load_session(session_id)
            if not translation_session:
                return jsonify({'error': 'Session not found'}), 404
            
            return jsonify({
                'session': translation_session.to_dict(),
                'progress': translation_session.get_progress_info()
            })
        
        @app.route('/api/auto-save', methods=['POST'])
        def auto_save():
            """Auto-save current translation."""
            session_id = session.get('translation_session_id')
            if not session_id:
                return jsonify({'error': 'No active session'}), 404
            
            data = request.get_json()
            if not data or 'translation' not in data:
                return jsonify({'error': 'Translation text required'}), 400
            
            translation = data['translation']
            translation_session = self.session_manager.load_session(session_id)
            
            if not translation_session:
                return jsonify({'error': 'Session not found'}), 404
            
            # Update translation without changing status (auto-save)
            current_item = translation_session.get_current_item()
            if current_item:
                current_item.target_text = translation
                current_item.updated_at = datetime.now()
                translation_session.last_activity = datetime.now()
                
                success = self.session_manager.auto_save_session(translation_session)
                return jsonify({'success': success, 'auto_saved': True})
            
            return jsonify({'error': 'No current item to save'}), 400
        
        @app.route('/api/validate', methods=['POST'])
        def validate_translation():
            """Validate a translation without saving."""
            data = request.get_json()
            if not data or 'translation' not in data:
                return jsonify({'error': 'Translation text required'}), 400
            
            translation = data['translation']
            validation_result = self.translation_service.validate_translation(translation)
            
            return jsonify(validation_result)
        
        @app.route('/api/usage')
        def get_usage_info():
            """Get translation API usage information."""
            usage_info = self.translation_service.get_translation_usage_info()
            if usage_info:
                return jsonify(usage_info)
            else:
                return jsonify({'error': 'Usage information not available'}), 503
        
        @app.route('/api/session/<session_id>/export')
        def export_session(session_id):
            """Export completed translations from a session."""
            translated_samples = self.translation_service.export_session_results(session_id)
            
            if translated_samples is None:
                return jsonify({'error': 'Session not found'}), 404
            
            # Convert to serializable format
            export_data = []
            for sample in translated_samples:
                export_data.append({
                    'original_id': sample.sample.id,
                    'source_dataset': sample.sample.source_dataset,
                    'original_text': sample.sample.original_text,
                    'translated_text': sample.translated_text,
                    'metadata': sample.translation_metadata,
                    'processing_timestamp': sample.processing_timestamp.isoformat()
                })
            
            return jsonify({
                'session_id': session_id,
                'exported_count': len(export_data),
                'translations': export_data
            })
        
        @app.route('/api/cleanup', methods=['POST'])
        def cleanup_sessions():
            """Clean up expired sessions."""
            data = request.get_json() or {}
            max_age_hours = data.get('max_age_hours', 24)
            
            cleaned_count = self.translation_service.cleanup_expired_sessions(max_age_hours)
            
            return jsonify({
                'cleaned_sessions': cleaned_count,
                'max_age_hours': max_age_hours
            })
        
        @app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        # Error handlers
        @app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Endpoint not found',
                    'message': f'The requested endpoint {request.path} was not found',
                    'timestamp': datetime.now().isoformat()
                }), 404
            return render_template('404.html'), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            import traceback
            error_id = str(uuid.uuid4())[:8]
            
            # Log the full error for debugging
            app.logger.error(f"Internal error {error_id}: {str(error)}")
            app.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Internal server error',
                    'error_id': error_id,
                    'message': 'An unexpected error occurred. Please try again.',
                    'timestamp': datetime.now().isoformat()
                }), 500
            return render_template('500.html', error_id=error_id), 500
        
        @app.errorhandler(TranslationError)
        def handle_translation_error(error):
            """Handle translation-specific errors."""
            error_id = str(uuid.uuid4())[:8]
            app.logger.error(f"Translation error {error_id}: {str(error)}")
            
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Translation error',
                    'error_id': error_id,
                    'message': str(error),
                    'timestamp': datetime.now().isoformat()
                }), 503
            
            return render_template('500.html', 
                                 error_message=f"Translation service error: {str(error)}",
                                 error_id=error_id), 503
        
        @app.errorhandler(ValueError)
        def handle_value_error(error):
            """Handle validation errors."""
            if request.path.startswith('/api/'):
                return jsonify({
                    'error': 'Validation error',
                    'message': str(error),
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            return render_template('500.html', 
                                 error_message=f"Validation error: {str(error)}"), 400
    
    def _create_demo_samples(self) -> List[Sample]:
        """Create demo samples for testing the translation interface."""
        return [
            Sample(
                id="sample_1",
                content={"text": "The patient presents with acute chest pain and shortness of breath."},
                source_dataset="demo_medical_dataset",
                original_text="The patient presents with acute chest pain and shortness of breath."
            ),
            Sample(
                id="sample_2",
                content={"text": "Blood pressure is elevated at 180/100 mmHg."},
                source_dataset="demo_medical_dataset", 
                original_text="Blood pressure is elevated at 180/100 mmHg."
            ),
            Sample(
                id="sample_3",
                content={"text": "Recommend immediate cardiac evaluation and monitoring."},
                source_dataset="demo_medical_dataset",
                original_text="Recommend immediate cardiac evaluation and monitoring."
            )
        ]
    
    def _create_demo_session(self, mode: str) -> TranslationSession:
        """Create a demo translation session for testing."""
        session_id = str(uuid.uuid4())
        processing_mode = ProcessingMode(mode)
        
        # Demo translation items
        demo_items = [
            TranslationItem(
                id="item_1",
                source_text="The patient presents with acute chest pain and shortness of breath.",
                auto_translation="Le patient présente une douleur thoracique aiguë et un essoufflement." if mode == ProcessingMode.SEMI_AUTOMATIC.value else None
            ),
            TranslationItem(
                id="item_2", 
                source_text="Blood pressure is elevated at 180/100 mmHg.",
                auto_translation="La pression artérielle est élevée à 180/100 mmHg." if mode == ProcessingMode.SEMI_AUTOMATIC.value else None
            ),
            TranslationItem(
                id="item_3",
                source_text="Recommend immediate cardiac evaluation and monitoring.",
                auto_translation="Recommander une évaluation cardiaque immédiate et une surveillance." if mode == ProcessingMode.SEMI_AUTOMATIC.value else None
            )
        ]
        
        return TranslationSession(
            session_id=session_id,
            mode=processing_mode,
            items=demo_items,
            total_items=len(demo_items)
        )
    
    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)
    
    def get_app(self) -> Flask:
        """Get the Flask application instance."""
        return self.app


def create_app(
    session_manager: Optional[SessionManager] = None,
    translation_service: Optional[WebTranslationService] = None
) -> Flask:
    """Factory function to create Flask application."""
    flask_app = FlaskTranslationApp(session_manager, translation_service)
    return flask_app.get_app()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)