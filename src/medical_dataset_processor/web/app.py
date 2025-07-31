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

from .models import (
    ProcessingMode, 
    TranslationItem, 
    TranslationSession, 
    SessionManager,
    TranslationStatus
)


class FlaskTranslationApp:
    """Flask application for translation interface."""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize the Flask application."""
        self.session_manager = session_manager or SessionManager()
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
                # Create a demo session for now (in real implementation, this would come from dataset)
                translation_session = self._create_demo_session(mode)
                session['translation_session_id'] = translation_session.session_id
                self.session_manager.save_session(translation_session)
            
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
            success = translation_session.update_current_translation(translation)
            
            if success:
                self.session_manager.save_session(translation_session)
                return jsonify({
                    'success': True,
                    'progress': translation_session.get_progress_info()
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
            
            # Create session with demo data (in real implementation, use actual dataset)
            translation_session = self._create_demo_session(mode)
            self.session_manager.save_session(translation_session)
            
            # Store session ID in Flask session
            session['translation_session_id'] = translation_session.session_id
            session.permanent = True
            
            return jsonify({
                'session_id': translation_session.session_id,
                'progress': translation_session.get_progress_info()
            }), 201
        
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
                return jsonify({'error': 'Endpoint not found'}), 404
            return render_template('404.html'), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Internal server error'}), 500
            return render_template('500.html'), 500
    
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


def create_app(session_manager: Optional[SessionManager] = None) -> Flask:
    """Factory function to create Flask application."""
    flask_app = FlaskTranslationApp(session_manager)
    return flask_app.get_app()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)