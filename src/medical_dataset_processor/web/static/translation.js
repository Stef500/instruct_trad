// Translation interface JavaScript
class TranslationInterface {
    constructor() {
        this.currentSessionId = null;
        this.currentMode = null;
        this.autoSaveTimeout = null;
        this.isNavigating = false;
        this.autoSaveInterval = 2000; // 2 seconds
        
        this.init();
    }
    
    init() {
        // Get session data from the page
        const sessionData = this.getSessionData();
        if (sessionData) {
            this.currentSessionId = sessionData.sessionId;
            this.currentMode = sessionData.mode;
        }
        
        this.bindEvents();
        this.updateButtonStates();
    }
    
    getSessionData() {
        // Try to get session data from meta tags or global variables
        const sessionIdMeta = document.querySelector('meta[name="session-id"]');
        const modeMeta = document.querySelector('meta[name="mode"]');
        
        if (sessionIdMeta && modeMeta) {
            return {
                sessionId: sessionIdMeta.content,
                mode: modeMeta.content
            };
        }
        
        // Fallback to global variables if they exist
        if (typeof currentSessionId !== 'undefined' && typeof currentMode !== 'undefined') {
            return {
                sessionId: currentSessionId,
                mode: currentMode
            };
        }
        
        return null;
    }
    
    bindEvents() {
        const translationInput = document.getElementById('translationInput');
        if (translationInput) {
            translationInput.addEventListener('input', () => this.handleInputChange());
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
        
        // Prevent accidental page refresh
        window.addEventListener('beforeunload', (e) => {
            if (this.hasUnsavedChanges()) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }
    
    handleInputChange() {
        this.showStatus('saving', 'Sauvegarde...');
        
        // Clear existing timeout
        if (this.autoSaveTimeout) {
            clearTimeout(this.autoSaveTimeout);
        }
        
        // Set new timeout for auto-save
        this.autoSaveTimeout = setTimeout(() => {
            this.saveTranslation(false);
        }, this.autoSaveInterval);
    }
    
    handleKeyboardShortcuts(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 's':
                    e.preventDefault();
                    this.saveTranslation(false);
                    break;
                case 'Enter':
                    e.preventDefault();
                    this.validateTranslation();
                    break;
            }
        }
        
        if (e.altKey) {
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigate('previous');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigate('next');
                    break;
            }
        }
    }
    
    async navigate(direction) {
        if (this.isNavigating) return;
        
        this.isNavigating = true;
        const btn = direction === 'next' ? 
            document.getElementById('nextBtn') : 
            document.getElementById('prevBtn');
        
        if (!btn) return;
        
        const originalText = btn.textContent;
        btn.textContent = 'Chargement...';
        btn.disabled = true;
        
        try {
            const response = await fetch(`/api/navigate/${direction}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateInterface(data.item, data.progress);
                this.showStatus('saved', 'Sauvegardé');
            } else {
                this.showStatus('error', data.error || 'Erreur de navigation');
            }
        } catch (error) {
            console.error('Navigation error:', error);
            this.showStatus('error', 'Erreur de connexion');
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
            this.isNavigating = false;
            this.updateButtonStates();
        }
    }
    
    async validateTranslation() {
        const success = await this.saveTranslation(true);
        if (success) {
            // Auto-navigate to next item after validation
            setTimeout(() => {
                this.navigate('next');
            }, 500);
        }
    }
    
    async saveTranslation(isValidation = false) {
        const translationInput = document.getElementById('translationInput');
        if (!translationInput) return false;
        
        const translation = translationInput.value.trim();
        
        if (!translation && isValidation) {
            this.showStatus('error', 'Veuillez saisir une traduction');
            return false;
        }
        
        this.showStatus('saving', 'Sauvegarde...');
        
        try {
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    translation: translation,
                    is_validation: isValidation
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showStatus('saved', isValidation ? 'Validé et sauvegardé' : 'Sauvegardé');
                if (data.progress) {
                    this.updateProgress(data.progress);
                }
                return true;
            } else {
                this.showStatus('error', data.error || 'Erreur de sauvegarde');
                return false;
            }
        } catch (error) {
            console.error('Save error:', error);
            this.showStatus('error', 'Erreur de connexion');
            return false;
        }
    }
    
    async clearTranslation() {
        const translationInput = document.getElementById('translationInput');
        if (!translationInput) return;
        
        if (this.currentMode === 'semi_automatic') {
            // In semi-automatic mode, restore auto-translation
            try {
                const data = await this.getCurrentItem();
                if (data && data.item && data.item.auto_translation) {
                    translationInput.value = data.item.auto_translation;
                } else {
                    translationInput.value = '';
                }
                this.showStatus('saved', 'Texte restauré');
            } catch (error) {
                translationInput.value = '';
                this.showStatus('saved', 'Texte effacé');
            }
        } else {
            // In manual mode, just clear the field
            translationInput.value = '';
            this.showStatus('saved', 'Texte effacé');
        }
        
        translationInput.focus();
    }
    
    async getCurrentItem() {
        try {
            const response = await fetch('/api/current');
            return await response.json();
        } catch (error) {
            console.error('Error getting current item:', error);
            return null;
        }
    }
    
    updateInterface(item, progress) {
        if (item) {
            const sourceText = document.getElementById('sourceText');
            if (sourceText) {
                sourceText.textContent = item.source_text;
            }
            
            const translationInput = document.getElementById('translationInput');
            if (translationInput) {
                if (this.currentMode === 'semi_automatic' && item.auto_translation) {
                    translationInput.value = item.target_text || item.auto_translation;
                } else {
                    translationInput.value = item.target_text || '';
                }
            }
        }
        
        if (progress) {
            this.updateProgress(progress);
        }
        
        this.updateButtonStates();
    }
    
    updateProgress(progress) {
        const currentItem = document.getElementById('currentItem');
        if (currentItem) {
            currentItem.textContent = progress.current_item;
        }
        
        const progressFill = document.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = progress.percentage + '%';
        }
    }
    
    async updateButtonStates() {
        try {
            const data = await this.getCurrentItem();
            if (data && data.progress) {
                const progress = data.progress;
                
                const prevBtn = document.getElementById('prevBtn');
                const nextBtn = document.getElementById('nextBtn');
                
                if (prevBtn) {
                    prevBtn.disabled = progress.current_item <= 1;
                }
                
                if (nextBtn) {
                    nextBtn.disabled = progress.current_item >= progress.total_items;
                }
            }
        } catch (error) {
            console.error('Error updating button states:', error);
        }
    }
    
    showStatus(type, message) {
        const statusIndicator = document.getElementById('statusIndicator');
        if (!statusIndicator) return;
        
        statusIndicator.className = `status-indicator status-${type}`;
        statusIndicator.textContent = message;
        
        // Auto-hide error messages after 5 seconds
        if (type === 'error') {
            setTimeout(() => {
                this.showStatus('saved', 'Sauvegardé');
            }, 5000);
        }
    }
    
    hasUnsavedChanges() {
        // Check if there are unsaved changes
        return this.autoSaveTimeout !== null;
    }
}

// Mode selection JavaScript
class ModeSelection {
    constructor() {
        this.init();
    }
    
    init() {
        this.bindEvents();
    }
    
    bindEvents() {
        const modeOptions = document.querySelectorAll('.mode-option');
        const submitBtn = document.getElementById('submitBtn');
        const modeForm = document.getElementById('modeForm');
        
        // Handle mode selection
        modeOptions.forEach(option => {
            option.addEventListener('click', () => {
                // Remove selected class from all options
                modeOptions.forEach(opt => opt.classList.remove('selected'));
                
                // Add selected class to clicked option
                option.classList.add('selected');
                
                // Check the radio button
                const radio = option.querySelector('input[type="radio"]');
                if (radio) {
                    radio.checked = true;
                }
                
                // Enable submit button
                if (submitBtn) {
                    submitBtn.disabled = false;
                }
            });
        });
        
        // Handle form submission
        if (modeForm) {
            modeForm.addEventListener('submit', (e) => {
                const selectedMode = document.querySelector('input[name="mode"]:checked');
                if (!selectedMode) {
                    e.preventDefault();
                    alert('Veuillez sélectionner un mode de traduction.');
                    return;
                }
                
                // Show loading state
                if (submitBtn) {
                    submitBtn.textContent = 'Chargement...';
                    submitBtn.disabled = true;
                }
            });
        }
    }
}

// Global functions for backward compatibility
function navigate(direction) {
    if (window.translationInterface) {
        window.translationInterface.navigate(direction);
    }
}

function validateTranslation() {
    if (window.translationInterface) {
        window.translationInterface.validateTranslation();
    }
}

function clearTranslation() {
    if (window.translationInterface) {
        window.translationInterface.clearTranslation();
    }
}

function startProcessing() {
    // For automatic mode
    alert('Le traitement automatique serait démarré ici.\n\nDans l\'implémentation complète, ceci déclencherait le pipeline de traduction existant.');
    
    // For demo purposes, redirect back to mode selection
    setTimeout(() => {
        window.location.href = '/';
    }, 2000);
}

// Initialize appropriate interface based on page
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('translationInput')) {
        // Translation interface page
        window.translationInterface = new TranslationInterface();
    } else if (document.getElementById('modeForm')) {
        // Mode selection page
        window.modeSelection = new ModeSelection();
    }
});