// Translation interface JavaScript
class TranslationInterface {
    constructor() {
        this.currentSessionId = null;
        this.currentMode = null;
        this.autoSaveTimeout = null;
        this.isNavigating = false;
        this.autoSaveInterval = 5000; // 5 seconds as per requirements
        this.lastSavedContent = '';
        this.currentProgress = null;
        this.validationInProgress = false;
        
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
        this.startAutoSave();
        
        // Initialize last saved content
        const translationInput = document.getElementById('translationInput');
        if (translationInput) {
            this.lastSavedContent = translationInput.value.trim();
        }
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
        
        // Save current translation before navigating
        await this.autoSaveCurrentTranslation();
        
        this.isNavigating = true;
        const btn = direction === 'next' ? 
            document.getElementById('nextBtn') : 
            document.getElementById('prevBtn');
        
        if (!btn || btn.disabled) {
            this.isNavigating = false;
            return;
        }
        
        const originalText = btn.textContent;
        btn.textContent = 'Chargement...';
        btn.disabled = true;
        
        // Disable all navigation buttons during navigation
        this.setNavigationButtonsState(true);
        
        try {
            const response = await fetch(`/api/navigate/${direction}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateInterface(data.item, data.progress);
                this.currentProgress = data.progress;
                this.lastSavedContent = data.item?.target_text || '';
                this.showStatus('saved', 'Navigation réussie');
                
                // Update progress indicators
                this.updateProgressIndicators(data.progress);
            } else {
                this.showStatus('error', data.error || 'Erreur de navigation');
            }
        } catch (error) {
            console.error('Navigation error:', error);
            this.showStatus('error', 'Erreur de connexion');
        } finally {
            btn.textContent = originalText;
            this.setNavigationButtonsState(false);
            this.isNavigating = false;
            this.updateButtonStates();
        }
    }
    
    async validateTranslation() {
        if (this.validationInProgress) return;
        
        this.validationInProgress = true;
        const validateBtn = document.getElementById('validateBtn');
        
        if (validateBtn) {
            const originalText = validateBtn.textContent;
            validateBtn.textContent = 'Validation...';
            validateBtn.disabled = true;
        }
        
        try {
            const translationInput = document.getElementById('translationInput');
            if (!translationInput) return false;
            
            const translation = translationInput.value.trim();
            
            // Validate translation content
            if (!translation) {
                this.showStatus('error', 'Veuillez saisir une traduction avant de valider');
                return false;
            }
            
            // Perform client-side validation
            const validationResult = this.performClientValidation(translation);
            if (!validationResult.isValid) {
                this.showStatus('error', validationResult.message);
                return false;
            }
            
            // Save the validated translation
            const success = await this.saveTranslation(true);
            if (success) {
                this.lastSavedContent = translation;
                this.showStatus('saved', 'Traduction validée et sauvegardée');
                
                // Auto-navigate to next item after validation if not on last item
                if (this.currentProgress && this.currentProgress.current_item < this.currentProgress.total_items) {
                    setTimeout(() => {
                        this.navigate('next');
                    }, 800);
                } else {
                    this.showStatus('saved', 'Dernière traduction validée - Session terminée');
                }
                return true;
            }
            return false;
        } catch (error) {
            console.error('Validation error:', error);
            this.showStatus('error', 'Erreur lors de la validation');
            return false;
        } finally {
            this.validationInProgress = false;
            if (validateBtn) {
                validateBtn.textContent = 'Valider';
                validateBtn.disabled = false;
            }
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
        
        // Show confirmation if there's significant content
        const currentContent = translationInput.value.trim();
        if (currentContent.length > 50) {
            const confirmed = confirm('Êtes-vous sûr de vouloir effacer cette traduction ?');
            if (!confirmed) return;
        }
        
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            const originalText = clearBtn.textContent;
            clearBtn.textContent = 'Effacement...';
            clearBtn.disabled = true;
            
            setTimeout(() => {
                clearBtn.textContent = originalText;
                clearBtn.disabled = false;
            }, 500);
        }
        
        if (this.currentMode === 'semi_automatic') {
            // In semi-automatic mode, restore auto-translation
            try {
                const data = await this.getCurrentItem();
                if (data && data.item && data.item.auto_translation) {
                    translationInput.value = data.item.auto_translation;
                    this.lastSavedContent = data.item.auto_translation;
                    this.showStatus('saved', 'Traduction automatique restaurée');
                } else {
                    translationInput.value = '';
                    this.lastSavedContent = '';
                    this.showStatus('saved', 'Texte effacé');
                }
            } catch (error) {
                translationInput.value = '';
                this.lastSavedContent = '';
                this.showStatus('saved', 'Texte effacé');
            }
        } else {
            // In manual mode, just clear the field
            translationInput.value = '';
            this.lastSavedContent = '';
            this.showStatus('saved', 'Texte effacé');
        }
        
        translationInput.focus();
        
        // Auto-save the cleared state
        await this.autoSaveCurrentTranslation();
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
        if (!statusIndicator) {
            // Fallback to console if no status indicator
            console.log(`Status [${type}]: ${message}`);
            return;
        }
        
        statusIndicator.className = `status-indicator status-${type}`;
        statusIndicator.textContent = message;
        
        // Show toast notification for important messages
        this.showToastNotification(type, message);
        
        // Auto-hide error messages after 5 seconds
        if (type === 'error') {
            setTimeout(() => {
                if (statusIndicator.textContent === message) {
                    this.showStatus('saved', 'Prêt');
                }
            }, 5000);
        }
    }
    
    showToastNotification(type, message) {
        // Create toast notification for better user feedback
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${this.getToastIcon(type)}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Add to page
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(toast);
        
        // Auto-remove after delay
        const delay = type === 'error' ? 8000 : 4000;
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, delay);
        
        // Animate in
        setTimeout(() => {
            toast.classList.add('toast-show');
        }, 100);
    }
    
    getToastIcon(type) {
        const icons = {
            'error': '⚠️',
            'saving': '💾',
            'saved': '✅',
            'warning': '⚡',
            'info': 'ℹ️'
        };
        return icons[type] || 'ℹ️';
    }
    
    showErrorDialog(title, message, details = null) {
        // Create modal error dialog for critical errors
        const modal = document.createElement('div');
        modal.className = 'error-modal';
        modal.innerHTML = `
            <div class="error-modal-content">
                <div class="error-modal-header">
                    <h3>${title}</h3>
                    <button class="error-modal-close" onclick="this.closest('.error-modal').remove()">×</button>
                </div>
                <div class="error-modal-body">
                    <p>${message}</p>
                    ${details ? `<details><summary>Détails techniques</summary><pre>${details}</pre></details>` : ''}
                </div>
                <div class="error-modal-footer">
                    <button onclick="this.closest('.error-modal').remove()" class="btn btn-primary">OK</button>
                    <button onclick="window.location.reload()" class="btn btn-secondary">Recharger la page</button>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Focus on close button for accessibility
        setTimeout(() => {
            const closeBtn = modal.querySelector('.error-modal-close');
            if (closeBtn) closeBtn.focus();
        }, 100);
    }
    
    hasUnsavedChanges() {
        // Check if there are unsaved changes
        const translationInput = document.getElementById('translationInput');
        if (!translationInput) return false;
        
        const currentContent = translationInput.value.trim();
        return currentContent !== this.lastSavedContent || this.autoSaveTimeout !== null;
    }
    
    async autoSaveCurrentTranslation() {
        // Auto-save current translation before navigation
        const translationInput = document.getElementById('translationInput');
        if (!translationInput) return;
        
        const currentContent = translationInput.value.trim();
        if (currentContent !== this.lastSavedContent) {
            try {
                const response = await fetch('/api/auto-save', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        translation: currentContent
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    this.lastSavedContent = currentContent;
                }
            } catch (error) {
                console.error('Auto-save error:', error);
            }
        }
    }
    
    setNavigationButtonsState(disabled) {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const validateBtn = document.getElementById('validateBtn');
        
        if (prevBtn) prevBtn.disabled = disabled;
        if (nextBtn) nextBtn.disabled = disabled;
        if (validateBtn) validateBtn.disabled = disabled;
    }
    
    updateProgressIndicators(progress) {
        // Update progress bar
        this.updateProgress(progress);
        
        // Update additional progress indicators
        const progressInfo = document.querySelector('.progress-info');
        if (progressInfo) {
            progressInfo.innerHTML = `Élément <span id="currentItem">${progress.current_item}</span> sur ${progress.total_items}`;
        }
        
        // Update progress percentage in visual bar
        const progressFill = document.querySelector('.progress-fill');
        if (progressFill) {
            progressFill.style.width = `${progress.percentage}%`;
            progressFill.setAttribute('aria-valuenow', progress.percentage);
        }
        
        // Add completion status if on last item
        if (progress.current_item === progress.total_items) {
            const statusIndicator = document.getElementById('statusIndicator');
            if (statusIndicator) {
                statusIndicator.classList.add('completion-ready');
            }
        }
    }
    
    performClientValidation(translation) {
        // Basic client-side validation
        if (!translation || translation.trim().length === 0) {
            return {
                isValid: false,
                message: 'La traduction ne peut pas être vide'
            };
        }
        
        // Check minimum length
        if (translation.trim().length < 2) {
            return {
                isValid: false,
                message: 'La traduction doit contenir au moins 2 caractères'
            };
        }
        
        // Check maximum length (reasonable limit)
        if (translation.length > 10000) {
            return {
                isValid: false,
                message: 'La traduction est trop longue (maximum 10 000 caractères)'
            };
        }
        
        // Check for suspicious patterns (only spaces, repeated characters)
        if (/^\s+$/.test(translation)) {
            return {
                isValid: false,
                message: 'La traduction ne peut contenir que des espaces'
            };
        }
        
        // Check for excessive repetition of single character
        const repeatedChar = /(.)\1{20,}/.test(translation);
        if (repeatedChar) {
            return {
                isValid: false,
                message: 'La traduction contient trop de caractères répétés'
            };
        }
        
        return {
            isValid: true,
            message: 'Validation réussie'
        };
    }
    
    startAutoSave() {
        // Start periodic auto-save
        if (this.autoSaveInterval > 0) {
            setInterval(() => {
                if (this.hasUnsavedChanges()) {
                    this.autoSaveCurrentTranslation();
                }
            }, this.autoSaveInterval);
        }
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

function toggleShortcuts() {
    const panel = document.getElementById('shortcutsPanel');
    if (panel) {
        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
        } else {
            panel.style.display = 'none';
        }
    }
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