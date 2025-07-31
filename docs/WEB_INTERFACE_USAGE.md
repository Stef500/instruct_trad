# Interface Web de Traduction - Guide d'Utilisation

## Vue d'Ensemble

L'interface web de traduction du Medical Dataset Processor offre trois modes de traduction interactifs pour traiter vos datasets médicaux :

- **Automatique** : Traitement entièrement automatisé via CLI
- **Semi-automatique** : Interface web avec traductions automatiques éditables
- **Manuel** : Interface web avec saisie manuelle complète

## Démarrage Rapide

### 1. Installation et Configuration

```bash
# Installer les dépendances
pip install -e .

# Configurer les clés API (requis)
export DEEPL_API_KEY="votre_cle_deepl"
export OPENAI_API_KEY="votre_cle_openai"  # optionnel pour certains modes
```

### 2. Lancement de l'Interface Web

```bash
# Lancer avec sélection de mode interactive
medical-dataset-processor web

# Ou spécifier directement le mode
medical-dataset-processor web --mode semi_automatic

# Avec options personnalisées
medical-dataset-processor web \
  --mode manual \
  --port 8080 \
  --target-language ES \
  --datasets-config mon_config.yaml
```

### 3. Accès à l'Interface

Une fois lancée, l'interface est accessible à :
- **URL locale** : http://localhost:5000 (ou port spécifié)
- **Vérification santé** : http://localhost:5000/api/health

## Modes de Traduction

### Mode Automatique
- **Usage** : Traitement batch sans intervention
- **Interface** : CLI uniquement
- **Recommandé pour** : Gros volumes, traductions de routine

```bash
medical-dataset-processor web --mode automatic
# Redirige vers la commande CLI standard
medical-dataset-processor process --translation-count 100
```

### Mode Semi-Automatique
- **Usage** : Révision et édition de traductions automatiques
- **Interface** : Web avec panneau source/cible
- **Recommandé pour** : Contrôle qualité, textes spécialisés

**Fonctionnalités** :
- Traduction automatique pré-remplie
- Édition en temps réel
- Sauvegarde automatique toutes les 5 secondes
- Navigation entre éléments
- Validation et passage automatique au suivant

### Mode Manuel
- **Usage** : Traduction entièrement manuelle
- **Interface** : Web avec champ de saisie vide
- **Recommandé pour** : Textes très spécialisés, contrôle total

**Fonctionnalités** :
- Saisie libre dans le panneau cible
- Sauvegarde automatique
- Navigation flexible
- Validation manuelle

## Interface Utilisateur

### Panneau Principal
```
┌─────────────────────────────────────────────────────────┐
│ Élément 2 sur 10                    ████████░░ 80%      │
├─────────────────────────────────────────────────────────┤
│ TEXTE SOURCE          │ TRADUCTION                      │
│                       │                                 │
│ The patient presents  │ Le patient présente une         │
│ with acute chest pain │ douleur thoracique aiguë et     │
│ and shortness of      │ un essoufflement.               │
│ breath.               │                                 │
│                       │                                 │
├─────────────────────────────────────────────────────────┤
│ [Précédent] [Effacer]     [Sauvegardé] [Valider] [Suivant] │
└─────────────────────────────────────────────────────────┘
```

### Contrôles et Raccourcis

#### Boutons
- **Précédent** : Revenir à l'élément précédent
- **Suivant** : Passer à l'élément suivant
- **Effacer** : Vider le champ de traduction (ou restaurer l'auto-traduction)
- **Valider** : Sauvegarder et passer au suivant automatiquement

#### Raccourcis Clavier
- `Ctrl+S` : Sauvegarde manuelle
- `Ctrl+Enter` : Valider la traduction
- `Alt+←` : Élément précédent
- `Alt+→` : Élément suivant

#### Indicateurs de Statut
- 🟢 **Sauvegardé** : Modifications enregistrées
- 🟡 **Sauvegarde...** : Sauvegarde en cours
- 🔴 **Erreur** : Problème de connexion ou validation

## Configuration Avancée

### Variables d'Environnement

```bash
# API et traduction
DEEPL_API_KEY=votre_cle_deepl
TARGET_LANGUAGE=FR                    # Code langue DeepL
TRANSLATION_MAX_RETRIES=3
TRANSLATION_BASE_DELAY=1.0

# Serveur web
WEB_HOST=0.0.0.0
WEB_PORT=5000
SECRET_KEY=votre_cle_secrete

# Session et sauvegarde
AUTO_SAVE_INTERVAL=5                  # secondes
MAX_SESSION_DURATION=86400            # 24 heures
```

### Options de Ligne de Commande

```bash
medical-dataset-processor web --help

Options:
  -m, --mode [automatic|semi_automatic|manual]
                                  Mode de traitement
  --deepl-key TEXT               Clé API DeepL
  -d, --datasets-config PATH     Fichier de configuration datasets
  --host TEXT                    Adresse d'écoute (défaut: 0.0.0.0)
  -p, --port INTEGER             Port d'écoute (défaut: 5000)
  --target-language TEXT         Langue cible (défaut: FR)
  --debug                        Mode debug Flask
  --help                         Afficher cette aide
```

## Gestion des Erreurs

### Erreurs Communes

#### "DeepL API key is required"
```bash
# Solution : Configurer la clé API
export DEEPL_API_KEY="votre_cle_deepl"
```

#### "Port already in use"
```bash
# Solution : Utiliser un port différent
medical-dataset-processor web --port 8080
```

#### "Session not found"
- **Cause** : Session expirée ou corrompue
- **Solution** : Recharger la page pour créer une nouvelle session

#### "Translation service error"
- **Cause** : Problème avec l'API DeepL
- **Solution** : Vérifier la clé API et la connectivité

### Récupération Automatique

L'interface inclut plusieurs mécanismes de récupération :

- **Sauvegarde automatique** : Toutes les 5 secondes
- **Restauration de session** : Au rechargement de page
- **Retry automatique** : Pour les erreurs temporaires
- **Notifications utilisateur** : Alertes visuelles pour tous les états

## Déploiement Docker

### Configuration Simple

```yaml
# docker-compose.yml
version: '3.8'
services:
  medical-processor-web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DEEPL_API_KEY=${DEEPL_API_KEY}
      - TARGET_LANGUAGE=FR
      - WEB_HOST=0.0.0.0
      - WEB_PORT=5000
    volumes:
      - ./data:/app/data
      - ./datasets.yaml:/app/datasets.yaml
```

### Lancement Docker

```bash
# Construire et lancer
docker-compose up --build

# En arrière-plan
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

## API REST

L'interface expose également une API REST pour l'intégration :

### Endpoints Principaux

```bash
# Santé du service
GET /api/health

# Session courante
GET /api/current
POST /api/save
GET /api/navigate/next
GET /api/navigate/previous

# Gestion des sessions
POST /api/session/create
GET /api/session/{id}
GET /api/session/{id}/export

# Utilitaires
POST /api/validate
GET /api/usage
POST /api/cleanup
```

### Exemple d'Usage API

```bash
# Créer une session
curl -X POST http://localhost:5000/api/session/create \
  -H "Content-Type: application/json" \
  -d '{"mode": "semi_automatic"}'

# Sauvegarder une traduction
curl -X POST http://localhost:5000/api/save \
  -H "Content-Type: application/json" \
  -d '{"translation": "Ma traduction"}'

# Exporter les résultats
curl http://localhost:5000/api/session/{session_id}/export
```

## Dépannage

### Logs et Debug

```bash
# Lancer en mode debug
medical-dataset-processor web --debug

# Vérifier les logs
tail -f logs/medical_dataset_processor_*.log
```

### Vérifications de Base

1. **Connectivité API**
   ```bash
   curl -H "Authorization: DeepL-Auth-Key ${DEEPL_API_KEY}" \
        https://api-free.deepl.com/v2/usage
   ```

2. **Configuration datasets**
   ```bash
   medical-dataset-processor validate --datasets-config datasets.yaml
   ```

3. **Santé du service**
   ```bash
   curl http://localhost:5000/api/health
   ```

## Support et Contribution

- **Issues** : Signaler les bugs via GitHub Issues
- **Documentation** : Consulter `/docs` pour plus de détails
- **Tests** : Lancer `pytest tests/` pour valider l'installation

---

*Guide mis à jour pour la version 0.1.0 du Medical Dataset Processor*