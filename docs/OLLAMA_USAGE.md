# Utilisation d'Ollama avec Medical Dataset Processor

Ce guide détaille l'utilisation d'Ollama avec CroissantLM pour le traitement local des datasets médicaux, permettant d'éviter les coûts des APIs cloud.

## Avantages du traitement local

- **Économies** : Pas de coûts d'API
- **Confidentialité** : Traitement local, données restent sur votre machine
- **Contrôle** : Paramètres entièrement configurables
- **Performance** : Pas de latence réseau (selon votre matériel)

## Prérequis

### Matériel recommandé

- **RAM** : 8GB minimum, 16GB recommandé
- **Stockage** : 4GB d'espace libre pour le modèle
- **CPU** : Processeur moderne (Intel i5/AMD Ryzen 5 ou supérieur)
- **GPU** : Optionnel mais recommandé pour de meilleures performances

### Logiciels requis

- **Ollama** : https://ollama.ai/
- **Python 3.8+** : Avec les dépendances du projet

## Installation

### 1. Installer Ollama

#### macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
1. Télécharger depuis https://ollama.ai/
2. Installer l'exécutable
3. Démarrer Ollama

### 2. Démarrer le serveur Ollama

```bash
# Démarrer le serveur
ollama serve

# Vérifier que le serveur fonctionne
curl http://localhost:11434/api/tags
```

### 3. Créer le modèle CroissantLM depuis le Modelfile

```bash
# Créer le modèle avec le template optimisé (peut prendre plusieurs minutes)
ollama create croissantlm Modelfile

# Vérifier l'installation
ollama list
```

### 4. Vérifier l'intégration

```bash
# Vérifier la configuration complète
medical-dataset-processor ollama --check

# Tester avec un exemple simple
python tests/test_ollama_integration.py
```

## Utilisation

### Commandes de base

#### Vérification de la configuration
```bash
medical-dataset-processor ollama --check
```

#### Création automatique du modèle depuis le Modelfile
```bash
medical-dataset-processor ollama --setup
```

#### Affichage des instructions
```bash
medical-dataset-processor ollama --instructions
```

### Traitement de datasets

#### Traitement simple avec Ollama
```bash
medical-dataset-processor process --use-ollama
```

#### Traitement avec options personnalisées
```bash
medical-dataset-processor process \
  --use-ollama \
  --ollama-model croissantlm \
  --ollama-url http://localhost:11434 \
  --translation-count 25 \
  --generation-count 25 \
  --output-dir ./output_ollama
```

#### Traitement de datasets spécifiques
```bash
medical-dataset-processor process \
  --use-ollama \
  --datasets-config datasets.yaml \
  --translation-count 10 \
  --generation-count 10
```

## Configuration

### Options Ollama disponibles

| Option | Description | Défaut |
|--------|-------------|---------|
| `--use-ollama` | Activer le mode Ollama | `False` |
| `--ollama-model` | Nom du modèle à utiliser | `croissantlm` |
| `--ollama-url` | URL du serveur Ollama | `http://localhost:11434` |

### Configuration avancée

#### Variables d'environnement
```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=croissantlm
```

#### Configuration dans le code
```python
from medical_dataset_processor import PipelineConfig

config = PipelineConfig(
    use_ollama=True,
    ollama_model_name="croissantlm",
    ollama_base_url="http://localhost:11434",
    translation_count=50,
    generation_count=50
)
```

## Performance et optimisation

### Paramètres de performance

Le processeur Ollama utilise des paramètres optimisés pour le traitement local :

- **Batch size** : 5 (plus petit que les APIs cloud)
- **Timeout** : 60 secondes par requête
- **Retries** : 3 tentatives en cas d'échec
- **Temperature** : 0.0 pour la traduction (déterministe), 0.7 pour la génération

### Optimisation des performances

#### Ajuster la mémoire
```bash
# Allouer plus de mémoire à Ollama
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/path/to/models
```

#### Utiliser un GPU (si disponible)
```bash
# Vérifier la disponibilité GPU
ollama run croissantlm "Test GPU" --gpu

# Forcer l'utilisation GPU
export OLLAMA_GPU_LAYERS=35
```

#### Optimiser le batch size
```python
# Dans votre configuration
config = PipelineConfig(
    use_ollama=True,
    batch_size=3,  # Réduire si problèmes de mémoire
    max_retries=2
)
```

## Dépannage

### Problèmes courants

#### 1. Ollama non installé
```bash
# Erreur : "ollama: command not found"
curl -fsSL https://ollama.ai/install.sh | sh
```

#### 2. Serveur Ollama non démarré
```bash
# Erreur : "Cannot connect to Ollama server"
ollama serve
```

#### 3. Modèle non disponible
```bash
# Erreur : "Model 'croissantlm' not available"
ollama create croissantlm Modelfile
```

#### 4. Mémoire insuffisante
```bash
# Erreur : "out of memory"
# Réduire le batch size ou fermer d'autres applications
```

#### 5. Timeout des requêtes
```bash
# Erreur : "Ollama API request timed out"
# Augmenter le timeout ou réduire la charge
```

### Logs et débogage

#### Activer les logs détaillés
```bash
medical-dataset-processor process --use-ollama --verbose
```

#### Vérifier les logs Ollama
```bash
# Logs du serveur Ollama
ollama serve --verbose

# Logs des requêtes
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"croissantlm","prompt":"Test"}'
```

#### Test de connectivité
```bash
# Test simple
curl http://localhost:11434/api/tags

# Test avec modèle
ollama run croissantlm "Bonjour, comment allez-vous ?"
```

## Comparaison avec les APIs cloud

| Aspect | Ollama (Local) | APIs Cloud |
|--------|----------------|------------|
| **Coût** | Gratuit | Payant par requête |
| **Latence** | Variable (matériel) | Faible |
| **Confidentialité** | Excellente | Dépend du fournisseur |
| **Fiabilité** | Dépend de votre infrastructure | Élevée |
| **Performance** | Dépend du matériel | Optimisée |
| **Maintenance** | Vous gérez | Gérée par le fournisseur |

## Cas d'usage recommandés

### Utiliser Ollama quand :
- Vous traitez de petits à moyens volumes de données
- La confidentialité est importante
- Vous voulez contrôler les coûts
- Vous avez un matériel suffisant

### Utiliser les APIs cloud quand :
- Vous traitez de gros volumes
- Vous avez besoin de haute disponibilité
- Vous n'avez pas de matériel suffisant
- La vitesse est critique

## Support et ressources

### Documentation officielle
- [Ollama Documentation](https://ollama.ai/docs)
- [CroissantLM Model](https://ollama.ai/library/croissantlm)

### Communauté
- [Ollama Discord](https://discord.gg/ollama)
- [GitHub Issues](https://github.com/ollama/ollama/issues)

### Outils utiles
- [Ollama Web UI](https://github.com/ollama-webui/ollama-webui)
- [Ollama Desktop](https://github.com/ollama/ollama-desktop)
