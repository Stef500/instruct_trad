# Medical Dataset Processor

Un système automatisé pour le traitement de datasets médicaux combinant traduction automatique via DeepL et génération de contenu via OpenAI GPT-4o-mini, avec support optionnel d'Ollama pour le traitement local.

## 📚 Documentation

La documentation complète se trouve dans le répertoire `docs/` :

- **[Interface Web de Traduction](docs/WEB_INTERFACE_USAGE.md)** - Guide complet pour l'interface web interactive
- **[Déploiement Docker](docs/DOCKER_DEPLOYMENT.md)** - Instructions de déploiement avec Docker
- **[Exemples de Configuration](docs/examples/)** - Fichiers d'exemple et cas d'usage

## Vue d'ensemble

Ce package automatise le traitement de datasets médicaux en:
- Récupérant des datasets médicaux depuis Hugging Face
- Traduisant 50 échantillons par dataset via l'API DeepL ou Ollama (local)
- Générant du contenu pour 50 autres échantillons via OpenAI GPT-4o-mini ou Ollama (local)
- Consolidant les résultats au format JSONL
- Créant un échantillon PDF pour relecture

### Modes de traitement

- **Cloud APIs** (par défaut): Utilise DeepL et OpenAI pour le traitement
- **Local Ollama**: Utilise CroissantLM via Ollama pour un traitement local sans coûts API

## Datasets supportés

- **MedQA**: Questions-réponses médicales cliniques
- **PubMedQA**: Questions-réponses basées sur PubMed
- **HealthSearchQA**: Questions-réponses sur la santé
- **MMLU Clinical**: Connaissances cliniques du benchmark MMLU

## Installation

### Prérequis

- Python 3.12+
- Clé API DeepL (https://www.deepl.com/pro-api) - optionnel si utilisation d'Ollama
- Clé API OpenAI (https://platform.openai.com/api-keys) - optionnel si utilisation d'Ollama
- Ollama (https://ollama.ai/) - requis pour le traitement local

### Installation du package

```bash
# Cloner le repository
git clone <repository-url>
cd medical-dataset-processor

# Installer les dépendances
pip install -e .

# Ou avec uv (recommandé)
uv pip install -e .
```

### Installation d'Ollama (pour le traitement local)

Si vous souhaitez utiliser le traitement local avec Ollama :

1. **Installer Ollama** :
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Télécharger depuis https://ollama.ai/
   ```

2. **Démarrer le serveur Ollama** :
   ```bash
   ollama serve
   ```

3. **Créer le modèle CroissantLM depuis le Modelfile** :
   ```bash
   ollama create croissantlm Modelfile
   ```

4. **Vérifier l'installation** :
   ```bash
   medical-dataset-processor ollama --check
   ```

### Configuration

1. Copiez le fichier de configuration d'exemple:
```bash
cp .env.example .env
```

2. Éditez le fichier `.env` avec vos clés API:
```bash
DEEPL_API_KEY=your_deepl_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Utilisation

### Interface en ligne de commande

Le CLI est organisé en sous-commandes pour différents types d'opérations :

```bash
# Voir toutes les commandes disponibles
medical-dataset-processor --help

# Voir les options d'une commande spécifique
medical-dataset-processor process --help
```

#### Commande principale : `process`

**Traitement avec APIs cloud (par défaut)**

```bash
# Traitement complet avec configuration par défaut
medical-dataset-processor process

# Spécifier des fichiers de configuration personnalisés
medical-dataset-processor process --datasets-config datasets.yaml --output-dir ./output

# Traitement avec options avancées
medical-dataset-processor process \
  --deepl-key YOUR_DEEPL_KEY \
  --openai-key YOUR_OPENAI_KEY \
  --translation-count 25 \
  --generation-count 25 \
  --target-language FR \
  --pdf-sample-size 50
```

**Traitement local avec Ollama**

```bash
# Traitement complet avec Ollama (traduction ET génération)
medical-dataset-processor process --use-ollama

# Utilisation d'Ollama uniquement pour la traduction
medical-dataset-processor process --use-ollama-for-translation --openai-key YOUR_KEY

# Utilisation d'Ollama uniquement pour la génération
medical-dataset-processor process --use-ollama-for-generation --deepl-key YOUR_KEY

# Traitement avec Ollama et options personnalisées
medical-dataset-processor process \
  --use-ollama \
  --ollama-model croissantlm \
  --ollama-url http://localhost:11434 \
  --translation-count 25 \
  --generation-count 25
```

#### Gestion d'Ollama : `ollama`

```bash
# Vérifier la configuration d'Ollama
medical-dataset-processor ollama --check

# Créer le modèle CroissantLM depuis le Modelfile
medical-dataset-processor ollama --setup

# Afficher les instructions d'installation
medical-dataset-processor ollama --instructions
```

#### Validation : `validate`

```bash
# Valider un fichier de configuration
medical-dataset-processor validate --datasets-config datasets.yaml
```

#### Interface web : `web`

```bash
# Démarrer l'interface web de traduction
medical-dataset-processor web --mode automatic --host 0.0.0.0 --port 5000
```

#### Statistiques : `show-stats`

```bash
# Afficher les statistiques d'un traitement précédent
medical-dataset-processor show-stats output/processing_stats.json
```

#### Version : `version`

```bash
# Afficher la version
medical-dataset-processor version
```

### Options de la commande `process`

#### Options de configuration
- `--datasets-config, -d`: Fichier de configuration YAML (défaut: `datasets.yaml`)
- `--deepl-key`: Clé API DeepL (ou variable d'environnement DEEPL_API_KEY)
- `--openai-key`: Clé API OpenAI (ou variable d'environnement OPENAI_API_KEY)
- `--output-dir, -o`: Répertoire de sortie (défaut: `output`)
- `--log-file`: Fichier de log personnalisé
- `--verbose, -v`: Activer les logs détaillés

#### Options de traitement
- `--translation-count, -t`: Nombre d'échantillons à traduire par dataset (défaut: 50)
- `--generation-count, -g`: Nombre d'échantillons à générer par dataset (défaut: 50)
- `--target-language`: Langue cible pour la traduction (code DeepL, défaut: FR)
- `--pdf-sample-size`: Nombre d'échantillons dans le PDF (défaut: 100)
- `--batch-size`: Taille des lots pour les requêtes API (défaut: 10)
- `--max-retries`: Nombre maximum de tentatives (défaut: 3)
- `--random-seed`: Graine aléatoire pour la reproductibilité

#### Options Ollama
- `--use-ollama`: Utiliser Ollama pour la traduction ET la génération
- `--use-ollama-for-translation`: Utiliser Ollama uniquement pour la traduction
- `--use-ollama-for-generation`: Utiliser Ollama uniquement pour la génération
- `--ollama-model`: Nom du modèle Ollama (défaut: `croissantlm`)
- `--ollama-url`: URL du serveur Ollama (défaut: `http://localhost:11434`)

#### Options de fichiers de sortie
- `--jsonl-filename`: Nom du fichier JSONL de sortie (défaut: `consolidated_dataset.jsonl`)
- `--pdf-filename`: Nom du fichier PDF de sortie (défaut: `sample_review.pdf`)
- `--stats-file`: Sauvegarder les statistiques dans un fichier JSON

#### Options spéciales
- `--dry-run`: Valider la configuration sans traiter
- `--version`: Afficher la version

### Utilisation programmatique

```python
from medical_dataset_processor import MedicalDatasetProcessor

# Initialisation
processor = MedicalDatasetProcessor(
    deepl_api_key="your_deepl_key",
    openai_api_key="your_openai_key"
)

# Traitement complet
results = processor.process_datasets(
    config_path="datasets.yaml",
    output_dir="./output"
)

# Traitement par étapes
datasets = processor.load_datasets("datasets.yaml")
translated = processor.translate_samples(datasets, samples_per_dataset=50)
generated = processor.generate_samples(datasets, samples_per_dataset=50)
consolidated = processor.consolidate_datasets(translated, generated)
processor.export_jsonl(consolidated, "output/dataset.jsonl")
processor.generate_pdf_sample(consolidated, "output/sample.pdf")
```

## Configuration

### Fichier datasets.yaml

Le fichier `datasets.yaml` définit les datasets à traiter:

```yaml
medqa:
  name: "medqa"
  source_type: "huggingface"
  source_path: "bigbio/med_qa"
  subset: "med_qa_en_bigbio_qa"
  text_fields: ["question", "answer"]
  description: "Medical Question Answering dataset"

processing_config:
  translation_samples_per_dataset: 50
  generation_samples_per_dataset: 50
  target_language: "fr"
  output_format: "jsonl"
```

### Variables d'environnement

Toutes les options peuvent être configurées via des variables d'environnement:

```bash
# APIs
DEEPL_API_KEY=your_key
OPENAI_API_KEY=your_key
TARGET_LANGUAGE=fr
OPENAI_MODEL=gpt-4o-mini

# Sortie
OUTPUT_JSONL_PATH=output/dataset.jsonl
OUTPUT_PDF_PATH=output/sample.pdf

# Performance
MAX_CONCURRENT_REQUESTS=5
MAX_RETRIES=3
REQUEST_TIMEOUT=30
```

## Format de sortie

### Fichier JSONL

Chaque ligne du fichier JSONL contient un échantillon traité:

```json
{
  "id": "medqa_001",
  "source_dataset": "medqa",
  "processing_type": "translation",
  "original_text": "What is the most common cause of...",
  "processed_text": "Quelle est la cause la plus fréquente de...",
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "api_used": "deepl",
    "confidence_score": 0.95
  }
}
```

### Échantillon PDF

Le PDF contient 100 échantillons aléatoires formatés pour relecture humaine avec:
- Texte original
- Texte traduit/généré
- Métadonnées de traitement
- Scores de qualité

## Cas d'usage et configurations

### Configuration mixte (recommandée)

Pour optimiser les coûts et la qualité, vous pouvez utiliser une configuration mixte :

- **Ollama pour la traduction** : Économies sur les coûts DeepL, qualité suffisante pour la plupart des cas
- **OpenAI pour la génération** : Meilleure qualité de génération de contenu

```bash
medical-dataset-processor process --use-ollama-for-translation --openai-key YOUR_KEY
```

### Traitement local complet

Pour une confidentialité maximale et des coûts nuls :

```bash
medical-dataset-processor process --use-ollama
```

### Traitement cloud complet

Pour une qualité et une fiabilité maximales :

```bash
medical-dataset-processor process --deepl-key YOUR_KEY --openai-key YOUR_KEY
```

## Gestion d'erreurs et reprise

Le système inclut une gestion robuste des erreurs:

- **Retry automatique**: Jusqu'à 3 tentatives par échantillon
- **Rate limiting**: Gestion automatique des limites d'API
- **Sauvegarde d'état**: Reprise possible après interruption
- **Logging détaillé**: Traçabilité complète des opérations

### Reprendre un traitement interrompu

Le système sauvegarde automatiquement les résultats partiels. Pour consulter l'état d'un traitement précédent, vous pouvez :

```bash
# Consulter les logs de traitement
tail -f logs/medical_dataset_processor.log

# Afficher les statistiques d'un traitement précédent
medical-dataset-processor show-stats output/processing_stats.json
```

## Logs et monitoring

Les logs sont sauvegardés dans `./logs/` avec:
- Progression du traitement
- Erreurs d'API détaillées
- Statistiques de performance
- Rapports de qualité

```bash
# Voir les logs en temps réel
tail -f logs/medical_dataset_processor.log

# Analyser les erreurs
grep "ERROR" logs/medical_dataset_processor.log
```

## Développement

### Structure du projet

```
src/medical_dataset_processor/
├── __init__.py
├── cli.py                 # Interface ligne de commande
├── pipeline.py           # Orchestrateur principal
├── models/
│   └── core.py          # Modèles de données
├── loaders/
│   └── dataset_loader.py # Chargement des datasets
├── processors/
│   ├── translation_processor.py
│   ├── generation_processor.py
│   ├── sample_selector.py
│   └── dataset_consolidator.py
├── exporters/
│   ├── jsonl_exporter.py
│   └── pdf_sample_generator.py
└── utils/
    ├── logging.py
    └── state_recovery.py
```

### Tests

```bash
# Lancer tous les tests
pytest

# Tests avec couverture
pytest --cov=src/medical_dataset_processor

# Tests d'intégration uniquement
pytest tests/test_*_integration.py
```

### Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Support

Pour des questions ou problèmes:
1. Consulter la documentation
2. Vérifier les issues existantes
3. Créer une nouvelle issue avec les détails du problème