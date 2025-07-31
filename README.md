# Medical Dataset Processor

Un système automatisé pour le traitement de datasets médicaux combinant traduction automatique via DeepL et génération de contenu via OpenAI GPT-4o-mini.

## Vue d'ensemble

Ce package automatise le traitement de datasets médicaux en:
- Récupérant des datasets médicaux depuis Hugging Face
- Traduisant 50 échantillons par dataset via l'API DeepL
- Générant du contenu pour 50 autres échantillons via OpenAI GPT-4o-mini
- Consolidant les résultats au format JSONL
- Créant un échantillon PDF pour relecture

## Datasets supportés

- **MedQA**: Questions-réponses médicales cliniques
- **PubMedQA**: Questions-réponses basées sur PubMed
- **HealthSearchQA**: Questions-réponses sur la santé
- **MMLU Clinical**: Connaissances cliniques du benchmark MMLU

## Installation

### Prérequis

- Python 3.8+
- Clé API DeepL (https://www.deepl.com/pro-api)
- Clé API OpenAI (https://platform.openai.com/api-keys)

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

```bash
# Traitement complet avec configuration par défaut
medical-dataset-processor

# Spécifier des fichiers de configuration personnalisés
medical-dataset-processor --config datasets.yaml --output-dir ./output

# Traitement avec options avancées
medical-dataset-processor \
  --datasets medqa,pubmedqa \
  --translation-samples 25 \
  --generation-samples 25 \
  --target-language fr \
  --pdf-samples 50
```

### Options de la CLI

- `--config`: Fichier de configuration YAML (défaut: `datasets.yaml`)
- `--output-dir`: Répertoire de sortie (défaut: `./output`)
- `--datasets`: Datasets à traiter (défaut: tous)
- `--translation-samples`: Nombre d'échantillons à traduire par dataset (défaut: 50)
- `--generation-samples`: Nombre d'échantillons à générer par dataset (défaut: 50)
- `--target-language`: Langue cible pour la traduction (défaut: fr)
- `--pdf-samples`: Nombre d'échantillons dans le PDF (défaut: 100)
- `--parallel`: Activer le traitement parallèle
- `--resume`: Reprendre un traitement interrompu

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

## Gestion d'erreurs et reprise

Le système inclut une gestion robuste des erreurs:

- **Retry automatique**: Jusqu'à 3 tentatives par échantillon
- **Rate limiting**: Gestion automatique des limites d'API
- **Sauvegarde d'état**: Reprise possible après interruption
- **Logging détaillé**: Traçabilité complète des opérations

### Reprendre un traitement interrompu

```bash
# Le système détecte automatiquement les traitements interrompus
medical-dataset-processor --resume

# Ou spécifier un fichier d'état
medical-dataset-processor --resume --state-file ./logs/processing_state.json
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