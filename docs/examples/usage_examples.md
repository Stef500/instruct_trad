# Exemples d'utilisation du Medical Dataset Processor

## Exemples de ligne de commande

### Traitement basique
```bash
# Traitement avec configuration par défaut
medical-dataset-processor

# Traitement avec fichier de configuration personnalisé
medical-dataset-processor --config examples/config_minimal.yaml
```

### Traitement sélectif
```bash
# Traiter seulement certains datasets
medical-dataset-processor --datasets medqa,pubmedqa

# Traiter avec moins d'échantillons pour test rapide
medical-dataset-processor --translation-samples 10 --generation-samples 10
```

### Options avancées
```bash
# Traitement parallèle avec configuration avancée
medical-dataset-processor \
  --config examples/config_advanced.yaml \
  --parallel \
  --max-workers 3 \
  --output-dir ./results

# Reprendre un traitement interrompu
medical-dataset-processor --resume --state-file ./logs/processing_state.json
```

## Exemples programmatiques

### Utilisation basique
```python
from medical_dataset_processor import MedicalDatasetProcessor
import os

# Configuration depuis variables d'environnement
processor = MedicalDatasetProcessor(
    deepl_api_key=os.getenv("DEEPL_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Traitement complet
results = processor.process_datasets(
    config_path="datasets.yaml",
    output_dir="./output"
)

print(f"Traitement terminé: {results.total_samples} échantillons traités")
```

### Traitement par étapes
```python
from medical_dataset_processor import MedicalDatasetProcessor
from medical_dataset_processor.loaders import DatasetLoader
from medical_dataset_processor.processors import SampleSelector

# Initialisation
processor = MedicalDatasetProcessor(
    deepl_api_key="your_deepl_key",
    openai_api_key="your_openai_key"
)

# Étape 1: Charger les datasets
loader = DatasetLoader()
datasets = loader.load_datasets("datasets.yaml")

# Étape 2: Sélectionner les échantillons
selector = SampleSelector()
translation_samples = {}
generation_samples = {}

for dataset_name, dataset in datasets.items():
    translation_samples[dataset_name] = selector.select_for_translation(
        dataset, count=50
    )
    generation_samples[dataset_name] = selector.select_for_generation(
        dataset, count=50, exclude=translation_samples[dataset_name]
    )

# Étape 3: Traitement
translated_results = processor.translate_samples(translation_samples)
generated_results = processor.generate_samples(generation_samples)

# Étape 4: Consolidation et export
consolidated = processor.consolidate_datasets(translated_results, generated_results)
processor.export_jsonl(consolidated, "output/dataset.jsonl")
processor.generate_pdf_sample(consolidated, "output/sample.pdf", sample_size=100)
```

### Traitement avec gestion d'erreurs personnalisée
```python
from medical_dataset_processor import MedicalDatasetProcessor
from medical_dataset_processor.utils import ProcessingLogger
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = ProcessingLogger("custom_processing")

try:
    processor = MedicalDatasetProcessor(
        deepl_api_key="your_deepl_key",
        openai_api_key="your_openai_key",
        logger=logger
    )
    
    # Traitement avec callbacks personnalisés
    results = processor.process_datasets(
        config_path="datasets.yaml",
        output_dir="./output",
        on_sample_processed=lambda sample: print(f"Traité: {sample.id}"),
        on_error=lambda error: logger.log_error(error),
        on_progress=lambda progress: print(f"Progression: {progress}%")
    )
    
except Exception as e:
    logger.log_error(f"Erreur critique: {e}")
    # Générer rapport d'erreurs
    error_report = logger.generate_error_report()
    print(f"Rapport d'erreurs sauvegardé: {error_report.path}")
```

### Traitement parallèle personnalisé
```python
from medical_dataset_processor import MedicalDatasetProcessor
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def process_dataset_async(processor, dataset_name, config):
    """Traitement asynchrone d'un dataset"""
    try:
        result = await processor.process_dataset_async(dataset_name, config)
        return result
    except Exception as e:
        print(f"Erreur pour {dataset_name}: {e}")
        return None

async def main():
    processor = MedicalDatasetProcessor(
        deepl_api_key="your_deepl_key",
        openai_api_key="your_openai_key"
    )
    
    # Charger la configuration
    datasets_config = processor.load_config("datasets.yaml")
    
    # Traitement parallèle de tous les datasets
    tasks = [
        process_dataset_async(processor, name, config)
        for name, config in datasets_config.items()
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Consolidation des résultats
    all_results = [r for r in results if r is not None]
    consolidated = processor.consolidate_results(all_results)
    
    # Export final
    processor.export_jsonl(consolidated, "output/parallel_dataset.jsonl")

# Exécution
asyncio.run(main())
```

## Exemples de configuration

### Configuration pour test rapide
```yaml
# config_test.yaml - Pour développement et tests
medqa:
  name: "medqa"
  source_type: "huggingface"
  source_path: "bigbio/med_qa"
  subset: "med_qa_en_bigbio_qa"
  text_fields: ["question", "answer"]

processing_config:
  translation_samples_per_dataset: 5
  generation_samples_per_dataset: 5
  target_language: "fr"
  pdf_sample_size: 10
```

### Configuration pour production
```yaml
# config_production.yaml - Pour traitement complet
medqa:
  name: "medqa"
  source_type: "huggingface"
  source_path: "bigbio/med_qa"
  subset: "med_qa_en_bigbio_qa"
  text_fields: ["question", "answer"]

pubmedqa:
  name: "pubmedqa"
  source_type: "huggingface"
  source_path: "bigbio/pubmed_qa"
  subset: "pubmed_qa_labeled_fold0_bigbio_qa"
  text_fields: ["question", "context", "final_decision"]

healthsearchqa:
  name: "healthsearchqa"
  source_type: "huggingface"
  source_path: "keivalya/HealthSearchQA"
  text_fields: ["question", "answer"]

mmlu_clinical:
  name: "mmlu_clinical"
  source_type: "huggingface"
  source_path: "cais/mmlu"
  subset: "clinical_knowledge"
  text_fields: ["question", "choices", "answer"]

processing_config:
  translation_samples_per_dataset: 50
  generation_samples_per_dataset: 50
  target_language: "fr"
  output_format: "jsonl"
  pdf_sample_size: 100
  
  performance:
    max_concurrent_requests: 10
    enable_caching: true
    batch_size: 20
```

## Scripts d'automatisation

### Script de traitement par lots
```bash
#!/bin/bash
# batch_process.sh - Traitement de plusieurs configurations

configs=("config_minimal.yaml" "config_advanced.yaml")
output_base="./batch_results"

for config in "${configs[@]}"; do
    echo "Traitement avec $config..."
    output_dir="${output_base}/$(basename $config .yaml)"
    mkdir -p "$output_dir"
    
    medical-dataset-processor \
        --config "examples/$config" \
        --output-dir "$output_dir" \
        --parallel
    
    echo "Terminé: $config -> $output_dir"
done

echo "Traitement par lots terminé!"
```

### Script de monitoring
```bash
#!/bin/bash
# monitor_processing.sh - Surveillance du traitement

log_file="logs/medical_dataset_processor.log"

echo "Surveillance du traitement en cours..."
echo "Appuyez sur Ctrl+C pour arrêter"

while true; do
    if [ -f "$log_file" ]; then
        # Afficher les statistiques
        echo "=== Statistiques $(date) ==="
        grep "Processed:" "$log_file" | tail -5
        grep "ERROR" "$log_file" | wc -l | xargs echo "Erreurs totales:"
        echo ""
    fi
    sleep 30
done
```

Ces exemples couvrent les cas d'usage les plus courants et peuvent être adaptés selon vos besoins spécifiques.