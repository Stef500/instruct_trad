# Project: instruct_trad - Medical Dataset Processor

## Architecture Overview
- **Language**: Python
- **Type**: CLI tool with optional web interface
- **Domain**: Medical dataset processing and translation
- **Deployment**: Docker-based with uv package management

## Core Components
- **Pipeline**: Medical dataset processing pipeline
- **Processors**: Generation, translation, consolidation, sampling
- **Loaders**: Dataset loading from various sources
- **Exporters**: JSONL, PDF sample generation
- **Web Interface**: Flask-based manual translation interface
- **Ollama Integration**: Local LLM for generation and translation

## Key Technologies
- **Package Manager**: uv (modern Python package manager)
- **Web Framework**: Flask
- **LLM Integration**: Ollama
- **Containerization**: Docker with multi-stage builds
- **Data Format**: JSONL for medical datasets

## Project Structure
```
src/medical_dataset_processor/
├── cli.py                    # Command-line interface
├── pipeline.py              # Main processing pipeline
├── processors/              # Data processing modules
├── loaders/                 # Dataset loading
├── exporters/              # Output generation
├── web/                    # Web interface
└── utils/                  # Utilities (logging, ollama, recovery)
```

## Recent Development Focus
- **Ollama Translation**: Added translation-only mode
- **French Support**: Bilingual processing capabilities
- **Web Interface**: Manual translation workflow
- **Docker Optimization**: Improved containerization
- **State Recovery**: Process resumption capabilities

## Configuration
- **Main Config**: datasets.yaml
- **Docker**: docker-compose.yml with environment variables
- **Examples**: docs/examples/ with various configuration patterns