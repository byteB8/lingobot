# LingoBot - Healthcare Information Assistant

A domain-specific chatbot that provides general health FAQs, lifestyle tips, and hospital information using RAG (Retrieval-Augmented Generation) architecture.

## Overview

LingBot is a healthcare information assistant that helps users with:
- General health questions and FAQs
- Healthy lifestyle tips and recommendations
- Hospital information and services
- Educational health guidance

The system uses intent classification, vector search, and LLM generation to provide accurate, context-aware responses while maintaining safety through disclaimers and content filtering.

## Architecture

- **Intent Classification**: TF-IDF based classification to route queries
- **Vector Search**: FAISS indices for semantic search of FAQ and tips
- **Hospital Lookup**: Direct keyword-based lookup for hospital information
- **LLM Generation**: Qwen2-1.5B model for response generation
- **Safety Filtering**: Content validation and educational disclaimers

## Data Sources

- **FAQ Data**: 16,412 medical Q&A pairs from MedQuad dataset
- **Lifestyle Tips**: 79 health and wellness tips from WHO, CDC, NIH
- **Hospital Data**: Lingo Healthcare facility information and services

## Installation

1. **Create virtual environment:**
```bash
python3 -m venv lingoEnv
```

2. **Activate virtual environment:**
```bash
source lingoEnv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

1. **Preprocess data:**
```bash
python3 src/data_processing/data_preprocessor.py
```

2. **Create vector indices:**
```bash
python3 scripts/create_indices.py
```

3. **Run the application:**
```bash
python3 main.py
```

4. **Access interface:**
Open browser to `http://localhost:7860`

## Example Queries

- "What are the symptoms of flu?"
- "How much sleep should I get?"
- "What are the hospital's working hours?"
- "Give me tips for healthy eating"
- "How to book an appointment?"

## Project Structure

```
task1/
├── src/
│   ├── data_processing/     # Data preprocessing and indexing
│   ├── pipeline/           # Intent classification and routing
│   └── deployment/         # Gradio interface
├── data/                   # Processed data and indices
├── scripts/               # Utility scripts
├── main.py               # Application entry point
└── requirements.txt      # Dependencies
```

## License

This project is for educational purposes only and not a substitute for professional medical advice.
