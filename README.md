# LLM-Enhanced Clinical Decision Support with FHIR

A medical AI evaluation framework that tests the MedGemma language model's ability to diagnose diabetes and recommend treatments based on patient data and clinical guidelines.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Start LM Studio
- Download and install [LM Studio](https://lmstudio.ai/)
- Load the `medgemma-4b-it-mlx` model
- Start the local server on port 1234

### 2. FHIR Data Collection
The system uses HAPI FHIR server to collect real patient data:

```bash
python cdr.py --cohort-size 10
```

This will:
- Connect to HAPI FHIR server (https://hapi.fhir.org/baseR4)
- Search for patients with diabetes-related lab values (A1C, glucose)
- Collect comprehensive clinical data following ADA 2023 guidelines
- Generate patient data files automatically

### 3. Generate Clinical Recommendations
Create ground truth recommendations using the clinical decision support system:

```bash
python cdr.py --batch-process data/
```

This processes all patient files and generates expert clinical recommendations based on ADA 2023 guidelines.

### 4. Prepare Additional Files
Ensure your `data/` directory contains:
- Medical guidelines in `diabetes_rules.md`
- System prompt in `system_prompt.md`:
```
You are a clinical decision support assistant built for licensed physicians to assist in diagnosis support and treatment recommendations. Your audience is practicing clinicians, so you may assume full medical literacy and should avoid lay explanations.
```

## Running the Evaluation

Execute the main evaluation script:

```bash
python "Medgemma 4b.py"
```

**Note**: The first run may take longer as it builds the RAG vector index from medical guidelines.

## Core Components

### 1. **Medgemma 4b.py** - Main AI Evaluation Engine
- RAG-powered guideline retrieval for each patient case
- Few-shot vs zero-shot learning comparison
- Automated testing of diabetes diagnosis accuracy
- Performance evaluation against clinical ground truth

### 2. **cdr.py** - Clinical Decision Rules Engine  
- Processes patient data through comprehensive CDSS
- Generates clinical alerts, medication recommendations, and treatment goals
- Implements ADA 2023 diabetes care standards
- Creates ground truth recommendations for AI comparison

### 3. **fhir_resources.py** - FHIR Data Collection
- Connects to HAPI FHIR server for real patient data
- Searches for diabetes-related lab values and clinical indicators
- Processes comprehensive clinical panels (A1C, glucose, lipids, kidney function)
- Follows clinical data standards for interoperability

### 4. **text_extraction_frompdf.py** - Document Processing
- Extracts text from medical PDF guidelines and documents
- Processes clinical images through OCR
- Integrates visual medical data into the knowledge base

## What It Does

- **Collects real patient data** from HAPI FHIR server with diabetes-related lab values
- **Generates clinical recommendations** using ADA 2023 guidelines through CDSS engine
- **Tests MedGemma AI model** on diabetes diagnosis and treatment decisions
- **Compares AI responses** against expert clinical decision support recommendations  
- **Uses RAG system** to retrieve relevant medical guidelines for each case
- **Validates clinical rules** against established diabetes care standards

## Test Cases

- **Normal Patient** (A1C: 5.6%) - Should classify as normal
- **Diabetes Patient** (A1C: 7.0%) - Should classify as diabetes

## Output

Results are saved to timestamped files like `medgemma_test_results_20250815_143022.txt` containing:
- Model classifications and reasoning
- Medication recommendations
- Comparison with ground truth
- Performance evaluation

## Requirements

- Python 3.8+
- LM Studio with MedGemma model
- Internet access for HAPI FHIR server
- 8GB+ RAM for model inference
