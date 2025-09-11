import os, re, json, time, pathlib, requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

#  PATHS
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "medgemma-4b-it-mlx"

DOC_PATH = pathlib.Path("data/diabetes_rules.md")
SYSTEM_PROMPT_PATH = pathlib.Path("data/system_prompt.md")
IMAGE_FOLDER = Path("data/diabetes_images")
CDR_GROUND_TRUTH_DIR = Path("data/cdss_recommendations")
PATIENT_DATA_DIR = Path("data")

# PARAMETERS
MODEL_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 800,
    "top_p": 0.9,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.2
}

MAX_CHARS_PER_CALL = 50000

FEW_SHOT_EXAMPLES = {
    "normal_example": "48367822",  # A1C=5.5% (Normal)
    "diabetes_example": "LDME"  # A1C=6.7% (Diabetes)
}

TEST_PATIENTS = {
    "normal_test": "48367727",  # A1C=5.6% (Normal)
    "diabetes_test": "CDME"  # A1C=7.0% (Diabetes)
}

# RAG SYSTEM

class SimpleRAG:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.vectors = None

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence end
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > start + self.chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks

    def build_index(self, full_text: str):
        self.chunks = self.chunk_text(full_text)
        self.vectors = self.vectorizer.fit_transform(self.chunks)
        print(f" Created {len(self.chunks)} chunks")

    def get_relevant_chunks(self, query: str, top_k=6) -> str:
        if self.vectors is None:
            return ""

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                relevant_chunks.append(self.chunks[idx])

        return "\n\n".join(relevant_chunks)


# Initialize RAG system
rag = SimpleRAG()

#  SYSTEM PROMPT

def load_system_prompt() -> str:
    return read_if_exists(SYSTEM_PROMPT_PATH)

#  UTILITY FUNCTIONS
def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def read_if_exists(p: Path) -> str:
    return clean_text(p.read_text(encoding="utf-8")) if p.exists() else ""


def lm_studio_ocr_image(image_path: Path) -> str:
    try:
        url = 'https://api.ocr.space/parse/image'
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            data = {
                'apikey': 'helloworld',
                'language': 'eng',
                'detectOrientation': 'true',
                'scale': 'true',
                'OCREngine': '2'
            }
            r = requests.post(url, files=files, data=data, timeout=30)

        if r.status_code == 200:
            result = r.json()
            if result.get('ParsedResults'):
                text = result['ParsedResults'][0]['ParsedText']
                return text.strip() if text else ""
        return ""
    except Exception:
        return ""


def ocr_folder(folder: Path) -> str:
    if not folder.exists():
        return ""
    parts: List[str] = ["# OCR FROM IMAGES"]
    for img in sorted(folder.glob("*")):
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            continue
        try:
            Image.open(img).verify()
        except Exception as e:
            parts.append(f"## {img.name}\n[Skip: not a valid image: {e}]")
            continue
        try:
            txt = lm_studio_ocr_image(img)
            txt = clean_text(txt) if txt else ""
            parts.append(f"## {img.name}\n{(txt or '[No text extracted]')}")
        except Exception as e:
            parts.append(f"## {img.name}\n[OCR error: {e}]")
    return "\n\n".join(parts).strip()


def load_patient_data(patient_id: str) -> Dict[str, Any]:
    file_path = PATIENT_DATA_DIR / f"{patient_id}.diabetes_cdss.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def load_cdr_ground_truth(patient_id: str) -> Dict[str, Any]:
    file_path = CDR_GROUND_TRUTH_DIR / f"{patient_id}.cdss_recommendations.json"
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_patient_summary(patient_data: Dict) -> str:
    summary = patient_data.get("diabetes_summary", {})

    a1c_val = summary.get("a1c", {}).get("value", "N/A") if summary.get("a1c") else "N/A"

    glucose_data = summary.get("glucose", {})
    fpg_val = "N/A"
    if glucose_data and glucose_data.get("fasting_plasma"):
        fpg_val = glucose_data["fasting_plasma"].get("value", "N/A")

    lipids_data = summary.get("lipids", {})
    ldl_val = "N/A"
    if lipids_data and lipids_data.get("ldl"):
        ldl_val = lipids_data["ldl"].get("value", "N/A")

    return f"""Patient ID: {patient_data.get("patient", {}).get("id", "Unknown")}
A1C: {a1c_val}%
Fasting Glucose: {fpg_val} mg/dL
LDL Cholesterol: {ldl_val} mg/dL"""


def extract_cdr_key_info(cdr_data: Dict) -> Dict[str, Any]:
    classification = "normal"
    for alert in cdr_data.get('clinical_alerts', []):
        if 'Diabetes diagnosed' in alert['message']:
            classification = "diabetes"
            break
        elif 'Prediabetes' in alert['message']:
            classification = "prediabetes"
            break

    medications = [med['medication_class'] for med in cdr_data.get('medication_recommendations', [])]

    return {
        'classification': classification,
        'medications': medications,
        'summary': cdr_data.get('summary', '')
    }


def call_medgemma(user_prompt: str, system_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **MODEL_PARAMS
    }

    try:
        r = requests.post(LM_STUDIO_URL, json=payload, timeout=90)

        if r.status_code != 200:
            return f"Error: LM Studio returned status {r.status_code}: {r.text}"

        data = r.json()
        if 'choices' in data and len(data['choices']) > 0:
            response = data['choices'][0]['message']['content']
            return response.strip()

        return f"Unexpected response format: {data}"

    except Exception as e:
        return f"Error calling MedGemma: {e}"

# 6. RAG FUNCTIONS
def setup_rag_system():
    guidelines_text = read_if_exists(DOC_PATH)
    ocr_text = ocr_folder(IMAGE_FOLDER)
    full_text = f"{guidelines_text}\n\n{ocr_text}"

    rag.build_index(full_text)
    return rag


def zero_shot_evaluation_with_rag(patient_id: str, system_prompt: str):
    print(f"\n WITHOUT EXAMPLES evaluation for patient {patient_id}")

    patient_data = load_patient_data(patient_id)
    patient_summary = extract_patient_summary(patient_data)

    # Create query from patient data for retrieval
    query = f"diabetes diagnosis A1C {patient_data['diabetes_summary']['a1c']['value']} glucose classification treatment"

    # Get relevant guidelines chunks
    # UPDATED LINE: inject ADA A1C thresholds up front, then append retrieved chunks (top_k=6)
    relevant_guidelines = ("Diagnostic thresholds (ADA 2023): Normal A1C < 5.7%; Prediabetes 5.7–6.4%; Diabetes ≥ 6.5%.\n\n"
                           + rag.get_relevant_chunks(query, top_k=6))

    user_prompt = f"""Based on the relevant diabetes guidelines, analyze this patient:

RELEVANT GUIDELINES:
{relevant_guidelines}

PATIENT DATA:
{patient_summary}

Provide:
1. CLASSIFICATION: normal/prediabetes/diabetes
2. RATIONALE: Brief reasoning based on guidelines
3. MEDICATIONS: Recommended medications or "none"  
4. MANAGEMENT: Top 3 priorities"""

    response = call_medgemma(user_prompt, system_prompt)

    print(f"Response for {patient_id}:")
    print(response)

    return response


def create_few_shot_context_with_rag() -> str:

    # Get relevant guidelines for examples
    normal_query = "normal A1C 5.5 classification"
    diabetes_query = "diabetes A1C 6.7 treatment medication metformin"

    normal_guidelines = rag.get_relevant_chunks(normal_query, top_k=3)
    diabetes_guidelines = rag.get_relevant_chunks(diabetes_query, top_k=3)

    # Load examples
    normal_data = load_patient_data(FEW_SHOT_EXAMPLES["normal_example"])
    diabetes_data = load_patient_data(FEW_SHOT_EXAMPLES["diabetes_example"])

    few_shot_context = f"""Guidelines: {normal_guidelines}

EXAMPLE 1: Normal Patient
{extract_patient_summary(normal_data)}
OUTPUT: CLASSIFICATION: normal, RATIONALE: A1C 5.5% is below 5.7% threshold per ADA guidelines, MEDICATIONS: none, MANAGEMENT: routine preventive care, lifestyle counseling, annual screening

Guidelines: {diabetes_guidelines}

EXAMPLE 2: Diabetes Patient  
{extract_patient_summary(diabetes_data)}
OUTPUT: CLASSIFICATION: diabetes, RATIONALE: A1C 6.7% meets diabetes criteria (≥6.5%) per ADA 2023, MEDICATIONS: metformin first-line therapy and ACE inhibitor for kidney protection, MANAGEMENT: comprehensive diabetes care plan, A1C target <7%, diabetes education referral

Now analyze new patient:"""

    return few_shot_context


def get_user_scenario() -> str:
    scenario = "Analyze this patient according to ADA 2023 guidelines and provide treatment recommendations"
    print(f" Using scenario: {scenario}")
    return scenario


def test_with_user_scenario_rag(patient_id: str, user_scenario: str, system_prompt: str):
    """Test patient with user-provided scenario using RAG"""

    patient_data = load_patient_data(patient_id)
    patient_summary = extract_patient_summary(patient_data)

    # Create query for RAG
    query = f"{user_scenario} diabetes A1C {patient_data['diabetes_summary']['a1c']['value']} treatment"

    # Get relevant chunks
    relevant_guidelines = rag.get_relevant_chunks(query, top_k=3)

    # Create few-shot context with RAG
    few_shot_context = create_few_shot_context_with_rag()

    # Create final prompt
    user_prompt = f"""{few_shot_context}

NEW SCENARIO: "{user_scenario}"
RELEVANT GUIDELINES:
{relevant_guidelines}

PATIENT DATA:
{patient_summary}

Provide your analysis in the same format as the examples above:"""

    response = call_medgemma(user_prompt, system_prompt)

    print(f"\nWITH EXAMPLES response for {patient_id} with scenario: '{user_scenario}'")
    print("=" * 60)
    print(response)
    print("-" * 60)

    return response


# ===========================
# 7. MAIN EXECUTION
# ===========================

def main():
    print("Starting MedGemma CDSS with RAG")
    print("=" * 60)

    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"medgemma_test_results_{timestamp}.txt"

    # Initialize results storage
    all_results = []

    system_prompt = load_system_prompt()
    if not system_prompt:
        print("Error: Could not load system prompt")
        return

    # Setup RAG system
    print("Building vector index...")
    setup_rag_system()

    # Test patients
    for patient_type, patient_id in TEST_PATIENTS.items():
        print(f"\n{'=' * 60}")
        print(f"TESTING {patient_type.upper()}: {patient_id}")
        print(f"{'=' * 60}")

        patient_results = {
            "patient_type": patient_type,
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Get automated scenario for few-shot
            user_scenario = get_user_scenario()
            patient_results["user_scenario"] = user_scenario

            # WITH TRAINING EXAMPLES (FIRST)
            print("Running evaluation WITH training examples...")
            with_examples_response = test_with_user_scenario_rag(patient_id, user_scenario, system_prompt)
            patient_results["with_examples_response"] = with_examples_response

            # WITHOUT TRAINING EXAMPLES (SECOND)
            print("Running evaluation WITHOUT training examples...")
            without_examples_response = zero_shot_evaluation_with_rag(patient_id, system_prompt)
            patient_results["without_examples_response"] = without_examples_response

            # Show ground truth
            cdr_data = load_cdr_ground_truth(patient_id)
            cdr_info = extract_cdr_key_info(cdr_data)
            patient_results["ground_truth"] = cdr_info

            print(f"\n Ground Truth: {cdr_info['classification']}")
            print(f"Medications: {cdr_info['medications']}")

        except Exception as e:
            print(f" Error: {e}")
            patient_results["error"] = str(e)

        # Add to results
        all_results.append(patient_results)

    # Save all results to file
    print(f"\n Saving results to {output_file}")
    with open(output_file, 'w') as f:
        f.write("MEDGEMMA CDSS TEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Model: {MODEL_NAME}\n\n")

        for i, result in enumerate(all_results, 1):
            f.write(f"\n{'=' * 50}\n")
            f.write(f"TEST {i}: {result['patient_type'].upper()}\n")
            f.write(f"Patient ID: {result['patient_id']}\n")
            f.write(f"{'=' * 50}\n")

            if "error" in result:
                f.write(f"ERROR: {result['error']}\n")
                continue

            f.write(f"\nUSER SCENARIO: {result.get('user_scenario', 'N/A')}\n")

            f.write(f"\nWITH EXAMPLES RESPONSE:\n{'-' * 30}\n")
            f.write(f"{result.get('with_examples_response', 'No response')}\n")

            f.write(f"\nWITHOUT EXAMPLES RESPONSE:\n{'-' * 30}\n")
            f.write(f"{result.get('without_examples_response', 'No response')}\n")

            f.write(f"\nGROUND TRUTH:\n{'-' * 30}\n")
            gt = result.get('ground_truth', {})
            f.write(f"Classification: {gt.get('classification', 'N/A')}\n")
            f.write(f"Medications: {gt.get('medications', 'N/A')}\n")
            f.write(f"Summary: {gt.get('summary', 'N/A')}\n")

    print(f" Results saved to: {output_file}")
    print(" RAG evaluation complete!")


if __name__ == "__main__":
    main()
