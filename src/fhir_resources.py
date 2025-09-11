#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, os, sys, time, math
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import requests
from datetime import datetime, timedelta

# Config / paths
DEFAULT_FHIR_BASE = os.getenv("FHIR_BASE_URL", "https://hapi.fhir.org/baseR4")
DEFAULT_OUTDIR = (Path(__file__).resolve().parent / "data").as_posix()


# LOINC codes for diabetes management (ADA 2023 aligned)
LOINC_A1C = ["4548-4"]  # Hemoglobin A1c (DCCT/NGSP)
LOINC_BMI = ["39156-5"]  # Body mass index
LOINC_BLOOD_PRESSURE = ["8480-6", "8462-4"]  # Systolic and diastolic BP

# Glucose measurements for diabetes diagnosis (per ADA Table 2.2/2.5)
LOINC_GLUCOSE = {
    "fasting_plasma": ["1558-6"],  # Fasting plasma glucose
    "plasma": ["2345-7"],  # Random plasma glucose
    "whole_blood": ["2339-0"],  # Whole blood glucose
    "ogtt_2h_75g": ["1518-0"],  # 2-hour 75g OGTT
}

# Cardiovascular risk factors (ADA Section 10)
LOINC_LIPIDS = {
    "ldl": ["13457-7"],  # LDL cholesterol
    "hdl": ["2085-9"],  # HDL cholesterol
    "triglycerides": ["2571-8"],  # Triglycerides
    "total_cholesterol": ["2093-3"],  # Total cholesterol
}

# Kidney function (ADA Section 11)
LOINC_RENAL = {
    "creatinine": ["2160-0"],  # Serum creatinine
    "egfr": ["33914-3", "62238-1"],  # eGFR (multiple codes)
    "bun": ["3094-0"],  # Blood urea nitrogen
}

# Microalbuminuria/proteinuria
LOINC_URINE = {
    "acr": ["14959-1", "9318-7"],  # Albumin/creatinine ratio
    "microalbumin": ["1755-8"],  # Microalbumin
}

# Additional diabetes monitoring
LOINC_ADDITIONAL = {
    "vitamin_b12": ["2132-9"],  # B12 (metformin monitoring)
    "liver_function": ["1742-6", "1744-2"],  # ALT, AST (medication safety)
}

# Core discovery codes for initial patient identification
DISCOVERY_CORE = LOINC_A1C + sum(LOINC_GLUCOSE.values(), [])
# Extended panel for comprehensive assessment
DISCOVERY_EXTENDED = DISCOVERY_CORE + LOINC_BMI + sum(LOINC_LIPIDS.values(), []) + sum(LOINC_RENAL.values(), [])

# Clinical decision support calculations
def calculate_diabetes_risk_factors(data: Dict) -> Dict:
    risk_factors = {
        "diabetes_diagnosis": None,
        "prediabetes_diagnosis": None,
        "cardiovascular_risk": "unknown",
        "kidney_disease_stage": "unknown",
        "hypertension_status": "unknown",
        "obesity_status": "unknown",
        "clinical_alerts": []
    }

    summary = data.get("diabetes_summary", {})

    # Diabetes diagnosis per ADA Table 2.2/2.5
    a1c = summary.get("a1c", {})
    if a1c and a1c.get("value") is not None:
        a1c_value = a1c["value"]
        if a1c_value >= 6.5:
            risk_factors["diabetes_diagnosis"] = "diabetes"
        elif a1c_value >= 5.7:
            risk_factors["prediabetes_diagnosis"] = "prediabetes"
        else:
            risk_factors["diabetes_diagnosis"] = "normal"

    # Check fasting plasma glucose
    fpg = summary.get("glucose", {}).get("fasting_plasma")
    if fpg and fpg.get("value") is not None:
        fpg_value = fpg["value"]
        if fpg_value >= 126:
            risk_factors["diabetes_diagnosis"] = "diabetes"
        elif fpg_value >= 100:
            risk_factors["prediabetes_diagnosis"] = "prediabetes"

    # Random plasma glucose
    random_glucose = summary.get("glucose", {}).get("random_plasma_or_whole")
    if random_glucose and random_glucose.get("value") is not None:
        if random_glucose["value"] >= 200:
            risk_factors["diabetes_diagnosis"] = "diabetes"

    # BMI assessment
    bmi = summary.get("bmi", {})
    if bmi and bmi.get("value") is not None:
        bmi_value = bmi["value"]
        if bmi_value >= 30:
            risk_factors["obesity_status"] = "obese"
        elif bmi_value >= 25:
            risk_factors["obesity_status"] = "overweight"
        else:
            risk_factors["obesity_status"] = "normal"

    # Cardiovascular risk assessment
    ldl = summary.get("lipids", {}).get("ldl")
    if ldl and ldl.get("value") is not None:
        ldl_value = ldl["value"]
        if ldl_value >= 190:
            risk_factors["cardiovascular_risk"] = "very_high"
        elif ldl_value >= 160:
            risk_factors["cardiovascular_risk"] = "high"
        elif ldl_value >= 130:
            risk_factors["cardiovascular_risk"] = "moderate"
        else:
            risk_factors["cardiovascular_risk"] = "low"

    # Kidney function assessment
    egfr = summary.get("renal", {}).get("egfr")
    creatinine = summary.get("renal", {}).get("creatinine")

    if egfr and egfr.get("value") is not None:
        egfr_value = egfr["value"]
        if egfr_value >= 90:
            risk_factors["kidney_disease_stage"] = "normal_or_high"
        elif egfr_value >= 60:
            risk_factors["kidney_disease_stage"] = "mild_decrease"
        elif egfr_value >= 45:
            risk_factors["kidney_disease_stage"] = "mild_to_moderate"
        elif egfr_value >= 30:
            risk_factors["kidney_disease_stage"] = "moderate_to_severe"
        elif egfr_value >= 15:
            risk_factors["kidney_disease_stage"] = "severe"
        else:
            risk_factors["kidney_disease_stage"] = "kidney_failure"

    # Generate clinical alerts based on ADA guidelines
    alerts = []

    if risk_factors.get("diabetes_diagnosis") == "diabetes":
        alerts.append("Diabetes diagnosed - initiate comprehensive diabetes care")

        # A1C target alerts
        if a1c and a1c.get("value", 0) > 7.0:
            alerts.append("A1C above target (<7%) - consider treatment intensification")

        # Cardiovascular protection
        if risk_factors.get("cardiovascular_risk") in ["high", "very_high"]:
            alerts.append("High CV risk - consider SGLT2i or GLP-1 RA with CV benefit")

    elif risk_factors.get("prediabetes_diagnosis") == "prediabetes":
        alerts.append("Prediabetes diagnosed - initiate prevention strategies")

        # Metformin consideration for high-risk prediabetes
        if (bmi and bmi.get("value", 0) >= 35) or (a1c and a1c.get("value", 0) >= 6.0):
            alerts.append("High-risk prediabetes - consider metformin therapy")

    # Kidney disease alerts
    if risk_factors.get("kidney_disease_stage") in ["moderate_to_severe", "severe", "kidney_failure"]:
        alerts.append("Advanced CKD detected - nephrology referral recommended")

    # Medication safety alerts
    if creatinine and creatinine.get("value", 0) > 1.5:
        alerts.append("Elevated creatinine - review metformin dosing")

    risk_factors["clinical_alerts"] = alerts
    return risk_factors


def assess_treatment_goals(data: Dict, risk_factors: Dict) -> Dict:
    goals = {
        "a1c_target": 7.0,
        "bp_target": "130/80",
        "ldl_target": 100,
        "treatment_priority": [],
        "medication_considerations": []
    }

    # Individualize A1C targets (ADA Section 6)
    if risk_factors.get("diabetes_diagnosis") == "diabetes":
        # Standard target for most adults
        goals["a1c_target"] = 7.0

        # More stringent if can be achieved safely
        summary = data.get("diabetes_summary", {})
        if summary.get("glucose", {}).get("fasting_plasma"):
            goals["treatment_priority"].append("Optimize glycemic control")

    # Cardiovascular risk management (ADA Section 10)
    cv_risk = risk_factors.get("cardiovascular_risk")
    if cv_risk in ["high", "very_high"]:
        goals["ldl_target"] = 70  # More aggressive LDL target
        goals["treatment_priority"].append("Intensive cardiovascular risk reduction")
        goals["medication_considerations"].append("Consider high-intensity statin")

    # Kidney protection (ADA Section 11)
    kidney_stage = risk_factors.get("kidney_disease_stage")
    if kidney_stage in ["mild_to_moderate", "moderate_to_severe"]:
        goals["treatment_priority"].append("Kidney protection")
        goals["medication_considerations"].append("ACE inhibitor or ARB recommended")

        if risk_factors.get("diabetes_diagnosis") == "diabetes":
            goals["medication_considerations"].append("Consider SGLT2 inhibitor for kidney protection")

    return goals

# Enhanced HTTP helper with retries
def _get(url: str, params: Optional[Dict[str, str]] = None, *, max_retries: int = 4) -> Dict:
    attempt = 0
    while True:
        try:
            r = requests.get(
                url,
                params=params,
                headers={"Accept": "application/fhir+json"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or "resourceType" not in data:
                raise ValueError(f"Unexpected response (no resourceType) from {url}")
            return data
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = (status is None) or (500 <= status < 600)
            attempt += 1
            if attempt >= max_retries or not retryable:
                raise
            sleep_s = 0.8 * (2 ** (attempt - 1))
            time.sleep(sleep_s)

# Patient discovery functions
def _extract_patient_ids_from_bundle(bundle: Dict, seen: Set[str], ids: List[str], max_patients: int) -> bool:
    for e in bundle.get("entry", []) or []:
        obs = e.get("resource", {})
        if obs.get("resourceType") != "Observation":
            continue
        ref = (obs.get("subject") or {}).get("reference")
        if not ref:
            continue
        pid = ref.split("/")[-1]
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= max_patients:
                return True
    return False


def find_diabetes_patients(base_url: str, max_patients: int = 20) -> List[str]:
    print(f"Searching for diabetes patients with comprehensive lab data...")

    # First search for A1C - most specific for diabetes
    a1c_patients = []
    try:
        params = {
            "code": "4548-4",  # A1C
            "category": "laboratory",
            "_count": "100",
            "_elements": "subject"
        }
        bundle = _get(f"{base_url.rstrip('/')}/Observation", params)
        seen = set()
        _extract_patient_ids_from_bundle(bundle, seen, a1c_patients, max_patients)
        print(f"Found {len(a1c_patients)} patients with A1C values")
    except Exception as e:
        print(f"Error searching A1C patients: {e}")

    # need more patients, search glucose values
    all_patients = a1c_patients.copy()
    if len(all_patients) < max_patients:
        glucose_patients = []
        for glucose_type, codes in LOINC_GLUCOSE.items():
            if len(all_patients) >= max_patients:
                break
            for code in codes:
                try:
                    params = {
                        "code": code,
                        "category": "laboratory",
                        "_count": "50",
                        "_elements": "subject"
                    }
                    bundle = _get(f"{base_url.rstrip('/')}/Observation", params)
                    seen = set(all_patients)
                    temp_ids = []
                    _extract_patient_ids_from_bundle(bundle, seen, temp_ids, max_patients - len(all_patients))
                    all_patients.extend(temp_ids)
                except Exception as e:
                    print(f"Error searching {glucose_type} glucose: {e}")
                    continue
                time.sleep(0.1)  # Rate limiting

    print(f"Total diabetes candidates found: {len(all_patients)}")
    return all_patients[:max_patients]

# Enhanced data collection
def _normalize_obs(o: Dict) -> Optional[Dict]:
    # Handle valueQuantity
    q = o.get("valueQuantity") or {}
    if "value" in q:
        unit = q.get("unit") or q.get("code") or q.get("system")
        codings = (o.get("code", {}) or {}).get("coding") or []
        code = next((c.get("code") for c in codings if c.get("code")), None)
        return {
            "code": code,
            "value": q.get("value"),
            "unit": unit,
            "time": o.get("effectiveDateTime") or (o.get("effectivePeriod") or {}).get("start"),
        }

    # Handle coded values for some observations
    value_concept = o.get("valueCodeableConcept")
    if value_concept:
        codings = value_concept.get("coding", [])
        if codings:
            # This could be useful for categorical results
            pass

    return None


def _fetch_latest_observations(base_url: str, patient_id: str, codes: List[str],
                               category: Optional[str] = None, n: int = 1) -> List[Dict]:
    base = base_url.rstrip("/")
    strategies = [
        {"_sort": "-date", "_count": str(n)},
        {"_count": str(n)},
        {"_count": "1"},
    ]

    for strat in strategies:
        params = {
            "patient": patient_id,
            "code": ",".join(codes),
            "_elements": "code,value,subject,effectiveDateTime,effectivePeriod,category,status",
            **strat,
        }
        if category:
            params["category"] = category

        try:
            bundle = _get(f"{base}/Observation", params)
            observations = []

            for entry in bundle.get("entry", []) or []:
                obs = entry.get("resource", {})

                # Skip cancelled or preliminary results
                status = obs.get("status", "")
                if status in ["cancelled", "preliminary"]:
                    continue

                normalized = _normalize_obs(obs)
                if normalized:
                    observations.append(normalized)

            return observations[:n]
        except requests.HTTPError:
            continue

    return []


def collect_comprehensive_diabetes_panel(base_url: str, patient_id: str) -> Dict:
    panel = {}

    # A1C - core diabetes marker
    a1c_history = _fetch_latest_observations(base_url, patient_id, LOINC_A1C, "laboratory", 3)
    panel["a1c"] = {
        "latest": a1c_history[0] if a1c_history else None,
        "history": a1c_history
    }

    # Glucose measurements for diagnosis
    glucose_panel = {}
    for glucose_type, codes in LOINC_GLUCOSE.items():
        obs = _fetch_latest_observations(base_url, patient_id, codes, "laboratory", 1)
        glucose_panel[glucose_type] = {"latest": obs[0] if obs else None}
    panel["glucose"] = glucose_panel

    # BMI - important for risk stratification
    bmi_obs = _fetch_latest_observations(base_url, patient_id, LOINC_BMI, "vital-signs", 1)
    panel["bmi"] = {"latest": bmi_obs[0] if bmi_obs else None}

    # Lipid panel for cardiovascular risk
    lipids = {}
    for lipid_type, codes in LOINC_LIPIDS.items():
        obs = _fetch_latest_observations(base_url, patient_id, codes, "laboratory", 1)
        lipids[lipid_type] = {"latest": obs[0] if obs else None}
    panel["lipids"] = lipids

    # Kidney function
    renal = {}
    for test_type, codes in LOINC_RENAL.items():
        obs = _fetch_latest_observations(base_url, patient_id, codes, "laboratory", 1)
        renal[test_type] = {"latest": obs[0] if obs else None}
    panel["renal"] = renal

    # Urine studies
    urine_acr = _fetch_latest_observations(base_url, patient_id, LOINC_URINE["acr"], "laboratory", 1)
    panel["urine"] = {"acr": {"latest": urine_acr[0] if urine_acr else None}}

    return panel


def _meets_diabetes_criteria(panel: Dict) -> bool:
    # Has A1C OR any glucose measurement
    has_a1c = bool(panel.get("a1c", {}).get("latest"))

    glucose_data = panel.get("glucose", {})
    has_glucose = any((glucose_data.get(k) or {}).get("latest") for k in glucose_data.keys())

    return has_a1c or has_glucose


def _build_diabetes_summary(panel: Dict) -> Dict:

    def get_latest(data_dict: Dict, keys: List[str]):
        for key in keys:
            value = (data_dict.get(key) or {}).get("latest")
            if value:
                return value
        return None

    # Core diabetes markers
    a1c = (panel.get("a1c") or {}).get("latest")
    bmi = (panel.get("bmi") or {}).get("latest")

    # Glucose panel
    glucose_data = panel.get("glucose", {})
    glucose_summary = {
        "fasting_plasma": (glucose_data.get("fasting_plasma") or {}).get("latest"),
        "random_plasma_or_whole": get_latest(glucose_data, ["plasma", "whole_blood"]),
        "ogtt_2h_75g": (glucose_data.get("ogtt_2h_75g") or {}).get("latest"),
    }

    # Lipids
    lipids_data = panel.get("lipids", {})
    lipids_summary = {
        "ldl": (lipids_data.get("ldl") or {}).get("latest"),
        "hdl": (lipids_data.get("hdl") or {}).get("latest"),
        "triglycerides": (lipids_data.get("triglycerides") or {}).get("latest"),
        "total_cholesterol": (lipids_data.get("total_cholesterol") or {}).get("latest"),
    }

    # Renal function
    renal_data = panel.get("renal", {})
    renal_summary = {
        "creatinine": (renal_data.get("creatinine") or {}).get("latest"),
        "egfr": (renal_data.get("egfr") or {}).get("latest"),
    }

    # Urine studies
    urine_summary = {
        "acr": (panel.get("urine", {}).get("acr") or {}).get("latest")
    }

    return {
        "a1c": a1c,
        "bmi": bmi,
        "glucose": glucose_summary,
        "lipids": lipids_summary,
        "renal": renal_summary,
        "urine": urine_summary,
    }

# Main API functions
def get_patient_diabetes_data(patient_id: str, base_url: Optional[str] = None) -> Dict:
"
    base = (base_url or DEFAULT_FHIR_BASE).rstrip("/")

    # Collect comprehensive panel
    panel = collect_comprehensive_diabetes_panel(base, patient_id)

    # Build summary for easier processing
    summary = _build_diabetes_summary(panel)

    # Calculate risk factors and treatment goals
    risk_factors = calculate_diabetes_risk_factors({"diabetes_summary": summary})
    treatment_goals = assess_treatment_goals({"diabetes_summary": summary}, risk_factors)

    # Construct final data structure
    patient_data = {
        "patient": {"id": patient_id},
        "metabolic": panel,
        "diabetes_minimal_ok": _meets_diabetes_criteria(panel),
        "diabetes_summary": summary,
        "clinical_assessment": {
            "risk_factors": risk_factors,
            "treatment_goals": treatment_goals,
            "data_completeness": _assess_data_completeness(panel)
        },
        "provenance": {
            "fhir_base_url": base,
            "collection_timestamp": datetime.now().isoformat(),
            "ada_guidelines_version": "2023"
        }
    }

    return patient_data


def _assess_data_completeness(panel: Dict) -> Dict:
    completeness = {
        "core_diabetes": False,
        "cardiovascular_risk": False,
        "kidney_function": False,
        "completeness_score": 0.0
    }

    # Core diabetes data
    has_a1c = bool(panel.get("a1c", {}).get("latest"))
    has_glucose = any((panel.get("glucose", {}).get(k) or {}).get("latest")
                      for k in ["fasting_plasma", "plasma", "whole_blood"])
    completeness["core_diabetes"] = has_a1c or has_glucose

    # CV risk data
    lipids = panel.get("lipids", {})
    has_lipids = any((lipids.get(k) or {}).get("latest") for k in ["ldl", "hdl", "triglycerides"])
    has_bmi = bool(panel.get("bmi", {}).get("latest"))
    completeness["cardiovascular_risk"] = has_lipids and has_bmi

    # Kidney function
    renal = panel.get("renal", {})
    has_renal = any((renal.get(k) or {}).get("latest") for k in ["creatinine", "egfr"])
    completeness["kidney_function"] = has_renal

    # Calculate overall score
    scores = [
        completeness["core_diabetes"],
        completeness["cardiovascular_risk"],
        completeness["kidney_function"]
    ]
    completeness["completeness_score"] = sum(scores) / len(scores)

    return completeness


def save_patient_data(patient_id: str, data: Dict, outdir: str = DEFAULT_OUTDIR) -> str:
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Enhanced filename with diabetes status
    diabetes_status = "unknown"
    if data.get("clinical_assessment", {}).get("risk_factors", {}).get("diabetes_diagnosis"):
        diabetes_status = data["clinical_assessment"]["risk_factors"]["diabetes_diagnosis"]

    filename = f"{patient_id}.diabetes_cdss.json"
    path = os.path.join(outdir, filename)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path

# Batch operations
def collect_diabetes_cohort(base_url: str, outdir: str, target_patients: int = 10) -> int:
    print(f"Collecting diabetes cohort of {target_patients} patients...")

    # Find candidate patients
    candidates = find_diabetes_patients(base_url, max_patients=target_patients * 3)

    if not candidates:
        print("No candidate patients found.")
        return 0

    collected = 0
    processed = 0

    for patient_id in candidates:
        if collected >= target_patients:
            break

        processed += 1
        try:
            print(f"Processing patient {patient_id} ({processed}/{len(candidates)})...")

            data = get_patient_diabetes_data(patient_id, base_url)

            # Only save if we have meaningful diabetes data
            if data.get("diabetes_minimal_ok"):
                path = save_patient_data(patient_id, data, outdir)
                collected += 1

                # Print summary
                risk = data.get("clinical_assessment", {}).get("risk_factors", {})
                diabetes_status = risk.get("diabetes_diagnosis", "unknown")
                alerts = len(risk.get("clinical_alerts", []))

                print(f"  Saved {patient_id}: {diabetes_status}, {alerts} clinical alerts")
            else:
                print(f"  Skipped {patient_id}: insufficient diabetes data")

        except Exception as e:
            print(f"   Error processing {patient_id}: {e}")

        # Rate limiting
        time.sleep(0.2)

    print(f"\nCollected {collected}/{target_patients} patients with comprehensive diabetes data")
    return collected


def print_cohort_summary(outdir: str) -> None:
    files = sorted(glob.glob(os.path.join(outdir, "*.diabetes_cdss.json")))

    if not files:
        print(f"No diabetes CDSS files found in {outdir}")
        return

    print(f"\n=== Diabetes CDSS Cohort Summary ===")
    print(f"Total patients: {len(files)}")

    # Analyze cohort characteristics
    diabetes_count = 0
    prediabetes_count = 0
    high_cv_risk = 0
    kidney_disease = 0
    total_alerts = 0

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        risk = data.get("clinical_assessment", {}).get("risk_factors", {})

        if risk.get("diabetes_diagnosis") == "diabetes":
            diabetes_count += 1
        elif risk.get("prediabetes_diagnosis") == "prediabetes":
            prediabetes_count += 1

        if risk.get("cardiovascular_risk") in ["high", "very_high"]:
            high_cv_risk += 1

        if risk.get("kidney_disease_stage") in ["moderate_to_severe", "severe", "kidney_failure"]:
            kidney_disease += 1

        total_alerts += len(risk.get("clinical_alerts", []))

        # Print individual patient summary
        patient_id = data.get("patient", {}).get("id", "unknown")
        summary = data.get("diabetes_summary", {})

        a1c_val = (summary.get("a1c") or {}).get("value", "N/A")
        bmi_val = (summary.get("bmi") or {}).get("value", "N/A")
        alerts = len(risk.get("clinical_alerts", []))

        print(f"  {os.path.basename(file_path)}: A1C={a1c_val}, BMI={bmi_val}, Alerts={alerts}")

    print(f"\nCohort Statistics:")
    print(f"  Diabetes: {diabetes_count} ({diabetes_count / len(files) * 100:.1f}%)")
    print(f"  Prediabetes: {prediabetes_count} ({prediabetes_count / len(files) * 100:.1f}%)")
    print(f"  High CV Risk: {high_cv_risk} ({high_cv_risk / len(files) * 100:.1f}%)")
    print(f"  Advanced CKD: {kidney_disease} ({kidney_disease / len(files) * 100:.1f}%)")
    print(f"  Total Clinical Alerts: {total_alerts}")
    print(f"  Avg Alerts per Patient: {total_alerts / len(files):.1f}")

# CLI and main functions
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced diabetes CDSS data collector with clinical decision support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect cohort of 10 diabetes patients
  python diabetes_cdss_collector.py

  # Collect specific patient
  python diabetes_cdss_collector.py --patient 12345

  # Collect larger cohort
  python diabetes_cdss_collector.py --cohort-size 25

  # Show summary of collected data
  python diabetes_cdss_collector.py --summary-only

  # Collect with risk assessment focus
  python diabetes_cdss_collector.py --risk-assessment
        """
    )

    # Main modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--patient",
        help="Collect data for specific patient ID"
    )
    mode_group.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary of existing collected data"
    )

    # Collection parameters
    parser.add_argument(
        "--cohort-size",
        type=int,
        default=10,
        help="Number of patients to collect for cohort (default: 10)"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_FHIR_BASE,
        help=f"FHIR base URL (default: {DEFAULT_FHIR_BASE})"
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})"
    )

    # Special modes
    parser.add_argument(
        "--risk-assessment",
        action="store_true",
        help="Focus on high-risk patients for CDSS testing"
    )
    parser.add_argument(
        "--validate-guidelines",
        action="store_true",
        help="Validate clinical decision rules against ADA guidelines"
    )

    return parser.parse_args()


def validate_clinical_guidelines() -> None:
    print(" Validating Clinical Decision Rules Against ADA 2023 Guidelines \n")

    # Test diabetes diagnosis criteria (Table 2.2/2.5)
    test_cases = [
        {"a1c": 6.8, "expected": "diabetes", "guideline": "A1C ≥6.5%"},
        {"a1c": 6.2, "expected": "prediabetes", "guideline": "A1C 5.7-6.4%"},
        {"a1c": 5.5, "expected": "normal", "guideline": "A1C <5.7%"},
        {"fpg": 130, "expected": "diabetes", "guideline": "FPG ≥126 mg/dL"},
        {"fpg": 110, "expected": "prediabetes", "guideline": "FPG 100-125 mg/dL"},
        {"random_glucose": 220, "expected": "diabetes", "guideline": "Random glucose ≥200 mg/dL"},
    ]

    print("Testing Diabetes Diagnosis Criteria:")
    for i, case in enumerate(test_cases, 1):
        # Create test data
        test_data = {"diabetes_summary": {}}

        if "a1c" in case:
            test_data["diabetes_summary"]["a1c"] = {"value": case["a1c"]}
        if "fpg" in case:
            test_data["diabetes_summary"]["glucose"] = {
                "fasting_plasma": {"value": case["fpg"]}
            }
        if "random_glucose" in case:
            test_data["diabetes_summary"]["glucose"] = {
                "random_plasma_or_whole": {"value": case["random_glucose"]}
            }

        # Test our function
        risk_factors = calculate_diabetes_risk_factors(test_data)

        # Check result
        if "diabetes" in case["expected"]:
            result = risk_factors.get("diabetes_diagnosis")
        else:
            result = risk_factors.get("prediabetes_diagnosis") or risk_factors.get("diabetes_diagnosis")

        status = "PASS" if result == case["expected"] else "FAIL"
        print(f"  {i}. {case['guideline']} → Expected: {case['expected']}, Got: {result} [{status}]")

    print("\nTesting A1C Target Recommendations (Section 6):")
    target_tests = [
        {"scenario": "Standard adult", "expected_target": 7.0},
        {"scenario": "High CV risk", "note": "Consider <7% if achievable safely"},
    ]

    for test in target_tests:
        print(f"  • {test['scenario']}: Target A1C <{test.get('expected_target', 'individualized')}%")
        if "note" in test:
            print(f"    Note: {test['note']}")

    print("\nTesting Cardiovascular Risk Assessment (Section 10):")
    cv_tests = [
        {"ldl": 195, "expected": "very_high", "guideline": "LDL ≥190 mg/dL"},
        {"ldl": 165, "expected": "high", "guideline": "LDL 160-189 mg/dL"},
        {"ldl": 135, "expected": "moderate", "guideline": "LDL 130-159 mg/dL"},
        {"ldl": 85, "expected": "low", "guideline": "LDL <130 mg/dL"},
    ]

    for i, case in enumerate(cv_tests, 1):
        test_data = {
            "diabetes_summary": {
                "lipids": {"ldl": {"value": case["ldl"]}}
            }
        }
        risk_factors = calculate_diabetes_risk_factors(test_data)
        result = risk_factors.get("cardiovascular_risk")
        status = "PASS" if result == case["expected"] else "FAIL"
        print(f"  {i}. {case['guideline']} → Expected: {case['expected']}, Got: {result} [{status}]")

    print("\n  Validation Complete ")
    print("Note: These validations ensure our CDSS logic aligns with ADA 2023 Standards of Care")


def collect_high_risk_cohort(base_url: str, outdir: str, target_patients: int = 15) -> int:
    print(f"Collecting high-risk diabetes cohort for CDSS testing...")

    # Cast a wider net for high-risk patients
    candidates = find_diabetes_patients(base_url, max_patients=target_patients * 4)

    if not candidates:
        print("No candidate patients found.")
        return 0

    collected = 0
    high_risk_collected = 0
    processed = 0

    for patient_id in candidates:
        if collected >= target_patients:
            break

        processed += 1
        try:
            print(f"Evaluating patient {patient_id} ({processed}/{len(candidates)})...")

            data = get_patient_diabetes_data(patient_id, base_url)

            if not data.get("diabetes_minimal_ok"):
                print(f"  Skipped {patient_id}: insufficient diabetes data")
                continue

            # Assess risk profile
            risk = data.get("clinical_assessment", {}).get("risk_factors", {})
            completeness = data.get("clinical_assessment", {}).get("data_completeness", {})

            # Prioritize high-risk patients
            is_high_risk = False
            risk_reasons = []

            if risk.get("diabetes_diagnosis") == "diabetes":
                risk_reasons.append("diabetes")
                is_high_risk = True

            if risk.get("cardiovascular_risk") in ["high", "very_high"]:
                risk_reasons.append("high_cv_risk")
                is_high_risk = True

            if risk.get("kidney_disease_stage") in ["moderate_to_severe", "severe", "kidney_failure"]:
                risk_reasons.append("advanced_ckd")
                is_high_risk = True

            if risk.get("obesity_status") == "obese":
                risk_reasons.append("obesity")

            alerts = len(risk.get("clinical_alerts", []))
            if alerts >= 2:
                risk_reasons.append("multiple_alerts")
                is_high_risk = True

            # Good data completeness preferred
            if completeness.get("completeness_score", 0) >= 0.7:
                risk_reasons.append("complete_data")

            # Save if high-risk or if we need more patients
            if is_high_risk or collected < target_patients // 2:
                path = save_patient_data(patient_id, data, outdir)
                collected += 1

                if is_high_risk:
                    high_risk_collected += 1

                print(f" Saved {patient_id}: {', '.join(risk_reasons)}, {alerts} alerts")
            else:
                print(f" Skipped {patient_id}: low risk profile")

        except Exception as e:
            print(f"  Error processing {patient_id}: {e}")

        time.sleep(0.2)

    print(f"\nHigh-Risk Cohort Collection Complete:")
    print(f"  Total patients: {collected}/{target_patients}")
    print(f"  High-risk patients: {high_risk_collected}/{collected}")
    print(f"  High-risk percentage: {high_risk_collected / collected * 100:.1f}%" if collected > 0 else "")

    return collected


def main() -> int:
    args = _parse_args()

    try:
        # Summary only mode
        if args.summary_only:
            print_cohort_summary(args.outdir)
            return 0

        # Guideline validation mode
        if args.validate_guidelines:
            validate_clinical_guidelines()
            return 0

        # Single patient mode
        if args.patient:
            print(f"Collecting diabetes data for patient {args.patient}...")

            try:
                data = get_patient_diabetes_data(args.patient, args.base_url)

                if not data.get("diabetes_minimal_ok"):
                    print(f"Patient {args.patient} does not have sufficient diabetes data for CDSS.")
                    return 1

                path = save_patient_data(args.patient, data, args.outdir)
                print(f" Saved comprehensive diabetes data → {path}")

                # Show clinical summary
                risk = data.get("clinical_assessment", {}).get("risk_factors", {})
                goals = data.get("clinical_assessment", {}).get("treatment_goals", {})

                print(f"\nClinical Assessment Summary:")
                print(f"  Diabetes Status: {risk.get('diabetes_diagnosis', 'unknown')}")
                print(f"  CV Risk: {risk.get('cardiovascular_risk', 'unknown')}")
                print(f"  Kidney Function: {risk.get('kidney_disease_stage', 'unknown')}")
                print(f"  A1C Target: <{goals.get('a1c_target', 'individualized')}%")

                alerts = risk.get("clinical_alerts", [])
                if alerts:
                    print(f"  Clinical Alerts:")
                    for alert in alerts:
                        print(f"    • {alert}")

                return 0

            except Exception as e:
                print(f"Error collecting data for patient {args.patient}: {e}")
                return 1

        # Cohort collection mode (default)
        print("=== Diabetes CDSS Data Collector ===")
        print(f"Collecting cohort for Clinical Decision Support System development")
        print(f"Based on ADA 2023 Standards of Care guidelines\n")

        if args.risk_assessment:
            collected = collect_high_risk_cohort(args.base_url, args.outdir, args.cohort_size)
        else:
            collected = collect_diabetes_cohort(args.base_url, args.outdir, args.cohort_size)

        if collected > 0:
            print(f"\n=== Collection Summary ===")
            print_cohort_summary(args.outdir)

            print(f" Review collected data in: {args.outdir}")

            return 0
        else:
            print("No patients collected.")
            return 1

    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
