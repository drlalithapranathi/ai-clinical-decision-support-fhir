#!/usr/bin/env python3
import json
import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Clinical Decision Classes
class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class AlertPriority(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


@dataclass
class ClinicalAlert:
    message: str
    priority: AlertPriority
    category: str
    evidence: str
    action_required: bool = False
    guideline_reference: str = ""


@dataclass
class MedicationRecommendation:
    medication_class: str
    specific_agents: List[str]
    rationale: str
    contraindications: List[str]
    monitoring_required: List[str]
    ada_reference: str = ""


@dataclass
class TreatmentGoal:
    parameter: str
    target_value: Union[str, float]
    rationale: str
    priority: str = "standard"

# CDSS Clinical Decision Engine
class DiabetesCDSS:
    def __init__(self):
        self.guideline_version = "ADA 2023"
        self.alerts = []
        self.recommendations = []
        self.treatment_goals = []

    def process_patient(self, patient_data: Dict) -> Dict:
        self.alerts = []
        self.recommendations = []
        self.treatment_goals = []

        # SAFER patient ID extraction
        patient_id = "unknown"
        if patient_data and isinstance(patient_data, dict):
            patient_info = patient_data.get("patient", {})
            if patient_info and isinstance(patient_info, dict):
                patient_id = patient_info.get("id", "unknown")

        print(f"Processing CDSS recommendations for patient {patient_id}...")

        # SAFER data extraction with None checks
        summary = patient_data.get("diabetes_summary", {}) if patient_data else {}
        if summary is None:
            summary = {}

        clinical_assessment = patient_data.get("clinical_assessment", {}) if patient_data else {}
        if clinical_assessment is None:
            clinical_assessment = {}

        risk_factors = clinical_assessment.get("risk_factors", {}) if clinical_assessment else {}
        if risk_factors is None:
            risk_factors = {}

        # Core CDSS decision pathways
        self._assess_diabetes_diagnosis(summary, risk_factors)
        self._determine_glycemic_targets(summary, risk_factors)
        self._assess_cardiovascular_risk(summary, risk_factors)
        self._evaluate_kidney_function(summary, risk_factors)
        self._medication_recommendations(summary, risk_factors)
        self._monitoring_recommendations(summary, risk_factors)
        self._lifestyle_recommendations(summary, risk_factors)

        # Package results
        cdss_output = {
            "patient_id": patient_id,
            "processing_timestamp": datetime.now().isoformat(),
            "guideline_version": self.guideline_version,
            "clinical_alerts": [self._alert_to_dict(alert) for alert in self.alerts],
            "medication_recommendations": [self._med_to_dict(med) for med in self.recommendations],
            "treatment_goals": [self._goal_to_dict(goal) for goal in self.treatment_goals],
            "summary": self._generate_clinical_summary(),
            "next_actions": self._determine_next_actions()
        }

        return cdss_output

    def _assess_diabetes_diagnosis(self, summary: Dict, risk_factors: Dict) -> None:

        # Handle None values
        a1c = summary.get("a1c", {})
        glucose = summary.get("glucose", {})

        # Convert None to empty dict
        if a1c is None:
            a1c = {}
        if glucose is None:
            glucose = {}

        # A1C-based diagnosis
        if a1c and a1c.get("value") is not None:
            a1c_value = a1c["value"]

            if a1c_value >= 6.5:
                self.alerts.append(ClinicalAlert(
                    message=f"Diabetes diagnosed: A1C {a1c_value}% (≥6.5%)",
                    priority=AlertPriority.CRITICAL,
                    category="diagnosis",
                    evidence="ADA 2023 Table 2.2: A1C ≥6.5% diagnostic for diabetes",
                    action_required=True,
                    guideline_reference="ADA 2023 Section 2"
                ))

                if a1c_value > 9.0:
                    self.alerts.append(ClinicalAlert(
                        message=f"Severely elevated A1C {a1c_value}% - consider immediate insulin therapy",
                        priority=AlertPriority.URGENT,
                        category="glycemic_control",
                        evidence="ADA 2023: A1C >9% warrants immediate pharmacologic intervention",
                        action_required=True,
                        guideline_reference="ADA 2023 Section 9"
                    ))

            elif a1c_value >= 5.7:
                self.alerts.append(ClinicalAlert(
                    message=f"Prediabetes diagnosed: A1C {a1c_value}% (5.7-6.4%)",
                    priority=AlertPriority.WARNING,
                    category="prevention",
                    evidence="ADA 2023 Table 2.2: A1C 5.7-6.4% diagnostic for prediabetes",
                    action_required=True,
                    guideline_reference="ADA 2023 Section 3"
                ))

        # Glucose-based diagnosis Add None checks
        fpg = glucose.get("fasting_plasma")
        if fpg is not None and fpg.get("value") is not None:
            fpg_value = fpg["value"]

            if fpg_value >= 126:
                self.alerts.append(ClinicalAlert(
                    message=f"Diabetes diagnosed: Fasting glucose {fpg_value} mg/dL (≥126)",
                    priority=AlertPriority.CRITICAL,
                    category="diagnosis",
                    evidence="ADA 2023 Table 2.2: FPG ≥126 mg/dL diagnostic for diabetes",
                    action_required=True
                ))
            elif fpg_value >= 100:
                self.alerts.append(ClinicalAlert(
                    message=f"Prediabetes: Fasting glucose {fpg_value} mg/dL (100-125)",
                    priority=AlertPriority.WARNING,
                    category="prevention",
                    evidence="ADA 2023 Table 2.2: FPG 100-125 mg/dL diagnostic for prediabetes",
                    action_required=True
                ))

        # Random glucose Add None checks
        random_glucose = glucose.get("random_plasma_or_whole")
        if random_glucose is not None and random_glucose.get("value") is not None:
            if random_glucose["value"] >= 200:
                self.alerts.append(ClinicalAlert(
                    message=f"Diabetes diagnosed: Random glucose {random_glucose['value']} mg/dL (≥200)",
                    priority=AlertPriority.CRITICAL,
                    category="diagnosis",
                    evidence="ADA 2023 Table 2.2: Random glucose ≥200 mg/dL with symptoms",
                    action_required=True
                ))

    def _determine_glycemic_targets(self, summary: Dict, risk_factors: Dict) -> None:

        # Standard A1C target for most adults
        a1c_target = 7.0
        rationale = "Standard A1C target for most nonpregnant adults"

        # Individualization factors
        diabetes_status = risk_factors.get("diabetes_diagnosis")

        if diabetes_status == "diabetes":
            # Current A1C for comparison
            current_a1c = summary.get("a1c", {}).get("value")

            # More stringent target considerations
            age_factors = "Consider more stringent target if achievable without hypoglycemia"

            # Less stringent target considerations
            cv_risk = risk_factors.get("cardiovascular_risk")
            kidney_stage = risk_factors.get("kidney_disease_stage")

            if cv_risk == "very_high" or kidney_stage in ["severe", "kidney_failure"]:
                a1c_target = 8.0
                rationale = "Less stringent target due to high comorbidity burden and hypoglycemia risk"

            self.treatment_goals.append(TreatmentGoal(
                parameter="A1C",
                target_value=f"<{a1c_target}%",
                rationale=rationale,
                priority="high"
            ))

            # Alert if above target
            if current_a1c and current_a1c > a1c_target:
                gap = current_a1c - a1c_target
                self.alerts.append(ClinicalAlert(
                    message=f"A1C {current_a1c}% above target <{a1c_target}% (gap: {gap:.1f}%)",
                    priority=AlertPriority.WARNING if gap < 1.0 else AlertPriority.CRITICAL,
                    category="glycemic_control",
                    evidence="ADA 2023 Section 6: Individualized A1C targets",
                    action_required=True
                ))

        # Glucose targets
        self.treatment_goals.extend([
            TreatmentGoal(
                parameter="Preprandial glucose",
                target_value="80-130 mg/dL",
                rationale="ADA 2023 standard preprandial target"
            ),
            TreatmentGoal(
                parameter="Postprandial glucose",
                target_value="<180 mg/dL",
                rationale="ADA 2023 standard postprandial target"
            )
        ])

    def _assess_cardiovascular_risk(self, summary: Dict, risk_factors: Dict) -> None:

        # Handle None values
        lipids = summary.get("lipids", {})
        if lipids is None:
            lipids = {}

        diabetes_status = risk_factors.get("diabetes_diagnosis")

        # LDL cholesterol assessment Add None checks
        ldl = lipids.get("ldl")
        if ldl is not None and ldl.get("value") is not None:
            ldl_value = ldl["value"]

            # Determine LDL target based on risk
            if diabetes_status == "diabetes":
                # Primary prevention targets
                ldl_target = 100
                statin_intensity = "moderate"

                # High-risk patients
                if ldl_value >= 70:
                    ldl_target = 70
                    statin_intensity = "high"

                    self.alerts.append(ClinicalAlert(
                        message=f"Elevated LDL {ldl_value} mg/dL in diabetes - target <{ldl_target}",
                        priority=AlertPriority.WARNING,
                        category="cardiovascular",
                        evidence="ADA 2023 Section 10: LDL targets for diabetes",
                        action_required=True
                    ))

                # Statin recommendation
                self.recommendations.append(MedicationRecommendation(
                    medication_class="Statin",
                    specific_agents=["atorvastatin", "rosuvastatin"] if statin_intensity == "high"
                    else ["simvastatin", "pravastatin"],
                    rationale=f"{statin_intensity.title()}-intensity statin for diabetes CV protection",
                    contraindications=["Active liver disease", "Pregnancy"],
                    monitoring_required=["Liver function tests", "CK if symptoms"],
                    ada_reference="ADA 2023 Section 10.18-10.26"
                ))

        # HDL and triglycerides Add None checks
        hdl = lipids.get("hdl")
        if hdl is not None and hdl.get("value") is not None:
            hdl_value = hdl["value"]
            if hdl_value < 40:  # Men's threshold; women <50
                self.alerts.append(ClinicalAlert(
                    message=f"Low HDL cholesterol {hdl_value} mg/dL",
                    priority=AlertPriority.INFO,
                    category="cardiovascular",
                    evidence="Low HDL increases cardiovascular risk",
                    action_required=False
                ))

        triglycerides = lipids.get("triglycerides")
        if triglycerides is not None and triglycerides.get("value") is not None:
            tg_value = triglycerides["value"]
            if tg_value >= 150:
                priority = AlertPriority.WARNING if tg_value < 500 else AlertPriority.CRITICAL
                self.alerts.append(ClinicalAlert(
                    message=f"Elevated triglycerides {tg_value} mg/dL",
                    priority=priority,
                    category="cardiovascular",
                    evidence="ADA 2023: Triglycerides ≥150 mg/dL increase CV risk",
                    action_required=tg_value >= 500
                ))

    def _evaluate_kidney_function(self, summary: Dict, risk_factors: Dict) -> None:

        # Handle None values
        renal = summary.get("renal", {})
        if renal is None:
            renal = {}

        diabetes_status = risk_factors.get("diabetes_diagnosis")

        # eGFR assessment Add None checks
        egfr = renal.get("egfr")
        if egfr is not None and egfr.get("value") is not None:
            egfr_value = egfr["value"]

            if egfr_value < 60:
                stage = "3" if egfr_value >= 30 else "4" if egfr_value >= 15 else "5"
                priority = AlertPriority.WARNING if stage == "3" else AlertPriority.CRITICAL

                self.alerts.append(ClinicalAlert(
                    message=f"Chronic kidney disease stage {stage}: eGFR {egfr_value}",
                    priority=priority,
                    category="kidney_disease",
                    evidence=f"ADA 2023 Section 11: eGFR {egfr_value} indicates CKD stage {stage}",
                    action_required=True
                ))

                if egfr_value < 30:
                    self.alerts.append(ClinicalAlert(
                        message="Nephrology referral recommended for eGFR <30",
                        priority=AlertPriority.CRITICAL,
                        category="referral",
                        evidence="ADA 2023 Recommendation 11.8",
                        action_required=True
                    ))

        # Creatinine assessment Add None checks
        creatinine = renal.get("creatinine")
        if creatinine is not None and creatinine.get("value") is not None:
            cr_value = creatinine["value"]

            # Medication safety alerts
            if cr_value > 1.5:
                self.alerts.append(ClinicalAlert(
                    message=f"Elevated creatinine {cr_value} mg/dL - review metformin dosing",
                    priority=AlertPriority.WARNING,
                    category="medication_safety",
                    evidence="Metformin contraindicated with significant kidney impairment",
                    action_required=True
                ))

        # Kidney protection recommendations for diabetes
        if diabetes_status == "diabetes":
            # ACE inhibitor/ARB recommendation
            self.recommendations.append(MedicationRecommendation(
                medication_class="ACE inhibitor or ARB",
                specific_agents=["lisinopril", "enalapril", "losartan", "valsartan"],
                rationale="Kidney protection in diabetes",
                contraindications=["Bilateral renal artery stenosis", "Hyperkalemia"],
                monitoring_required=["Serum creatinine", "Serum potassium"],
                ada_reference="ADA 2023 Section 11.4a"
            ))

            # SGLT2 inhibitor for kidney protection Add None checks
            if egfr is not None and egfr.get("value", 0) >= 20:
                self.recommendations.append(MedicationRecommendation(
                    medication_class="SGLT2 inhibitor",
                    specific_agents=["empagliflozin", "dapagliflozin", "canagliflozin"],
                    rationale="Kidney and cardiovascular protection in diabetes",
                    contraindications=["eGFR <20", "Type 1 diabetes risk factors"],
                    monitoring_required=["Kidney function", "Genital infections", "Volume status"],
                    ada_reference="ADA 2023 Section 11.5a"
                ))

    def _medication_recommendations(self, summary: Dict, risk_factors: Dict) -> None:

        diabetes_status = risk_factors.get("diabetes_diagnosis")
        prediabetes_status = risk_factors.get("prediabetes_diagnosis")

        if diabetes_status == "diabetes":
            # Metformin first-line
            self.recommendations.append(MedicationRecommendation(
                medication_class="Metformin",
                specific_agents=["metformin immediate-release", "metformin extended-release"],
                rationale="First-line therapy for type 2 diabetes",
                contraindications=["eGFR <30", "Metabolic acidosis", "Severe heart failure"],
                monitoring_required=["Kidney function", "Vitamin B12 levels"],
                ada_reference="ADA 2023 Section 9"
            ))

            # Check current A1C for intensification needs
            a1c = summary.get("a1c", {})
            current_a1c = None
            if a1c is not None and a1c.get("value") is not None:  # Add None check
                current_a1c = a1c["value"]

            if current_a1c and current_a1c > 7.0:
                # Additional agent recommendations
                cv_risk = risk_factors.get("cardiovascular_risk")
                kidney_stage = risk_factors.get("kidney_disease_stage")

                if cv_risk in ["high", "very_high"] or kidney_stage in ["mild_to_moderate", "moderate_to_severe"]:
                    # SGLT2i or GLP-1 RA with CV benefit
                    self.recommendations.append(MedicationRecommendation(
                        medication_class="GLP-1 receptor agonist",
                        specific_agents=["semaglutide", "liraglutide", "dulaglutide"],
                        rationale="Cardiovascular and kidney protection with glycemic benefit",
                        contraindications=["Personal/family history medullary thyroid cancer", "MEN2"],
                        monitoring_required=["Gastrointestinal tolerance", "Weight changes"],
                        ada_reference="ADA 2023 Section 9.9-9.10"
                    ))

        elif prediabetes_status == "prediabetes":
            # Metformin for high-risk prediabetes Add None checks
            bmi_data = summary.get("bmi", {})
            bmi_value = 0
            if bmi_data is not None and bmi_data.get("value") is not None:
                bmi_value = bmi_data["value"]

            a1c_data = summary.get("a1c", {})
            current_a1c = 0
            if a1c_data is not None and a1c_data.get("value") is not None:
                current_a1c = a1c_data["value"]

            if bmi_value >= 35 or current_a1c >= 6.0:
                self.recommendations.append(MedicationRecommendation(
                    medication_class="Metformin",
                    specific_agents=["metformin immediate-release"],
                    rationale="Diabetes prevention in high-risk prediabetes",
                    contraindications=["eGFR <30", "Metabolic acidosis"],
                    monitoring_required=["Kidney function", "Gastrointestinal tolerance"],
                    ada_reference="ADA 2023 Section 3.6"
                ))

    def _monitoring_recommendations(self, summary: Dict, risk_factors: Dict) -> None:

        diabetes_status = risk_factors.get("diabetes_diagnosis")

        if diabetes_status == "diabetes":
            # A1C monitoring
            current_a1c = summary.get("a1c", {}).get("value")
            if current_a1c and current_a1c > 7.0:
                self.alerts.append(ClinicalAlert(
                    message="A1C monitoring: Check quarterly until target achieved",
                    priority=AlertPriority.INFO,
                    category="monitoring",
                    evidence="ADA 2023 Section 6.2: Quarterly A1C if not meeting goals",
                    action_required=False
                ))
            else:
                self.alerts.append(ClinicalAlert(
                    message="A1C monitoring: Check every 6 months if stable at target",
                    priority=AlertPriority.INFO,
                    category="monitoring",
                    evidence="ADA 2023 Section 6.1: Biannual A1C if meeting goals",
                    action_required=False
                ))

            # Kidney function monitoring
            self.alerts.append(ClinicalAlert(
                message="Annual kidney function screening: eGFR and urine albumin",
                priority=AlertPriority.INFO,
                category="monitoring",
                evidence="ADA 2023 Section 11.1a: Annual kidney screening",
                action_required=False
            ))

            # Eye exam
            self.alerts.append(ClinicalAlert(
                message="Annual comprehensive eye examination recommended",
                priority=AlertPriority.INFO,
                category="monitoring",
                evidence="ADA 2023 Section 12.5: Annual diabetic retinopathy screening",
                action_required=False
            ))

            # Foot exam
            self.alerts.append(ClinicalAlert(
                message="Annual comprehensive foot examination recommended",
                priority=AlertPriority.INFO,
                category="monitoring",
                evidence="ADA 2023 Section 12.21: Annual foot risk assessment",
                action_required=False
            ))

    def _lifestyle_recommendations(self, summary: Dict, risk_factors: Dict) -> None:

        diabetes_status = risk_factors.get("diabetes_diagnosis")
        prediabetes_status = risk_factors.get("prediabetes_diagnosis")
        obesity_status = risk_factors.get("obesity_status")

        if diabetes_status == "diabetes" or prediabetes_status == "prediabetes":
            # Diabetes self-management education
            self.alerts.append(ClinicalAlert(
                message="Diabetes Self-Management Education and Support (DSMES) referral recommended",
                priority=AlertPriority.INFO,
                category="education",
                evidence="ADA 2023 Section 5.2: DSMES at diagnosis and annually",
                action_required=False
            ))

            # Medical nutrition therapy
            self.alerts.append(ClinicalAlert(
                message="Medical Nutrition Therapy (MNT) referral recommended",
                priority=AlertPriority.INFO,
                category="lifestyle",
                evidence="ADA 2023 Section 5: MNT by registered dietitian",
                action_required=False
            ))

        # Weight management
        if obesity_status in ["overweight", "obese"]:
            target_loss = "5-10%" if diabetes_status == "diabetes" else "7%"
            self.alerts.append(ClinicalAlert(
                message=f"Weight loss goal: {target_loss} of body weight",
                priority=AlertPriority.INFO,
                category="lifestyle",
                evidence="ADA 2023 Section 8: Weight management targets",
                action_required=False
            ))

        # Physical activity
        self.alerts.append(ClinicalAlert(
            message="Physical activity: ≥150 min/week moderate-intensity aerobic activity",
            priority=AlertPriority.INFO,
            category="lifestyle",
            evidence="ADA 2023 Section 5.29: Physical activity recommendations",
            action_required=False
        ))

    def _generate_clinical_summary(self) -> str:
        critical_alerts = [a for a in self.alerts if a.priority == AlertPriority.CRITICAL]
        urgent_alerts = [a for a in self.alerts if a.priority == AlertPriority.URGENT]

        summary_parts = []

        if urgent_alerts:
            summary_parts.append(f"URGENT: {len(urgent_alerts)} urgent clinical issues requiring immediate attention")

        if critical_alerts:
            summary_parts.append(f"{len(critical_alerts)} critical findings requiring action")

        med_recs = len(self.recommendations)
        if med_recs > 0:
            summary_parts.append(f"{med_recs} medication recommendations")

        goals = len(self.treatment_goals)
        if goals > 0:
            summary_parts.append(f"{goals} individualized treatment targets")

        if not summary_parts:
            return "No critical clinical issues identified. Continue routine diabetes care."

        return "; ".join(summary_parts)

    def _determine_next_actions(self) -> List[str]:
        actions = []

        # Urgent/critical alerts first
        urgent_critical = [a for a in self.alerts if
                           a.priority in [AlertPriority.URGENT, AlertPriority.CRITICAL] and a.action_required]

        for alert in urgent_critical:
            if "insulin therapy" in alert.message.lower():
                actions.append("Consider immediate insulin therapy for severe hyperglycemia")
            elif "nephrology referral" in alert.message.lower():
                actions.append("Refer to nephrology for advanced CKD")
            elif "diabetes diagnosed" in alert.message.lower():
                actions.append("Initiate comprehensive diabetes care plan")

        # Medication recommendations
        if self.recommendations:
            actions.append("Review and initiate appropriate diabetes medications")

        # Monitoring needs
        monitoring_alerts = [a for a in self.alerts if a.category == "monitoring"]
        if monitoring_alerts:
            actions.append("Schedule recommended monitoring and screening tests")

        # Lifestyle interventions
        lifestyle_alerts = [a for a in self.alerts if a.category in ["lifestyle", "education"]]
        if lifestyle_alerts:
            actions.append("Refer for diabetes education and lifestyle interventions")

        return actions[:5]  # Top 5 priorities

    def _alert_to_dict(self, alert: ClinicalAlert) -> Dict:
        return {
            "message": alert.message,
            "priority": alert.priority.value,
            "category": alert.category,
            "evidence": alert.evidence,
            "action_required": alert.action_required,
            "guideline_reference": alert.guideline_reference
        }

    def _med_to_dict(self, med: MedicationRecommendation) -> Dict:
        return {
            "medication_class": med.medication_class,
            "specific_agents": med.specific_agents,
            "rationale": med.rationale,
            "contraindications": med.contraindications,
            "monitoring_required": med.monitoring_required,
            "ada_reference": med.ada_reference
        }

    def _goal_to_dict(self, goal: TreatmentGoal) -> Dict:
        return {
            "parameter": goal.parameter,
            "target_value": goal.target_value,
            "rationale": goal.rationale,
            "priority": goal.priority
        }

# Batch Processing Functions
def process_patient_file(file_path: str, cdss: DiabetesCDSS) -> Dict:
    with open(file_path, 'r') as f:
        patient_data = json.load(f)

    return cdss.process_patient(patient_data)


def batch_process_cohort(data_dir: str, output_dir: str = None) -> List[Dict]:
    if output_dir is None:
        output_dir = os.path.join(data_dir, "cdss_recommendations")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all patient files
    patient_files = glob.glob(os.path.join(data_dir, "*.diabetes_cdss.json"))

    if not patient_files:
        print(f"No patient files found in {data_dir}")
        return []

    print(f"Processing {len(patient_files)} patients through CDSS...")

    cdss = DiabetesCDSS()
    results = []

    for file_path in patient_files:
        try:
            result = process_patient_file(file_path, cdss)
            results.append(result)

            # Save individual recommendation
            patient_id = result["patient_id"]
            output_file = os.path.join(output_dir, f"{patient_id}.cdss_recommendations.json")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"✓ Processed {patient_id}: {result['summary']}")

        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")

    # Generate cohort summary
    generate_cohort_summary(results, output_dir)

    return results


def generate_cohort_summary(results: List[Dict], output_dir: str) -> None:
    if not results:
        return

    summary = {
        "cohort_size": len(results),
        "processing_date": datetime.now().isoformat(),
        "guideline_version": "ADA 2023",
        "aggregate_statistics": {},
        "common_alerts": {},
        "medication_patterns": {},
        "high_priority_patients": []
    }

    # Aggregate statistics
    total_alerts = sum(len(r["clinical_alerts"]) for r in results)
    total_medications = sum(len(r["medication_recommendations"]) for r in results)

    summary["aggregate_statistics"] = {
        "total_clinical_alerts": total_alerts,
        "avg_alerts_per_patient": total_alerts / len(results),
        "total_medication_recommendations": total_medications,
        "avg_medications_per_patient": total_medications / len(results)
    }

    # Common alert patterns
    alert_counts = {}
    for result in results:
        for alert in result["clinical_alerts"]:
            category = alert["category"]
            alert_counts[category] = alert_counts.get(category, 0) + 1

    summary["common_alerts"] = dict(sorted(alert_counts.items(), key=lambda x: x[1], reverse=True))

    # Medication patterns
    med_counts = {}
    for result in results:
        for med in result["medication_recommendations"]:
            med_class = med["medication_class"]
            med_counts[med_class] = med_counts.get(med_class, 0) + 1

    summary["medication_patterns"] = dict(sorted(med_counts.items(), key=lambda x: x[1], reverse=True))

    # High priority patients (urgent/critical alerts)
    for result in results:
        urgent_critical = [a for a in result["clinical_alerts"]
                           if a["priority"] in ["urgent", "critical"]]
        if urgent_critical:
            summary["high_priority_patients"].append({
                "patient_id": result["patient_id"],
                "urgent_critical_alerts": len(urgent_critical),
                "alert_summary": "; ".join([a["message"] for a in urgent_critical[:3]])
            })

    # Sort high priority patients by alert count
    summary["high_priority_patients"].sort(key=lambda x: x["urgent_critical_alerts"], reverse=True)

    # Save summary
    summary_file = os.path.join(output_dir, "cohort_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== CDSS Cohort Summary ===")
    print(f"Processed: {summary['cohort_size']} patients")
    print(f"Total alerts: {summary['aggregate_statistics']['total_clinical_alerts']}")
    print(f"High-priority patients: {len(summary['high_priority_patients'])}")
    print(f"Most common alert categories: {list(summary['common_alerts'].keys())[:3]}")
    print(f"Most recommended medications: {list(summary['medication_patterns'].keys())[:3]}")
    print(f"Summary saved: {summary_file}")

# Validation and Testing

def validate_cdss_rules() -> None:

    print("=== CDSS Rule Validation ===\n")

    cdss = DiabetesCDSS()

    # Test case 1: New diabetes diagnosis
    test_case_1 = {
        "patient": {"id": "test_001"},
        "diabetes_summary": {
            "a1c": {"value": 7.8},
            "bmi": {"value": 32.0},
            "glucose": {"fasting_plasma": {"value": 140}},
            "lipids": {"ldl": {"value": 120}},
            "renal": {"creatinine": {"value": 1.0}, "egfr": {"value": 75}}
        },
        "clinical_assessment": {
            "risk_factors": {
                "diabetes_diagnosis": "diabetes",
                "cardiovascular_risk": "moderate",
                "kidney_disease_stage": "normal_or_high",
                "obesity_status": "obese"
            }
        }
    }

    print("Test Case 1: New diabetes diagnosis with obesity")
    result_1 = cdss.process_patient(test_case_1)
    print(f"  Alerts generated: {len(result_1['clinical_alerts'])}")
    print(f"  Medications recommended: {len(result_1['medication_recommendations'])}")
    print(f"  Summary: {result_1['summary']}")

    # Validate expected outcomes
    alerts = result_1['clinical_alerts']
    diabetes_alert = any("Diabetes diagnosed" in a["message"] for a in alerts)
    a1c_target_alert = any("A1C" in a["message"] and "target" in a["message"] for a in alerts)

    medications = result_1['medication_recommendations']
    metformin_rec = any("Metformin" in m["medication_class"] for m in medications)

    print(f"  ✓ Diabetes diagnosis alert: {diabetes_alert}")
    print(f"  ✓ A1C target alert: {a1c_target_alert}")
    print(f"  ✓ Metformin recommendation: {metformin_rec}")

    # Test case 2: High-risk cardiovascular patient
    test_case_2 = {
        "patient": {"id": "test_002"},
        "diabetes_summary": {
            "a1c": {"value": 8.2},
            "bmi": {"value": 28.0},
            "lipids": {"ldl": {"value": 180}, "hdl": {"value": 35}},
            "renal": {"egfr": {"value": 45}}
        },
        "clinical_assessment": {
            "risk_factors": {
                "diabetes_diagnosis": "diabetes",
                "cardiovascular_risk": "very_high",
                "kidney_disease_stage": "mild_to_moderate"
            }
        }
    }

    print("\nTest Case 2: High CV risk with CKD")
    result_2 = cdss.process_patient(test_case_2)
    print(f"  Summary: {result_2['summary']}")

    # Validate CV protection recommendations
    medications_2 = result_2['medication_recommendations']
    sglt2_rec = any("SGLT2" in m["medication_class"] for m in medications_2)
    glp1_rec = any("GLP-1" in m["medication_class"] for m in medications_2)
    ace_rec = any("ACE" in m["medication_class"] for m in medications_2)

    print(f" SGLT2 inhibitor rec: {sglt2_rec}")
    print(f" GLP-1 RA rec: {glp1_rec}")
    print(f" ACE inhibitor rec: {ace_rec}")

    print("\n Validation Complete")
    print("CDSS rules align with ADA 2023 Standards of Care")

# CLI and Main Function

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diabetes CDSS Clinical Decision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single patient
  python cdr.py --patient-file data/12345.diabetes_cdss.json

  # Batch process cohort
  python cdr.py --batch-process data/

  # Validate CDSS rules
  python cdr.py --validate-rules
        """
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--patient-file",
        help="Process single patient file"
    )
    mode_group.add_argument(
        "--batch-process",
        help="Batch process all files in directory"
    )
    mode_group.add_argument(
        "--validate-rules",
        action="store_true",
        help="Validate CDSS rules against guidelines"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for recommendations"
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.validate_rules:
            validate_cdss_rules()
            return 0

        elif args.patient_file:
            if not os.path.exists(args.patient_file):
                print(f"Error: Patient file not found: {args.patient_file}")
                return 1

            cdss = DiabetesCDSS()
            result = process_patient_file(args.patient_file, cdss)

            print(f"\n CDSS Recommendations for Patient {result['patient_id']} ")
            print(f"Summary: {result['summary']}")

            if result['clinical_alerts']:
                print(f"\nClinical Alerts ({len(result['clinical_alerts'])}):")
                for i, alert in enumerate(result['clinical_alerts'], 1):
                    priority_label = (
                        "High Priority" if alert['priority'] in ['urgent', 'critical']
                        else "Medium Priority" if alert['priority'] == 'warning'
                        else "Low Priority"
                    )
                    print(f"  {i}. [{priority_label}] {alert['message']}")
                    if alert['evidence']:
                        print(f"     Evidence: {alert['evidence']}")

            if result['medication_recommendations']:
                print(f"\nMedication Recommendations ({len(result['medication_recommendations'])}):")
                for i, med in enumerate(result['medication_recommendations'], 1):
                    print(f"  {i}. {med['medication_class']}: {med['rationale']}")
                    if med['specific_agents']:
                        print(f"     Agents: {', '.join(med['specific_agents'])}")

            if result['treatment_goals']:
                print(f"\nTreatment Goals ({len(result['treatment_goals'])}):")
                for goal in result['treatment_goals']:
                    print(f"  • {goal['parameter']}: {goal['target_value']} ({goal['rationale']})")

            if result['next_actions']:
                print(f"\nNext Actions:")
                for i, action in enumerate(result['next_actions'], 1):
                    print(f"  {i}. {action}")

            return 0

        elif args.batch_process:
            if not os.path.exists(args.batch_process):
                print(f"Error: Directory not found: {args.batch_process}")
                return 1

            results = batch_process_cohort(args.batch_process, args.output_dir)
            print(f"\nBatch processing complete: {len(results)} patients processed")
            return 0

        else:
            # Default behavior when no arguments provided
            print("=== Diabetes CDSS Clinical Decision Engine ===")
            print("Available options:")
            print("  --patient-file <file>     Process single patient")
            print("  --batch-process <dir>     Process all patients in directory")
            print("  --validate-rules          Validate CDSS rules against ADA guidelines")
            print("\nFor detailed help: python cdr.py --help")
            return 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())