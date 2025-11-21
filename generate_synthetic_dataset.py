"""Generate large synthetic EHR dataset and display translated examples."""
import torch
import logging
import json
import csv
from pathlib import Path
from collections import Counter

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_with_structure_constraints
from visit_structure_sampler import VisitStructureSampler

# Comprehensive ICD-9 code descriptions
ICD9_DESCRIPTIONS = {
    # Cardiovascular - Ischemic Heart Disease
    '41011': 'Acute ST elevation myocardial infarction, anterior wall',
    '41041': 'Acute ST elevation myocardial infarction, inferior wall',
    '41071': 'Subendocardial infarction',
    '4111': 'Intermediate coronary syndrome (unstable angina)',
    '4139': 'Chronic ischemic heart disease, unspecified',
    '41401': 'Coronary atherosclerosis of native coronary artery',
    '4148': 'Chronic ischemic heart disease, other specified',
    '4412': 'Coronary atherosclerosis of unspecified vessel type',

    # Cardiovascular - Heart Failure & Arrhythmias
    '42781': 'Sinoatrial node dysfunction',
    '42731': 'Atrial fibrillation',
    '42741': 'Atrial fibrillation',
    '4240': 'Congestive heart failure, unspecified',
    '4241': 'Left heart failure',
    '42821': 'Acute systolic heart failure',
    '42823': 'Acute on chronic systolic heart failure',
    '42831': 'Acute diastolic heart failure',
    '4280': 'Congestive heart failure, unspecified',

    # Cardiovascular - Valvular & Other
    '4241': 'Left heart failure',
    '42490': 'Unspecified disease of endocardium',
    '4254': 'Cardiomyopathy, other primary',
    '4260': 'Atrioventricular block, complete',
    '4270': 'Paroxysmal supraventricular tachycardia',

    # Vascular Disease
    '4400': 'Atherosclerosis of aorta',
    '4404': 'Chronic total occlusion of artery of the extremities',
    '44020': 'Atherosclerosis of native arteries of extremities',
    '44023': 'Atherosclerosis of bypass graft of extremities',
    '4408': 'Other peripheral vascular disease',
    '4412': 'Coronary atherosclerosis',
    '4414': 'Coronary atherosclerosis of artery bypass graft',
    '4419': 'Chronic total occlusion of coronary artery',

    # Cerebrovascular
    '43310': 'Occlusion and stenosis of basilar artery',
    '43320': 'Occlusion of cerebral arteries',
    '43330': 'Occlusion of multiple and bilateral precerebral arteries',
    '43381': 'Acute cerebrovascular insufficiency',
    '43400': 'Cerebral thrombosis without infarction',
    '43411': 'Cerebral embolism with cerebral infarction',
    '43491': 'Cerebral artery occlusion with cerebral infarction',

    # Hypertension
    '4011': 'Benign essential hypertension',
    '4019': 'Unspecified essential hypertension',
    '40390': 'Hypertensive chronic kidney disease',
    '40490': 'Hypertensive heart and chronic kidney disease',

    # Diabetes
    '25000': 'Diabetes mellitus type II without complication',
    '25001': 'Diabetes mellitus type I without complication',
    '25040': 'Diabetes with renal manifestations, type II',
    '25050': 'Diabetes with ophthalmic manifestations, type II',
    '25060': 'Diabetes with neurological manifestations, type II',
    '25070': 'Diabetes with peripheral circulatory disorders, type II',
    '25080': 'Diabetes with other specified manifestations, type II',

    # Renal
    '5845': 'Acute kidney failure with tubular necrosis',
    '5849': 'Acute kidney failure, unspecified',
    '5853': 'Chronic kidney disease, Stage III',
    '5854': 'Chronic kidney disease, Stage IV',
    '5855': 'Chronic kidney disease, Stage V',
    '5856': 'End stage renal disease',
    '5990': 'Urinary tract infection, site not specified',

    # Respiratory
    '4168': 'Chronic pulmonary heart disease, unspecified',
    '486': 'Pneumonia, organism unspecified',
    '496': 'Chronic airway obstruction, not elsewhere classified',
    '5119': 'Unspecified pleural effusion',
    '51881': 'Acute respiratory failure',
    '51884': 'Acute and chronic respiratory failure',
    '5185': 'Pulmonary insufficiency following trauma and surgery',

    # Gastrointestinal & Hepatic
    '5712': 'Alcoholic cirrhosis of liver',
    '5715': 'Cirrhosis of liver without mention of alcohol',
    '5719': 'Unspecified chronic liver disease',
    '5723': 'Portal hypertension',
    '57410': 'Cholelithiasis (gallstones)',
    '5781': 'Hepatomegaly',
    '5789': 'Ascites',

    # Hematologic & Oncologic
    '2189': 'Benign neoplasm of other specified sites',
    '2851': 'Acute posthemorrhagic anemia',
    '2859': 'Anemia, unspecified',
    '28521': 'Anemia in chronic kidney disease',
    '2859': 'Anemia, unspecified',

    # Metabolic & Endocrine
    '2720': 'Pure hypercholesterolemia',
    '2724': 'Hyperlipidemia, unspecified',
    '2762': 'Acidosis',
    '2765': 'Volume depletion (dehydration)',
    '2766': 'Fluid overload',
    '2767': 'Hyperpotassemia (hyperkalemia)',
    '2768': 'Hypopotassemia (hypokalemia)',
    '27651': 'Dehydration',

    # Mental Health & Neurological
    '2939': 'Unspecified transient mental disorder',
    '3040': 'Anxiety state, unspecified',
    '311': 'Depressive disorder',
    '3310': "Alzheimer's disease",
    '3319': 'Dementia, unspecified',
    '3481': 'Anoxic brain damage',
    '3510': 'Idiopathic peripheral autonomic neuropathy',
    '3572': 'Polyneuropathy in diabetes',

    # Ophthalmologic
    '36201': 'Background diabetic retinopathy',
    '36203': 'Proliferative diabetic retinopathy',
    '3659': 'Retinal disorder, unspecified',
    '3661': 'Hypertensive retinopathy',
    '36523': 'Retinal vascular occlusion, branch artery',
    '3663': 'Retinal edema',
    '3689': 'Retinal disorder, unspecified',
    '3963': 'Aphakia (absence of lens)',
    '3968': 'Cataract, unspecified',

    # Otolaryngologic
    '38900': 'Otitis media, unspecified',

    # Musculoskeletal
    '71535': 'Osteoarthrosis, localized, primary, pelvic region',
    '71595': 'Osteoarthrosis, unspecified, pelvic region and thigh',
    '71946': 'Arthropathy, unspecified, lower leg',
    '7213': 'Lumbosacral spondylosis without myelopathy',
    '72190': 'Spondylosis, unspecified',

    # Dermatologic
    '6826': 'Cellulitis of leg',
    '68110': 'Cellulitis of finger',
    '7070': 'Pressure ulcer, unspecified stage',

    # Genitourinary
    '5990': 'Urinary tract infection',
    '59651': 'Cystostomy infection',
    '6002': 'Benign prostatic hyperplasia with urinary obstruction',
    '60300': 'Acute prostatitis',
    '63411': 'Incomplete uterine prolapse',

    # Complications & Injuries
    '8363': 'Nervous system complications from device/implant',
    '83801': 'Infection due to other vascular device/implant',
    '9971': 'Cardiac complications following procedure',
    '9982': 'Accidental puncture or laceration during procedure',
    '9985': 'Postoperative infection',
    '9997': 'Iatrogenic pulmonary embolism and infarction',
    '81323': 'Closed fracture of fourth metacarpal bone',

    # External Causes
    'E8152': 'Accidental fall from building or structure',
    'E9344': 'Accident caused by machinery',

    # Lab Findings & Symptoms
    '78837': 'Dysphagia, pharyngeal phase',
    '79092': 'Abnormal coagulation profile',
    '7907': 'Bacteremia',
    '78959': 'Other ascites',

    # Procedural/Status Codes
    'V4281': 'Bone marrow replaced by transplant',
    'V4501': 'Cardiac pacemaker in situ',
    'V4582': 'Status post percutaneous transluminal coronary angioplasty',
    'V5861': 'Long-term use of anticoagulants',

    # Birth weight codes (neonatal - age-inappropriate for adults)
    '76503': 'Extremely low birth weight, 500-749g [NEONATAL]',
    '76514': 'Extremely low birth weight, 1000-1249g [NEONATAL]',
    '76518': 'Extremely low birth weight, 1500-1749g [NEONATAL]',
    '76524': 'Other preterm infant, 2000-2499g [NEONATAL]',
    '76525': 'Other preterm infant, 2000-2499g [NEONATAL]',
    '76529': 'Other preterm infant, unspecified weight [NEONATAL]',

    # Common V codes
    'V182': 'Asymptomatic HIV infection status',
    'V290': 'Observation for suspected infectious condition',
    'V3001': 'Single liveborn, born before hospital admission',
    'V5391': 'Tracheostomy status',
    'V708': 'General medical examination',
    'V721': 'Observation following accident',

    # Other
    '4162': 'Chronic pulmonary embolism',
    '30181': 'Alcohol abuse, continuous',
    '86132': 'Drug-induced mental disorders',
    '2357': 'Dysthymic disorder',
    '217': 'Benign neoplasm of breast',
}

def translate_code(code: str) -> str:
    """Translate ICD-9 code to human-readable description."""
    return ICD9_DESCRIPTIONS.get(code, f"Unknown code")


def main():
    """Generate large synthetic dataset and display examples."""
    # Setup
    logging.basicConfig(level=logging.WARNING)
    config = Config.from_defaults()
    device = torch.device("cpu")

    print("="*80)
    print("SYNTHETIC EHR DATASET GENERATION")
    print("="*80)
    print()

    # Load data
    print("Loading MIMIC-III data...", flush=True)
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logging.getLogger("data_loader"),
        num_patients=config.data.num_patients
    )
    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Initialize visit structure sampler
    print("Initializing visit structure sampler...", flush=True)
    structure_sampler = VisitStructureSampler(patient_records, seed=42)
    print(f"  {structure_sampler}", flush=True)

    # Load 20-epoch model
    print("Loading 20-epoch model...", flush=True)
    checkpoint_path = "/scratch/jalenj4/promptehr_checkpoints/best_model.pt"
    model = load_trained_model(checkpoint_path, tokenizer, config, device, logging.getLogger("model"))

    # Generate 200 patients (CPU-friendly)
    n_patients = 200
    print(f"\nGenerating {n_patients} synthetic patients...", flush=True)
    print("(This will take 2-3 minutes)\n")

    all_patients = []
    all_codes = []

    for i in range(n_patients):
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_patients} patients...", flush=True)

        # Sample realistic visit structure
        target_structure = structure_sampler.sample_structure()

        result = generate_patient_with_structure_constraints(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_structure=target_structure,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )

        demo = result['demographics']
        patient_data = {
            'patient_id': f"SYNTH_{i+1:04d}",
            'age': demo['age'],
            'sex': 'M' if demo['sex'] == 0 else 'F',
            'num_visits': result['num_visits'],
            'visits': result['generated_visits']
        }
        all_patients.append(patient_data)

        # Collect codes for diversity analysis
        for visit in result['generated_visits']:
            all_codes.extend(visit)

    print(f"\n  Complete! Generated {n_patients} patients.")

    # Save to CSV
    output_file = Path("synthetic_patients_200.csv")
    print(f"\nSaving to {output_file}...", flush=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'age', 'sex', 'num_visits', 'visit_num', 'diagnosis_codes'])

        for patient in all_patients:
            for visit_idx, visit_codes in enumerate(patient['visits']):
                codes_str = ';'.join(visit_codes)
                writer.writerow([
                    patient['patient_id'],
                    f"{patient['age']:.1f}",
                    patient['sex'],
                    patient['num_visits'],
                    visit_idx + 1,
                    codes_str
                ])

    print(f"  Saved {len(all_patients)} patients to {output_file}")

    # Diversity analysis
    unique_codes = set(all_codes)
    code_freq = Counter(all_codes)

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}\n")
    print(f"  Total patients: {len(all_patients)}")
    print(f"  Total visits: {sum(p['num_visits'] for p in all_patients)}")
    print(f"  Total diagnosis codes: {len(all_codes)}")
    print(f"  Unique codes: {len(unique_codes)}")
    print(f"  Vocabulary coverage: {len(unique_codes)/6985*100:.2f}%")
    print(f"  Average codes per patient: {len(all_codes)/n_patients:.1f}")
    print(f"  Average visits per patient: {sum(p['num_visits'] for p in all_patients)/n_patients:.2f}")

    # Display 20 examples with full translations
    print(f"\n{'='*80}")
    print("SAMPLE PATIENTS (20 Examples with Full ICD-9 Translations)")
    print(f"{'='*80}\n")

    for i in range(20):
        patient = all_patients[i]
        sex_str = 'Male' if patient['sex'] == 'M' else 'Female'

        print(f"╔═══ {patient['patient_id']} {'═'*62}")
        print(f"║  Age: {patient['age']:.1f} years")
        print(f"║  Sex: {sex_str}")
        print(f"║  Visits: {patient['num_visits']}")
        print(f"╚{'═'*78}")
        print()

        if patient['num_visits'] == 0:
            print("  (No visits generated)")
        else:
            for visit_idx, visit_codes in enumerate(patient['visits']):
                print(f"  📋 Visit {visit_idx + 1}: {len(visit_codes)} diagnoses")
                print()
                for code in visit_codes:
                    description = translate_code(code)
                    # Mark if unknown
                    if description == "Unknown code":
                        print(f"     • {code:8s} [{code} - not in translation dictionary]")
                    else:
                        print(f"     • {code:8s} {description}")
                print()

        print("-" * 80)
        print()

    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")
    print(f"Full dataset: {output_file}")
    print(f"Total patients: {n_patients}")
    print(f"Unique diagnosis codes: {len(unique_codes)}")
    print()


if __name__ == "__main__":
    main()
