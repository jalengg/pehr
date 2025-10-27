"""
Generate and translate reconstructed patient records for human readability.
"""
import logging
import sys
import torch
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer
from generate import load_trained_model, generate_patient_sequence_conditional


# ICD-9 code descriptions (subset of common codes)
ICD9_DESCRIPTIONS = {
    # Neonatal codes
    'V3000': 'Single liveborn, born in hospital',
    'V3001': 'Single liveborn, born before admission',
    'V3101': 'Twin birth, mate liveborn, born in hospital',
    'V053': 'Prophylactic vaccination against viral hepatitis',
    'V290': 'Observation for suspected infectious condition',
    'V502': 'Routine circumcision',
    '7742': 'Other preterm infants',
    '769': 'Respiratory distress syndrome in newborn',
    '7461': 'Transitory tachypnea of newborn',
    '7454': 'Abnormal weight loss (neonatal)',
    '76515': 'Extreme immaturity, 1000-1249 grams',
    '76518': 'Extreme immaturity, 1500-1749 grams',
    '76525': 'Other preterm infants, 1000-1249 grams',
    '76527': 'Other preterm infants, 1500-1749 grams',
    '77081': 'Primary apnea',

    # Adult cardiovascular
    '42979': 'Atrial fibrillation',
    '42731': 'Atrial fibrillation',
    '4019': 'Hypertension, unspecified',
    '41401': 'Coronary atherosclerosis of native coronary artery',
    '412': 'Old myocardial infarction',
    '4139': 'Chronic ischemic heart disease, unspecified',
    '4240': 'Mitral valve disorders',
    '4280': 'Congestive heart failure, unspecified',
    '4241': 'Aortic valve disorders',
    '42831': 'Acute systolic heart failure',
    '42842': 'Chronic combined systolic and diastolic heart failure',
    '4270': 'Cardiac dysrhythmias',
    '42653': 'Ventricular premature beats',
    '4168': 'Other chronic pulmonary heart diseases',
    '431': 'Intracerebral hemorrhage',
    '4412': 'Dissection of aorta, thoracic',

    # Diabetes
    '25000': 'Diabetes mellitus without mention of complication, type II',
    '250.00': 'Diabetes mellitus without mention of complication, type II',
    '3572': 'Polyneuropathy in diabetes',

    # Metabolic
    '2720': 'Pure hypercholesterolemia',
    '2724': 'Other and unspecified hyperlipidemia',
    '2749': 'Disorder of lipoid metabolism, unspecified',
    '2449': 'Hypothyroidism, unspecified',
    '2765': 'Hypovolemia',
    '2768': 'Hypopotassemia',
    '2761': 'Hyposmolality and/or hyponatremia',
    '2762': 'Acidosis',
    '2773': 'Amyloidosis',

    # Renal
    '5855': 'Chronic kidney disease, Stage V',
    '5849': 'Acute kidney failure, unspecified',
    '5845': 'Acute kidney failure with lesion of tubular necrosis',
    '5990': 'Urinary tract infection, site not specified',
    '5601': 'Paralytic ileus',

    # Respiratory
    '496': 'Chronic airway obstruction, not elsewhere classified',
    '486': 'Pneumonia, organism unspecified',
    '49121': 'Obstructive chronic bronchitis with acute exacerbation',
    '49322': 'Chronic obstructive asthma with acute exacerbation',
    '49390': 'Asthma, unspecified type, unspecified',
    '515': 'Postinflammatory pulmonary fibrosis',
    '5119': 'Unspecified pleural effusion',

    # GI/Hepatic
    '5712': 'Alcoholic cirrhosis of liver',
    '5723': 'Portal hypertension',
    '5728': 'Other sequelae of chronic liver disease',
    '570': 'Acute and subacute necrosis of liver',
    '5070': 'Esophageal varices with bleeding',
    '53081': 'Esophageal reflux',
    '5570': 'Acute vascular insufficiency of intestine',

    # Hematologic
    '2851': 'Acute posthemorrhagic anemia',
    '2859': 'Anemia, unspecified',
    '2869': 'Coagulation defect, unspecified',
    '2875': 'Thrombocytopenia, unspecified',
    '2879': 'Hemorrhagic disorder, unspecified',

    # Infectious
    '0389': 'Unspecified septicemia',
    '0388': 'Other specified septicemias',
    '0417': 'Pseudomonas infection in conditions classified elsewhere',
    '07054': 'Hepatitis C without hepatic coma, chronic',

    # Mental/Neurological
    '311': 'Depressive disorder, not elsewhere classified',
    '30301': 'Acute alcoholic intoxication in alcohol dependence, continuous',
    '30411': 'Sedative, hypnotic or anxiolytic abuse, continuous',
    '30500': 'Alcohol abuse, unspecified',
    '34690': 'Migraine, unspecified',
    '34291': 'Hemiplegia, affecting unspecified side',

    # Trauma/Injury
    '9982': 'Accidental puncture or laceration during a procedure',
    '9974': 'Digestive system complications',
    '99702': 'Iatrogenic cerebrovascular infarction or hemorrhage',
    '99662': 'Infection and inflammatory reaction due to other vascular device',
    '99674': 'Other complications of internal orthopedic device',
    '9971': 'Cardiac complications',
    '99811': 'Hemorrhage complicating a procedure',
    '9992': 'Other and unspecified infection',

    # Other
    'V420': 'Kidney replaced by transplant',
    'V441': 'Gastrostomy status',
    'V4501': 'Cardiac pacemaker in situ',
    'V4582': 'Percutaneous transluminal coronary angioplasty status',
    'V422': 'Heart valve replaced by transplant',
    'V4589': 'Other postprocedural states',
    'V4975': 'Status post hip arthroplasty',
    'V1005': 'Personal history of malignant neoplasm of large intestine',
    'V1006': 'Personal history of malignant neoplasm of rectum',
    'V1052': 'Personal history of malignant neoplasm of bladder',
    'V1087': 'Personal history of malignant neoplasm of thyroid',
    'V1201': 'Personal history of nutritional deficiency',
    'V5861': 'Long-term (current) use of anticoagulants',
    'V0259': 'Need for prophylactic vaccination against other viral diseases',
    'V090': 'Infection with microorganisms resistant to penicillins',

    # Symptoms
    '7806': 'Fever',
    '7843': 'Lack of coordination',
    '7905': 'Other abnormal serum enzyme levels',
    '78039': 'Other convulsions',
    '79093': 'Elevated blood pressure reading without diagnosis of hypertension',

    # Injuries
    'E9479': 'Unspecified adverse effect of drug',
    'E9504': 'Suicide and self-inflicted poisoning by other specified drugs',
    'E9581': 'Suicide and self-inflicted injuries by other specified means',
    'E9342': 'Accident caused by powered hand tools',

    # Dermatologic
    '6826': 'Cellulitis and abscess of leg',
    '68110': 'Cellulitis and abscess of finger, unspecified',

    # Musculoskeletal
    '71536': 'Osteoarthrosis, localized, primary, lower leg',
    '72704': 'Lumbago',
    '4590': 'Hemorrhoids, unspecified',
    '70722': 'Pressure ulcer, stage II',
    '73382': 'Pathologic fracture of tibia and fibula',

    # Oncologic
    '19889': 'Secondary malignant neoplasm of other specified sites',

    # Endocrine
    '2387': 'Neoplasm of uncertain behavior of other lymphatic and hematopoietic tissues',
    '2448': 'Other specified acquired hypothyroidism',

    # Cardiac procedures/findings
    '41011': 'Acute transmural myocardial infarction of anterior wall',
    '41091': 'Acute myocardial infarction of unspecified site',
    '41071': 'Subendocardial infarction',
    '43411': 'Cerebral embolism with cerebral infarction',
    '43310': 'Occlusion and stenosis of precerebral arteries',
    '43889': 'Other late effects of cerebrovascular disease',
    '44020': 'Atherosclerosis of native arteries of the extremities',
    '44023': 'Atherosclerosis of native arteries with gangrene',
    '4148': 'Other specified forms of chronic ischemic heart disease',
    '45620': 'Hemorrhage, unspecified',
    '45829': 'Other chronic venous embolism and thrombosis',

    # GI continued
    '51881': 'Acute respiratory failure following trauma and surgery',
    '55221': 'Abscess of appendix',

    # Lab/diagnostic
    '28521': 'Anemia in chronic kidney disease',
    '28981': 'Primary hypercoagulable state',
    '29181': 'Other alcohol dependence',
}


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("translate")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def translate_code(code: str) -> str:
    """Translate ICD-9 code to human-readable description."""
    return ICD9_DESCRIPTIONS.get(code, f"[Unknown: {code}]")


def main():
    """Generate and translate sample reconstructions."""
    logger = setup_logging()

    # Load configuration
    config = Config.from_defaults()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    logger.info("Loading MIMIC-III data...")
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logging.getLogger("data_loader"),
        num_patients=config.data.num_patients
    )

    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Load model
    checkpoint_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_trained_model(checkpoint_path, tokenizer, config, device, logging.getLogger("model"))

    # Get test patients
    num_total = len(patient_records)
    num_train = int(num_total * 0.7)
    num_val = int(num_total * 0.15)
    test_patients = patient_records[num_train + num_val:]

    # Generate and translate 10 examples
    output_path = Path("reconstruction_results/reconstruction_results_translated.txt")
    output_path.parent.mkdir(exist_ok=True)

    logger.info(f"\nGenerating 10 translated reconstruction examples...")
    logger.info(f"Output: {output_path}\n")

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("RECONSTRUCTED PATIENT RECORDS - TRANSLATED\n")
        f.write("Multi-task Learning with Age & Sex Prediction\n")
        f.write("=" * 100 + "\n\n")

        for i, patient in enumerate(test_patients[:10], 1):
            # Generate reconstruction
            result = generate_patient_sequence_conditional(
                model=model,
                tokenizer=tokenizer,
                target_patient=patient,
                device=device,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                prompt_prob=0.5,
                max_codes_per_visit=20
            )

            demo = result['demographics']

            # Write header
            header = f"Patient {i} | Age: {demo['age']:.0f} | Sex: {demo['gender']}"
            f.write("-" * 100 + "\n")
            f.write(header + "\n")
            f.write("-" * 100 + "\n\n")

            # Write each visit
            for visit_idx, (gen_codes, tgt_codes, prompt_codes) in enumerate(
                zip(result['generated_visits'], result['target_visits'], result['prompt_codes']), 1
            ):
                f.write(f"VISIT {visit_idx}:\n\n")

                # Prompt codes (input)
                f.write(f"  Input Codes (Prompt): {len(prompt_codes)} codes\n")
                for code in prompt_codes[:5]:  # Show first 5
                    f.write(f"    • {code:8s} - {translate_code(code)}\n")
                if len(prompt_codes) > 5:
                    f.write(f"    ... and {len(prompt_codes) - 5} more\n")
                f.write("\n")

                # Target codes (ground truth)
                f.write(f"  Target Codes (Ground Truth): {len(tgt_codes)} codes\n")
                for code in tgt_codes[:5]:
                    f.write(f"    • {code:8s} - {translate_code(code)}\n")
                if len(tgt_codes) > 5:
                    f.write(f"    ... and {len(tgt_codes) - 5} more\n")
                f.write("\n")

                # Generated codes (reconstruction)
                f.write(f"  Generated Codes (Reconstruction): {len(gen_codes)} codes\n")

                # Check for duplicates
                seen = set()
                duplicates = []
                for code in gen_codes:
                    if code in seen:
                        duplicates.append(code)
                    seen.add(code)

                # Check for age-inappropriate codes
                age_issues = []
                neonatal_prefixes = ['V30', 'V31', '76', '77', 'V502', 'V290']
                adult_codes = {'42979': 18, '3572': 10, '34690': 10}

                for code in gen_codes:
                    # Check neonatal codes
                    if demo['age'] > 1:
                        for prefix in neonatal_prefixes:
                            if code.startswith(prefix):
                                age_issues.append((code, f"neonatal code for age {demo['age']:.0f}"))
                                break

                    # Check adult minimum ages
                    if code in adult_codes and demo['age'] < adult_codes[code]:
                        age_issues.append((code, f"requires age >={adult_codes[code]}, got {demo['age']:.0f}"))

                for code in gen_codes[:10]:  # Show first 10
                    desc = translate_code(code)
                    markers = []
                    if code in duplicates:
                        markers.append("⚠ DUPLICATE")
                    if any(code == issue_code for issue_code, _ in age_issues):
                        issue_msg = next(msg for ic, msg in age_issues if ic == code)
                        markers.append(f"⚠ AGE: {issue_msg}")

                    marker_str = f" [{', '.join(markers)}]" if markers else ""
                    f.write(f"    • {code:8s} - {desc}{marker_str}\n")

                if len(gen_codes) > 10:
                    f.write(f"    ... and {len(gen_codes) - 10} more\n")

                # Compute Jaccard
                jaccard = len(set(gen_codes) & set(tgt_codes)) / len(set(gen_codes) | set(tgt_codes))
                f.write(f"\n  Jaccard Similarity: {jaccard:.3f}\n")

                if duplicates:
                    f.write(f"  ⚠ Duplicates: {len(duplicates)} ({', '.join(set(duplicates))})\n")
                if age_issues:
                    f.write(f"  ⚠ Age-inappropriate: {len(age_issues)} codes\n")

                f.write("\n")

            f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("END OF TRANSLATED RECONSTRUCTIONS\n")
        f.write("=" * 100 + "\n")

    logger.info(f"✓ Translations saved to {output_path}")
    logger.info("\nFirst example preview:")
    logger.info("-" * 80)

    # Print first example to console
    with open(output_path, 'r') as f:
        lines = f.readlines()
        in_first_patient = False
        line_count = 0
        for line in lines:
            if "Patient 1 |" in line:
                in_first_patient = True
            if in_first_patient:
                print(line.rstrip())
                line_count += 1
                if line_count > 50:  # Print first 50 lines of first patient
                    break


if __name__ == "__main__":
    main()
