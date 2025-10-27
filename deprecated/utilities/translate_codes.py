"""
Translate ICD-9 diagnosis codes to human-readable descriptions.
Uses a built-in ICD-9 code mapping for common codes.
"""
import re
from pathlib import Path


# Common ICD-9 codes and descriptions
ICD9_DESCRIPTIONS = {
    # Newborn/Birth codes
    'V3000': 'Single liveborn, born in hospital',
    'V3001': 'Single liveborn, born before admission',
    'V3101': 'Twin birth, mate liveborn, born in hospital',
    'V053': 'Need for prophylactic vaccination - viral hepatitis',
    'V290': 'Observation for suspected infectious condition',
    'V502': 'Routine circumcision',

    # Cardiovascular
    '4019': 'Hypertension, unspecified',
    '40301': 'Hypertensive chronic kidney disease, benign',
    '41401': 'Coronary atherosclerosis of native coronary artery',
    '4139': 'Other chronic ischemic heart disease',
    '412': 'Old myocardial infarction',
    '4240': 'Mitral valve disorders',
    '42731': 'Atrial fibrillation',
    '42789': 'Other cardiac dysrhythmias',
    '42979': 'Atrial fibrillation',
    '428.0': 'Congestive heart failure, unspecified',
    '4295': 'Other heart disease',
    '42091': 'Acute myocarditis',

    # Diabetes/Endocrine
    '250.00': 'Diabetes mellitus without mention of complication, type II',
    '2449': 'Unspecified acquired hypothyroidism',
    '2639': 'Unspecified protein-calorie malnutrition',
    '2720': 'Pure hypercholesterolemia',
    '2749': 'Unspecified disorder of lipoid metabolism',
    '2765': 'Hypokalemia',
    '2768': 'Hypopotassemia',
    '2761': 'Hyposmolality and/or hyponatremia',

    # Respiratory
    '486': 'Pneumonia, organism unspecified',
    '49390': 'Asthma, unspecified type, unspecified',
    '48283': 'Chronic respiratory failure',

    # Renal
    '584.9': 'Acute kidney failure, unspecified',
    '585.9': 'Chronic kidney disease, unspecified',
    '5855': 'Chronic kidney disease, Stage V',
    '51881': 'Acute kidney failure',
    '5180': 'Acute kidney failure with lesion of tubular necrosis',
    '5990': 'Urinary tract infection, site not specified',
    '5991': 'Urinary tract infection, site not specified',
    '5198': 'Other specified disorders of kidney and ureter',

    # Injury/Trauma
    '8052': 'Closed fracture of nasal bones',
    '8054': 'Closed fracture of malar and maxillary bones',
    '8056': 'Closed fracture of orbital floor (blow-out)',
    '8080': 'Closed fracture of acetabulum',
    '8082': 'Closed fracture of pubis',
    '8248': 'Multiple open fractures of hand bones',
    '81241': 'Fracture of unspecified phalanx or phalanges of foot',
    '83500': 'Unspecified intracapsular fracture of neck of femur',

    # External causes
    'E8160': 'Motor vehicle traffic accident involving collision',
    'E8217': 'Accidental drowning and submersion',
    'E8494': 'Accidents caused by other specified machinery',
    'E8889': 'Unspecified fall',
    'E9500': 'Suicide and self-inflicted poisoning',
    'E912': 'Inhalation and ingestion of food causing obstruction',

    # Infectious
    '0389': 'Unspecified septicemia',
    '570': 'Acute and subacute necrosis of liver',

    # Neonatal
    '769': 'Respiratory distress syndrome in newborn',
    '7423': 'Neonatal jaundice associated with preterm delivery',
    '7454': 'Abnormal weight loss',
    '7461': 'Transitory tachypnea of newborn',
    '74365': 'Late metabolic acidosis of newborn',
    '7626': 'Feeding problems in newborn',
    '7706': 'Other fetal and neonatal jaundice',
    '7742': 'Other preterm infants',
    '75501': 'Disruption of cesarean wound',
    '76515': 'Extreme immaturity, 1000-1249 grams',
    '76519': 'Extreme immaturity, 2000-2499 grams',
    '76525': 'Other preterm infants, 1000-1249 grams',
    '76528': 'Other preterm infants, 2000-2499 grams',
    '77989': 'Other specified conditions originating in perinatal period',

    # Mental/Substance
    '9654': 'Cocaine dependence, unspecified',
    '9982': 'Tracheostomy status',
    '9961': 'Mechanical complication of cardiac device',
    '99592': 'Other postoperative infection',
    '99602': 'Infection and inflammatory reaction due to other vascular device',

    # Other
    '1175': 'Tuberculosis of other specified organs',
    '1539': 'Malignant neoplasm of colon, unspecified site',
    '1603': 'Malignant neoplasm of stomach, fundus',
    '2859': 'Anemia, unspecified',
    '3313': 'Communicating hydrocephalus',
    '34690': 'Migraine, unspecified',
    '35789': 'Idiopathic peripheral autonomic neuropathy',
    '3572': 'Polyneuropathy in diabetes',
    '3688': 'Other specified cerebral degenerations',
    '4582': 'Hypotension, unspecified',
    '43811': 'Crohn\'s disease of large intestine',
    '43491': 'Gastrojejunal ulcer, chronic without obstruction',
    '46451': 'Intestinal obstruction, unspecified',
    '5070': 'Acute pancreatitis',
    '5793': 'Intestinal malabsorption, unspecified',
    '70400': 'Rheumatoid arthritis',
    '70703': 'Osteoarthrosis, localized, primary, forearm',
    '7098': 'Other specified arthropathies',
    '73025': 'Polyostotic fibrous dysplasia',
    '7907': 'Bacteremia',
    '80224': 'Closed fracture of nasal bones with other facial bones',
    '8026': 'Closed fracture of orbital floor',
    '82101': 'Fracture of thoracic vertebra without mention of spinal cord injury',
    '85200': 'Pneumonitis due to inhalation of food or vomitus',
    '90253': 'Late effect of accident due to motor vehicle',
    '9331': 'Cerebral anoxia',
    'V0381': 'Need for prophylactic vaccination against Hemophilus influenza',
    'V1582': 'Personal history of tobacco use',
    'V420': 'Kidney replaced by transplant',
}


def translate_code(code: str) -> str:
    """Translate ICD-9 code to description.

    Args:
        code: ICD-9 diagnosis code.

    Returns:
        Human-readable description or 'Unknown' if not in dictionary.
    """
    # Remove any numpy string wrapper
    code_clean = str(code).replace("np.str_('", "").replace("')", "").strip()

    if code_clean in ICD9_DESCRIPTIONS:
        return ICD9_DESCRIPTIONS[code_clean]
    else:
        return f"Unknown ICD-9 code: {code_clean}"


def translate_reconstruction_file(input_file: str, output_file: str):
    """Translate reconstruction results to human-readable format.

    Args:
        input_file: Path to reconstruction_results.txt
        output_file: Path to output translated file
    """
    with open(input_file, 'r') as f:
        content = f.read()

    # Parse and translate
    output_lines = []
    output_lines.append("# TRANSLATED ICD-9 CODES\n")
    output_lines.append("# Human-Readable Reconstruction Results\n\n")

    for line in content.split('\n'):
        output_lines.append(line + '\n')

        # Translate Target Codes
        if 'Target Codes' in line:
            codes_match = re.findall(r"'([^']+)'", line)
            if codes_match:
                output_lines.append("Target Descriptions:\n")
                for code in codes_match:
                    desc = translate_code(code)
                    output_lines.append(f"  - {code}: {desc}\n")

        # Translate Prompt Codes
        elif 'Prompt Codes' in line:
            codes_match = re.findall(r"'([^']+)'", line)
            if codes_match:
                output_lines.append("Prompt Descriptions:\n")
                for code in codes_match:
                    desc = translate_code(code)
                    output_lines.append(f"  - {code}: {desc}\n")

        # Translate Generated Codes
        elif 'Generated Codes' in line:
            codes_match = re.findall(r"'([^']+)'", line)
            if codes_match:
                output_lines.append("Generated Descriptions:\n")
                for code in codes_match:
                    desc = translate_code(code)
                    output_lines.append(f"  - {code}: {desc}\n")

    # Write output
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"Translated results saved to: {output_file}")


if __name__ == '__main__':
    input_file = 'reconstruction_results/reconstruction_results.txt'
    output_file = 'reconstruction_results/reconstruction_results_translated.txt'

    if Path(input_file).exists():
        translate_reconstruction_file(input_file, output_file)
    else:
        print(f"Error: {input_file} not found")
