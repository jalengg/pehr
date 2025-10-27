"""
Decode generated patient sequences into readable medical diagnoses.
"""
import re

# Comprehensive ICD-9 code descriptions (curated common codes)
ICD9_CODES = {
    # Infectious and Parasitic Diseases (001-139)
    "075": "Infectious mononucleosis",
    "0389": "Septicemia, unspecified",
    "0839": "Other specified intestinal infections",

    # Neoplasms (140-239)
    "1419": "Malignant neoplasm of tongue, unspecified",
    "1642": "Malignant neoplasm of upper lobe, bronchus or lung",
    "1174": "Malignant neoplasm of breast (female), unspecified",

    # Endocrine, Nutritional, Metabolic (240-279)
    "25000": "Diabetes mellitus without mention of complication, type II or unspecified, not stated as uncontrolled",
    "25060": "Diabetes with neurological manifestations, type II or unspecified, not stated as uncontrolled",
    "25071": "Diabetes with peripheral circulatory disorders, type I, not stated as uncontrolled",
    "25082": "Diabetes with other specified manifestations, type II or unspecified, not stated as uncontrolled",
    "2518": "Other specified acquired hypothyroidism",
    "2553": "Disorders of adrenal medulla",
    "2733": "Amyloidosis",
    "2734": "Hypercalcemia",
    "2858": "Other and unspecified anemias",
    "2939": "Nutritional deficiency, unspecified",
    "2989": "Other and unspecified endocrine disorders",

    # Blood and Blood-Forming Organs (280-289)
    "28319": "Chronic lymphoid leukemia, without mention of having achieved remission",

    # Mental Disorders (290-319)
    "3010": "Personality disorder, unspecified",
    "3090": "Adjustment disorder, unspecified",
    "3310": "Alzheimer's disease",
    "3558": "Other forms of epilepsy",
    "3556": "Infantile spasms",

    # Nervous System (320-389)
    "3574": "Sleep apnea, unspecified",
    "3688": "Other specified cerebral degenerations",
    "36210": "Background diabetic retinopathy",

    # Circulatory System (390-459)
    "4019": "Unspecified essential hypertension",
    "4029": "Unspecified hypertensive heart disease",
    "40301": "Hypertensive chronic kidney disease, malignant, with chronic kidney disease stage I through stage IV, or unspecified",
    "4254": "Other primary cardiomyopathies",
    "4267": "Cardiac dysrhythmias, unspecified",
    "42731": "Atrial fibrillation",
    "4280": "Congestive heart failure, unspecified",
    "4295": "Other ill-defined heart diseases",
    "42971": "Paroxysmal ventricular tachycardia",
    "43882": "Other peripheral vascular disease",
    "44329": "Chronic venous insufficiency (CVI)",
    "45189": "Other venous embolism and thrombosis",
    "4847": "Other ventricular premature beats (VPB)",
    "46410": "Acute arterial embolism and thrombosis of unspecified site",
    "46451": "Acute arterial embolism and thrombosis of iliac artery",

    # Respiratory System (460-519)
    "486": "Pneumonia, organism unspecified",
    "5071": "Influenza with other respiratory manifestations",
    "5080": "Influenza with pneumonia",
    "5081": "Influenza with other manifestations",
    "5109": "Chronic obstructive pulmonary disease, unspecified",
    "51881": "Acute respiratory failure",
    "5192": "Emphysema, unspecified",
    "5198": "Other diseases of lung, not elsewhere classified",

    # Digestive System (520-579)
    "53641": "Duodenitis",
    "5370": "Diverticulosis of small intestine without mention of hemorrhage",
    "5641": "Irritable bowel syndrome",
    "5650": "Anal fissure",
    "5651": "Anal fistula",
    "5695": "Abscess of intestine",
    "5570": "Acute vascular insufficiency of intestine",

    # Genitourinary System (580-629)
    "5849": "Acute kidney failure, unspecified",
    "5856": "End stage renal disease",
    "59589": "Other specified disorders resulting from impaired renal function",
    "6089": "Female infertility, unspecified",
    "6144": "Endometriosis of pelvic peritoneum",

    # Pregnancy, Childbirth (630-679)
    "V3201": "Twin pregnancy with antenatal problem",

    # Skin (680-709)
    "7061": "Other regional lymphadenitis",
    "7098": "Other disorders of skin and subcutaneous tissue",
    "7140": "Rheumatoid arthritis",

    # Musculoskeletal (710-739)
    "71107": "Pyogenic arthritis, ankle and foot",
    "7313": "Pathological fracture of vertebrae",
    "7458": "Other acquired deformities of limbs",
    "7534": "Hallux valgus (acquired)",

    # Congenital Anomalies (740-759)
    "75567": "Congenital anomalies of spinal cord",

    # Perinatal (760-779)
    "76077": "Late vomiting of pregnancy",
    "76407": "Polyhydramnios",
    "76496": "Other multiple gestation",
    "76522": "Fetal distress first noted during labor and delivery in liveborn infant",
    "76525": "Extreme immaturity",
    "76526": "Other preterm infants",

    # Symptoms, Signs, and Ill-Defined Conditions (780-799)
    "7718": "Shortness of breath",
    "7778": "Other edema",
    "7795": "Other nonspecific abnormal findings of blood chemistry",
    "78072": "Dysphagia, oropharyngeal phase",
    "78650": "Chest pain, unspecified",
    "78839": "Other symptoms involving urinary system",
    "7902": "Abnormal reflex",

    # Injury and Poisoning (800-999)
    "80221": "Fracture of nasal bones, closed",
    "80229": "Fracture of face bones, closed, other",
    "80426": "Multiple fractures involving both lower limbs, lower with upper limb, and lower limb(s) with rib(s) and sternum, closed",
    "80505": "Closed fracture of C5-C7",
    "80605": "Closed fracture of T5-T6",
    "80625": "Closed fracture of T7-T8",
    "80700": "Closed fracture of ribs, unspecified",
    "8072": "Fracture of nasal bones, unspecified",
    "8600": "Traumatic pneumothorax without mention of open wound into thorax",
    "8731": "Open wound of scalp, without mention of complication",
    "81341": "Sprains and strains of wrist",
    "85142": "Sprains and strains of knee",

    # E Codes - External Causes (E800-E999)
    "E8120": "Other and unspecified water transport accident, injuring unspecified person",
    "E8493": "Accidents occurring in residential institution",
    "E8528": "Accidental poisoning by other specified drugs and medicinal substances",
    "E8538": "Accidental poisoning by other specified gases and vapors",

    # V Codes - Supplementary Classification (V01-V91)
    "V0251": "Need for prophylactic vaccination and inoculation against influenza",
    "V0381": "Need for prophylactic vaccination and inoculation against Hemophilus influenzae, type B [Hib]",
    "V4569": "Other postprocedural status, unspecified",
    "V4573": "Acquired absence of intestine (large) (small)",
    "V4579": "Other acquired absence of organ",
    "V8381": "Body Mass Index 25.0-25.9, adult",
    "V8541": "Body Mass Index 40.0-44.9, adult",
    "V8801": "Body Mass Index less than 19, pediatric",

    # Common codes from generated sequences
    "3556": "Other forms of epilepsy and recurrent seizures",
    "37943": "Other and unspecified disorders of the autonomic nervous system",
    "4295": "Other ill-defined heart diseases",
    "5224": "Chronic gastric ulcer without mention of hemorrhage or perforation, without mention of obstruction",
    "5258": "Gastrojejunal ulcer, unspecified as acute or chronic, without mention of hemorrhage or perforation, without mention of obstruction",
    "5968": "Other and unspecified disorders of kidney and ureter",
    "7098": "Other disorders of skin and subcutaneous tissue",
    "7313": "Pathological fracture of vertebrae",
    "72888": "Other disorders of muscle, ligament, and fascia",
    "73022": "Fracture of femur, closed",
    "34680": "Migraine, unspecified, without mention of intractable migraine",
    "42091": "Arteriovenous fistula, acquired",
    "47832": "Other skin rash",
    "56211": "Diverticulitis of colon without mention of hemorrhage",
    "56941": "Intestinal bypass",
    "07020": "Varicella with other specified complications",
    "23879": "Other specified endocrine tumors",
    "29633": "Bipolar II disorder",
    "30560": "Alcohol abuse, unspecified",
    "30928": "Other and unspecified alcohol-induced mental disorders",
    "92400": "Fracture of carpal bone(s), unspecified, closed",
    "92810": "Contusion of face, scalp, and neck except eye(s)",
    "9053": "Late effect of fracture of neck of femur",
    "9092": "Late effect of injury to the trunk",
    "9211": "Burn of face, head, and neck, erythema [first degree]",
    "9663": "Poisoning by antiallergic and antiemetic drugs",
    "9701": "Burn confined to eye and adnexa",
    "9754": "Superficial injury of fingers",
    "9950": "Certain adverse effects not elsewhere classified, unspecified",
    "99791": "Other and unspecified systemic manifestations of infection",
    "99831": "Other and unspecified infection",
    "99939": "Other and unspecified complications of medical care, not elsewhere classified",
    "82032": "Fracture of malar and maxillary bones, closed",
    "82521": "Fracture of angle of mandible, closed",
    "83905": "Disturbance of skin sensation",
    "87320": "Open wound of scalp, without mention of complication",
    "87349": "Other open wound of face",
    "87363": "Open wound of cheek",
    "77083": "Asphyxia and hypoxemia",
}


def get_code_description(code):
    """Get description for ICD-9 code."""
    # Remove leading zeros for lookup
    code_clean = code.lstrip('0')

    # Try exact match
    if code in ICD9_CODES:
        return ICD9_CODES[code]

    # Try without leading zeros
    if code_clean in ICD9_CODES:
        return ICD9_CODES[code_clean]

    # Try adding leading zero
    if f"0{code}" in ICD9_CODES:
        return ICD9_CODES[f"0{code}"]

    # Return code itself if not found
    return f"[{code} - description not available]"


def decode_patient_sequence(sequence_str, demographics):
    """Decode a patient sequence into readable format."""
    tokens = sequence_str.split()

    visits = []
    current_visit = []

    for token in tokens:
        if token == "<v>":
            current_visit = []
        elif token in ["<\\v>", "<END>"]:
            if current_visit:
                visits.append(current_visit)
                current_visit = []
        elif token not in ["<BOS>", "<s>", "</s>"]:
            # It's an ICD-9 code
            current_visit.append(token)

    # Add last visit if exists
    if current_visit:
        visits.append(current_visit)

    # Print patient info
    print("\n" + "="*80)
    print(f"PATIENT: {demographics}")
    print("="*80)

    for i, visit in enumerate(visits, 1):
        print(f"\n--- VISIT {i} ({len(visit)} diagnosis codes) ---\n")
        for j, code in enumerate(visit, 1):
            desc = get_code_description(code)
            print(f"  {j:2d}. {code:8s} - {desc}")

    print(f"\nTotal visits: {len(visits)}")
    print(f"Total diagnosis codes: {sum(len(v) for v in visits)}")
    print(f"Average codes per visit: {sum(len(v) for v in visits) / len(visits):.1f}")


def main():
    # Generated sequences from the output
    patients = [
        {
            "demo": "65.0yo WHITE M",
            "seq": "<BOS> <v> 4295 82521 7098 42971 3010 5968 82521 7718 E8493 73022 9053 E8120 <END>"
        },
        {
            "demo": "45.0yo BLACK F",
            "seq": "<BOS> <v> 7795 8600 8072 7313 43882 99791 53641 99939 42091 25082 1174 78839 87320 73022 V8381 <\\v> <v> 9053 75567 99939 V0251 6089 76522 V0381 5198 9053 7098 V4573 73022 7140 <END>"
        },
        {
            "demo": "30.0yo HISPANIC M",
            "seq": "<BOS> <v> 92400 5968 78839 76407 87349 42971 7718 5258 7313 76496 V4569 7313 0839 9701 5224 <\\v> <v> 5258 99939 76496 3010 37943 V3201 3010 73022 80221 8731 76522 73022 5695 72888 3090 <\\v> <v> 3010 25071 73022 25071 9053 37943 99831 83905 42091 4295 28319 80625 5641 7778 5641"
        },
        {
            "demo": "70.0yo WHITE F",
            "seq": "<BOS> <v> 7098 9053 5370 99939 6089 E8493 3688 42091 5968 E8538 4267 5224 30560 5370 E8493 <\\v> <v> 83905 5080 99791 9950 92400 76525 42091 7098 46451 07020 7718 7098 76496 E8493 34680 <\\v> <v> 99791 25071 4295 80426 E8493 46410 7061 85142 3010 25071 81341 87320 6089 E8493 99831"
        },
        {
            "demo": "55.0yo ASIAN M",
            "seq": "<BOS> <v> 45189 99939 V8541 99939 76496 5080 7718 76526 7718 4254 3310 76496 4254 73022 5650 <\\v> <v> 53641 77083 5968 4847 V4573 25071 87320 V0381 7098 80229 9092 4295 46451 25071 V0381 <\\v> <v> 7313 30928 87320 5192 5258 2733 5370 V0381 5081 29633 80605 87349 7098 2518 99831"
        },
        {
            "demo": "60.0yo BLACK M",
            "seq": "<BOS> <v> 5198 E8493 25071 87320 76525 9053 87363 7718 42091 4295 37943 <END>"
        },
        {
            "demo": "25.0yo HISPANIC F",
            "seq": "<BOS> <v> 3556 3010 99831 3010 7098 9211 7718 56941 1419 7098 80221 E8528 30560 44329 83905 <\\v> <v> 76496 7534 9754 76525 73022 87320 73022 3558 30560 3010 6144 80505 80625 37943 83905 <\\v> <v> 75567 7458 92400 36210 1419 6089 76407 34680 85142 7098 80625 73022 6089 47832 73022"
        },
        {
            "demo": "50.0yo OTHER F",
            "seq": "<BOS> <v> 45189 9053 5570 5109 46410 92810 7313 9663 E8493 76496 V4569 87320 E8120 2989 99939 <\\v> <v> 76526 99939 76496 E8120 2939 3090 34680 3574 5370 7098 4295 76496 99831 59589 5641 <\\v> <v> 1642 3090 25082 83905 36210 5060 56211 3010 4295 76496 7098 5081 6089 2518 46410"
        },
        {
            "demo": "75.0yo WHITE M",
            "seq": "<BOS> <v> 23879 99791 9950 5224 7718 82032 E8493 99939 2553 7718 36210 75567 <END>"
        },
        {
            "demo": "80.0yo WHITE F",
            "seq": "<BOS> <v> 56211 5224 80700 2858 8731 99791 V8801 5651 36210 5224 92810 76525 76077 V0381 E8493 <\\v> <v> E8493 9053 5224 5370 V4573 7098 E8538 5370 5695 7061 46451 76496 5370 71107 30928 <\\v> <v> 75567 25071 30928 V4579 78839 7313 5071 83905 3010 4295 2734 80229 5224 5060 075"
        }
    ]

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  GENERATED SYNTHETIC PATIENTS - MEDICAL TRANSLATION".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    for i, patient in enumerate(patients, 1):
        decode_patient_sequence(patient["seq"], patient["demo"])

    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  TRANSLATION COMPLETE".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    main()
