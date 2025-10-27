"""
Analyze generated patient sequences - decode ICD-9 codes and check quality.
"""
import logging
import sys
from pathlib import Path

from config import Config
from data_loader import load_mimic_data
from code_tokenizer import DiagnosisCodeTokenizer


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("analyzer")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def decode_icd9_code(code: str) -> str:
    """Decode ICD-9 code to description (simplified mapping of common codes)."""
    # Common ICD-9 codes with descriptions
    icd9_descriptions = {
        # Cardiovascular
        "4019": "Hypertension, unspecified",
        "4280": "Congestive heart failure, unspecified",
        "42731": "Atrial fibrillation",
        "41401": "Coronary atherosclerosis",
        "V4573": "Acquired absence of intestine",
        "V0381": "Need for prophylactic vaccination",

        # Diabetes
        "25000": "Diabetes mellitus without complication",
        "25060": "Diabetes with neurological manifestations",
        "25071": "Diabetes with peripheral circulatory disorders",
        "25082": "Diabetes with unspecified complication",

        # Renal
        "5849": "Acute kidney failure, unspecified",
        "585": "Chronic kidney disease (CKD)",
        "5859": "Chronic kidney disease, unspecified",
        "5856": "End stage renal disease",

        # Respiratory
        "4280": "Congestive heart failure",
        "49390": "Asthma, unspecified",
        "486": "Pneumonia, organism unspecified",
        "51881": "Acute respiratory failure",

        # General symptoms
        "7840": "Headache",
        "78650": "Chest pain, unspecified",
        "78959": "Other ascites",
        "7802": "Syncope and collapse",
    }

    return icd9_descriptions.get(code, f"[{code}]")


def main():
    logger = setup_logging()

    # Load vocabulary
    config = Config.from_defaults()
    patient_records, vocab = load_mimic_data(
        patients_path=config.data.patients_path,
        admissions_path=config.data.admissions_path,
        diagnoses_path=config.data.diagnoses_path,
        logger=logger,
        num_patients=config.data.num_patients
    )

    tokenizer = DiagnosisCodeTokenizer(vocab)

    # Example generated sequences from the output
    sequences = [
        {
            "id": 1,
            "demo": "65.0yo WHITE M",
            "sequence": "<BOS> <v> 4295 82521 7098 42971 3010 5968 82521 7718 E8493 73022 9053 E8120 <END>"
        },
        {
            "id": 2,
            "demo": "45.0yo BLACK F",
            "sequence": "<BOS> <v> 7795 8600 8072 7313 43882 99791 53641 99939 42091 25082 1174 78839 87320 73022 V8381 <\\v> <v> 9053 75567 99939 V0251 6089 76522 V0381 5198 9053 7098 V4573 73022 7140 <END>"
        },
        {
            "id": 8,
            "demo": "60.0yo BLACK M",
            "sequence": "<BOS> <v> 5198 E8493 25071 87320 76525 9053 87363 7718 42091 4295 37943 <END>"
        }
    ]

    logger.info("=" * 80)
    logger.info("GENERATED PATIENT ANALYSIS")
    logger.info("=" * 80)

    for seq_data in sequences:
        logger.info(f"\nPatient {seq_data['id']}: {seq_data['demo']}")
        logger.info("─" * 80)

        # Parse sequence
        tokens = seq_data["sequence"].split()
        visits = []
        current_visit = []

        for token in tokens:
            if token == "<v>":
                current_visit = []
            elif token == "<\\v>" or token == "<END>":
                if current_visit:
                    visits.append(current_visit)
                    current_visit = []
            elif token not in ["<BOS>", "<s>", "</s>"]:
                # It's a code ID
                try:
                    code_id = int(token)
                    # Decode from tokenizer
                    if code_id >= tokenizer.code_offset:
                        vocab_idx = code_id - tokenizer.code_offset
                        if vocab_idx in tokenizer.vocab.idx2code:
                            icd_code = tokenizer.vocab.idx2code[vocab_idx]
                            current_visit.append(icd_code)
                except ValueError:
                    pass

        # Add last visit if exists
        if current_visit:
            visits.append(current_visit)

        # Display visits
        logger.info(f"Number of visits: {len(visits)}")
        for i, visit in enumerate(visits, 1):
            logger.info(f"\nVisit {i} ({len(visit)} codes):")
            for code in visit[:10]:  # Show first 10 codes
                description = decode_icd9_code(code)
                logger.info(f"  {code:8s} - {description}")
            if len(visit) > 10:
                logger.info(f"  ... and {len(visit) - 10} more codes")

        logger.info("")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
