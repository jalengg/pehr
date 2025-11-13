"""
Quick medical validity check for generated patients from hierarchical model.
"""
import re
from collections import Counter

# Age-inappropriate codes (examples)
AGE_RULES = {
    # Neonatal (age < 1)
    '770': (0, 0.1, 'neonatal respiratory'),
    '779': (0, 0.1, 'neonatal'),

    # Pediatric (age < 18)
    '315': (0, 18, 'developmental delays'),

    # Pregnancy (age 12-55, female only)
    '630': (12, 55, 'pregnancy - molar'),
    '640': (12, 55, 'pregnancy - hemorrhage'),
    'V22': (12, 55, 'pregnancy - normal'),
}

# Sex-inappropriate codes (examples)
SEX_RULES = {
    # Male-only
    '600': ('M', 'prostate hyperplasia'),
    '601': ('M', 'prostatitis'),
    '602': ('M', 'prostate disorders'),

    # Female-only
    '614': ('F', 'pelvic inflammatory disease'),
    '615': ('F', 'uterine inflammatory disease'),
    '616': ('F', 'vaginal/vulvar inflammation'),
    '630': ('F', 'molar pregnancy'),
    '640': ('F', 'pregnancy hemorrhage'),
}

def check_age_appropriate(code, age):
    """Check if code is age-appropriate."""
    prefix = code[:3] if len(code) >= 3 else code

    if prefix in AGE_RULES:
        min_age, max_age, desc = AGE_RULES[prefix]
        if not (min_age <= age <= max_age):
            return False, f"Age {age} inappropriate for {desc} ({code})"
    return True, None

def check_sex_appropriate(code, sex):
    """Check if code is sex-appropriate."""
    prefix = code[:3] if len(code) >= 3 else code

    if prefix in SEX_RULES:
        required_sex, desc = SEX_RULES[prefix]
        if sex != required_sex:
            return False, f"Sex {sex} inappropriate for {desc} ({code})"
    return True, None

def parse_generated_file(filename):
    """Parse generated patients file."""
    with open(filename) as f:
        content = f.read()

    patients = []
    pattern = r'Patient \d+:\n  Age: ([\d.]+)\n  Sex: ([MF])\n  Categories: (\d+)\n  Codes: (\d+)\n  Expansion ratio: ([\d.]+)\n  Codes: ([0-9, ]+)'

    for match in re.finditer(pattern, content):
        age = float(match.group(1))
        sex = match.group(2)
        codes = match.group(6).split(', ')
        patients.append({'age': age, 'sex': sex, 'codes': codes})

    return patients

def main():
    print("=" * 80)
    print("Medical Validity Check - Generated Hierarchical Patients")
    print("=" * 80)

    # Parse generated patients
    patients = parse_generated_file('generated_patients_hierarchical.txt')
    print(f"\nLoaded {len(patients)} generated patients")

    # Check for duplicates within patients
    total_codes = 0
    duplicates = 0
    for p in patients:
        total_codes += len(p['codes'])
        unique_codes = len(set(p['codes']))
        if unique_codes < len(p['codes']):
            duplicates += len(p['codes']) - unique_codes

    print(f"Total code instances: {total_codes}")
    print(f"Duplicate codes within patients: {duplicates} ({duplicates/total_codes*100:.2f}%)")

    # Check age appropriateness
    age_violations = []
    for i, p in enumerate(patients):
        for code in p['codes']:
            is_valid, msg = check_age_appropriate(code, p['age'])
            if not is_valid:
                age_violations.append(f"Patient {i+1}: {msg}")

    print(f"\nAge-inappropriate codes: {len(age_violations)}")
    if age_violations:
        print("Examples:")
        for v in age_violations[:5]:
            print(f"  {v}")

    # Check sex appropriateness
    sex_violations = []
    for i, p in enumerate(patients):
        for code in p['codes']:
            is_valid, msg = check_sex_appropriate(code, p['sex'])
            if not is_valid:
                sex_violations.append(f"Patient {i+1}: {msg}")

    print(f"\nSex-inappropriate codes: {len(sex_violations)}")
    if sex_violations:
        print("Examples:")
        for v in sex_violations[:5]:
            print(f"  {v}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total patients: {len(patients)}")
    print(f"Total codes: {total_codes}")
    print(f"Duplicates: {duplicates} ({duplicates/total_codes*100:.1f}%)")
    print(f"Age violations: {len(age_violations)} ({len(age_violations)/total_codes*100:.1f}%)")
    print(f"Sex violations: {len(sex_violations)} ({len(sex_violations)/total_codes*100:.1f}%)")

    # Validity rates
    valid_codes = total_codes - len(age_violations) - len(sex_violations)
    print(f"\nMedical validity rate: {valid_codes/total_codes*100:.1f}%")
    print(f"Age-appropriate rate: {(total_codes - len(age_violations))/total_codes*100:.1f}%")
    print(f"Sex-appropriate rate: {(total_codes - len(sex_violations))/total_codes*100:.1f}%")

if __name__ == "__main__":
    main()
