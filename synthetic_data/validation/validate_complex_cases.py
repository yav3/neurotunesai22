"""
Validation suite for complex synthetic patient cases.
Checks: structural integrity, genetic variants, ICD-10 codes, demographics,
biomarker plausibility, multi-omics consistency, disease-specific rules,
cross-case distribution, diagnostic journeys, and treatments.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parent.parent / "quarantine" / "SYNTHETIC_complex_cases.json"

KNOWN_GENES = {
    "APOE", "APP", "PSEN1", "PSEN2", "MAPT", "GRN", "C9orf72", "TREM2", "CLU",
    "BIN1", "LRRK2", "SNCA", "GBA", "PARK2", "PINK1", "DJ1", "VPS35", "SOD1",
    "FUS", "TARDBP", "TBK1", "OPTN", "NEK1", "HTT", "SCN1A", "SCN2A", "SCN8A",
    "KCNQ2", "KCNA1", "KCNB1", "TSC1", "TSC2", "DEPDC5", "SLC2A1", "GRIN2A",
    "GABRA1", "GABRG2", "NOTCH3", "ATP7B", "NPC1", "NPC2", "PRNP", "MECP2",
    "FMR1", "DYRK1A", "GFAP", "LMNB1", "EIF2B1", "EIF2B2", "EIF2B3", "EIF2B4",
    "EIF2B5", "ABCD1", "GALC", "ARSA", "GLA", "HEXA", "HEXB", "SMPD1", "ASAH1",
    "CLN3", "CLN5", "CLN6", "CLN8", "PPT1", "TPP1", "DNMT1", "SETX", "ATXN1",
    "ATXN2", "ATXN3", "ATXN7", "TBP", "CACNA1A", "SPG4", "SPG7", "SPG11",
    "REEP1", "ATL1", "KIF5A", "DCTN1", "SMN1", "SMN2", "DYNC1H1", "KCNC3",
    "PRRT2", "CHRNA4", "CHRNB2", "CDKL5", "ARX", "STXBP1", "PCDH19", "POLG",
    "SURF1", "NDUFS1", "NDUFV1", "MT_TL1", "MT_ND4", "MT_ATP6", "SCN9A",
    "TRPV1", "ATP1A2", "KCNK18", "TRESK", "HTRA1", "COL4A1", "COL4A2", "ADA2",
    "MTHFR", "CBS", "HFE", "TBX1", "SHANK3", "NRXN1", "CNTNAP2", "CHD8",
    "ADNP", "SYNGAP1", "ANK2", "ARID1B", "KMT2A", "CTNNB1", "WDR45", "GNAO1",
    "SLC6A3", "TH", "DDC", "DBH", "PNKD", "GCH1", "SGCE", "THAP1", "TOR1A",
    "ANO3", "GNAL", "ACTB", "TUBB4A", "KIF1A", "VEGFA", "HES1", "SYP",
    "SNAP25", "SLC17A7", "C3", "BACE1", "SLC6A4", "BDNF", "FKBP5", "CYP2D6",
    "CYP2C19", "ABCB1", "GAD1", "GAD2", "SLC1A2", "GRIN1", "GRIN2B", "DRD2",
    "DRD4", "HTR2A", "COMT", "DISC1", "NRG1", "DTNBP1", "AQP4", "MOG",
    "MBP", "PLP1", "NEFH", "NEFL", "NEFM", "CHAT", "ACHE", "BCHE",
    "PRKN", "PARK7", "ATP13A2", "FBXO7", "PLA2G6", "SYNJ1", "DNAJC6",
    "RAB39B", "CHCHD2", "VCP", "UBQLN2", "SQSTM1", "CHMP2B", "ANG",
    "VAPB", "FIG4", "ALS2", "SIGMAR1", "ERBB4", "SS18L1", "HNRNPA1",
    "MATR3", "TUBA4A", "CCNF", "KIF5A", "GLB1", "IDUA", "IDS", "SGSH",
    "NAGLU", "HGSNAT", "GNS", "GUSB", "ARSB", "GNPTAB", "GNPTG",
    "MCOLN1", "LAMP2", "CSTB", "EPM2A", "NHLRC1", "KCNT1", "SLC13A5",
    "WWOX", "SPTAN1", "IQSEC2", "SCN1B", "GABRD", "EFHC1", "LGI1",
    "RELN", "FLNA", "DCX", "LIS1", "TSEN54", "EXOSC3", "RARS2",
    "CASK", "FOXG1", "ZEB2", "TCF4", "MEF2C", "MED12", "ANKRD11",
    "EP300", "CREBBP", "EHMT1", "KANSL1", "DEAF1", "SIN3A",
    "MT-TL1", "16p11.2", "22q11.2",
}

VALID_PATHOGENICITY = {
    "pathogenic", "likely_pathogenic", "risk_factor",
    "uncertain_significance", "benign",
}
VALID_ZYGOSITY = {
    "heterozygous", "homozygous", "hemizygous", "compound_heterozygous",
    "mosaic", "heteroplasmic",
}
VALID_INHERITANCE = {
    "autosomal_dominant", "autosomal_recessive", "x_linked",
    "mitochondrial", "de_novo", "complex", "somatic",
}
VALID_CATEGORIES = {"zebra", "comorbidity", "mimic", "genetic_overlap"}
VALID_SEXES = {"M", "F"}
ICD10_RE = re.compile(r"^[A-TV-Z]\d{2}(\.[A-Za-z0-9]{1,4})?$")
RSID_RE = re.compile(r"^rs\d+$")

REQUIRED_FIELDS = [
    "case_id", "category", "classification", "demographics",
    "comorbidities", "multi_omics", "clinical_narrative", "is_synthetic",
]

BIOMARKER_RANGES = {
    "mmse_score": (0, 30),
    "moca_score": (0, 30),
    "hamilton_depression_score": (0, 52),
    "updrs_motor_score": (0, 132),
    "blood_nfl_pg_ml": (1, 500),
    "csf_amyloid_beta42_pg_ml": (50, 2500),
    "eeg_theta_alpha_ratio": (0, 5),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Validator:
    def __init__(self, cases):
        self.cases = cases
        self.errors = []
        self.warnings = []

    def err(self, case_id, msg):
        self.errors.append(f"  [{case_id}] {msg}")

    def warn(self, case_id, msg):
        self.warnings.append(f"  [{case_id}] {msg}")

    # ------------------------------------------------------------------
    # 1. Structural integrity
    # ------------------------------------------------------------------
    def test_structural_integrity(self):
        print("=" * 60)
        print("TEST 1: Structural integrity")
        print("=" * 60)
        ok = True
        ids = []
        for c in self.cases:
            cid = c.get("case_id", "<missing>")
            ids.append(cid)
            for f in REQUIRED_FIELDS:
                if f not in c:
                    self.err(cid, f"missing required field '{f}'")
                    ok = False
            if not c.get("is_synthetic"):
                self.err(cid, "is_synthetic is not True")
                ok = False
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        if dupes:
            self.err("GLOBAL", f"duplicate case_ids: {dupes}")
            ok = False
        print(f"  Cases loaded: {len(self.cases)}")
        print(f"  Required fields present: {'PASS' if ok else 'FAIL'}")
        print(f"  No duplicate IDs: {'PASS' if not dupes else 'FAIL'}")
        return ok

    # ------------------------------------------------------------------
    # 2. Genetic variant validation
    # ------------------------------------------------------------------
    def test_genetic_variants(self):
        print("\n" + "=" * 60)
        print("TEST 2: Genetic variant validation")
        print("=" * 60)
        ok = True
        total_variants = 0
        unknown_genes = set()
        bad_rsids = 0
        bad_pathogenicity = 0
        bad_zygosity = 0
        bad_inheritance = 0
        bad_af = 0

        for c in self.cases:
            cid = c.get("case_id", "?")
            variants = c.get("genetic_variants", [])
            if not isinstance(variants, list):
                variants = []
            for v in variants:
                total_variants += 1
                gene = v.get("gene", "")
                if gene and gene not in KNOWN_GENES:
                    unknown_genes.add(gene)
                    self.warn(cid, f"gene '{gene}' not in known set")
                rsid = v.get("rsid", "")
                if rsid and not RSID_RE.match(rsid):
                    bad_rsids += 1
                    self.err(cid, f"invalid rsID format: '{rsid}'")
                    ok = False
                path = v.get("pathogenicity", "")
                if path and path not in VALID_PATHOGENICITY:
                    bad_pathogenicity += 1
                    self.err(cid, f"invalid pathogenicity: '{path}'")
                    ok = False
                zyg = v.get("zygosity", "")
                if zyg and zyg not in VALID_ZYGOSITY:
                    bad_zygosity += 1
                    self.err(cid, f"invalid zygosity: '{zyg}'")
                    ok = False
                inh = v.get("inheritance", "")
                if inh and inh not in VALID_INHERITANCE:
                    bad_inheritance += 1
                    self.err(cid, f"invalid inheritance: '{inh}'")
                    ok = False
                af = _num(v.get("gnomad_af"))
                if af is not None and (af < 0 or af > 0.5):
                    bad_af += 1
                    self.err(cid, f"gnomAD AF out of range: {af}")
                    ok = False

        print(f"  Total variants: {total_variants}")
        print(f"  Unknown genes (warnings): {len(unknown_genes)}")
        if unknown_genes:
            print(f"    {sorted(unknown_genes)[:20]}")
        print(f"  Bad rsIDs: {bad_rsids}")
        print(f"  Bad pathogenicity: {bad_pathogenicity}")
        print(f"  Bad zygosity: {bad_zygosity}")
        print(f"  Bad inheritance: {bad_inheritance}")
        print(f"  Bad allele freq: {bad_af}")
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 3. ICD-10 code validation
    # ------------------------------------------------------------------
    def test_icd10_codes(self):
        print("\n" + "=" * 60)
        print("TEST 3: ICD-10 code validation")
        print("=" * 60)
        ok = True
        total = 0
        bad = 0
        for c in self.cases:
            cid = c.get("case_id", "?")
            codes = []
            cls = c.get("classification", {})
            ic = cls.get("icd10_codes", {})
            if isinstance(ic, dict):
                p = ic.get("primary", "")
                if p:
                    codes.append(p)
                for s in ic.get("secondary", []):
                    codes.append(s)
            elif isinstance(ic, list):
                codes.extend(ic)
            elif isinstance(ic, str):
                codes.append(ic)
            for cm in c.get("comorbidities", []):
                code = cm.get("icd10", "")
                if code:
                    codes.append(code)
            for code in codes:
                total += 1
                if not ICD10_RE.match(code):
                    bad += 1
                    self.err(cid, f"invalid ICD-10 format: '{code}'")
                    ok = False
        print(f"  Total ICD-10 codes checked: {total}")
        print(f"  Invalid format: {bad}")
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 4. Demographic consistency
    # ------------------------------------------------------------------
    def test_demographics(self):
        print("\n" + "=" * 60)
        print("TEST 4: Demographic consistency")
        print("=" * 60)
        ok = True
        for c in self.cases:
            cid = c.get("case_id", "?")
            dem = c.get("demographics", {})
            age = _num(dem.get("age"))
            if age is not None and (age < 0 or age > 120):
                self.err(cid, f"age {age} out of [0,120]")
                ok = False
            sex = dem.get("sex", "")
            if sex and sex not in VALID_SEXES:
                self.err(cid, f"invalid sex: '{sex}'")
                ok = False
            bmi = _num(dem.get("bmi"))
            if bmi is not None and (bmi < 10 or bmi > 60):
                self.err(cid, f"BMI {bmi} out of [10,60]")
                ok = False
            for cm in c.get("comorbidities", []):
                onset = _num(cm.get("onset_age"))
                if onset is not None and age is not None and onset > age:
                    self.err(cid, f"comorbidity onset_age {onset} > current age {age}: {cm.get('condition','?')}")
                    ok = False
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 5. Biomarker plausibility
    # ------------------------------------------------------------------
    def test_biomarkers(self):
        print("\n" + "=" * 60)
        print("TEST 5: Biomarker physiological plausibility")
        print("=" * 60)
        ok = True
        checked = 0
        for c in self.cases:
            cid = c.get("case_id", "?")
            bio = c.get("standard_biomarkers", {})
            if not bio:
                bio = {}
            for key, (lo, hi) in BIOMARKER_RANGES.items():
                val = _num(bio.get(key))
                if val is None:
                    continue
                checked += 1
                if val < lo or val > hi:
                    self.err(cid, f"{key}={val} out of [{lo},{hi}]")
                    ok = False
        print(f"  Biomarker values checked: {checked}")
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 6. Multi-omics consistency
    # ------------------------------------------------------------------
    def test_multi_omics(self):
        print("\n" + "=" * 60)
        print("TEST 6: Multi-omics consistency")
        print("=" * 60)
        ok = True
        for c in self.cases:
            cid = c.get("case_id", "?")
            mo = c.get("multi_omics", {})
            if not mo:
                continue
            tx = mo.get("transcriptomic", {})
            degs = tx.get("key_degs", [])
            for d in degs:
                lfc = _num(d.get("log2fc"))
                if lfc is not None and (lfc < -10 or lfc > 10):
                    self.err(cid, f"transcriptomic log2fc {lfc} out of [-10,10]")
                    ok = False
                padj = _num(d.get("padj"))
                if padj is not None and (padj < 0 or padj > 1):
                    self.err(cid, f"transcriptomic padj {padj} out of [0,1]")
                    ok = False
                dirn = d.get("direction", "")
                if dirn and dirn not in ("up", "down"):
                    self.err(cid, f"transcriptomic direction '{dirn}' invalid")
                    ok = False
            for section in ("proteomic", "metabolomic"):
                sec = mo.get(section, {})
                if isinstance(sec, dict):
                    for k, v in sec.items():
                        if k in ("reference", "platform", "note"):
                            continue
                        nv = _num(v) if not isinstance(v, dict) else _num(v.get("value", v.get("mean")))
                        if nv is not None and nv < 0:
                            self.err(cid, f"{section}.{k}={nv} is negative")
                            ok = False
            epi = mo.get("epigenomic", {})
            meth = _num(epi.get("global_methylation_pct", epi.get("global_dna_methylation_pct")))
            if meth is not None and (meth < 0 or meth > 100):
                self.err(cid, f"methylation {meth}% out of [0,100]")
                ok = False
            acc = _num(epi.get("horvath_age_acceleration_yrs",
                               epi.get("horvath_epigenetic_age_acceleration_yrs")))
            if acc is not None and (acc < -20 or acc > 20):
                self.err(cid, f"age acceleration {acc} out of [-20,20]")
                ok = False
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 7. Disease-specific internal consistency
    # ------------------------------------------------------------------
    def test_disease_consistency(self):
        print("\n" + "=" * 60)
        print("TEST 7: Disease-specific internal consistency")
        print("=" * 60)
        ok = True
        checks = 0
        for c in self.cases:
            cid = c.get("case_id", "?")
            cls = c.get("classification", {})
            primary = (cls.get("primary_diagnosis", "") or "").lower()
            bio = c.get("standard_biomarkers", {}) or {}
            mo = c.get("multi_omics", {}) or {}
            prot = mo.get("proteomic", {}) or {}

            def _get_bio(key):
                v = _num(bio.get(key))
                if v is not None:
                    return v
                return _num(prot.get(key))

            if "alzheimer" in primary:
                ab42 = _get_bio("csf_amyloid_beta42_pg_ml")
                if ab42 is not None:
                    checks += 1
                    if ab42 > 800:
                        self.warn(cid, f"AD case but CSF Abeta42={ab42} (expected <800)")
                mmse = _get_bio("mmse_score")
                if mmse is not None:
                    checks += 1
                    if mmse > 27:
                        self.warn(cid, f"AD case but MMSE={mmse} (expected <27)")

            if "parkinson" in primary:
                updrs = _get_bio("updrs_motor_score")
                if updrs is not None:
                    checks += 1
                    if updrs < 10:
                        self.warn(cid, f"PD case but UPDRS={updrs} (expected >10)")

            if "depress" in primary or "mdd" in primary:
                ham = _get_bio("hamilton_depression_score")
                if ham is not None:
                    checks += 1
                    if ham < 10:
                        self.warn(cid, f"MDD case but Hamilton={ham} (expected >10)")

            if "amyotrophic" in primary or "als" in primary:
                nfl = _get_bio("blood_nfl_pg_ml")
                if nfl is not None:
                    checks += 1
                    if nfl < 30:
                        self.warn(cid, f"ALS case but blood NfL={nfl} (expected >30)")

        print(f"  Disease-specific checks run: {checks}")
        print(f"  Warnings (soft): {sum(1 for w in self.warnings if 'case but' in w)}")
        print(f"  PASS: {ok}  (disease consistency uses soft warnings)")
        return ok

    # ------------------------------------------------------------------
    # 8. Cross-case validation
    # ------------------------------------------------------------------
    def test_cross_case(self):
        print("\n" + "=" * 60)
        print("TEST 8: Cross-case validation")
        print("=" * 60)
        ok = True
        cats = Counter(c.get("category", "unknown") for c in self.cases)
        print(f"  Category distribution: {dict(cats)}")
        for cat in VALID_CATEGORIES:
            if cats.get(cat, 0) == 0:
                self.err("GLOBAL", f"missing category: {cat}")
                ok = False
        min_counts = {"zebra": 10, "comorbidity": 10, "mimic": 5, "genetic_overlap": 5}
        for cat, mn in min_counts.items():
            if cats.get(cat, 0) < mn:
                self.warn("GLOBAL", f"category '{cat}' has {cats.get(cat,0)} cases (want >={mn})")

        sexes = set()
        ages = []
        for c in self.cases:
            dem = c.get("demographics", {})
            s = dem.get("sex")
            if s:
                sexes.add(s)
            a = _num(dem.get("age"))
            if a is not None:
                ages.append(a)
        if len(sexes) < 2:
            self.err("GLOBAL", f"only sexes {sexes} represented")
            ok = False
        if ages:
            print(f"  Age range: [{min(ages):.0f}, {max(ages):.0f}], mean={sum(ages)/len(ages):.1f}")
        print(f"  Sexes: {sexes}")

        sigs = set()
        for c in self.cases:
            dem = c.get("demographics", {})
            cls = c.get("classification", {})
            sig = (dem.get("age"), dem.get("sex"), cls.get("primary_diagnosis", ""))
            if sig in sigs:
                self.warn(c.get("case_id", "?"), f"duplicate signature {sig}")
            sigs.add(sig)

        print(f"  All categories present: {'PASS' if ok else 'FAIL'}")
        print(f"  Both sexes: {'PASS' if len(sexes) >= 2 else 'FAIL'}")
        return ok

    # ------------------------------------------------------------------
    # 9. Diagnostic journey consistency
    # ------------------------------------------------------------------
    def test_diagnostic_journey(self):
        print("\n" + "=" * 60)
        print("TEST 9: Diagnostic journey consistency")
        print("=" * 60)
        ok = True
        checked = 0
        for c in self.cases:
            cid = c.get("case_id", "?")
            dj = c.get("diagnostic_journey", {})
            if not dj:
                continue
            checked += 1
            delay = _num(dj.get("diagnostic_delay_years"))
            if delay is not None and delay < 0:
                self.err(cid, f"diagnostic_delay_years={delay} is negative")
                ok = False
            correct = dj.get("correct_diagnosis", {})
            correct_age = _num(correct.get("age")) if isinstance(correct, dict) else None
            misdiagnoses = dj.get("misdiagnoses", [])
            if isinstance(misdiagnoses, list):
                prev_age = 0
                for md in misdiagnoses:
                    md_age = _num(md.get("age"))
                    if md_age is not None:
                        if md_age < prev_age:
                            self.warn(cid, f"misdiagnosis age {md_age} < previous {prev_age}")
                        prev_age = md_age
                        if correct_age is not None and md_age > correct_age:
                            self.err(cid, f"misdiagnosis age {md_age} > correct diagnosis age {correct_age}")
                            ok = False
        print(f"  Journeys checked: {checked}")
        print(f"  PASS: {ok}")
        return ok

    # ------------------------------------------------------------------
    # 10. Treatment validation
    # ------------------------------------------------------------------
    def test_treatments(self):
        print("\n" + "=" * 60)
        print("TEST 10: Treatment validation")
        print("=" * 60)
        ok = True
        total = 0
        for c in self.cases:
            cid = c.get("case_id", "?")
            age = _num(c.get("demographics", {}).get("age"))
            treatments = c.get("treatment_history", [])
            if not isinstance(treatments, list):
                continue
            for t in treatments:
                total += 1
                dose = t.get("dose", t.get("dose_mg", ""))
                if not dose and dose != 0:
                    self.warn(cid, f"empty dose for drug '{t.get('drug','?')}'")
                start = _num(t.get("start_age"))
                if start is not None and age is not None and start > age:
                    self.err(cid, f"treatment start_age {start} > current age {age}")
                    ok = False
        print(f"  Treatments checked: {total}")
        print(f"  PASS: {ok}")
        return ok

    def run_all(self):
        results = {
            "structural_integrity": self.test_structural_integrity(),
            "genetic_variants": self.test_genetic_variants(),
            "icd10_codes": self.test_icd10_codes(),
            "demographics": self.test_demographics(),
            "biomarkers": self.test_biomarkers(),
            "multi_omics": self.test_multi_omics(),
            "disease_consistency": self.test_disease_consistency(),
            "cross_case": self.test_cross_case(),
            "diagnostic_journey": self.test_diagnostic_journey(),
            "treatments": self.test_treatments(),
        }

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for name, passed in results.items():
            print(f"  {name:30s}: {'PASS' if passed else 'FAIL'}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for e in self.errors[:50]:
                print(e)
            if len(self.errors) > 50:
                print(f"  ... and {len(self.errors) - 50} more")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for w in self.warnings[:30]:
                print(w)
            if len(self.warnings) > 30:
                print(f"  ... and {len(self.warnings) - 30} more")

        all_ok = all(results.values())
        print(f"\nOverall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
        return all_ok


def main():
    fp = Path(sys.argv[1]) if len(sys.argv) > 1 else CASES_PATH
    if not fp.exists():
        print(f"File not found: {fp}")
        print("Generate complex cases first: python3 -I generators/generate_complex_cases.py")
        return 1

    with open(fp) as f:
        data = json.load(f)

    cases = data.get("cases", [])
    if not cases:
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                cases = v
                break
    if not cases:
        print(f"No cases found in {fp}")
        return 1

    print(f"Loaded {len(cases)} complex cases from {fp.name}\n")
    v = Validator(cases)
    ok = v.run_all()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
