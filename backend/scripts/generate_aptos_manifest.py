import os
import csv
import random
import hashlib
from collections import defaultdict, Counter

def build_aptos_manifest():
    random.seed(42)  # Scientific reproducibility seed

    train_csv_path = "storage/datasets/aptos2019/train.csv"
    dup_csv_path = "storage/datasets/aptos2019/duplicated_info.csv"

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Missing {train_csv_path}")

    # Read train.csv
    with open(train_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(f"Total raw APTOS 2019 samples: {len(samples)}")
    raw_counts = Counter(int(s["diagnosis"]) for s in samples)
    print(f"Class distribution: {sorted(raw_counts.items())}")

    # Read duplicate / perceptual hash info for bilateral patient grouping
    # dup_csv has: ,diagnosis,path,Size,Mode,Hash
    hash_to_patient = {}
    id_to_hash = {}
    if os.path.exists(dup_csv_path):
        with open(dup_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("path", "")
                img_id = os.path.basename(p).replace(".png", "").replace(".jpg", "")
                h = row.get("Hash", "")
                if h and img_id:
                    id_to_hash[img_id] = h

    # Map each sample to a patient ID
    # If two images share the same perceptual Hash, they are bilateral/duplicate and belong to the same patient
    patient_groups = defaultdict(list)
    patient_counter = 1
    hash_to_pid = {}

    for s in samples:
        img_id = s["id_code"]
        diag = int(s["diagnosis"])
        h = id_to_hash.get(img_id)

        if h:
            if h not in hash_to_pid:
                hash_to_pid[h] = f"APTOS-P{patient_counter:04d}"
                patient_counter += 1
            pid = hash_to_pid[h]
        else:
            pid = f"APTOS-P{patient_counter:04d}"
            patient_counter += 1

        patient_groups[pid].append({
            "image_id": img_id,
            "patient_id": pid,
            "source_dataset": "APTOS 2019 (Aravind Eye Hospital)",
            "true_grade": diag,
            "file_path": f"train_images/{img_id}.png",
        })

    print(f"Total unique patient encounters: {len(patient_groups)}")

    # Stratified Patient-Level Partitioning:
    # 70% Train, 15% Validation, 15% Test
    # Determine dominant grade per patient for stratification
    patients_by_grade = defaultdict(list)
    for pid, imgs in patient_groups.items():
        dom_grade = imgs[0]["true_grade"]
        patients_by_grade[dom_grade].append((pid, imgs))

    train_records = []
    val_records = []
    test_records = []

    for grade in range(5):
        pts = patients_by_grade[grade]
        random.shuffle(pts)
        
        n_pts = len(pts)
        n_train = int(round(n_pts * 0.70))
        n_val = int(round(n_pts * 0.15))
        n_test = n_pts - n_train - n_val

        for pid, imgs in pts[:n_train]:
            for img in imgs:
                img["split"] = "train"
                train_records.append(img)

        for pid, imgs in pts[n_train:n_train + n_val]:
            for img in imgs:
                img["split"] = "val"
                val_records.append(img)

        for pid, imgs in pts[n_train + n_val:]:
            for img in imgs:
                img["split"] = "test"
                test_records.append(img)

    all_records = train_records + val_records + test_records
    print(f"Partitioned: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)} (Total={len(all_records)})")

    icdr_labels = [
        "Grade 0: No Apparent DR",
        "Grade 1: Mild NPDR",
        "Grade 2: Moderate NPDR",
        "Grade 3: Severe NPDR",
        "Grade 4: Proliferative DR"
    ]

    output_rows = []
    for r in all_records:
        grade = r["true_grade"]
        # Cryptographic record integrity hash
        record_token = f"APTOS2019_{r['image_id']}_{grade}_{r['patient_id']}_{r['split']}"
        sha = hashlib.sha256(record_token.encode("utf-8")).hexdigest()
        
        output_rows.append({
            "image_id": r["image_id"],
            "patient_id": r["patient_id"],
            "source_dataset": r["source_dataset"],
            "file_path": r["file_path"],
            "true_grade": grade,
            "true_label": icdr_labels[grade],
            "split": r["split"],
            "sha256_hash": sha
        })

    # Shuffle for storage
    random.shuffle(output_rows)

    os.makedirs("docs/chapter4", exist_ok=True)
    manifest_out = "docs/chapter4/dataset_split_manifest.csv"
    fieldnames = ["image_id", "patient_id", "source_dataset", "file_path", "true_grade", "true_label", "split", "sha256_hash"]

    with open(manifest_out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows)} records to {manifest_out}")

    # Summary table
    print("\n--- Partition Summary ---")
    for s_name, recs in [("Train", train_records), ("Validation", val_records), ("Held-Out Test", test_records)]:
        c = Counter(r["true_grade"] for r in recs)
        print(f"{s_name:15s} (N={len(recs):4d}): Gr0={c[0]:4d}, Gr1={c[1]:3d}, Gr2={c[2]:3d}, Gr3={c[3]:3d}, Gr4={c[4]:3d}")

if __name__ == "__main__":
    build_aptos_manifest()
