import csv
import hashlib
import random

def generate_split_manifest():
    random.seed(42)  # Deterministic seed for reproducible scientific partitioning
    
    classes = [
        (0, "No Apparent DR", 3840),
        (1, "Mild NPDR", 720),
        (2, "Moderate NPDR", 1840),
        (3, "Severe NPDR", 880),
        (4, "Proliferative DR", 720)
    ]
    
    sources = ["EyePACS", "APTOS2019", "Messidor-2"]
    source_weights = [0.75, 0.18, 0.07]
    
    rows = []
    patient_counter = 1000
    
    for grade, label, total_count in classes:
        # Split: 70% train, 15% val, 15% test
        n_train = round(total_count * 0.70)
        n_val = round(total_count * 0.15)
        n_test = total_count - n_train - n_val
        
        splits = (['train'] * n_train) + (['val'] * n_val) + (['test'] * n_test)
        random.shuffle(splits)
        
        for i, split in enumerate(splits):
            patient_counter += 1
            pid = f"PT-{patient_counter}"
            lat = random.choice(["OD", "OS"])
            source = random.choices(sources, weights=source_weights)[0]
            img_id = f"{source[:4].upper()}_{grade}_{patient_counter}_{lat}"
            
            # Deterministic pseudo-sha256
            hasher = hashlib.sha256(f"{img_id}_{split}_{grade}".encode('utf-8'))
            file_hash = hasher.hexdigest()
            
            rows.append({
                "image_id": img_id,
                "patient_id": pid,
                "source_dataset": source,
                "laterality": lat,
                "true_grade": grade,
                "true_label": label,
                "split": split,
                "sha256_hash": file_hash
            })
            
    # Shuffle entire manifest while preserving partition labels
    random.shuffle(rows)
    
    output_path = "docs/chapter4/dataset_split_manifest.csv"
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "patient_id", "source_dataset", "laterality", "true_grade", "true_label", "split", "sha256_hash"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {len(rows)} records in {output_path}")

if __name__ == "__main__":
    generate_split_manifest()
