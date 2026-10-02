import os
import zipfile
import hashlib

def build_zip():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_zip = os.path.join(repo_root, "CHAPTER_4_THESIS_SUBMISSION_EVIDENCE.zip")

    # Exclusions
    exclude_dirs = {
        ".venv", "venv", "node_modules", ".git", ".pytest_cache", 
        "__pycache__", ".vscode", ".idea", "dist_temp", "scratch"
    }
    exclude_extensions = {".pyc", ".pyo", ".pyd", ".tar.gz", ".zip", ".log"}
    exclude_files = {"CHAPTER_4_THESIS_SUBMISSION_EVIDENCE.zip", "cdss.tar.gz", "evidence.zip", "submission.zip"}

    print(f"Building clean submission archive: {out_zip}")
    file_count = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(repo_root):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]

            for file in files:
                if file in exclude_files:
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_extensions and file != "test_execution.log":
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_root)
                
                # Double check no venv or node_modules slipped in
                parts = rel_path.replace("\\", "/").split("/")
                if any(p in exclude_dirs for p in parts):
                    continue

                zf.write(full_path, rel_path)
                file_count += 1

    size_bytes = os.path.getsize(out_zip)
    
    # Compute SHA-256
    sha256 = hashlib.sha256()
    with open(out_zip, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    digest = sha256.hexdigest()

    print(f"\n=======================================================")
    print(f"SUBMISSION ARCHIVE GENERATED SUCCESSFULLY")
    print(f"=======================================================")
    print(f"Archive:   {out_zip}")
    print(f"Files:     {file_count:,}")
    print(f"Size:      {size_bytes:,} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"SHA-256:   {digest}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    build_zip()
