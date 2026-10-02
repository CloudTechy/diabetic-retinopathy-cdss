import os, zipfile, hashlib

TARGET_ZIP = 'THESIS_FINAL_QA_SOURCE.zip'

include_items = [
    'backend/app',
    'backend/tests',
    'backend/scripts',
    'backend/models/weights/efficientnet_b0_dr.pth',
    'backend/main.py',
    'backend/requirements.txt',
    'backend/Dockerfile',
    'backend/pytest.ini',
    'frontend/src',
    'frontend/package.json',
    'frontend/tsconfig.json',
    'frontend/tsconfig.node.json',
    'frontend/vite.config.ts',
    'frontend/tailwind.config.js',
    'frontend/postcss.config.js',
    'frontend/index.html',
    'frontend/Dockerfile',
    'frontend/nginx.conf',
    'docs/chapter4',
    'docker-compose.yml',
    'docker-compose.prod.yml',
    '.env.example',
]

exclude_substrings = [
    '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'dist', '.git',
    '.zip', '.tar.gz',
]

added = []
with zipfile.ZipFile(TARGET_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for item in include_items:
        if not os.path.exists(item):
            print(f'Warning: {item} does not exist!')
            continue
        if os.path.isfile(item):
            zf.write(item, item)
            added.append(item)
        elif os.path.isdir(item):
            for root, dirs, files in os.walk(item):
                dirs[:] = [d for d in dirs if d not in exclude_substrings and not d.startswith('.')]
                for file in files:
                    fp = os.path.join(root, file).replace(chr(92), '/')
                    if any(ex in fp for ex in exclude_substrings):
                        continue
                    zf.write(fp, fp)
                    added.append(fp)

sz = os.path.getsize(TARGET_ZIP)
h = hashlib.sha256()
with open(TARGET_ZIP, 'rb') as f:
    while chunk := f.read(65536):
        h.update(chunk)

print(f'Archive: {TARGET_ZIP}')
print(f'Files: {len(added)}')
print(f'Size: {sz:,} bytes ({sz/(1024*1024):.2f} MiB)')
print(f'SHA-256: {h.hexdigest()}')
