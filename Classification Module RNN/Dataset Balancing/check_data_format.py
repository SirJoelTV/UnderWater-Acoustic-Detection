"""Check actual data formats in the dataset."""
import os
from collections import defaultdict

print("Checking actual file formats in dataset...\n")

data_dir = r'D:\Main Project\UnderWater-Acoustic-Detection\data'
print(f"Data directory: {data_dir}\n")

file_types = defaultdict(list)
total_files = 0

for root, dirs, files in os.walk(data_dir):
    level = root.replace(data_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    rel_path = os.path.relpath(root, data_dir)
    
    if files:
        print(f"{indent}📁 {rel_path}/")
        for file in files[:10]:  # Show first 10 files
            ext = os.path.splitext(file)[1].lower()
            file_types[ext].append(os.path.join(root, file))
            total_files += 1
            file_size = os.path.getsize(os.path.join(root, file)) / 1e6  # MB
            print(f"{indent}  └─ {file[:50]:50s} ({file_size:.2f} MB) {ext}")
        
        if len(files) > 10:
            print(f"{indent}  ... and {len(files) - 10} more files")

print(f"\n{'='*70}")
print(f"Total files found: {total_files}")
print(f"\nFile type breakdown:")
for ext in sorted(file_types.keys(), key=lambda x: -len(file_types[x])):
    print(f"  {ext if ext else '(no extension)':15s}: {len(file_types[ext]):4d} files")

print(f"\n{'='*70}")
print("⚠️  If no .wav files found, your data might be in a different format!")
print("    Consider the following:")
print("    1. Are the files .mp3, .flac, or another audio format?")
print("    2. Do files have extensions? (might be extensionless)")
print("    3. Are they inside .crdownload or other temporary formats?")
