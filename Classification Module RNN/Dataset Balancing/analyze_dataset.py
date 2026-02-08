"""
Dataset analysis script to identify issues causing poor generalization.
"""
import os
import json
from collections import defaultdict
import librosa
import numpy as np
import config

def analyze_dataset():
    """Analyze dataset for common issues."""
    data_dir = config.DATA_DIR
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    # 1. Scan structure
    print("\n1. DIRECTORY STRUCTURE & FILE COUNT")
    print("-" * 70)
    
    class_counts = defaultdict(int)
    class_files = defaultdict(list)
    invalid_files = []
    total_duration = 0
    file_sizes = []
    
    for cat in os.listdir(data_dir):
        cat_path = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_path):
            continue
            
        print(f"\n  📂 {cat}/")
        cat_total = 0
        
        for subcat in os.listdir(cat_path):
            sub_path = os.path.join(cat_path, subcat)
            
            # Check if it's a directory
            if not os.path.isdir(sub_path):
                # It's a file, not a directory
                if subcat.endswith('.wav'):
                    try:
                        duration = librosa.get_sndfile_info(sub_path).duration
                        total_duration += duration
                        file_sizes.append(os.path.getsize(sub_path))
                        class_label = f"{cat}_{os.path.splitext(subcat)[0]}"
                        class_counts[class_label] += 1
                        class_files[class_label].append(sub_path)
                        cat_total += 1
                    except Exception as e:
                        invalid_files.append((sub_path, str(e)))
                continue
            
            # It's a directory
            subcat_files = [f for f in os.listdir(sub_path) if f.endswith('.wav')]
            subcat_count = 0
            
            for wav_file in subcat_files:
                wav_path = os.path.join(sub_path, wav_file)
                try:
                    # Verify it's a valid audio file
                    info = librosa.get_sndfile_info(wav_path)
                    duration = info.duration
                    
                    if duration < 1:  # Audio too short
                        invalid_files.append((wav_path, f"Too short: {duration}s"))
                        continue
                    
                    total_duration += duration
                    file_size = os.path.getsize(wav_path)
                    file_sizes.append(file_size)
                    
                    class_label = f"{cat}_{subcat}"
                    class_counts[class_label] += 1
                    class_files[class_label].append(wav_path)
                    subcat_count += 1
                    cat_total += 1
                    
                except Exception as e:
                    invalid_files.append((wav_path, f"Corrupt/Invalid: {str(e)}"))
            
            if subcat_count > 0:
                print(f"    └─ {subcat}: {subcat_count} files")
        
        if cat_total > 0:
            print(f"  Total: {cat_total} files")
    
    # 2. Dataset Statistics
    print("\n2. DATASET STATISTICS")
    print("-" * 70)
    
    total_files = sum(class_counts.values())
    print(f"✓ Total valid WAV files: {total_files}")
    print(f"✗ Invalid/Corrupt files: {len(invalid_files)}")
    print(f"  Total dataset duration: {total_duration/3600:.2f} hours")
    print(f"  Average file duration: {total_duration/max(1, total_files):.2f} seconds")
    print(f"  Average file size: {np.mean(file_sizes)/1e6:.2f} MB" if file_sizes else "  No files found")
    
    # 3. Class Distribution
    print("\n3. CLASS DISTRIBUTION (CRITICAL!)")
    print("-" * 70)
    
    if not class_counts:
        print("❌ NO AUDIO FILES FOUND! Check your data directory!")
        return
    
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nTotal unique classes: {len(class_counts)}")
    print(f"Imbalance ratio: {imbalance_ratio:.1f}x (Max/Min)")
    
    print("\nClass counts:")
    for class_name, count in sorted_classes:
        pct = 100 * count / total_files
        bar = "█" * int(pct / 2)
        issue = ""
        if count < 10:
            issue = " ⚠️  TOO SMALL (< 10 files)"
        elif count < 50:
            issue = " ⚠️  Small"
        print(f"  {class_name:30s}: {count:4d} ({pct:5.1f}%) {bar}")
        if issue:
            print(f"    {issue}")
    
    # 4. Potential Issues
    print("\n4. POTENTIAL ISSUES")
    print("-" * 70)
    
    issues = []
    
    # Issue 1: Too few files overall
    if total_files < 100:
        issues.append(f"❌ Dataset too small: {total_files} files (need at least 200-500)")
    elif total_files < 500:
        issues.append(f"⚠️  Dataset is small: {total_files} files (ideal: 500+)")
    
    # Issue 2: Class imbalance
    if imbalance_ratio > 10:
        issues.append(f"❌ SEVERE class imbalance: {imbalance_ratio:.1f}x ratio (should be <3x)")
    elif imbalance_ratio > 3:
        issues.append(f"⚠️  Class imbalance: {imbalance_ratio:.1f}x ratio (should be <3x)")
    
    # Issue 3: Too few classes
    if len(class_counts) < 2:
        issues.append("❌ Binary classification detected - ensure this is intentional")
    elif len(class_counts) < 4:
        issues.append("⚠️  Very few classes - may be too simple")
    
    # Issue 4: Classes with very few samples
    small_classes = [c for c, count in class_counts.items() if count < 20]
    if small_classes:
        issues.append(f"❌ {len(small_classes)} classes with <20 files: {small_classes}")
    
    # Issue 5: Corrupt files
    if invalid_files:
        issues.append(f"❌ {len(invalid_files)} corrupt/invalid files:")
        for path, error in invalid_files[:5]:
            issues.append(f"   - {os.path.basename(path)}: {error}")
        if len(invalid_files) > 5:
            issues.append(f"   ... and {len(invalid_files)-5} more")
    
    # Issue 6: Very short duration
    if total_duration / total_files < 5:
        issues.append(f"⚠️  Average duration too short: {total_duration/total_files:.1f}s (need 10-30s)")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ Dataset structure looks reasonable!")
    
    # 5. Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 70)
    
    recommendations = []
    
    if total_files < 300:
        recommendations.append("🔴 HIGH PRIORITY: Collect more data (need 300+, ideally 500+)")
    
    if imbalance_ratio > 5:
        recommendations.append("🔴 HIGH PRIORITY: Balance classes (use oversampling, undersampling, or class weights)")
    
    small_class_files = [f for c, f in class_files.items() if len(f) < 20]
    if small_class_files:
        recommendations.append("🟡 MEDIUM: Remove or merge classes with <20 files")
    
    if invalid_files:
        recommendations.append("🟡 MEDIUM: Remove corrupt files")
    
    if not recommendations:
        recommendations.append("✓ No major issues detected - check model architecture instead!")
    
    for rec in recommendations:
        print(rec)
    
    # 6. Summary for Training
    print("\n6. TRAIN/VAL SPLIT PREVIEW (80/20)")
    print("-" * 70)
    
    for class_name, count in sorted_classes[:5]:  # Show top 5
        train_size = int(0.8 * count)
        val_size = count - train_size
        print(f"  {class_name:30s}: Train={train_size}, Val={val_size}")
    
    print("\n" + "=" * 70)
    
    # Save analysis to JSON
    analysis_data = {
        'total_files': total_files,
        'invalid_files': len(invalid_files),
        'total_classes': len(class_counts),
        'total_duration_hours': total_duration / 3600,
        'imbalance_ratio': imbalance_ratio,
        'class_counts': dict(sorted_classes),
        'issues': issues,
        'recommendations': recommendations
    }
    
    with open('dataset_analysis.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    print("Analysis saved to dataset_analysis.json\n")

if __name__ == "__main__":
    analyze_dataset()
