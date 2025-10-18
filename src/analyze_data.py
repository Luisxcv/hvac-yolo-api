"""
Usage:
    python src/analyze_data.py --dataset datasets/HVAC_8_CLASS-4
"""

import os
import argparse
from collections import Counter
from glob import glob
import yaml

def analyze_labels(dataset_dir):
    labels_dir = os.path.join(dataset_dir, "train", "labels")
    label_files = glob(os.path.join(labels_dir, "*.txt"))
    counter = Counter()

    for f in label_files:
        with open(f) as lf:
            for line in lf:
                cls = line.strip().split()[0]
                counter[cls] += 1

    print(f"\nFound {len(label_files)} label files.")
    print("Class frequencies:")
    for cls, count in counter.items():
        print(f"  Class {cls}: {count}")

    if counter:
        max_c = max(counter.values())
        min_c = min(counter.values())
        ratio = max_c / min_c if min_c else 0
        print(f"\nImbalance ratio (max/min): {ratio:.2f}")
    else:
        print("No labels found — please check dataset path.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to YOLO dataset directory")
    args = parser.parse_args()
    analyze_labels(args.dataset)

if __name__ == "__main__":
    main()