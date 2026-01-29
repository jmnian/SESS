#!/usr/bin/env python3
"""Download MATH dataset from HuggingFace and save to data/MATH-data/.

The MATH dataset (Hendrycks et al.) contains competition mathematics problems
with step-by-step solutions. This script downloads the numeric subset which
has clean numerical answers for easier evaluation.

Usage:
    python scripts/download_math_dataset.py
"""

import json
import os
import sys

# Add OPRO root to path
OPRO_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, OPRO_ROOT_PATH)

from datasets import load_dataset


def main():
    output_dir = os.path.join(OPRO_ROOT_PATH, "data", "MATH-data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading MATH dataset from HuggingFace...")
    print("Dataset: jeggers/competition_math (numeric subset)")
    
    # Load the numeric subset which has clean numerical answers
    dataset = load_dataset("jeggers/competition_math", "numeric")
    
    print(f"\nDataset structure:")
    print(f"  Train split: {len(dataset['train'])} examples")
    print(f"  Test split: {len(dataset['test'])} examples")
    
    # Show sample
    print("\nSample problem:")
    sample = dataset['train'][0]
    print(f"  Problem: {sample['problem'][:200]}...")
    print(f"  Answer: {sample['extracted_solution']}")
    print(f"  Level: {sample.get('level', 'N/A')}")
    print(f"  Type: {sample.get('type', 'N/A')}")
    
    # Save train split
    train_data = []
    for item in dataset['train']:
        train_data.append({
            "problem": item['problem'],
            "solution": item['solution'],
            "answer": item['extracted_solution'],
            "level": item.get('level', ''),
            "type": item.get('type', ''),
        })
    
    train_path = os.path.join(output_dir, "math_train.json")
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"\nSaved {len(train_data)} training examples to {train_path}")
    
    # Save test split
    test_data = []
    for item in dataset['test']:
        test_data.append({
            "problem": item['problem'],
            "solution": item['solution'],
            "answer": item['extracted_solution'],
            "level": item.get('level', ''),
            "type": item.get('type', ''),
        })
    
    test_path = os.path.join(output_dir, "math_test.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Saved {len(test_data)} test examples to {test_path}")
    
    # Also save by difficulty level for potential stratified sampling
    levels = {}
    for item in train_data:
        level = item.get('level', 'unknown')
        if level not in levels:
            levels[level] = []
        levels[level].append(item)
    
    print(f"\nTraining set by difficulty level:")
    for level, items in sorted(levels.items()):
        print(f"  {level}: {len(items)} examples")
    
    # Save by type/category
    types = {}
    for item in train_data:
        t = item.get('type', 'unknown')
        if t not in types:
            types[t] = []
        types[t].append(item)
    
    print(f"\nTraining set by problem type:")
    for t, items in sorted(types.items()):
        print(f"  {t}: {len(items)} examples")
    
    print(f"\nDone! Data saved to {output_dir}")


if __name__ == "__main__":
    main()

