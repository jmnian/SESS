#!/usr/bin/env python3
"""Download and prepare the GPQA dataset.

GPQA (Graduate-Level Google-Proof Q&A Benchmark) is a challenging multiple-choice
question answering dataset requiring expert-level knowledge.

This script downloads the dataset from HuggingFace and creates separate JSON files
for each subset: Main, Extended, Expert, and Diamond.

Usage:
    # Set your HuggingFace token (GPQA is a gated dataset)
    export HF_TOKEN="your_token_here"
    python scripts/download_gpqa_dataset.py
"""

import json
import os
import random
import sys

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OPRO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, OPRO_ROOT)

# Global seed for shuffling choices
GLOBAL_SEED = 42


def download_and_prepare_gpqa():
    """Download GPQA from HuggingFace and prepare JSON files for each subset."""
    from datasets import load_dataset
    from huggingface_hub import login
    
    # Check for HF token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Using HuggingFace token from HF_TOKEN environment variable...")
        login(token=hf_token)
    else:
        print("Warning: No HF_TOKEN found. GPQA is a gated dataset - you may need to authenticate.")
        print("Set HF_TOKEN environment variable or run 'huggingface-cli login'")
    
    print("Downloading GPQA dataset from HuggingFace...")
    
    # The GPQA dataset has multiple configs:
    # - gpqa_main
    # - gpqa_extended  
    # - gpqa_diamond
    # Note: "Expert" validation may be part of the extended or a specific split
    
    output_dir = os.path.join(OPRO_ROOT, "data", "GPQA-data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Map of subset names to HuggingFace config names
    subsets = {
        "main": "gpqa_main",
        "extended": "gpqa_extended",
        "diamond": "gpqa_diamond",
    }
    
    for subset_name, config_name in subsets.items():
        print(f"\nProcessing {subset_name} subset (config: {config_name})...")
        
        try:
            # Load the dataset (token is automatically used after login)
            dataset = load_dataset("Idavidrein/gpqa", config_name)
            
            # GPQA typically has a 'train' split that contains all examples
            # Check available splits
            print(f"  Available splits: {list(dataset.keys())}")
            
            # Use the first available split (usually 'train')
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            
            print(f"  Number of examples: {len(data)}")
            
            # Examine the columns
            print(f"  Columns: {data.column_names}")
            
            # Print first example to understand structure
            if len(data) > 0:
                print(f"  Sample row keys: {list(data[0].keys())}")
            
            # Process and save the data
            formatted_examples = []
            
            for idx, row in enumerate(data):
                # Set seed for deterministic shuffling per question
                random.seed(GLOBAL_SEED + idx)
                
                # Extract question and answers
                # Column names may vary - try common patterns
                question = row.get("Question", row.get("question", "")).strip()
                correct_answer = row.get("Correct Answer", row.get("correct_answer", "")).strip()
                incorrect_1 = row.get("Incorrect Answer 1", row.get("incorrect_answer_1", row.get("incorrect_answer1", ""))).strip()
                incorrect_2 = row.get("Incorrect Answer 2", row.get("incorrect_answer_2", row.get("incorrect_answer2", ""))).strip()
                incorrect_3 = row.get("Incorrect Answer 3", row.get("incorrect_answer_3", row.get("incorrect_answer3", ""))).strip()
                
                # Create choices list with their original type (correct/incorrect)
                choices = [
                    {"text": correct_answer, "is_correct": True},
                    {"text": incorrect_1, "is_correct": False},
                    {"text": incorrect_2, "is_correct": False},
                    {"text": incorrect_3, "is_correct": False},
                ]
                
                # Shuffle choices
                random.shuffle(choices)
                
                # Assign letters A, B, C, D
                choice_labels = ["A", "B", "C", "D"]
                labeled_choices = {}
                correct_label = None
                
                for i, choice in enumerate(choices):
                    label = choice_labels[i]
                    labeled_choices[label] = choice["text"]
                    if choice["is_correct"]:
                        correct_label = label
                
                # Format as multiple choice question
                mc_question = question + "\n"
                for label in choice_labels:
                    mc_question += f"({label}) {labeled_choices[label]}\n"
                
                example = {
                    "input": mc_question.strip(),
                    "question_only": question,
                    "target": correct_label,  # A, B, C, or D
                    "correct_answer_text": correct_answer,
                    "choices": labeled_choices,
                    "original_idx": idx,
                }
                
                # Add metadata if available
                if "High-level domain" in row or "high_level_domain" in row:
                    example["domain"] = row.get("High-level domain", row.get("high_level_domain", ""))
                if "Subdomain" in row or "subdomain" in row:
                    example["subdomain"] = row.get("Subdomain", row.get("subdomain", ""))
                
                formatted_examples.append(example)
            
            # Save to JSON file
            output_file = os.path.join(output_dir, f"gpqa_{subset_name}.json")
            with open(output_file, "w") as f:
                json.dump(formatted_examples, f, indent=2)
            
            print(f"  Saved {len(formatted_examples)} examples to {output_file}")
            
            # Print sample
            if formatted_examples:
                print(f"\n  Sample formatted question:")
                print(f"  {formatted_examples[0]['input'][:200]}...")
                print(f"  Correct answer: {formatted_examples[0]['target']}")
            
        except Exception as e:
            print(f"  Error processing {subset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("GPQA dataset download and preparation complete!")
    print(f"Data saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    download_and_prepare_gpqa()

