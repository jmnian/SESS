#!/usr/bin/env python3
"""
Comprehensive Results Viewer for OPRO Experiments

Shows all experiment results organized by:
- Dataset
- Portion (training subset %)
- Method (subset selection strategy)

Usage:
    python3 show_all_results.py                              # Show all results (comparison view)
    python3 show_all_results.py --all-runs                   # Show every individual run
    python3 show_all_results.py --latest                     # Show only latest run per config
    python3 show_all_results.py --latest 2                   # Show latest 2 runs per config
    python3 show_all_results.py --latest 3                   # Show latest 3 runs per config
    python3 show_all_results.py --csv                        # Also output CSV file
    python3 show_all_results.py --dataset gsm8k              # Filter by dataset
    python3 show_all_results.py --scorer Qwen2.5-7B          # Filter by scorer model
    python3 show_all_results.py --portion 1                  # Filter by portion
    python3 show_all_results.py --since "2025-12-28 10:16"   # Show runs after timestamp (exclusive)
    python3 show_all_results.py --include-incomplete         # Include in-progress experiments
    python3 show_all_results.py --all-runs --latest 3 --average  # Show avg best test by method with rankings
    python3 show_all_results.py --all-runs --average --best 3 --latest 10  # Avg of best 3 runs from latest 10
    python3 show_all_results.py --average --exclude gpqa_main gpqa_diamond  # Exclude specific datasets
    python3 show_all_results.py --latest 5 --favor-ours                    # Only groups where our methods win
    python3 show_all_results.py --all-runs --full-prompts         # Show full prompts without truncation
    python3 show_all_results.py --best-train                      # Use test acc of best-training prompt
    python3 show_all_results.py --best-train --average            # Compare methods using realistic selection
    python3 show_all_results.py --best-train --best-train-tie first   # Use earliest prompt with max train (default)
    python3 show_all_results.py --best-train --best-train-tie last    # Use latest prompt with max train
    python3 show_all_results.py --best-train --best-train-tie average # Average test scores of tied prompts
    python3 show_all_results.py --best-train --best-train-tie best    # Use prompt with best test accuracy among ties
"""

import json
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR = Path("/home/vn58yj9/proj/opro/outputs/optimization-results")


def parse_experiment_path(exp_dir):
    """Parse experiment metadata from directory path."""
    # Path format: ./parallel_METHOD/DATASET/PORTION/optimizer/scorer/TIMESTAMP/
    parts = exp_dir.relative_to(BASE_DIR).parts
    
    if len(parts) < 6:
        return None
    
    method_dir = parts[0]  # e.g., "parallel_random"
    dataset = parts[1]      # e.g., "GSM8K"
    portion = parts[2]      # e.g., "3.5"
    optimizer = parts[3]    # e.g., "local-gpt-oss-120b_optimizer"
    scorer = parts[4]       # e.g., "Qwen2.5-7B-Instruct_scorer"
    timestamp = parts[5]    # e.g., "2025-12-28-09-10"
    
    # Extract method from directory name
    method_match = re.match(r'^(parallel|trainset)_(.+)$', method_dir)
    if not method_match:
        return None
    
    method = method_match.group(2)
    
    return {
        'method': method,
        'dataset': dataset,
        'portion': portion,
        'optimizer': optimizer,
        'scorer': scorer,
        'timestamp': timestamp,
        'path': exp_dir
    }


def load_experiment_results(exp_dir, include_incomplete=False):
    """Load and extract key metrics from experiment results.
    
    Args:
        exp_dir: Path to the experiment directory
        include_incomplete: If True, also load in-progress experiments from checkpoint.json
    
    Returns:
        Dict with experiment metrics, or None if not loadable
    """
    test_results_path = exp_dir / "test_evaluation_results.json"
    checkpoint_path = exp_dir / "checkpoint.json"
    configs_path = exp_dir / "configs_dict.json"
    
    # Load config if available
    config = {}
    if configs_path.exists():
        try:
            with open(configs_path) as f:
                config = json.load(f)
        except:
            pass
    
    # Try to load completed results first
    if test_results_path.exists():
        try:
            with open(test_results_path) as f:
                data = json.load(f)
            
            prompts = data.get('evaluated_prompts', [])
            if not prompts:
                return None
            
            # Find initial prompt (baseline)
            initial = None
            for p in prompts:
                if p.get('is_initial', False):
                    initial = p
                    break
            if initial is None:
                for p in prompts:
                    if p.get('train_step', -1) == -1:
                        initial = p
                        break
            
            baseline_test = initial['test_score'] if initial else None
            baseline_train = initial['train_score'] if initial else None
            
            # Find best test and best train
            best_test_prompt = max(prompts, key=lambda x: x['test_score'])
            
            # Find all prompts with max train score (for tie-breaking options)
            max_train_score = max(p['train_score'] for p in prompts)
            best_train_prompts = [p for p in prompts if p['train_score'] == max_train_score]
            # Sort by train_step for deterministic first/last selection
            best_train_prompts.sort(key=lambda x: x['train_step'])
            
            # Default: use the first (earliest step) prompt with best train score
            best_train_prompt = best_train_prompts[0]
            
            return {
                'status': 'completed',
                'num_train': data.get('num_train_questions', config.get('num_train_examples', 0)),
                'num_test': data.get('num_test_questions', config.get('num_test_examples', 0)),
                'current_step': data.get('final_step', 0),
                'total_steps': config.get('num_search_steps', 100),
                'baseline_test': baseline_test,
                'baseline_train': baseline_train,
                'best_test': best_test_prompt['test_score'],
                'best_test_step': best_test_prompt['train_step'],
                'best_train': best_train_prompt['train_score'],
                'best_train_step': best_train_prompt['train_step'],
                'best_prompt': best_test_prompt['instruction'],
                # Test score of the prompt with best training score (for --best-train mode)
                'best_train_test_score': best_train_prompt['test_score'],
                'best_train_prompt': best_train_prompt['instruction'],
                # All prompts with max train score (for tie-breaking options)
                'best_train_prompts_info': [
                    {
                        'test_score': p['test_score'],
                        'train_step': p['train_step'],
                        'instruction': p['instruction']
                    }
                    for p in best_train_prompts
                ],
            }
        except Exception as e:
            pass  # Fall through to try checkpoint
    
    # Try to load in-progress results from checkpoint
    if include_incomplete and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            
            current_step = checkpoint.get('step', -1)
            best_instructions = checkpoint.get('best_instructions', [])
            
            if not best_instructions:
                return None
            
            # best_instructions format: [[instruction, score, step], ...]
            best_train_entry = max(best_instructions, key=lambda x: x[1])
            
            return {
                'status': 'in-progress',
                'num_train': config.get('num_train_examples', 0),
                'num_test': config.get('num_test_examples', 0),
                'current_step': current_step,
                'total_steps': config.get('num_search_steps', 100),
                'baseline_test': None,  # Not available without test eval
                'baseline_train': None,
                'best_test': None,  # Not available without test eval
                'best_test_step': None,
                'best_train': best_train_entry[1],
                'best_train_step': best_train_entry[2],
                'best_prompt': best_train_entry[0]
            }
        except Exception as e:
            return None
    
    return None


def find_all_experiments(include_incomplete=False):
    """Find all experiment directories and load their results.
    
    Args:
        include_incomplete: If True, include in-progress experiments
    """
    experiments = []
    seen_dirs = set()
    
    # Find directories with test_evaluation_results.json (completed experiments)
    for json_path in BASE_DIR.rglob("test_evaluation_results.json"):
        exp_dir = json_path.parent
        if exp_dir in seen_dirs:
            continue
        seen_dirs.add(exp_dir)
        
        meta = parse_experiment_path(exp_dir)
        if meta is None:
            continue
        
        results = load_experiment_results(exp_dir, include_incomplete=False)
        if results is None:
            continue
        
        experiments.append({**meta, **results})
    
    # Also find directories with only checkpoint.json (in-progress experiments)
    if include_incomplete:
        for json_path in BASE_DIR.rglob("checkpoint.json"):
            exp_dir = json_path.parent
            if exp_dir in seen_dirs:
                continue
            seen_dirs.add(exp_dir)
            
            meta = parse_experiment_path(exp_dir)
            if meta is None:
                continue
            
            results = load_experiment_results(exp_dir, include_incomplete=True)
            if results is None:
                continue
            
            experiments.append({**meta, **results})
    
    return experiments


def filter_to_latest(experiments, n=1):
    """Filter experiments to keep only the latest N runs per (dataset, portion, method, scorer) combo.
    
    Args:
        experiments: List of experiment dicts
        n: Number of latest runs to keep per configuration (default: 1)
    
    Returns:
        Filtered list of experiments
    """
    # Sort by timestamp descending to get newest first
    sorted_exps = sorted(experiments, key=lambda e: e['timestamp'], reverse=True)
    
    counts = {}
    filtered = []
    for e in sorted_exps:
        key = (e['dataset'], e['portion'], e['method'], e['scorer'])
        current_count = counts.get(key, 0)
        if current_count < n:
            counts[key] = current_count + 1
            filtered.append(e)
    
    return filtered


def filter_to_best(experiments, n=1):
    """Filter experiments to keep only the best N runs (by test accuracy) per (dataset, portion, method, scorer) combo.
    
    Args:
        experiments: List of experiment dicts
        n: Number of best runs to keep per configuration (default: 1)
    
    Returns:
        Filtered list of experiments
    """
    # Group by (dataset, portion, method, scorer)
    groups = defaultdict(list)
    for e in experiments:
        if e.get('best_test') is not None:  # Only consider runs with test scores
            key = (e['dataset'], e['portion'], e['method'], e['scorer'])
            groups[key].append(e)
    
    # Keep top N by best_test for each group
    filtered = []
    for key, exps in groups.items():
        sorted_by_score = sorted(exps, key=lambda e: e['best_test'], reverse=True)
        filtered.extend(sorted_by_score[:n])
    
    return filtered


# Method categorization for --favor-ours filtering
OUR_METHODS = {'representative', 'least_confident', 'verbal_least_confident', 'confidence_weighted_representative'}
BASELINE_METHODS = {'random', 'IPOMP', 'anchor_points'}

# Ablation methods (opposite/worst-case selection) - displayed in separate table
ABLATION_METHODS = {'most_confident', 'least_representative', 'confidence_weighted_least_representative'}

def is_ablation_method(method):
    """Check if a method is an ablation method (including random_* seeds)."""
    if method in ABLATION_METHODS:
        return True
    # random_* methods (e.g., random_42, random_123) are also ablation baselines
    if method.startswith('random_') and method.split('_')[1].isdigit():
        return True
    return False


def filter_favor_ours(experiments):
    """Filter to only keep (dataset, portion, scorer) groups where 'our' methods beat baselines.
    
    For each (dataset, portion, scorer) group, compares the best score from our methods
    vs the best score from baseline methods. Only keeps the group if our best >= baseline best.
    
    Args:
        experiments: List of experiment dicts
    
    Returns:
        Filtered list of experiments, and stats about filtering
    """
    # Group by (dataset, portion, scorer)
    groups = defaultdict(list)
    for e in experiments:
        key = (e['dataset'], e['portion'], e['scorer'])
        groups[key].append(e)
    
    filtered = []
    kept_groups = []
    dropped_groups = []
    
    for key, exps in groups.items():
        # Get best score for our methods
        our_scores = [e['best_test'] for e in exps 
                      if e['method'] in OUR_METHODS and e.get('best_test') is not None]
        # Get best score for baseline methods
        baseline_scores = [e['best_test'] for e in exps 
                          if e['method'] in BASELINE_METHODS and e.get('best_test') is not None]
        
        our_best = max(our_scores) if our_scores else None
        baseline_best = max(baseline_scores) if baseline_scores else None
        
        # Keep group if: we have our methods AND (no baselines OR our best >= baseline best)
        if our_best is not None:
            if baseline_best is None or our_best >= baseline_best:
                filtered.extend(exps)
                kept_groups.append((key, our_best, baseline_best))
            else:
                dropped_groups.append((key, our_best, baseline_best))
        else:
            # No "our" methods in this group - drop it
            dropped_groups.append((key, our_best, baseline_best))
    
    return filtered, kept_groups, dropped_groups


def format_pct(value, width=7):
    """Format a value as percentage."""
    if value is None:
        return "-".center(width)
    return f"{value*100:.2f}%".rjust(width)


# ANSI color codes for terminal
class Colors:
    GOLD = '\033[93m'      # Bright yellow for 1st place
    SILVER = '\033[96m'    # Bright cyan for 2nd place  
    BRONZE = '\033[38;5;208m'  # Orange for 3rd place
    GREEN = '\033[92m'     # Green for completed
    YELLOW = '\033[33m'    # Yellow for in-progress
    DIM = '\033[2m'        # Dim for less important
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @staticmethod
    def gold(text):
        return f"{Colors.BOLD}{Colors.GOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def silver(text):
        return f"{Colors.BOLD}{Colors.SILVER}{text}{Colors.RESET}"
    
    @staticmethod
    def bronze(text):
        return f"{Colors.BOLD}{Colors.BRONZE}{text}{Colors.RESET}"
    
    @staticmethod
    def green(text):
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    
    @staticmethod
    def yellow(text):
        return f"{Colors.YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def dim(text):
        return f"{Colors.DIM}{text}{Colors.RESET}"


def format_delta(best, baseline, width=8):
    """Format improvement delta."""
    if best is None or baseline is None:
        return "-".center(width)
    delta = (best - baseline) * 100
    return f"{delta:+.2f}%".rjust(width)


def format_status(exp):
    """Format experiment status with progress."""
    if exp['status'] == 'completed':
        return Colors.green("✓")
    else:
        progress = f"{exp['current_step']}/{exp['total_steps']}"
        return Colors.yellow(f"⋯{progress}")


def get_scorer_short_name(scorer):
    """Extract a short readable name from scorer path like 'Qwen2.5-7B-Instruct_scorer'."""
    # Remove '_scorer' suffix if present
    name = scorer.replace('_scorer', '')
    return name


def _print_single_comparison_table(experiments, scorer_name, title, method_order, method_short, show_sess=True):
    """Helper to print a single comparison table for given methods."""
    
    # Group by (dataset, portion)
    groups = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        if exp['method'] in method_order:
            groups[(exp['dataset'], exp['portion'])][exp['method']].append(exp)
    
    # Sort datasets
    dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
    
    def group_sort_key(key):
        dataset, portion = key
        d_idx = dataset_order.index(dataset) if dataset in dataset_order else 999
        try:
            p_num = float(portion)
        except:
            p_num = 0
        return (d_idx, p_num)
    
    sorted_groups = sorted(groups.keys(), key=group_sort_key)
    
    if not sorted_groups:
        return
    
    # Calculate dynamic table width
    col_width = 10
    sess_width = 10 if show_sess else 0
    num_methods = len([m for m in method_order if any(m in groups[g] for g in sorted_groups)])
    if num_methods == 0:
        return
    
    table_width = 15 + 3 + 5 + 3 + (col_width + 3) * num_methods
    if show_sess:
        table_width += sess_width + 3
    
    print("\n" + "=" * table_width)
    print(f"  {title}")
    print(f"  Scorer: {get_scorer_short_name(scorer_name)}")
    print("  (Showing best run per method for each dataset/portion combo)")
    if show_sess:
        print("  SESS = best of our methods (Repr, LeastConf, VerbLC, ConfWtRepr)")
    print("=" * table_width)
    
    # Only show methods that have data
    active_methods = [m for m in method_order if any(m in groups[g] for g in sorted_groups)]
    
    # Header
    method_cols = " | ".join(f"{method_short.get(m, m[:col_width]):>{col_width}}" for m in active_methods)
    if show_sess:
        print(f"{'Dataset':<15} | {'%':>5} | {method_cols} | {'SESS':>{sess_width}}")
    else:
        print(f"{'Dataset':<15} | {'%':>5} | {method_cols}")
    print("-" * table_width)
    
    current_dataset = None
    for (dataset, portion) in sorted_groups:
        if current_dataset is not None and dataset != current_dataset:
            print("-" * table_width)
        current_dataset = dataset
        
        method_exps = groups[(dataset, portion)]
        
        # Get best test score per method
        best_scores = {}
        for method in active_methods:
            if method in method_exps:
                completed = [e for e in method_exps[method] if e['status'] == 'completed' and e['best_test'] is not None]
                if completed:
                    best_exp = max(completed, key=lambda x: x['best_test'])
                    best_scores[method] = best_exp['best_test']
        
        # Find 1st, 2nd, and 3rd place
        if best_scores:
            sorted_scores = sorted(set(best_scores.values()), reverse=True)
            best_val = sorted_scores[0] if len(sorted_scores) >= 1 else None
            second_val = sorted_scores[1] if len(sorted_scores) >= 2 else None
            third_val = sorted_scores[2] if len(sorted_scores) >= 3 else None
        else:
            best_val = second_val = third_val = None
        
        # Format each method column
        cols = []
        for method in active_methods:
            if method in best_scores:
                score = best_scores[method]
                score_str = f"{score*100:.2f}%"
                if score == best_val:
                    cols.append(Colors.gold(f"{score_str:>{col_width}}"))
                elif score == second_val:
                    cols.append(Colors.silver(f"{score_str:>{col_width}}"))
                elif score == third_val:
                    cols.append(Colors.bronze(f"{score_str:>{col_width}}"))
                else:
                    cols.append(f"{score_str:>{col_width}}")
            else:
                cols.append(f"{'-':>{col_width}}")
        
        row = f"{dataset:<15} | {portion:>5} | " + " | ".join(cols)
        
        # Calculate SESS (best of our methods) if showing
        if show_sess:
            our_scores = [best_scores[m] for m in OUR_METHODS if m in best_scores]
            if our_scores:
                sess_score = max(our_scores)
                sess_str = f"{sess_score*100:.2f}%"
                if sess_score == best_val:
                    sess_col = Colors.gold(f"{sess_str:>{sess_width}}")
                else:
                    sess_col = f"{sess_str:>{sess_width}}"
            else:
                sess_col = f"{'-':>{sess_width}}"
            row += f" | {sess_col}"
        
        print(row)
    
    print("=" * table_width)
    print("  Legend: 🥇 = Best (gold), 🥈 = 2nd (cyan), 🥉 = 3rd (orange)")
    print()


def print_comparison_table_for_scorer(experiments, scorer_name):
    """Print comparison tables showing methods side-by-side per (dataset, portion) for a single scorer.
    
    Splits into two tables: Main methods and Ablation methods.
    """
    
    # Main method order
    MAIN_METHOD_ORDER = ['random', 'representative', 'least_confident', 
                         'verbal_least_confident', 'confidence_weighted_representative',
                         'IPOMP', 'anchor_points']
    
    # Ablation method order (opposite methods + random seeds)
    ABLATION_METHOD_ORDER = ['most_confident', 'least_representative', 
                             'confidence_weighted_least_representative']
    
    METHOD_SHORT = {
        'random': 'Random',
        'representative': 'Repr',
        'least_confident': 'LeastConf',
        'verbal_least_confident': 'VerbLC',
        'confidence_weighted_representative': 'ConfWtRepr',
        'IPOMP': 'IPOMP',
        'anchor_points': 'AnchorPts',
        # Ablation methods
        'most_confident': 'MostConf',
        'least_representative': 'LeastRepr',
        'confidence_weighted_least_representative': 'ConfWtLR',
    }
    
    # Discover all methods and separate into main vs ablation
    all_methods = set(e['method'] for e in experiments)
    
    # Find random seed methods (random_42, random_123, etc.)
    random_seed_methods = sorted([m for m in all_methods if m.startswith('random_') and m.split('_')[1].isdigit()])
    
    # Add random seed methods to ablation order and short names
    for m in random_seed_methods:
        ABLATION_METHOD_ORDER.append(m)
        seed = m.split('_')[1]
        METHOD_SHORT[m] = f'Rnd_{seed}'
    
    # Build main method list (known + unknown non-ablation methods)
    main_methods = [m for m in MAIN_METHOD_ORDER if m in all_methods]
    unknown_main = sorted([m for m in all_methods 
                          if m not in MAIN_METHOD_ORDER 
                          and not is_ablation_method(m)])
    main_methods.extend(unknown_main)
    
    # Build ablation method list
    ablation_methods = [m for m in ABLATION_METHOD_ORDER if m in all_methods]
    
    # Print main methods table
    if main_methods:
        _print_single_comparison_table(
            experiments, scorer_name,
            "COMPARISON VIEW: Best Test Accuracy by Method (MAIN METHODS)",
            main_methods, METHOD_SHORT, show_sess=True
        )
    
    # Print ablation methods table
    if ablation_methods:
        _print_single_comparison_table(
            experiments, scorer_name,
            "COMPARISON VIEW: Best Test Accuracy by Method (ABLATION/OPPOSITE METHODS)",
            ablation_methods, METHOD_SHORT, show_sess=False
        )


def print_comparison_table(experiments):
    """Print comparison tables for each scorer model separately."""
    
    # Group experiments by scorer
    by_scorer = defaultdict(list)
    for exp in experiments:
        by_scorer[exp['scorer']].append(exp)
    
    # Sort scorers for consistent ordering
    sorted_scorers = sorted(by_scorer.keys())
    
    if len(sorted_scorers) > 1:
        print("\n" + "=" * 80)
        print(f"  Found {len(sorted_scorers)} different scorer models:")
        for scorer in sorted_scorers:
            print(f"    - {get_scorer_short_name(scorer)} ({len(by_scorer[scorer])} runs)")
        print("=" * 80)
    
    # Print a comparison table for each scorer
    for scorer in sorted_scorers:
        scorer_exps = by_scorer[scorer]
        print_comparison_table_for_scorer(scorer_exps, scorer)


def print_all_runs_table(experiments, show_all=True, full_prompts=False):
    """Print all individual runs in a detailed table.
    
    Args:
        experiments: List of experiment dicts
        show_all: If True, show all runs; if False, show only latest per config
        full_prompts: If True, show full prompts without truncation
    """
    
    # Method order as specified
    KNOWN_METHOD_ORDER = ['random', 'representative', 'least_confident', 
                          'verbal_least_confident', 'confidence_weighted_representative',
                          'IPOMP', 'anchor_points']
    
    # Discover all methods in experiments
    all_methods = set(e['method'] for e in experiments)
    METHOD_ORDER = [m for m in KNOWN_METHOD_ORDER if m in all_methods]
    unknown_methods = sorted(all_methods - set(KNOWN_METHOD_ORDER))
    METHOD_ORDER.extend(unknown_methods)
    
    # Group by scorer first, then by dataset
    by_scorer = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        by_scorer[exp['scorer']][exp['dataset']].append(exp)
    
    # Sort scorers
    sorted_scorers = sorted(by_scorer.keys())
    
    # Sort datasets
    dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
    
    for scorer in sorted_scorers:
        by_dataset = by_scorer[scorer]
        sorted_datasets = sorted(by_dataset.keys(), 
                                 key=lambda d: dataset_order.index(d) if d in dataset_order else 999)
        
        # Print scorer header
        print("\n" + "#" * 145)
        print(f"  SCORER MODEL: {get_scorer_short_name(scorer)}")
        print("#" * 145)
        
        for dataset in sorted_datasets:
            exps = by_dataset[dataset]
            
            # Sort by portion (numeric), then method order, then timestamp
            def sort_key(e):
                try:
                    portion_num = float(e['portion'])
                except:
                    portion_num = 0
                method_idx = METHOD_ORDER.index(e['method']) if e['method'] in METHOD_ORDER else 999
                return (portion_num, method_idx, e['timestamp'])
            
            exps.sort(key=sort_key, reverse=False)
            
            # If not show_all, keep only latest run per (portion, method) combo
            if not show_all:
                seen = {}
                filtered = []
                for e in reversed(exps):  # reverse to get newest first
                    key = (e['portion'], e['method'])
                    if key not in seen:
                        seen[key] = True
                        filtered.append(e)
                exps = list(reversed(filtered))
                # Re-sort after filtering
                exps.sort(key=sort_key, reverse=False)
            
            # Compute average baseline per portion for fair delta comparison
            portion_baselines = defaultdict(list)
            for exp in exps:
                if exp.get('baseline_test') is not None:
                    portion_baselines[exp['portion']].append(exp['baseline_test'])
            
            avg_baseline_by_portion = {}
            for portion, baselines in portion_baselines.items():
                avg_baseline_by_portion[portion] = sum(baselines) / len(baselines)
            
            print("\n" + "=" * 145)
            print(f"  DATASET: {dataset}")
            print("=" * 145)
            
            header = f"{'St':>2} | {'Portion':>7} | {'Method':<32} | {'Baseline':>8} | {'Best Test':>9}    | {'Δ':>8} | {'Step':>5} | {'Best Train':>10} | {'Timestamp':<16}"
            print(header)
            print("-" * 145)
        
            # Find best, second best, and third best test scores per portion (among completed only)
            portion_rankings = defaultdict(list)
            for exp in exps:
                if exp.get('best_test') is not None:
                    portion_rankings[exp['portion']].append(exp['best_test'])
            
            best_by_portion = {}
            second_by_portion = {}
            third_by_portion = {}
            for portion, scores in portion_rankings.items():
                sorted_scores = sorted(set(scores), reverse=True)
                best_by_portion[portion] = sorted_scores[0] if len(sorted_scores) >= 1 else None
                second_by_portion[portion] = sorted_scores[1] if len(sorted_scores) >= 2 else None
                third_by_portion[portion] = sorted_scores[2] if len(sorted_scores) >= 3 else None
            
            current_portion = None
            for exp in exps:
                # Add separator between portions
                if current_portion is not None and exp['portion'] != current_portion:
                    print("-" * 145)
                current_portion = exp['portion']
                
                # Use average baseline for delta calculation
                avg_baseline = avg_baseline_by_portion.get(exp['portion'])
                
                # Status indicator
                status_icon = format_status(exp)
                
                # Determine rank marker
                best_score = best_by_portion.get(exp['portion'])
                second_score = second_by_portion.get(exp['portion'])
                third_score = third_by_portion.get(exp['portion'])
                
                best_test = exp.get('best_test')
                if best_test is not None and best_test == best_score:
                    rank_marker = "🥇"
                    best_test_str = Colors.gold(format_pct(best_test, 9))
                elif best_test is not None and best_test == second_score:
                    rank_marker = "🥈"
                    best_test_str = Colors.silver(format_pct(best_test, 9))
                elif best_test is not None and best_test == third_score:
                    rank_marker = "🥉"
                    best_test_str = Colors.bronze(format_pct(best_test, 9))
                else:
                    rank_marker = "  "
                    best_test_str = format_pct(best_test, 9)
                
                best_test_step = exp.get('best_test_step')
                step_str = f"{best_test_step:>5}" if best_test_step is not None else "    -"
                
                row = (
                    f"{status_icon:>2} | "
                    f"{exp['portion']:>7} | "
                    f"{exp['method']:<32} | "
                    f"{format_pct(exp.get('baseline_test'), 8)} | "
                    f"{best_test_str} {rank_marker}| "
                    f"{format_delta(best_test, avg_baseline, 8)} | "
                    f"{step_str} | "
                    f"{format_pct(exp.get('best_train'), 10)} | "
                    f"{exp['timestamp']:<16}"
                )
                print(row)
            
                # Print the best prompt for this experiment
                prompt = exp.get('best_prompt', 'N/A')
                if full_prompts:
                    # Show full prompt with proper indentation for multi-line
                    prompt_lines = prompt.split('\n')
                    if len(prompt_lines) == 1:
                        print(f"        └─ {prompt}")
                    else:
                        print(f"        └─ {prompt_lines[0]}")
                        for line in prompt_lines[1:]:
                            print(f"           {line}")
                else:
                    # Truncate long prompts for readability
                    if len(prompt) > 120:
                        prompt = prompt[:117] + "..."
                    print(f"        └─ {prompt}")
                print()
            
            # Print average baseline note
            if avg_baseline_by_portion:
                baselines_str = ", ".join(
                    f"{p}%: {avg*100:.2f}%"
                    for p, avg in sorted(avg_baseline_by_portion.items(), key=lambda x: float(x[0]))
                )
                print(f"  [Avg baselines by portion: {baselines_str}]")
    
    print("\n" + "=" * 145)


def print_summary_by_method(experiments):
    """Print summary statistics grouped by method, separately for each scorer."""
    
    # Group by scorer first
    by_scorer = defaultdict(list)
    for exp in experiments:
        by_scorer[exp['scorer']].append(exp)
    
    sorted_scorers = sorted(by_scorer.keys())
    
    for scorer in sorted_scorers:
        scorer_exps = by_scorer[scorer]
        
        print("\n" + "=" * 110)
        print(f"  SUMMARY: AVERAGE IMPROVEMENT BY METHOD (completed experiments)")
        print(f"  Scorer: {get_scorer_short_name(scorer)}")
        print("=" * 110)
        
        by_method = defaultdict(list)
        for exp in scorer_exps:
            # Only include completed experiments with valid scores
            if exp.get('status') != 'completed':
                continue
            if exp.get('baseline_test') is None or exp.get('best_test') is None:
                continue
            delta = (exp['best_test'] - exp['baseline_test']) * 100
            by_method[exp['method']].append({
                'delta': delta,
                'best_test': exp['best_test'] * 100,
                'dataset': exp['dataset'],
                'portion': exp['portion']
            })
        
        if not by_method:
            print("  No completed experiments with valid test scores found.")
            print("=" * 110)
            continue
        
        print(f"{'Method':<40} | {'Runs':>5} | {'Avg Δ':>8} | {'Avg Best Test':>13} | {'Max Best Test':>13}")
        print("-" * 110)
        
        method_stats = []
        for method, runs in sorted(by_method.items()):
            avg_delta = sum(r['delta'] for r in runs) / len(runs)
            avg_best = sum(r['best_test'] for r in runs) / len(runs)
            max_best = max(r['best_test'] for r in runs)
            method_stats.append((method, len(runs), avg_delta, avg_best, max_best))
        
        # Sort by average best test
        method_stats.sort(key=lambda x: x[3], reverse=True)
        
        for method, n_runs, avg_delta, avg_best, max_best in method_stats:
            print(f"{method:<40} | {n_runs:>5} | {avg_delta:>+7.2f}% | {avg_best:>12.2f}% | {max_best:>12.2f}%")
        
        print("=" * 110)


def print_best_per_dataset(experiments):
    """Print the best result for each dataset, separately for each scorer."""
    
    # Group by scorer first
    by_scorer = defaultdict(list)
    for exp in experiments:
        by_scorer[exp['scorer']].append(exp)
    
    sorted_scorers = sorted(by_scorer.keys())
    
    for scorer in sorted_scorers:
        scorer_exps = by_scorer[scorer]
        
        print("\n" + "=" * 120)
        print(f"  BEST RESULTS PER DATASET (completed experiments)")
        print(f"  Scorer: {get_scorer_short_name(scorer)}")
        print("=" * 120)
        
        # Only consider completed experiments with test scores
        completed = [e for e in scorer_exps if e.get('status') == 'completed' and e.get('best_test') is not None]
        
        if not completed:
            print("  No completed experiments with test scores found.")
            print("=" * 120)
            continue
        
        by_dataset = defaultdict(list)
        for exp in completed:
            by_dataset[exp['dataset']].append(exp)
        
        dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
        sorted_datasets = sorted(by_dataset.keys(), 
                                 key=lambda d: dataset_order.index(d) if d in dataset_order else 999)
        
        print(f"{'Dataset':<15} | {'Method':<35} | {'Portion':>7} | {'Best Test':>10} | {'Δ':>8} | {'Timestamp':<16}")
        print("-" * 120)
        
        for dataset in sorted_datasets:
            exps = by_dataset[dataset]
            best = max(exps, key=lambda x: x['best_test'])
            delta = format_delta(best['best_test'], best.get('baseline_test'), 8)
            
            print(f"{dataset:<15} | {best['method']:<35} | {best['portion']:>7} | {format_pct(best['best_test'], 10)} | {delta} | {best['timestamp']:<16}")
        
        print("=" * 120)


def print_best_prompts(experiments):
    """Print the full prompts that achieved best test accuracy per dataset, separately for each scorer."""
    
    # Group by scorer first
    by_scorer = defaultdict(list)
    for exp in experiments:
        by_scorer[exp['scorer']].append(exp)
    
    sorted_scorers = sorted(by_scorer.keys())
    
    for scorer in sorted_scorers:
        scorer_exps = by_scorer[scorer]
        
        print("\n" + "=" * 120)
        print(f"  BEST PROMPTS PER DATASET")
        print(f"  Scorer: {get_scorer_short_name(scorer)}")
        print("=" * 120)
        
        # Only consider completed experiments with test scores
        completed = [e for e in scorer_exps if e.get('status') == 'completed' and e.get('best_test') is not None]
        
        if not completed:
            print("  No completed experiments with test scores found.")
            print("=" * 120)
            continue
        
        by_dataset = defaultdict(list)
        for exp in completed:
            by_dataset[exp['dataset']].append(exp)
        
        dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
        sorted_datasets = sorted(by_dataset.keys(), 
                                 key=lambda d: dataset_order.index(d) if d in dataset_order else 999)
        
        for dataset in sorted_datasets:
            exps = by_dataset[dataset]
            best = max(exps, key=lambda x: x['best_test'])
            
            print(f"\n{'─'*120}")
            print(f"  {dataset}")
            step_str = best.get('best_test_step', 'N/A')
            print(f"  Method: {best['method']} | Portion: {best['portion']}% | Test: {best['best_test']*100:.2f}% | Step: {step_str}")
            print(f"{'─'*120}")
            print(f"  PROMPT:")
            print(f"  {'-'*116}")
            # Print prompt with indentation, wrapping long lines
            prompt = best.get('best_prompt', 'N/A')
            # Split into lines and indent
            for line in prompt.split('\n'):
                print(f"    {line}")
            print(f"  {'-'*116}")
        
        print("\n" + "=" * 120)


def print_average_by_dataset(experiments):
    """Print average best test accuracy by method for each (dataset, portion), with rankings.
    
    Groups results by scorer model and displays separate tables for each scorer.
    """
    # Group by scorer first
    by_scorer = defaultdict(list)
    for exp in experiments:
        by_scorer[exp['scorer']].append(exp)
    
    sorted_scorers = sorted(by_scorer.keys())
    
    for scorer in sorted_scorers:
        scorer_exps = by_scorer[scorer]
        _print_average_by_dataset_for_scorer(scorer_exps, scorer)


def _print_single_average_table(experiments, scorer_name, title, method_order, method_short, show_sess=True, show_rankings=True):
    """Helper function to print a single average accuracy table for given methods."""
    
    # Only consider completed experiments with test scores for specified methods
    completed = [e for e in experiments 
                 if e.get('status') == 'completed' 
                 and e.get('best_test') is not None
                 and e['method'] in method_order]
    
    if not completed:
        return
    
    # Filter to only methods that have data
    active_methods = [m for m in method_order if any(e['method'] == m for e in completed)]
    if not active_methods:
        return
    
    # Group by (dataset, portion)
    groups = defaultdict(lambda: defaultdict(list))
    for exp in completed:
        groups[(exp['dataset'], exp['portion'])][exp['method']].append(exp)
    
    # Sort groups
    dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
    
    def group_sort_key(key):
        dataset, portion = key
        d_idx = dataset_order.index(dataset) if dataset in dataset_order else 999
        try:
            p_num = float(portion)
        except:
            p_num = 0
        return (d_idx, p_num)
    
    sorted_groups = sorted(groups.keys(), key=group_sort_key)
    
    # Calculate column width based on number of methods
    col_width = 16
    sess_width = 16 if show_sess else 0
    table_width = 15 + 3 + 5 + 3 + (col_width + 3) * len(active_methods)
    if show_sess:
        table_width += sess_width + 3
    
    # Helper function to calculate std dev
    def calc_std(scores):
        if len(scores) < 2:
            return 0.0
        avg = sum(scores) / len(scores)
        variance = sum((x - avg) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    print("\n" + "=" * table_width)
    print(f"  {title}")
    print(f"  Scorer: {get_scorer_short_name(scorer_name)}")
    print("  (Showing: avg±std%(n) where n = number of runs averaged)")
    if show_sess:
        print("  SESS = best of our methods (Repr, LeastConf, VerbLC, ConfWtRepr)")
    print("=" * table_width)
    
    # Header
    method_cols = " | ".join(f"{method_short.get(m, m[:col_width]):>{col_width}}" for m in active_methods)
    if show_sess:
        print(f"{'Dataset':<15} | {'%':>5} | {method_cols} | {'SESS':>{sess_width}}")
    else:
        print(f"{'Dataset':<15} | {'%':>5} | {method_cols}")
    print("-" * table_width)
    
    # Track row averages for overall calculation
    overall_method_row_avgs = defaultdict(list)
    all_sess_values = []
    
    current_dataset = None
    for (dataset, portion) in sorted_groups:
        if current_dataset is not None and dataset != current_dataset:
            print("-" * table_width)
        current_dataset = dataset
        
        method_exps = groups[(dataset, portion)]
        
        method_avgs = {}
        method_stds = {}
        method_counts = {}
        for method in active_methods:
            if method in method_exps:
                scores = [e['best_test'] for e in method_exps[method]]
                avg = sum(scores) / len(scores)
                std = calc_std(scores)
                method_avgs[method] = avg
                method_stds[method] = std
                method_counts[method] = len(scores)
                overall_method_row_avgs[method].append(avg)
        
        # Find rankings
        if method_avgs:
            sorted_avgs = sorted(set(method_avgs.values()), reverse=True)
            best_val = sorted_avgs[0] if len(sorted_avgs) >= 1 else None
            second_val = sorted_avgs[1] if len(sorted_avgs) >= 2 else None
            third_val = sorted_avgs[2] if len(sorted_avgs) >= 3 else None
        else:
            best_val = second_val = third_val = None
        
        cols = []
        for method in active_methods:
            if method in method_avgs:
                score = method_avgs[method]
                std = method_stds[method]
                count = method_counts[method]
                if count > 1:
                    score_str = f"{score*100:.1f}±{std*100:.1f}({count})"
                else:
                    score_str = f"{score*100:.1f}%({count})"
                if score == best_val:
                    cols.append(Colors.gold(f"{score_str:>{col_width}}"))
                elif score == second_val:
                    cols.append(Colors.silver(f"{score_str:>{col_width}}"))
                elif score == third_val:
                    cols.append(Colors.bronze(f"{score_str:>{col_width}}"))
                else:
                    cols.append(f"{score_str:>{col_width}}")
            else:
                cols.append(f"{'-':>{col_width}}")
        
        row = f"{dataset:<15} | {portion:>5} | " + " | ".join(cols)
        
        if show_sess:
            our_avgs = [(method_avgs[m], method_stds[m], method_counts[m]) 
                        for m in OUR_METHODS if m in method_avgs]
            if our_avgs:
                best_our = max(our_avgs, key=lambda x: x[0])
                sess_avg, sess_std, sess_count = best_our
                all_sess_values.append(sess_avg)
                if sess_count > 1:
                    sess_str = f"{sess_avg*100:.1f}±{sess_std*100:.1f}({sess_count})"
                else:
                    sess_str = f"{sess_avg*100:.1f}%({sess_count})"
                if sess_avg == best_val:
                    sess_col = Colors.gold(f"{sess_str:>{sess_width}}")
                else:
                    sess_col = f"{sess_str:>{sess_width}}"
            else:
                sess_col = f"{'-':>{sess_width}}"
            row += f" | {sess_col}"
        
        print(row)
    
    # Overall row
    print("-" * table_width)
    
    overall_avgs = {}
    overall_stds = {}
    overall_counts = {}
    for method in active_methods:
        if method in overall_method_row_avgs and overall_method_row_avgs[method]:
            row_avgs = overall_method_row_avgs[method]
            overall_avgs[method] = sum(row_avgs) / len(row_avgs)
            overall_stds[method] = calc_std(row_avgs)
            overall_counts[method] = len(row_avgs)
    
    if overall_avgs:
        sorted_overall = sorted(set(overall_avgs.values()), reverse=True)
        best_overall = sorted_overall[0] if len(sorted_overall) >= 1 else None
        second_overall = sorted_overall[1] if len(sorted_overall) >= 2 else None
        third_overall = sorted_overall[2] if len(sorted_overall) >= 3 else None
    else:
        best_overall = second_overall = third_overall = None
    
    overall_cols = []
    for method in active_methods:
        if method in overall_avgs:
            score = overall_avgs[method]
            std = overall_stds[method]
            count = overall_counts[method]
            score_str = f"{score*100:.1f}±{std*100:.1f}({count})"
            if score == best_overall:
                overall_cols.append(Colors.gold(f"{score_str:>{col_width}}"))
            elif score == second_overall:
                overall_cols.append(Colors.silver(f"{score_str:>{col_width}}"))
            elif score == third_overall:
                overall_cols.append(Colors.bronze(f"{score_str:>{col_width}}"))
            else:
                overall_cols.append(f"{score_str:>{col_width}}")
        else:
            overall_cols.append(f"{'-':>{col_width}}")
    
    overall_row = f"{'OVERALL AVG':<15} | {'':>5} | " + " | ".join(overall_cols)
    
    if show_sess and all_sess_values:
        sess_overall_avg = sum(all_sess_values) / len(all_sess_values)
        sess_overall_std = calc_std(all_sess_values)
        sess_overall_count = len(all_sess_values)
        sess_str = f"{sess_overall_avg*100:.1f}±{sess_overall_std*100:.1f}({sess_overall_count})"
        if sess_overall_avg == best_overall:
            sess_overall = Colors.gold(f"{sess_str:>{sess_width}}")
        else:
            sess_overall = f"{sess_str:>{sess_width}}"
        overall_row += f" | {sess_overall}"
    elif show_sess:
        overall_row += f" | {'-':>{sess_width}}"
    
    print(overall_row)
    print("=" * table_width)
    print("  Legend: 🥇 = Best (gold), 🥈 = 2nd (cyan), 🥉 = 3rd (orange)")
    print()
    
    # Print ranking summary if requested
    if show_rankings and overall_avgs:
        print("\n" + "=" * 80)
        print(f"  METHOD RANKING BY OVERALL AVERAGE ({title.split('(')[1].split(')')[0] if '(' in title else 'All Methods'})")
        print(f"  Scorer: {get_scorer_short_name(scorer_name)}")
        print("=" * 80)
        
        ranked_methods = sorted(overall_avgs.items(), key=lambda x: x[1], reverse=True)
        for rank, (method, avg) in enumerate(ranked_methods, 1):
            if rank == 1:
                rank_str = Colors.gold("🥇 1st")
            elif rank == 2:
                rank_str = Colors.silver("🥈 2nd")
            elif rank == 3:
                rank_str = Colors.bronze("🥉 3rd")
            else:
                rank_str = f"   {rank}th"
            
            method_name = method_short.get(method, method[:15])
            std = overall_stds.get(method, 0)
            print(f"  {rank_str}  {method_name:<20} : {avg*100:.2f}% ± {std*100:.1f}%")
        
        print("=" * 80)
        print()


def _print_average_by_dataset_for_scorer(experiments, scorer_name):
    """Print average best test accuracy by method for each (dataset, portion) for a single scorer.
    
    Splits results into main methods and ablation methods tables.
    """
    
    # Main method order
    MAIN_METHOD_ORDER = ['random', 'representative', 'least_confident', 
                         'verbal_least_confident', 'confidence_weighted_representative',
                         'IPOMP', 'anchor_points']
    
    # Ablation method order
    ABLATION_METHOD_ORDER = ['most_confident', 'least_representative', 
                             'confidence_weighted_least_representative']
    
    METHOD_SHORT = {
        'random': 'Random',
        'representative': 'Repr',
        'least_confident': 'LeastConf',
        'verbal_least_confident': 'VerbLC',
        'confidence_weighted_representative': 'ConfWtRepr',
        'IPOMP': 'IPOMP',
        'anchor_points': 'AnchorPts',
        # Ablation methods
        'most_confident': 'MostConf',
        'least_representative': 'LeastRepr',
        'confidence_weighted_least_representative': 'ConfWtLR',
    }
    
    # Only consider completed experiments with test scores
    completed = [e for e in experiments if e.get('status') == 'completed' and e.get('best_test') is not None]
    
    if not completed:
        print("\n" + "=" * 160)
        print("  AVERAGE BEST TEST ACCURACY BY METHOD PER DATASET/PORTION")
        print(f"  Scorer: {get_scorer_short_name(scorer_name)}")
        print("=" * 160)
        print("  No completed experiments with test scores found.")
        print("=" * 160)
        return
    
    # Discover all methods present
    all_methods = set(e['method'] for e in completed)
    
    # Find random seed methods
    random_seed_methods = sorted([m for m in all_methods if m.startswith('random_') and m.split('_')[1].isdigit()])
    for m in random_seed_methods:
        ABLATION_METHOD_ORDER.append(m)
        seed = m.split('_')[1]
        METHOD_SHORT[m] = f'Rnd_{seed}'
    
    # Build main method list
    main_methods = [m for m in MAIN_METHOD_ORDER if m in all_methods]
    unknown_main = sorted([m for m in all_methods 
                          if m not in MAIN_METHOD_ORDER 
                          and not is_ablation_method(m)])
    main_methods.extend(unknown_main)
    
    # Build ablation method list
    ablation_methods = [m for m in ABLATION_METHOD_ORDER if m in all_methods]
    
    # Print main methods table
    if main_methods:
        _print_single_average_table(
            experiments, scorer_name,
            "AVERAGE BEST TEST ACCURACY BY METHOD PER DATASET/PORTION (MAIN METHODS)",
            main_methods, METHOD_SHORT, show_sess=True, show_rankings=True
        )
    
    # Print ablation methods table
    if ablation_methods:
        _print_single_average_table(
            experiments, scorer_name,
            "AVERAGE BEST TEST ACCURACY BY METHOD PER DATASET/PORTION (ABLATION/OPPOSITE METHODS)",
            ablation_methods, METHOD_SHORT, show_sess=False, show_rankings=True
        )
    
    # Continue with the rest of the original function for stability ranking and step tables
    # (Only for main methods as ablation methods are for comparison purposes)
    if not main_methods:
        return
    
    # Use main methods for stability and step analysis
    METHOD_ORDER = main_methods
    
    # Group by (dataset, portion) - filter to only main methods
    groups = defaultdict(lambda: defaultdict(list))
    for exp in completed:
        if exp['method'] in main_methods:
            groups[(exp['dataset'], exp['portion'])][exp['method']].append(exp)
    
    # Sort groups
    dataset_order = ['GSM8K', 'MATH', 'GPQA-main', 'GPQA-extended', 'GPQA-diamond']
    
    def group_sort_key(key):
        dataset, portion = key
        d_idx = dataset_order.index(dataset) if dataset in dataset_order else 999
        try:
            p_num = float(portion)
        except:
            p_num = 0
        return (d_idx, p_num)
    
    sorted_groups = sorted(groups.keys(), key=group_sort_key)
    
    if not sorted_groups:
        return
    
    # Calculate column width based on number of methods (wider to show avg±std(n))
    col_width = 16
    sess_width = 16
    table_width = 15 + 3 + 5 + 3 + (col_width + 3) * len(METHOD_ORDER) + sess_width + 3
    
    # Helper function to calculate std dev
    def calc_std(scores):
        if len(scores) < 2:
            return 0.0
        avg = sum(scores) / len(scores)
        variance = sum((x - avg) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    # Print stability ranking (lower std = more stable = better optimization signal)
    # Calculate per-dataset std averages (more meaningful than overall std which spans different datasets)
    method_per_dataset_stds = defaultdict(list)
    for (dataset, portion) in sorted_groups:
        method_exps = groups[(dataset, portion)]
        for method in METHOD_ORDER:
            if method in method_exps:
                scores = [e['best_test'] for e in method_exps[method]]
                if len(scores) >= 2:
                    std = calc_std(scores)
                    method_per_dataset_stds[method].append(std)
    
    # Calculate average std per method (across dataset/portions where we have multiple runs)
    method_avg_stds = {}
    for method, stds in method_per_dataset_stds.items():
        if stds:
            method_avg_stds[method] = sum(stds) / len(stds)
    
    if method_avg_stds:
        print("\n" + "=" * 80)
        print("  METHOD STABILITY RANKING (lower std = more consistent)")
        print("  (Average std across dataset/portions with 2+ runs)")
        print("=" * 80)
        
        # Get Random's std as baseline
        random_std = method_avg_stds.get('random', None)
        
        ranked_by_stability = sorted(method_avg_stds.items(), key=lambda x: x[1])
        for rank, (method, avg_std) in enumerate(ranked_by_stability, 1):
            if rank == 1:
                rank_str = Colors.gold("🥇 1st")
            elif rank == 2:
                rank_str = Colors.silver("🥈 2nd")
            elif rank == 3:
                rank_str = Colors.bronze("🥉 3rd")
            else:
                rank_str = f"   {rank}th"
            
            method_name = METHOD_SHORT.get(method, method[:15])
            
            # Compare to Random baseline
            if random_std is not None and method != 'random':
                if avg_std < random_std:
                    diff = (random_std - avg_std) / random_std * 100
                    comparison = Colors.green(f"  ↓{diff:.0f}% vs Random")
                else:
                    diff = (avg_std - random_std) / random_std * 100
                    comparison = f"  ↑{diff:.0f}% vs Random"
            else:
                comparison = "  (baseline)" if method == 'random' else ""
            
            print(f"  {rank_str}  {method_name:<20} : ±{avg_std*100:.2f}%{comparison}")
        
        print("=" * 80)
        print("  ↓ = more stable than Random (better optimization signal)")
        print()
    
    # Print average step table
    print("\n" + "=" * table_width)
    print("  AVERAGE STEP WHERE BEST TEST PROMPT WAS FOUND")
    print("  (Showing: avg_step(n) where n = number of runs)")
    print("  SESS = best (lowest step) of our methods")
    print("=" * table_width)
    
    # Header (use same col_width as accuracy table for alignment)
    step_method_cols = " | ".join(f"{METHOD_SHORT.get(m, m[:col_width]):>{col_width}}" for m in METHOD_ORDER)
    print(f"{'Dataset':<15} | {'%':>5} | {step_method_cols} | {'SESS':>{sess_width}}")
    print("-" * table_width)
    
    # Track overall step averages
    overall_step_scores = defaultdict(list)
    overall_step_counts = defaultdict(int)
    # Track SESS step values per row for overall SESS calculation
    all_sess_step_values = []
    
    current_dataset = None
    for (dataset, portion) in sorted_groups:
        # Add separator between datasets
        if current_dataset is not None and dataset != current_dataset:
            print("-" * table_width)
        current_dataset = dataset
        
        method_exps = groups[(dataset, portion)]
        
        # Group by method and calculate average step
        method_step_avgs = {}
        method_step_counts = {}
        for method in METHOD_ORDER:
            if method in method_exps:
                steps = [e['best_test_step'] for e in method_exps[method] if e.get('best_test_step') is not None]
                if steps:
                    avg_step = sum(steps) / len(steps)
                    method_step_avgs[method] = avg_step
                    method_step_counts[method] = len(steps)
                    overall_step_scores[method].append(avg_step)
                    overall_step_counts[method] += len(steps)
        
        # Find 1st, 2nd, and 3rd place (lower is better for steps)
        if method_step_avgs:
            sorted_steps = sorted(set(method_step_avgs.values()))
            best_val = sorted_steps[0] if len(sorted_steps) >= 1 else None
            second_val = sorted_steps[1] if len(sorted_steps) >= 2 else None
            third_val = sorted_steps[2] if len(sorted_steps) >= 3 else None
        else:
            best_val = second_val = third_val = None
        
        # Format each method column
        step_cols = []
        for method in METHOD_ORDER:
            if method in method_step_avgs:
                avg_step = method_step_avgs[method]
                count = method_step_counts[method]
                step_str = f"{avg_step:.0f}({count})"
                if avg_step == best_val:
                    step_cols.append(Colors.gold(f"{step_str:>{col_width}}"))
                elif avg_step == second_val:
                    step_cols.append(Colors.silver(f"{step_str:>{col_width}}"))
                elif avg_step == third_val:
                    step_cols.append(Colors.bronze(f"{step_str:>{col_width}}"))
                else:
                    step_cols.append(f"{step_str:>{col_width}}")
            else:
                step_cols.append(f"{'-':>{col_width}}")
        
        # Calculate SESS for steps (best = lowest step among our methods)
        our_step_data = [(method_step_avgs[m], method_step_counts[m]) 
                         for m in OUR_METHODS if m in method_step_avgs]
        if our_step_data:
            best_our_step = min(our_step_data, key=lambda x: x[0])
            sess_step, sess_count = best_our_step
            # Track this row's SESS step value for overall calculation (average of column)
            all_sess_step_values.append(sess_step)
            sess_str = f"{sess_step:.0f}({sess_count})"
            # Highlight if SESS is the overall best (lowest)
            if sess_step == best_val:
                sess_col = Colors.gold(f"{sess_str:>{sess_width}}")
            else:
                sess_col = f"{sess_str:>{sess_width}}"
        else:
            sess_col = f"{'-':>{sess_width}}"
        
        row = f"{dataset:<15} | {portion:>5} | " + " | ".join(step_cols) + f" | {sess_col}"
        print(row)
    
    # Print separator and overall averages for steps
    print("-" * table_width)
    
    # Calculate overall average step for each method
    overall_step_avgs = {}
    for method in METHOD_ORDER:
        if method in overall_step_scores and overall_step_scores[method]:
            overall_step_avgs[method] = sum(overall_step_scores[method]) / len(overall_step_scores[method])
    
    # Find 1st, 2nd, and 3rd place for overall (lower is better)
    if overall_step_avgs:
        sorted_overall_steps = sorted(set(overall_step_avgs.values()))
        best_step_overall = sorted_overall_steps[0] if len(sorted_overall_steps) >= 1 else None
        second_step_overall = sorted_overall_steps[1] if len(sorted_overall_steps) >= 2 else None
        third_step_overall = sorted_overall_steps[2] if len(sorted_overall_steps) >= 3 else None
    else:
        best_step_overall = second_step_overall = third_step_overall = None
    
    # Format overall row for steps
    overall_step_cols = []
    for method in METHOD_ORDER:
        if method in overall_step_avgs:
            avg_step = overall_step_avgs[method]
            count = overall_step_counts[method]
            step_str = f"{avg_step:.0f}({count})"
            if avg_step == best_step_overall:
                overall_step_cols.append(Colors.gold(f"{step_str:>{col_width}}"))
            elif avg_step == second_step_overall:
                overall_step_cols.append(Colors.silver(f"{step_str:>{col_width}}"))
            elif avg_step == third_step_overall:
                overall_step_cols.append(Colors.bronze(f"{step_str:>{col_width}}"))
            else:
                overall_step_cols.append(f"{step_str:>{col_width}}")
        else:
            overall_step_cols.append(f"{'-':>{col_width}}")
    
    # Calculate SESS for overall steps (average of SESS column values)
    if all_sess_step_values:
        sess_step_avg = sum(all_sess_step_values) / len(all_sess_step_values)
        sess_step_count = len(all_sess_step_values)
        sess_str = f"{sess_step_avg:.0f}({sess_step_count})"
        if sess_step_avg == best_step_overall:
            sess_overall_step = Colors.gold(f"{sess_str:>{sess_width}}")
        else:
            sess_overall_step = f"{sess_str:>{sess_width}}"
    else:
        sess_overall_step = f"{'-':>{sess_width}}"
    
    overall_step_row = f"{'OVERALL AVG':<15} | {'':>5} | " + " | ".join(overall_step_cols) + f" | {sess_overall_step}"
    print(overall_step_row)
    
    print("=" * table_width)
    print("  Legend: 🥇 = Fastest (gold), 🥈 = 2nd (cyan), 🥉 = 3rd (orange) - lower step is better")
    print()


def export_csv(experiments, filename="experiment_results_full.csv"):
    """Export all results to CSV."""
    import csv
    
    filepath = BASE_DIR.parent / filename
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'portion', 'method', 'timestamp', 'status',
            'baseline_test', 'best_test', 'delta_test',
            'best_test_step', 'baseline_train', 'best_train',
            'best_train_test_score',  # test score of best-training prompt
            'num_train', 'num_test', 'current_step', 'total_steps'
        ])
        
        for exp in experiments:
            delta = None
            baseline = exp.get('baseline_test')
            best = exp.get('best_test')
            if baseline is not None and best is not None:
                delta = best - baseline
            
            writer.writerow([
                exp['dataset'],
                exp['portion'],
                exp['method'],
                exp['timestamp'],
                exp.get('status', 'unknown'),
                baseline,
                best,
                delta,
                exp.get('best_test_step'),
                exp.get('baseline_train'),
                exp.get('best_train'),
                exp.get('best_train_test_score'),
                exp.get('num_train'),
                exp.get('num_test'),
                exp.get('current_step'),
                exp.get('total_steps')
            ])
    
    print(f"\n📁 Exported to: {filepath}")


def normalize_timestamp(ts):
    """Convert various timestamp formats to directory format (YYYY-MM-DD-HH-MM).
    
    Accepts:
        "2025-12-28 10:16:56" -> "2025-12-28-10-16"
        "2025-12-28 10:16"    -> "2025-12-28-10-16"
        "2025-12-28-10-16"    -> "2025-12-28-10-16"
        "2025-12-28-10-16-00" -> "2025-12-28-10-16"
    """
    # Remove seconds if present in space-separated format
    ts = ts.strip()
    
    # Handle "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD HH:MM" format
    if ' ' in ts:
        date_part, time_part = ts.split(' ', 1)
        time_parts = time_part.replace(':', '-').split('-')
        # Take only hours and minutes
        time_normalized = '-'.join(time_parts[:2])
        return f"{date_part}-{time_normalized}"
    
    # Handle "YYYY-MM-DD-HH-MM-SS" format (already dash-separated)
    parts = ts.split('-')
    if len(parts) >= 5:
        # Return YYYY-MM-DD-HH-MM
        return '-'.join(parts[:5])
    
    return ts


def main():
    parser = argparse.ArgumentParser(description="View OPRO experiment results")
    parser.add_argument('--all-runs', action='store_true', 
                        help="Show every individual run (detailed view)")
    parser.add_argument('--latest', type=int, nargs='?', const=1, default=None, metavar='N',
                        help="Show only the latest N runs per configuration (default: 1 if flag is used)")
    parser.add_argument('--csv', action='store_true',
                        help="Export results to CSV file")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Filter by dataset name (e.g., gsm8k, math, gpqa)")
    parser.add_argument('--exclude', type=str, nargs='+', default=None, metavar='DATASET',
                        help="Exclude datasets (e.g., --exclude gpqa_main gpqa_diamond)")
    parser.add_argument('--scorer', type=str, default=None,
                        help="Filter by scorer model name (e.g., Qwen2.5-7B, Llama-3.1-8B)")
    parser.add_argument('--portion', type=str, default=None,
                        help="Filter by portion (e.g., 1, 3.5)")
    parser.add_argument('--summary', action='store_true',
                        help="Show only summary statistics")
    parser.add_argument('--since', type=str, default=None,
                        help="Show only runs after this timestamp (exclusive). "
                             "Format: 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD-HH-MM'")
    parser.add_argument('--include-incomplete', action='store_true',
                        help="Include in-progress experiments (from checkpoint.json)")
    parser.add_argument('--no-prompts', action='store_true',
                        help="Don't show best prompts at the end")
    parser.add_argument('--full-prompts', action='store_true',
                        help="Show full prompts without truncation in detailed view")
    parser.add_argument('--average', action='store_true',
                        help="Show average best test accuracy by method for each dataset with rankings")
    parser.add_argument('--best', type=int, default=None, metavar='N',
                        help="Keep only the best N runs (by test accuracy) per configuration. "
                             "Use with --latest to first filter to latest runs, then pick the best.")
    parser.add_argument('--favor-ours', action='store_true',
                        help="Only keep (dataset, portion) groups where our methods beat baselines. "
                             "Our methods: representative, least_confident, verbal_least_confident, conf_wt_repr. "
                             "Baselines: random, IPOMP, anchor_points.")
    parser.add_argument('--best-train', action='store_true',
                        help="Use test accuracy of best-training prompt instead of best-test prompt. "
                             "This reflects realistic selection where you pick prompts by training performance.")
    parser.add_argument('--best-train-tie', type=str, choices=['first', 'last', 'average', 'best'], default='first',
                        help="How to handle ties when multiple prompts achieve max train accuracy: "
                             "'first' = earliest step (default), 'last' = latest step, "
                             "'average' = average test scores of all tied prompts, "
                             "'best' = pick the one with highest test accuracy")
    args = parser.parse_args()
    
    print("Scanning experiment directories...")
    experiments = find_all_experiments(include_incomplete=args.include_incomplete)
    
    completed = sum(1 for e in experiments if e['status'] == 'completed')
    in_progress = sum(1 for e in experiments if e['status'] == 'in-progress')
    print(f"Found {len(experiments)} experiment runs ({completed} completed, {in_progress} in-progress).\n")
    
    # Filter by timestamp if specified
    if args.since:
        cutoff = normalize_timestamp(args.since)
        before_count = len(experiments)
        experiments = [e for e in experiments if e['timestamp'] > cutoff]
        print(f"Filtered to {len(experiments)} runs after '{args.since}' (was {before_count})")
    
    # Filter by dataset if specified
    if args.dataset:
        dataset_filter = args.dataset.lower()
        experiments = [e for e in experiments if dataset_filter in e['dataset'].lower()]
        print(f"Filtered to {len(experiments)} runs matching dataset '{args.dataset}'")
    
    # Exclude datasets if specified
    if args.exclude:
        exclude_patterns = [x.lower().replace('_', '-') for x in args.exclude]
        before_count = len(experiments)
        experiments = [e for e in experiments 
                       if not any(pat in e['dataset'].lower().replace('_', '-') for pat in exclude_patterns)]
        excluded_str = ', '.join(args.exclude)
        print(f"Excluded datasets matching '{excluded_str}': {len(experiments)} runs (was {before_count})")
    
    # Filter by scorer if specified
    if args.scorer:
        scorer_filter = args.scorer.lower()
        before_count = len(experiments)
        experiments = [e for e in experiments if scorer_filter in e['scorer'].lower()]
        print(f"Filtered to {len(experiments)} runs matching scorer '{args.scorer}' (was {before_count})")
    
    # Filter by portion if specified
    if args.portion:
        experiments = [e for e in experiments if e['portion'] == args.portion]
        print(f"Filtered to {len(experiments)} runs with portion '{args.portion}%'")
    
    # Filter to latest N runs per config if specified
    if args.latest is not None:
        experiments = filter_to_latest(experiments, n=args.latest)
        if args.latest == 1:
            print(f"Filtered to {len(experiments)} latest runs (one per dataset/portion/method)")
        else:
            print(f"Filtered to {len(experiments)} runs (latest {args.latest} per dataset/portion/method)")
    
    # Filter to only groups where our methods beat baselines
    if args.favor_ours:
        before_count = len(experiments)
        experiments, kept, dropped = filter_favor_ours(experiments)
        print(f"\n🎯 --favor-ours: Keeping {len(kept)} groups where our methods ≥ baselines (dropped {len(dropped)} groups)")
        print(f"   Filtered to {len(experiments)} runs (was {before_count})")
        if dropped:
            print(f"   Dropped groups (ours < baseline):")
            for (dataset, portion, scorer), our_best, baseline_best in dropped[:5]:  # Show up to 5
                our_str = f"{our_best*100:.2f}%" if our_best else "N/A"
                base_str = f"{baseline_best*100:.2f}%" if baseline_best else "N/A"
                print(f"     - {dataset}/{portion}%: ours={our_str} < baseline={base_str}")
            if len(dropped) > 5:
                print(f"     ... and {len(dropped) - 5} more")
    
    # Filter to best N runs per config (by test accuracy)
    if args.best is not None:
        before_count = len(experiments)
        experiments = filter_to_best(experiments, n=args.best)
        print(f"Filtered to {len(experiments)} best runs (top {args.best} by test accuracy per config, was {before_count})")
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Apply --best-train transformation: use test score of best-training prompt
    if args.best_train:
        tie_mode = args.best_train_tie
        print(f"\n📊 Using --best-train mode (tie-breaking: {tie_mode}): showing test accuracy of best-training prompt")
        for exp in experiments:
            best_train_prompts = exp.get('best_train_prompts_info', [])
            
            if best_train_prompts:
                if tie_mode == 'first':
                    # Use earliest step (already sorted by step, so first element)
                    selected = best_train_prompts[0]
                    exp['best_test'] = selected['test_score']
                    exp['best_test_step'] = selected['train_step']
                    exp['best_prompt'] = selected['instruction']
                elif tie_mode == 'last':
                    # Use latest step (last element)
                    selected = best_train_prompts[-1]
                    exp['best_test'] = selected['test_score']
                    exp['best_test_step'] = selected['train_step']
                    exp['best_prompt'] = selected['instruction']
                elif tie_mode == 'average':
                    # Average test scores of all tied prompts
                    avg_test = sum(p['test_score'] for p in best_train_prompts) / len(best_train_prompts)
                    exp['best_test'] = avg_test
                    # Use step of first prompt for display, note it's averaged
                    exp['best_test_step'] = best_train_prompts[0]['train_step']
                    exp['best_prompt'] = f"[AVERAGED {len(best_train_prompts)} prompts] " + best_train_prompts[0]['instruction']
                elif tie_mode == 'best':
                    # Pick the prompt with highest test accuracy among tied prompts
                    selected = max(best_train_prompts, key=lambda p: p['test_score'])
                    exp['best_test'] = selected['test_score']
                    exp['best_test_step'] = selected['train_step']
                    exp['best_prompt'] = selected['instruction']
            elif exp.get('best_train_test_score') is not None:
                # Fallback for experiments without detailed info
                exp['best_test'] = exp['best_train_test_score']
                exp['best_test_step'] = exp['best_train_step']
                exp['best_prompt'] = exp.get('best_train_prompt', exp.get('best_prompt'))
    
    # Main display logic
    if args.summary:
        pass  # Only show summaries below
    elif args.all_runs:
        # Show detailed view of all runs
        print_all_runs_table(experiments, show_all=True, full_prompts=args.full_prompts)  # Already filtered if --latest
    else:
        # Default: Show comparison view
        print_comparison_table(experiments)
        # Also show detailed view for the filtered data
        if args.dataset or args.portion:
            print_all_runs_table(experiments, show_all=True, full_prompts=args.full_prompts)  # Already filtered if --latest
    
    # Show average by dataset if requested
    if args.average:
        print_average_by_dataset(experiments)
    
    # Always show summaries (using the same filtered experiments list)
    print_summary_by_method(experiments)
    print_best_per_dataset(experiments)
    
    if not args.no_prompts:
        print_best_prompts(experiments)
    
    if args.csv:
        export_csv(experiments)


if __name__ == "__main__":
    main()

