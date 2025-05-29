#!/usr/bin/env python3
"""
Standalone evaluation script for model outputs.

Expected input format (pickle file with DataFrame):
- output_text: The model's response text
- ground_truth: The expected answer
- dataset: Dataset name (e.g., 'gpqa', 'mmlu_pro', 'math500', 'livecodebench', 'aime')
- question: The question text

Output adds two columns:
- extracted_answer: Parsed answer from model output
- prompt_accuracy: 100.0 if correct, 0.0 if incorrect
"""
import sys
import os
import argparse
import logging
import pickle
import re
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_multiple_choice(text: str, max_option: str = 'D') -> Optional[str]:
    """Parse multiple choice answer (A-D or A-J)"""
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    if text.startswith(("['", '["')) and text.endswith(("']", '"]')):
        text = text[2:-2].strip()
    text = text.replace(r'\n', '\n').replace(r'\'', "'")

    # Look for explicit answer format
    match = re.search(r"\b(?:ANSWER|FINAL\s*ANSWER)\b\s*[:=]?\s*(?:\(?\s*([A-" + max_option + r"])\s*\)?)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback to any letter
    match = re.search(r"\b([A-" + max_option + r"])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_boxed_math(text: str) -> Optional[str]:
    """Parse \\boxed{answer} format"""
    if not isinstance(text, str):
        return None
    
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    
    # Find matching brace
    depth, i = 0, idx + 7  # len(r"\boxed{")
    content_start = i
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[content_start:i].strip()
            depth -= 1
        i += 1
    return None


def parse_aime_answer(text: str) -> Optional[int]:
    """Parse AIME integer answer (0-999)"""
    if not isinstance(text, str):
        return None
    
    # Try \boxed{digits}
    match = re.search(r"\\boxed{\s*(\d+)\s*}", text)
    if match:
        try:
            val = int(match.group(1))
            return val if 0 <= val <= 999 else None
        except ValueError:
            pass
    
    # Fallback to Answer: format
    match = re.search(r"Answer:\s*(\d+)(?!\.)\b", text, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            return val if 0 <= val <= 999 else None
        except ValueError:
            pass
    return None


def parse_code(text: str) -> Optional[str]:
    """Parse code from ```python code block"""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def evaluate_multiple_choice(parsed: Optional[str], ground_truth: str, valid_options: str) -> bool:
    """Evaluate multiple choice answer"""
    return (parsed and ground_truth and
            isinstance(parsed, str) and isinstance(ground_truth, str) and
            parsed.upper() in valid_options and
            parsed.upper() == ground_truth.upper())


def evaluate_math500(parsed: Optional[str], ground_truth: str) -> bool:
    """Evaluate MATH-500 using PRM800K grader"""
    if not parsed or not ground_truth:
        return False
    
    try:
        # Import PRM800K grader
        prm800k_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules", "prm800", "prm800k"))
        logger.debug(f"Looking for PRM800K at: {prm800k_path}")
        logger.debug(f"Path exists: {os.path.exists(prm800k_path)}")
        
        if prm800k_path not in sys.path:
            sys.path.insert(0, prm800k_path)
            logger.debug(f"Added to sys.path: {prm800k_path}")
        
        from grading.grader import grade_answer
        return grade_answer(given_answer=str(parsed), ground_truth=str(ground_truth))
    except ImportError as e:
        logger.error(f"Failed to import PRM800K grader: {e}")
        logger.error(f"PRM800K path: {prm800k_path}")
        logger.error(f"Current sys.path entries containing 'prm': {[p for p in sys.path if 'prm' in p]}")
        return False
    except Exception as e:
        logger.error(f"Error in PRM800K evaluation: {e}")
        return False


def evaluate_aime(parsed: Optional[int], ground_truth: int) -> bool:
    """Evaluate AIME integer answer"""
    try:
        return int(parsed) == int(ground_truth)
    except (ValueError, TypeError):
        return False


def evaluate_livecodebench(code: Optional[str], question_id: str) -> bool:
    """Evaluate LiveCodeBench code generation"""
    if not isinstance(code, str) or not code or not isinstance(question_id, str):
        return False
    
    lcb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules", "LiveCodeBench"))
    if not os.path.isdir(lcb_dir):
        raise FileNotFoundError(f"LiveCodeBench submodule required at: {lcb_dir}")
    
    original_cwd = os.getcwd()
    temp_dir = None
    needs_path_removal = False
    
    try:
        os.chdir(lcb_dir)
        if lcb_dir not in sys.path:
            sys.path.insert(0, lcb_dir)
            needs_path_removal = True
        
        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.runner.scenario_router import build_prompt_benchmark, sort_and_extract_save_results, get_metrics
        
        temp_dir = f"/tmp/temp_lcb_eval_{question_id}_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
        
        mock_args = argparse.Namespace(
            scenario=Scenario.codegeneration, release_version="release_v1",
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=60, num_workers=1, num_process_evaluate=1,
            model_name="standalone_eval", output_dir=temp_dir,
            prompt_type="custom", continue_existing=False, evaluate=True
        )
        
        # Suppress progress bars and verbose output during evaluation
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            full_benchmark, _ = build_prompt_benchmark(mock_args)
            benchmark_map = {inst.question_id: inst for inst in full_benchmark}
            
            instance = benchmark_map.get(question_id)
            if not instance:
                return False
            
            save_results = [instance.insert_output([code], [code])]
            _, combined_results = sort_and_extract_save_results(mock_args.scenario, save_results)
            _, instance_results, _ = get_metrics(mock_args.scenario, mock_args, [instance], combined_results)
            graded = extract_instance_results(instance_results)
        
        return graded[0][0] if graded and graded[0] else False
    
    except ImportError as e:
        raise ImportError("LiveCodeBench lcb_runner modules required") from e
    except Exception:
        return False
    finally:
        os.chdir(original_cwd)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if needs_path_removal and lcb_dir in sys.path:
            sys.path.remove(lcb_dir)


# Dataset evaluation dispatch
DATASET_EVALUATORS = {
    'gpqa': {
        'parse': lambda text: parse_multiple_choice(text, 'D'),
        'evaluate': lambda parsed, gt: evaluate_multiple_choice(parsed, gt, 'ABCD')
    },
    'mmlu_pro': {
        'parse': lambda text: parse_multiple_choice(text, 'J'),
        'evaluate': lambda parsed, gt: evaluate_multiple_choice(parsed, gt, 'ABCDEFGHIJ')
    },
    'math500': {
        'parse': parse_boxed_math,
        'evaluate': evaluate_math500
    },
    'aime': {
        'parse': parse_aime_answer,
        'evaluate': evaluate_aime
    },
    'livecodebench': {
        'parse': parse_code,
        'evaluate': evaluate_livecodebench
    }
}


def get_evaluator(dataset_name: str) -> Optional[Dict]:
    """Get evaluator functions for dataset"""
    dataset_lower = dataset_name.lower()
    for key, evaluator in DATASET_EVALUATORS.items():
        if key in dataset_lower:
            return evaluator
    return None


def process_row(row: pd.Series) -> Dict[str, Any]:
    """Process a single row and return extracted answer and accuracy"""
    result = {'extracted_answer': None, 'prompt_accuracy': 0.0}
    
    try:
        dataset_name = str(row.get('dataset', ''))
        raw_output = str(row.get('output_text', '')) if not pd.isna(row.get('output_text', '')) else ''
        ground_truth = row.get('ground_truth')
        
        evaluator = get_evaluator(dataset_name)
        if not evaluator:
            logger.warning(f"No evaluator for dataset '{dataset_name}'")
            return result
        
        # Parse answer
        extracted = evaluator['parse'](raw_output)
        result['extracted_answer'] = extracted
        
        # Evaluate if we have both parsed answer and ground truth
        if extracted is not None and not pd.isna(ground_truth):
            is_correct = evaluator['evaluate'](extracted, ground_truth)
            result['prompt_accuracy'] = 100.0 if is_correct else 0.0
    
    except Exception as e:
        logger.error(f"Error processing row: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--input-file", help="Input pickle file")
    parser.add_argument("--output-file", help="Output pickle file (defaults to <input-file>_evaluated.pkl)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    output_file = args.output_file or f"{os.path.splitext(args.input_file)[0]}_evaluated.pkl"
    
    logger.info(f"Processing: {args.input_file}")
    logger.info(f"Output: {output_file}")
    
    try:
        # Load data
        with open(args.input_file, 'rb') as f:
            df = pickle.load(f)
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input file does not contain a DataFrame")
        
        # Check required columns
        required_cols = ['output_text', 'dataset', 'ground_truth']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {available_cols}")
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Process all rows
        df_output = df.copy()
        df_output['extracted_answer'] = None
        df_output['prompt_accuracy'] = 0.0
        
        for idx in tqdm(df_output.index, desc="Processing rows"):
            result = process_row(df_output.loc[idx])
            df_output.at[idx, 'extracted_answer'] = result['extracted_answer']
            df_output.at[idx, 'prompt_accuracy'] = result['prompt_accuracy']
        
        # Summary
        evaluated = df_output['extracted_answer'].notna().sum()
        correct = (df_output['prompt_accuracy'] > 0).sum()
        accuracy = (correct / evaluated * 100) if evaluated > 0 else 0
        
        logger.info(f"\nResults: {evaluated} evaluated, {correct} correct, {accuracy:.2f}% accuracy")
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump(df_output, f)
        
        logger.info("Done!")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()