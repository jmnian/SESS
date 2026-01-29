# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Final answer parser for reasoning tasks.

The common forms of outputs to be parsed are like:
- "the answer: XXX"
- "XXX is the answer"
- "XXX is the final/right/correct answer"
"""

import dataclasses
import math
import re
import string
from typing import Dict, List, Sequence

import immutabledict

all_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_letters_list = set([f'({l})' for l in all_letters])  # ['(a)', ...]

_WORD_TO_NUM = immutabledict.ImmutableOrderedDict({
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
})
SPECIAL_NUM_CHARS = frozenset({'.', '/', ','})
# The logic for identifying patterns for the answer behind:
# First check if the primary patterns are in the string, then if not, check the
# secondary ones.
FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY = [
    'answer is ', 'answer: ', 'answer is: ',
    'response: ', 'response is ', 'result: ', 'result is ',
    'final answer: ', 'final answer is ',
    'therefore, the answer is ', 'thus, the answer is ',
    'so the answer is ', 'hence the answer is ',
]
FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY = ['is: ', 'are: ']
FINAL_ANSWER_AHEAD_PATTERNS = [
    ' is the correct answer',
    ' is the right answer',
    ' is the final answer',
    ' is the answer',
]
GSM8K_ANSWER = '#### '
# the Boolean symbols appeared in BBH tasks
BOOLEAN_SYMBOLS = [['false', 'true'], ['no', 'yes'], ['invalid', 'valid']]

MULTILINGUAL_QUESTION_DELIMITER = {
    'bn': {
        'Q': '\u09aa\u09cd\u09b0\u09b6\u09cd\u09a8: ',
        'A': (
            '\u09a7\u09be\u09aa\u09c7 \u09a7\u09be\u09aa\u09c7 '
            '\u0989\u09a4\u09cd\u09a4\u09b0: '
        ),
        'Direct A': '\u0989\u09a4\u09cd\u09a4\u09b0: ',
    },
    'de': {
        'Q': 'Frage: ',
        'A': 'Schritt-f\u00fcr-Schritt-Antwort: ',
        'Direct A': 'Antwort: ',
    },
    'en': {
        'Q': 'Question: ',
        'A': 'Step-by-Step Answer: ',
        'Direct A': 'Answer: ',
    },
    'es': {
        'Q': 'Pregunta: ',
        'A': 'Respuesta paso a paso: ',
        'Direct A': 'Respuesta: ',
    },
    'fr': {
        'Q': 'Question : ',
        'A': 'R\u00e9ponse \u00e9tape par \u00e9tape : ',
        'Direct A': 'R\u00e9ponse : ',
    },
    'ja': {
        'Q': '\u554f\u984c\uff1a',
        'A': '\u30b9\u30c6\u30c3\u30d7\u3054\u3068\u306e\u7b54\u3048\uff1a',
        'Direct A': '\u7b54\u3048\uff1a',
    },
    'ru': {
        'Q': '\u0417\u0430\u0434\u0430\u0447\u0430: ',
        'A': '\u041f\u043e\u0448\u0430\u0433\u043e\u0432\u043e\u0435 '
             '\u0440\u0435\u0448\u0435\u043d\u0438\u0435: ',
        'Direct A': '\u0440\u0435\u0448\u0435\u043d\u0438\u0435: ',
    },
    'sw': {
        'Q': 'Swali: ',
        'A': 'Jibu la Hatua kwa Hatua: ',
        'Direct A': 'Jibu: ',
    },
    'te': {
        'Q': '\u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28: ',
        'A': '\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40\u0c17\u0c3e '
             '\u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02: ',
        'Direct A': '\u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02: ',
    },
    'th': {
        'Q':
            '\u0e42\u0e08\u0e17\u0e22\u0e4c: ',
        'A':
            '\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35\u0e25\u0e30\u0e02\u0e31\u0e49\u0e19\u0e15\u0e2d\u0e19: ',  # pylint: disable=g-line-too-long
        'Direct A':
            '\u0e04\u0e33\u0e15\u0e2d\u0e1a\u0e17\u0e35: ',
    },
    'zh': {
        'Q': '\u95ee\u9898\uff1a',
        'A': '\u9010\u6b65\u89e3\u7b54\uff1a',
        'Direct A': '\u89e3\u7b54\uff1a',
    },
}
initial_keys = list(MULTILINGUAL_QUESTION_DELIMITER.keys())
for language in initial_keys:
  if language == 'en':
    continue
  MULTILINGUAL_QUESTION_DELIMITER[f'{language}-en'] = (
      MULTILINGUAL_QUESTION_DELIMITER['en']
  )

LANGUAGES = list(MULTILINGUAL_QUESTION_DELIMITER.keys())
NEXT_QUESTION_DELIMITERS = [
    d['Q'] for d in MULTILINGUAL_QUESTION_DELIMITER.values()
] + ['Q:']


def _is_float(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def remove_punctuation_from_string(input_string):
  output_string = input_string.translate(
      str.maketrans('', '', string.punctuation)
  )
  return output_string


def _extract_bracketed_choice_from_string(prediction, treat_as_number=False):
  """Extract bracketed ABCD...XYZ choices if there's exactly one bracketed choice.

  This is designed for multiple choice questions where answers are formatted
  as (A), (B), (C), (D). It should NOT be used for numeric answers.

  Args:
    prediction (str): the unprocessed prediction.
    treat_as_number (bool): if True, skip extraction (numeric answers shouldn't
      have bracketed letter extraction which could pick up variables like (x), (t), etc.)

  Returns:
    prediction (str): the processed prediction.
  """
  # For numeric predictions, don't try to extract bracketed choices
  # This avoids extracting mathematical variables like (T), (x), (n) as answers
  if treat_as_number:
    return prediction
  
  prediction_lower = prediction.lower()
  
  # Only look for traditional multiple choice options (a), (b), (c), (d)
  # Not all 26 letters - this avoids extracting (t), (n), (x) as math variables
  mc_options = {'(a)', '(b)', '(c)', '(d)', '(e)'}
  
  # Find all EXACT matches of bracketed choice letters like (a), (b), etc.
  # Use word boundary matching to avoid matching in words
  exact_bracketed_pattern = r'\([a-e]\)(?=\s|$|[,.:;!?\'\"])'
  exact_matches = re.findall(exact_bracketed_pattern, prediction_lower)
  
  # Also check for standalone bracketed letters at the end of text
  end_pattern = r'\([a-e]\)$'
  end_match = re.search(end_pattern, prediction_lower.strip())
  if end_match and end_match.group(0) not in exact_matches:
    exact_matches.append(end_match.group(0))
  
  # Filter to only actual MC options (a-e)
  exact_matches = [m for m in exact_matches if m in mc_options]
  
  # Only extract if exactly one unique bracketed letter is found
  unique_matches = set(exact_matches)
  if len(unique_matches) == 1:
    return list(unique_matches)[0]
  
  return prediction


def _extract_multiple_choice_answer(prediction: str, return_diagnostic: bool = False) -> str | None | tuple:
  """Extract a multiple choice answer (A/B/C/D) from model output.
  
  This function is designed for multiple choice questions where the answer
  is a single letter A, B, C, or D. It looks for common answer patterns,
  prioritizing patterns that appear at the END of the response since that's
  where models typically put their final answer.
  
  Args:
    prediction: The model's output text.
    return_diagnostic: If True, returns (answer, diagnostic_info) tuple.
    
  Returns:
    A lowercase single letter (a/b/c/d) if a clear answer is found, else None.
    If return_diagnostic=True, returns (answer, {"method": str, "matched_text": str, "position": str})
  """
  prediction_lower = prediction.lower().strip()
  
  def _return_result(answer, method, matched_text, position):
    if return_diagnostic:
      return (answer, {"method": method, "matched_text": matched_text, "position": position})
    return answer
  
  # Valid answer choices
  valid_choices = {'a', 'b', 'c', 'd'}
  
  # FIRST: Remove text contamination (model generating new Q&A after answer)
  # Look for patterns like "\boxed{X}...Q:" and truncate at Q:
  # This handles cases where model gives answer then starts a new question
  contamination_patterns = [
      r'(\\boxed\{[a-d]\}[^Q]*?)Q:',  # Boxed answer followed by Q:
      r'(\\boxed\{[a-d]\}[^Q]*?)\nQ:',  # With newline
      r'(\([a-d]\)\s*\.?\s*)Q:',  # (A). Q: pattern
  ]
  for contam_pattern in contamination_patterns:
    contam_match = re.search(contam_pattern, prediction_lower, re.IGNORECASE | re.DOTALL)
    if contam_match:
      # Found contamination - truncate the prediction at the Q:
      truncate_pos = contam_match.end(1)  # Keep up to the answer, remove Q: and after
      prediction_lower = prediction_lower[:truncate_pos].strip()
      break
  
  # SECOND: Check for boxed answers FIRST (highest priority - very explicit)
  # Search the ENTIRE text for boxed answers, not just end section
  boxed_patterns = [
      (r'\\boxed\{([a-d])\}', "boxed"),
      (r'\\boxed\{\(?([a-d])\)?\}', "boxed_paren"),
      (r'\\boxed\{\\text\{\(?([a-d])\)?[^}]*\}\}', "boxed_text"),  # \boxed{\text{(A)...}}
  ]
  for pattern, method_name in boxed_patterns:
    matches = list(re.finditer(pattern, prediction_lower, re.IGNORECASE))
    if matches:
      # Take the LAST boxed answer (most likely to be the final answer)
      match = matches[-1]
      answer = match.group(1).lower()
      if answer in valid_choices:
        matched_text = match.group(0)
        return _return_result(answer, method_name, matched_text, "full_text_boxed")
  
  # Patterns that explicitly indicate a final answer, ordered by priority
  # We search from the END of the response for better accuracy
  final_answer_patterns = [
      # Explicit "the answer is X" patterns
      (r'(?:the\s+)?(?:correct\s+)?answer\s+is[:\s]*\(?([a-d])\)?', "answer_is"),
      (r'(?:the\s+)?(?:correct\s+)?answer[:\s]+\(?([a-d])\)?', "answer_colon"),
      (r'(?:the\s+)?(?:final\s+)?answer[:\s]+\(?([a-d])\)?', "final_answer"),
      # "Answer=X" or "Answer = X" format (explicit equals sign)
      (r'answer\s*=\s*\(?([a-d])\)?', "answer_equals"),
      # "Therefore X" / "Thus X" / "Hence X" patterns  
      (r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:correct\s+)?(?:answer\s+is\s+)?(?:option\s+)?\(?([a-d])\)?', "therefore_thus"),
      # "Option X is correct" patterns
      (r'option\s+\(?([a-d])\)?\s+is\s+(?:the\s+)?(?:correct|right)', "option_is_correct"),
      (r'\(?([a-d])\)?\s+is\s+(?:the\s+)?(?:correct|right)\s+(?:answer|option|choice)', "x_is_correct"),
      # "I choose X" / "I select X" patterns
      (r'(?:i\s+)?(?:choose|select|pick)\s+(?:option\s+)?\(?([a-d])\)?', "i_choose"),
      # Standalone answer at end: "Answer: A" or just "A" as last word
      (r'answer[:\s]*\(?([a-d])\)?\.?\s*$', "answer_at_end"),
      # Just a letter in parentheses at the very end
      (r'\(([a-d])\)\s*\.?\s*$', "paren_at_end"),
  ]
  
  # Search for patterns, prioritizing the END of the response
  # Split into last ~500 chars for final answer patterns
  end_section = prediction_lower[-500:] if len(prediction_lower) > 500 else prediction_lower
  
  for pattern, method_name in final_answer_patterns:
    # First try to find in the end section
    matches = list(re.finditer(pattern, end_section, re.IGNORECASE))
    if matches:
      # Take the LAST match (most likely to be the final answer)
      match = matches[-1]
      answer = match.group(1).lower()
      if answer in valid_choices:
        matched_text = match.group(0)
        return _return_result(answer, f"{method_name}_end", matched_text, "end_500")
  
  # If not found in end section, try the full text but only for explicit patterns
  for pattern, method_name in final_answer_patterns[:6]:  # Only most explicit patterns
    matches = list(re.finditer(pattern, prediction_lower, re.IGNORECASE))
    if matches:
      match = matches[-1]
      answer = match.group(1).lower()
      if answer in valid_choices:
        matched_text = match.group(0)
        return _return_result(answer, f"{method_name}_full", matched_text, "full_text")
  
  # Last resort: look for "The answer is (A)" style anywhere, taking the last one
  explicit_answer_match = list(re.finditer(
      r'(?:the\s+)?answer\s+is[:\s]*\(?([a-d])\)?', 
      prediction_lower, re.IGNORECASE
  ))
  if explicit_answer_match:
    match = explicit_answer_match[-1]
    answer = match.group(1).lower()
    if answer in valid_choices:
      return _return_result(answer, "explicit_answer_is", match.group(0), "full_text")
  
  # Additional fallback: Look for standalone (A), (B), (C), (D) in the last portion
  # This handles cases where model just outputs "(B)" or similar at the end
  last_portion = prediction_lower[-200:] if len(prediction_lower) > 200 else prediction_lower
  standalone_pattern = r'\(([a-d])\)'
  standalone_matches = list(re.finditer(standalone_pattern, last_portion))
  if standalone_matches:
    # Take the last standalone bracketed letter
    match = standalone_matches[-1]
    answer = match.group(1).lower()
    if answer in valid_choices:
      return _return_result(answer, "standalone_paren", match.group(0), "last_200")
  
  # Final fallback: If the last non-whitespace character is a letter A-D (possibly with period)
  # This handles cases like "...the answer is D." where D is at the very end
  last_char_match = re.search(r'([a-d])\.?\s*$', last_portion)
  if last_char_match:
    potential_answer = last_char_match.group(1).lower()
    # Only accept if there's some indication this is an answer (not just a word ending in a-d)
    # Check if preceded by common answer indicators
    before_match = last_portion[:last_char_match.start()]
    if re.search(r'(?:answer|option|choice|is|select|pick)\s*(?:is\s+)?\s*$', before_match, re.IGNORECASE):
      return _return_result(potential_answer, "trailing_letter", last_char_match.group(0), "last_200")
  
  # Look for answer option references like "a) content" or "(a) content"
  # This handles cases where model echoes an option like "a) polya tail"
  # which indicates it's referencing/selecting option A
  # We look for these patterns but exclude list patterns like "a), b), c)"
  option_ref_pattern = r'(?:^|\s|\n)\(?([a-d])\)[\s:]+\w'
  option_matches = list(re.finditer(option_ref_pattern, prediction_lower))
  if option_matches:
    # Check if this looks like a list (multiple a), b), c) in sequence)
    # by checking if there are multiple different letters in sequence
    letters_found = [m.group(1) for m in option_matches]
    matched_texts = [m.group(0) for m in option_matches]
    # If we have sequential letters like a, b, c, d - it's likely a list, not an answer
    # But if we have just one or repeated same letter, it's likely an answer reference
    unique_letters = set(letters_found)
    if len(unique_letters) == 1:
      # Only one letter mentioned (possibly multiple times) - likely the answer
      return _return_result(letters_found[-1], "option_ref_single", matched_texts[-1], "full_text")
    elif len(unique_letters) <= 2:
      # Could be comparing two options - take the last one mentioned
      return _return_result(letters_found[-1], "option_ref_compare", matched_texts[-1], "full_text")
    # If 3+ different letters, it's probably a list - don't extract
  
  # No answer found
  if return_diagnostic:
    return (None, {"method": "no_match", "matched_text": None, "position": None})
  return None


def get_normalized_prediction(prediction: str,
                              *,
                              treat_as_number: bool,
                              num_decimals: int = 0,
                              treat_as_bool: bool = False,
                              treat_as_multiple_choice: bool = False) -> str:
  """Returns a normalized prediction for use in `number_included_accuracy`.

  Args:
    prediction: The original model prediction.
    treat_as_number: Whether to treat the prediction as a number (and perform
      additional post-processing relevant to numbers, such as stripping of units
      or normalization of thousand separators, etc.).
    num_decimals: Number of decimal places to which to round the answer. Only
      applicable when treat_as_number==True.
    treat_as_bool: Whether to treat the prediction as a Boolean object. Only set
      it to True when the target is Boolean. The parser will then convert an 0/1
      answer to False/True.
    treat_as_multiple_choice: Whether to treat the prediction as a multiple
      choice answer (A/B/C/D). When True, uses specialized extraction logic
      to find the final answer choice.

  Returns:
    A normalized answer string that can be directly compared with the normalized
    golden answer in order to determine the `number_included_accuracy`.
  """
  
  # For multiple choice questions, use specialized extraction first
  if treat_as_multiple_choice:
    mc_answer = _extract_multiple_choice_answer(prediction)
    if mc_answer:
      return mc_answer
    # If no clear answer found, fall through to default parsing
    # but still try to extract a single letter at the end

  prediction_parsed = prediction.lower().strip()
  
  # EARLY EXIT: If the output starts with a number (possibly with sign/currency),
  # extract it directly. This handles cases where the model outputs the answer first,
  # then continues with garbage text like "You are an AI assistant..." or
  # instructions like "If the answer is not a whole number, round to..."
  # Without this check, patterns like "answer is not" could incorrectly extract "not".
  if treat_as_number:
    # Check if first line starts with a number (with optional sign/currency)
    first_line = prediction_parsed.split('\n')[0].strip()
    leading_num_match = re.match(
        r'^[-+]?\s*[$€£]?\s*(\d[\d,.\s]*\d|\d)',  # Match number at start
        first_line
    )
    if leading_num_match:
      # The answer is at the start - extract just this number
      # Check if there's garbage text after a newline that might confuse later parsing
      if '\n' in prediction_parsed:
        rest_of_text = prediction_parsed.split('\n', 1)[1].lower()
        # If rest contains confusing patterns, just use the first line number
        confusing_patterns = [
            'you are an ai', 'ai assistant', 'do not write',
            'answer is not', 'if the answer', 'round to',
        ]
        if any(p in rest_of_text for p in confusing_patterns):
          prediction_parsed = first_line
  
  # Strip markdown and LaTeX formatting that modern LLMs often use
  # Do this early before any other parsing
  
  # Handle LaTeX \boxed{answer} format (used by GPT models)
  # IMPORTANT: Find the LAST boxed match, not the first, as models often have
  # intermediate results boxed and the final answer is typically the last one
  # Use a pattern that handles nested braces like \boxed{\frac{3}{4}}
  boxed_pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
  boxed_matches = list(re.finditer(boxed_pattern, prediction_parsed))
  if boxed_matches:
    # Take the LAST boxed answer
    prediction_parsed = boxed_matches[-1].group(1)
  
  # Handle mixed numbers FIRST (before symbolic eval)
  # Mixed numbers like 3\frac{1}{2} should be 3 + 1/2 = 3.5, NOT 3 * 1/2 = 1.5
  # sympy's latex parser treats these as multiplication, so we handle them specially
  def _eval_mixed_frac_early(match):
    """Evaluate a mixed number (integer + fraction) to decimal."""
    whole = match.group(1)
    numer = match.group(2) 
    denom = match.group(3)
    try:
      return str(float(whole) + float(numer) / float(denom))
    except (ValueError, ZeroDivisionError):
      return match.group(0)
  
  prediction_parsed = re.sub(
      r'(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}', 
      _eval_mixed_frac_early, 
      prediction_parsed
  )
  
  # Try to evaluate symbolic LaTeX expressions with our own simple parser
  # This handles expressions like \sqrt{2}, 4\pi, \sqrt[3]{8}, 2^3, etc.
  # We use our own parser instead of sympy's parse_latex to avoid potential hangs
  def _try_symbolic_eval(latex_expr):
    r"""Try to evaluate a LaTeX expression symbolically and return float.
    
    Uses a simple custom parser instead of sympy to avoid potential hangs.
    Handles: \sqrt{n}, \sqrt[k]{n}, n\pi, a^b, and combinations.
    """
    if not latex_expr:
      return None
    
    # Guard: Length limit and basic sanity checks
    if len(latex_expr) > 50:
      return None
    
    # Skip if it contains text-like content
    if any(x in latex_expr.lower() for x in [
        'text', 'therefore', 'answer', 'the ', ' is ', '\\begin', '\\end'
    ]):
      return None
    
    # Check if expression contains symbolic elements that need evaluation
    if not re.search(r'\\sqrt|\\pi|\^', latex_expr):
      return None
    
    try:
      expr = latex_expr.strip()
      result = None
      
      # Pattern 1: Simple \sqrt{n} -> sqrt(n)
      match = re.fullmatch(r'\\sqrt\{(\d+(?:\.\d+)?)\}', expr)
      if match:
        return math.sqrt(float(match.group(1)))
      
      # Pattern 2: n\sqrt{m} -> n * sqrt(m)
      match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*\\sqrt\{(\d+(?:\.\d+)?)\}', expr)
      if match:
        return float(match.group(1)) * math.sqrt(float(match.group(2)))
      
      # Pattern 3: \sqrt[k]{n} -> n^(1/k)
      match = re.fullmatch(r'\\sqrt\[(\d+)\]\{(\d+(?:\.\d+)?)\}', expr)
      if match:
        k = float(match.group(1))
        n = float(match.group(2))
        return n ** (1.0 / k)
      
      # Pattern 4: \pi or n\pi
      match = re.fullmatch(r'(\d+(?:\.\d+)?)?\s*\\pi', expr)
      if match:
        coef = float(match.group(1)) if match.group(1) else 1.0
        return coef * math.pi
      
      # Pattern 5: \frac{\pi}{n} or \frac{m\pi}{n}
      match = re.fullmatch(r'\\frac\{(\d+)?\\pi\}\{(\d+(?:\.\d+)?)\}', expr)
      if match:
        numer_coef = float(match.group(1)) if match.group(1) else 1.0
        denom = float(match.group(2))
        return (numer_coef * math.pi) / denom
      
      # Pattern 6: a^b (simple power)
      match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*\^\s*\{?(\d+(?:\.\d+)?)\}?', expr)
      if match:
        base = float(match.group(1))
        exp = float(match.group(2))
        return base ** exp
      
      # Pattern 7: a + b\sqrt{c} or a - b\sqrt{c}
      match = re.fullmatch(
          r'(\d+(?:\.\d+)?)\s*([\+\-])\s*(\d+(?:\.\d+)?)\s*\\sqrt\{(\d+(?:\.\d+)?)\}',
          expr
      )
      if match:
        a = float(match.group(1))
        op = match.group(2)
        b = float(match.group(3))
        c = float(match.group(4))
        if op == '+':
          return a + b * math.sqrt(c)
        else:
          return a - b * math.sqrt(c)
      
      # Pattern 8: \sqrt{n}\pi or n\sqrt{m}\pi
      match = re.fullmatch(r'(\d+(?:\.\d+)?)?\s*\\sqrt\{(\d+(?:\.\d+)?)\}\s*\\pi', expr)
      if match:
        coef = float(match.group(1)) if match.group(1) else 1.0
        n = float(match.group(2))
        return coef * math.sqrt(n) * math.pi
      
      return None
      
    except (ValueError, ZeroDivisionError, OverflowError):
      return None
  
  # Try symbolic evaluation first for expressions with sqrt, pi, etc.
  symbolic_result = _try_symbolic_eval(prediction_parsed)
  if symbolic_result is not None:
    # Successfully evaluated symbolic expression
    if treat_as_number:
      prediction_parsed = str(symbolic_result)
    else:
      prediction_parsed = str(symbolic_result)
  
  # Handle LaTeX \frac{a}{b} - convert to decimal BEFORE stripping other LaTeX
  # Note: Mixed numbers like 3\frac{1}{2} are already handled above before symbolic eval
  def _eval_frac(match):
    """Evaluate a LaTeX fraction to decimal."""
    numer = match.group(1)
    denom = match.group(2)
    try:
      return str(float(numer) / float(denom))
    except (ValueError, ZeroDivisionError):
      return match.group(0)  # Return original if can't evaluate
  
  # Handle simple fractions: \frac{3}{4}
  prediction_parsed = re.sub(
      r'\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}', 
      _eval_frac, 
      prediction_parsed
  )
  
  # Handle \text{...} LaTeX command - extract content
  prediction_parsed = re.sub(r'\\text\{([^}]*)\}', r'\1', prediction_parsed)
  
  # Handle inline math $answer$ format
  prediction_parsed = re.sub(r'\$([^$]+)\$', r'\1', prediction_parsed)
  
  # Handle scientific notation with × symbol: 2.5 × 10^3
  def _eval_scientific(match):
    """Evaluate scientific notation to decimal."""
    base = match.group(1)
    exp = match.group(2)
    try:
      return str(float(base) * (10 ** float(exp)))
    except (ValueError, OverflowError):
      return match.group(0)
  
  prediction_parsed = re.sub(
      r'(\d+(?:\.\d+)?)\s*[×x]\s*10\s*\^\s*\{?(\d+)\}?',
      _eval_scientific,
      prediction_parsed,
      flags=re.IGNORECASE
  )
  
  # Remove LaTeX display math delimiters and other LaTeX artifacts
  latex_artifacts = [
      r'\\[', r'\\]',  # Display math delimiters
      r'\\(', r'\\)',  # Inline math delimiters
      r'\\{', r'\\}',  # Escaped braces
      r'\\,', r'\\;', r'\\!', r'\\:',  # LaTeX spacing
      r'\\quad', r'\\qquad',  # LaTeX spacing
      r'\\cdot', r'\\times',  # Math operators
      r'\\sqrt',  # Math functions (will leave remnants but okay)
  ]
  for artifact in latex_artifacts:
    prediction_parsed = prediction_parsed.replace(artifact, ' ')
  
  # Remove any remaining backslash followed by special chars
  prediction_parsed = re.sub(r'\\[^a-z0-9\s]', '', prediction_parsed)
  
  # Remove bold (**text** or __text__) and italic (*text* or _text_)
  prediction_parsed = re.sub(r'\*\*([^*]+)\*\*', r'\1', prediction_parsed)  # **bold**
  prediction_parsed = re.sub(r'__([^_]+)__', r'\1', prediction_parsed)      # __bold__
  prediction_parsed = re.sub(r'\*([^*]+)\*', r'\1', prediction_parsed)      # *italic*
  prediction_parsed = re.sub(r'_([^_]+)_', r'\1', prediction_parsed)        # _italic_
  
  # Also strip any remaining stray * or _ at word boundaries
  prediction_parsed = re.sub(r'\*+', '', prediction_parsed)
  prediction_parsed = prediction_parsed.strip()

  # Use lowercased prediction for case-insensitive pattern matching
  prediction_lower = prediction.lower()
  FINAL_ANSWER_BEHIND_PATTERNS = (  # pylint: disable=invalid-name
      FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY  # pylint: disable=g-long-ternary
      if any(
          [item in prediction_lower for item in FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY]
      )
      else FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY
  )
  DELIMITERS_FOR_ANSWER_BEHIND = (  # pylint: disable=invalid-name
      [d['A'] for d in MULTILINGUAL_QUESTION_DELIMITER.values()]
      + [GSM8K_ANSWER]
      + FINAL_ANSWER_BEHIND_PATTERNS
  )
  DELIMITERS_FOR_ANSWER_AHEAD = FINAL_ANSWER_AHEAD_PATTERNS   # pylint: disable=invalid-name

  # If the model tries to keep generating a new question, remove that additional
  # text.
  for next_question_delimiter in NEXT_QUESTION_DELIMITERS:
    prediction_parsed = prediction_parsed.split(
        next_question_delimiter.strip().lower()
    )[0]

  answer_indicated = False
  for answer_delimiter in DELIMITERS_FOR_ANSWER_BEHIND:
    if answer_delimiter.lower() in prediction_parsed:
      prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[-1]
      answer_indicated = True

  for answer_delimiter in DELIMITERS_FOR_ANSWER_AHEAD:
    if answer_delimiter.lower() in prediction_parsed:
      prediction_parsed = prediction_parsed.split(answer_delimiter.lower())[0]
      answer_indicated = True

  prediction_parsed = prediction_parsed.strip()

  # Specific handling for a case that appears in one of the chain-of-thought
  # ablation experiments, where the rationale comes after final answer.
  prediction_parsed = prediction_parsed.split('this is the solution:')[0]

  # Remove trailing punctuation and special characters
  # This handles: "hour.", "days'", "old)", "archie'", etc.
  trailing_chars_to_strip = '.\'\"`)!?;:,'
  while prediction_parsed and prediction_parsed[-1] in trailing_chars_to_strip:
    prediction_parsed = prediction_parsed[:-1]
  
  # Also remove leading special characters
  leading_chars_to_strip = '(\'\"[`'
  while prediction_parsed and prediction_parsed[0] in leading_chars_to_strip:
    prediction_parsed = prediction_parsed[1:]
  
  prediction_parsed = prediction_parsed.strip()

  # extract the bracketed choices: "(A) apple" -> "(a)"
  # Only for non-numeric predictions - avoids extracting math variables like (t), (x)
  prediction_parsed = _extract_bracketed_choice_from_string(
      prediction_parsed, treat_as_number=treat_as_number
  )

  def _parse_without_treating_as_number(prediction_parsed):
    prediction_parsed = prediction_parsed.split('.')[0]
    return prediction_parsed

  def _parse_with_treating_as_number(prediction_parsed):
    prediction_parsed = prediction_parsed.split('=')[-1]
    
    # Handle negative numbers with currency: -$50 -> -50
    # Capture the sign before stripping currency symbols
    is_negative = False
    neg_currency_match = re.match(r'\s*-\s*[$€£]', prediction_parsed)
    if neg_currency_match:
      is_negative = True
      prediction_parsed = prediction_parsed[neg_currency_match.end():]
    
    for c in ['$', ',', '%', '€', '£']:
      prediction_parsed = prediction_parsed.replace(c, '')
    
    # Handle colon split carefully - only for ratio patterns like "3:4"
    # NOT for label patterns like "Total: 1500" where the number is AFTER the colon
    colon_parts = prediction_parsed.split(':')
    if len(colon_parts) > 1:
      after_colon = ':'.join(colon_parts[1:]).strip()
      before_colon = colon_parts[0].strip()
      
      # Check if this looks like a ratio pattern: "...X:Y" where X and Y are numbers
      # e.g., "the ratio is 3:4" should give "3", "time 2:30" should give "2"
      before_ends_with_num = re.search(r'(\d+(?:\.\d+)?)\s*$', before_colon)
      after_starts_with_num = re.match(r'^(\d+(?:\.\d+)?)', after_colon)
      
      if before_ends_with_num and after_starts_with_num:
        # This is a ratio pattern like "3:4" - take the first number
        prediction_parsed = before_ends_with_num.group(1)
      elif any(c.isdigit() for c in after_colon):
        # Number is after colon - this is a label pattern like "Total: 1500"
        prediction_parsed = after_colon
      elif any(c.isdigit() for c in before_colon):
        # Number is before colon only - take it
        prediction_parsed = before_colon
      # If neither has digits, just keep the original
    
    prediction_parsed = prediction_parsed.strip()
    
    # Handle space-separated thousands: "1 234 567" -> "1234567"
    # Only merge if we see a pattern of 3-digit groups
    space_num_match = re.match(r'^(\d{1,3}(?:\s+\d{3})+)(?:\s|$)', prediction_parsed)
    if space_num_match:
      # Found space-separated number format
      num_str = space_num_match.group(1).replace(' ', '')
      rest = prediction_parsed[space_num_match.end():]
      prediction_parsed = num_str + rest

    # 'eight' -> '8'.
    for word, num in _WORD_TO_NUM.items():
      if word in prediction_parsed:
        prediction_parsed = prediction_parsed.replace(word, str(num))

    corrected_answer = False

    # Helper to check if a string contains any digit
    def _contains_digit(s):
      return any(c.isdigit() for c in s)
    
    if not corrected_answer:  # If no calculator errors were made.
      # '5600 pounds' -> '5600'; 'the 6th' -> '6'.
      if answer_indicated:
        # Take the first token that has numerical values.
        parts = prediction_parsed.split(' ')
      else:
        # Take the last token that has numerical values.
        parts = list(reversed(prediction_parsed.split(' ')))

      prediction_parsed = parts[0]  # Default
      for part in parts:
        # Look for tokens that contain digits, not just "not alphabetic"
        # This avoids selecting things like "archie'", "\\]", "ph.d", etc.
        if _contains_digit(part):
          prediction_parsed = part
          break

      # '156kgs' -> 156. '823-yard' -> 823.
      while prediction_parsed and prediction_parsed[-1].isalpha():
        prediction_parsed = prediction_parsed[:-1]
      if prediction_parsed and prediction_parsed[-1] == '-':
        prediction_parsed = prediction_parsed[:-1]

    # Apply negative sign if detected earlier
    if is_negative and prediction_parsed and not prediction_parsed.startswith('-'):
      prediction_parsed = '-' + prediction_parsed

    if _is_float(prediction_parsed):
      prediction_parsed_float = round(float(prediction_parsed), num_decimals)
      prediction_parsed = '{:.{num_decimals}f}'.format(
          prediction_parsed_float, num_decimals=num_decimals)
    else:
      if re.search(r'(-?\d+)(?!.*\d)', prediction_parsed):
        prediction_parsed = re.search(r'(-?\d+)(?!.*\d)', prediction_parsed)[0]
    return prediction_parsed

  # If not expecting a Boolean result
  if not treat_as_bool:
    # If not expecting a number, then return the extracted answer as-is.
    if not treat_as_number:
      # String predictions may try to continue the sentence.
      prediction_parsed = _parse_without_treating_as_number(prediction_parsed)

    else:  # If expecting a number, do post-processing.
      prediction_parsed = _parse_with_treating_as_number(prediction_parsed)
  else:
    prediction_parsed_as_not_number = _parse_without_treating_as_number(
        prediction_parsed
    )
    prediction_parsed_as_number = _parse_with_treating_as_number(
        prediction_parsed
    )
    if not any(
        [prediction_parsed_as_not_number in item for item in BOOLEAN_SYMBOLS]
    ):
      if prediction_parsed_as_number in {'0', '1'}:
        prediction_parsed = str(bool(int(prediction_parsed_as_number))).lower()
      if prediction_parsed_as_not_number in {'0', '1'}:
        prediction_parsed = str(
            bool(int(prediction_parsed_as_not_number))
        ).lower()
    else:
      prediction_parsed = prediction_parsed_as_not_number
    # remove punctuations like ":" and then strip
    prediction_parsed = remove_punctuation_from_string(
        prediction_parsed
    ).strip()

  return prediction_parsed


@dataclasses.dataclass
class NormalizationResult:
  """Bundle of return values of get_normalized_target_and_prediction.

  Attributes:
    target: Normalized target string, suitable for direct comparison with the
      normalized prediction.
    prediction: Normalized prediction string, suitable for direct comparison
      with the normalized target.
    treat_as_number: Whether it was determined to treat the prediction as a
      number (and perform additional post-processing relevant to numbers, such
      as stripping of units or normalization of thousand separators, etc.).
    num_decimals: Number of decimal places to which it was determined to round
      the answer. Only relevant when treat_as_number==True.
  """
  target: str
  prediction: str
  treat_as_number: bool
  num_decimals: int


def get_normalized_target_and_prediction(
    target: str,
    prediction: str
    ) -> NormalizationResult:
  """Returns a normalized target and prediction for `number_included_accuracy`.

  Args:
    target: Target (i.e., golden answer). The function will automatically
      perform light normalization on the target, such as stripping off any
      answer indication prefixes like "The answer is".
    prediction: Original model prediction. The function will automatically
      normalize the prediction by stripping off trailing punctuation and any
      answer indication prefixes like "The answer is". If the target is numeric,
      will further strip units and round to the same precision as the target.

  Returns:
    The normalized target and prediction, along with related information
    indicating the types of normalization that were performed.
  """

  def _any_list_item_in_string(test_list, test_string):
    return any(item in test_string for item in test_list)

  primary_after_patterns_in_target = _any_list_item_in_string(
      FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY, target
  )
  secondary_after_patterns_in_target = _any_list_item_in_string(
      FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY, target
  )
  target = target.lower()
  if (
      primary_after_patterns_in_target
      or (
          secondary_after_patterns_in_target
          and not primary_after_patterns_in_target
      )
      or _any_list_item_in_string(FINAL_ANSWER_AHEAD_PATTERNS, target)
      or GSM8K_ANSWER in target
  ):
    if primary_after_patterns_in_target:
      target = re.split(
          r'|'.join(FINAL_ANSWER_BEHIND_PATTERNS_PRIMARY), target
      )[-1]
    elif (
        secondary_after_patterns_in_target
        and not primary_after_patterns_in_target
    ):
      target = re.split(
          r'|'.join(FINAL_ANSWER_BEHIND_PATTERNS_SECONDARY), target
      )[-1]
    target = re.split(r'|'.join(FINAL_ANSWER_AHEAD_PATTERNS), target)[0]
    target = target.split(GSM8K_ANSWER)[-1]
    if (
        target
        and target[-1] in [';', ',', '.']
        and _is_float(target[:-1])
    ):
      target = target[:-1]

  treat_as_number = _is_float(target)
  if treat_as_number and '.' in target:
    num_decimals = len(target.split('.')[-1])
  else:
    num_decimals = 0
  
  # Check if target is a multiple choice answer (single letter A-D)
  treat_as_multiple_choice = (
      target.strip().lower() in {'a', 'b', 'c', 'd'} and 
      len(target.strip()) == 1
  )

  normalized_prediction = get_normalized_prediction(
      prediction,
      treat_as_number=treat_as_number,
      num_decimals=num_decimals,
      treat_as_multiple_choice=treat_as_multiple_choice)

  return NormalizationResult(
      target=target,
      prediction=normalized_prediction,
      treat_as_number=treat_as_number,
      num_decimals=num_decimals)


def number_included_accuracy_list(
    targets: Sequence[str],
    predictions: Sequence[str],
) -> List[bool]:
  """Returns a list of booleans for if the target is anywhere in the prediction.

  Args:
    targets: Targets (i.e., golden answers).
    predictions: Original model predictions (before normalization).
  """

  correct_list = []
  for prediction, target in zip(predictions, targets):
    normalization_result = get_normalized_target_and_prediction(
        target=target, prediction=prediction)

    # If answer is not a number, then look for exact match.
    if not normalization_result.treat_as_number:
      correct_list.append(
          normalization_result.target == normalization_result.prediction)

    else:  # If the target is a number, then compare numerically.
      correct = False  # pylint: disable=unused-variable
      try:
        prediction_parsed_float = round(
            float(normalization_result.prediction),
            normalization_result.num_decimals)
        correct = (
            abs(prediction_parsed_float - float(normalization_result.target)) <=
            1e-5)
      except ValueError:
        correct = False
      except IndexError:
        correct = False
      correct_list.append(correct)
  return correct_list


def number_included_accuracy(targets: Sequence[str],
                             predictions: Sequence[str]) -> Dict[str, float]:
  """Special accuracy for if the target is anywhere in the prediction."""

  correct_list = number_included_accuracy_list(targets, predictions)

  correct_list_with_calc = number_included_accuracy_list(
      targets, predictions)

  return {
      'accuracy':
          sum(correct_list) / len(correct_list) * 100,
      'accuracy_with_calc':
          sum(correct_list_with_calc) / len(correct_list_with_calc) * 100
  }
