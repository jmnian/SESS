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
"""The utility functions for prompting GPT and Google Cloud models."""

import time
from datasets import Dataset
import openai
from tqdm import tqdm


# Global variable to cache the pipeline
_PIPELINE_CACHE = {
    "pipeline": None,
    "model_name": None,
}

# Global variable to cache model and tokenizer for direct generation
_MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
}


def load_local_pipeline(model_name):
  """Load a local model pipeline with caching."""
  global _PIPELINE_CACHE
  
  # Return cached pipeline if already loaded
  if (_PIPELINE_CACHE["pipeline"] is not None and 
      _PIPELINE_CACHE["model_name"] == model_name):
    return _PIPELINE_CACHE["pipeline"]
  
  from transformers import pipeline, AutoTokenizer
  
  print(f"Loading local model pipeline: {model_name}...")
  
  # Load tokenizer with left padding for decoder-only models
  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      trust_remote_code=True,
      padding_side='left',  # Required for decoder-only models
  )
  
  # Set pad token if not set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  pipe = pipeline(
      "text-generation",
      model=model_name,
      tokenizer=tokenizer,
      device_map="auto",
      trust_remote_code=True,
  )
  
  # Cache the pipeline
  _PIPELINE_CACHE["pipeline"] = pipe
  _PIPELINE_CACHE["model_name"] = model_name
  
  print(f"Pipeline for {model_name} loaded successfully!")
  return pipe


def load_local_model_and_tokenizer(model_name):
  """Load model and tokenizer directly (no pipeline) for faster batched generation."""
  global _MODEL_CACHE
  
  # Return cached model if already loaded
  if (_MODEL_CACHE["model"] is not None and 
      _MODEL_CACHE["model_name"] == model_name):
    return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]
  
  from transformers import AutoModelForCausalLM, AutoTokenizer
  import torch
  
  print(f"Loading local model directly: {model_name}...")
  
  # Load tokenizer with left padding for decoder-only models
  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      trust_remote_code=True,
      padding_side='left',  # Required for decoder-only models
  )
  
  # Set pad token if not set
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  
  # Load model with automatic device mapping
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      trust_remote_code=True,
      torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
  )
  model.eval()
  
  # Cache the model and tokenizer
  _MODEL_CACHE["model"] = model
  _MODEL_CACHE["tokenizer"] = tokenizer
  _MODEL_CACHE["model_name"] = model_name
  
  print(f"Model {model_name} loaded successfully!")
  return model, tokenizer


def call_local_model_single_prompt(
    prompt,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_decode_steps=512,
    temperature=0.8,
    num_decodes=1,
):
  """Call a local model on GPU with a single prompt using transformers pipeline."""
  pipe = load_local_pipeline(model_name)
  
  # If num_decodes > 1, generate multiple sequences in one call
  if num_decodes > 1:
    results = pipe(
        prompt,
        max_new_tokens=max_decode_steps,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=num_decodes,
        return_full_text=False,
    )
    outputs_list = [r["generated_text"].strip() for r in results]
  else:
    # Single decode
    result = pipe(
        prompt,
        max_new_tokens=max_decode_steps,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.95,
        num_return_sequences=1,
        return_full_text=False,
    )
    outputs_list = [result[0]["generated_text"].strip()]
  
  return outputs_list


def call_local_model_func(
    inputs,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_decode_steps=512,
    temperature=0.8,
    num_decodes=1,
    batch_size=8,  # Process multiple prompts at once for efficiency
):
  """Call a local model on GPU with a list of input strings, using datasets for max efficiency.
  
  Args:
    inputs: Single string or list of strings to process
    model_name: Name of the model to use
    max_decode_steps: Maximum tokens to generate
    temperature: Sampling temperature
    num_decodes: Number of outputs per input (only works with single inputs currently)
    batch_size: Number of inputs to process simultaneously
    
  Returns:
    List of generated outputs
  """
  if isinstance(inputs, str):
    inputs = [inputs]
  
  # Handle empty input case
  if len(inputs) == 0:
    return []
  
  pipe = load_local_pipeline(model_name)
  
  # If num_decodes > 1, we can't batch efficiently
  if num_decodes > 1:
    all_outputs = []
    for input_str in inputs:
      outputs = call_local_model_single_prompt(
          input_str,
          model_name=model_name,
          max_decode_steps=max_decode_steps,
          temperature=temperature,
          num_decodes=num_decodes,
      )
      all_outputs.extend(outputs)
    return all_outputs
  
  # Use datasets library for maximum efficiency with KeyDataset wrapper
  # Create a dataset from the inputs
  dataset = Dataset.from_dict({"text": inputs})
  
  # Use KeyDataset to properly iterate over the dataset with the pipeline
  from transformers.pipelines.pt_utils import KeyDataset
  
  # Process the entire dataset through the pipeline with progress bar
  # This is MUCH faster than manual batching
  results = []
  for out in tqdm(
      pipe(
          KeyDataset(dataset, "text"),
          max_new_tokens=max_decode_steps,
          temperature=temperature,
          do_sample=temperature > 0,
          top_p=0.95,
          return_full_text=False,
          batch_size=batch_size,
      ),
      total=len(inputs),
      desc="Processing batches",
      unit="prompt",
  ):
    results.append(out[0]["generated_text"].strip())
  
  return results


def call_local_model_func_fast(
    inputs,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_decode_steps=512,
    temperature=0.8,
    num_decodes=1,
    batch_size=128,
):
  """Fast batched generation using direct model.generate() - no Pipeline/Dataset overhead.
  
  This is significantly faster than the pipeline-based approach because:
  1. No Dataset/KeyDataset wrapper overhead
  2. Direct tokenization and model.generate() calls
  3. Batch processing similar to _compute_answer_log_likelihood_batch
  
  Args:
    inputs: Single string or list of strings to process
    model_name: Name of the model to use
    max_decode_steps: Maximum tokens to generate
    temperature: Sampling temperature
    num_decodes: Number of outputs per input
    batch_size: Number of inputs to process simultaneously
    
  Returns:
    List of generated outputs
  """
  import torch
  
  if isinstance(inputs, str):
    inputs = [inputs]
  
  # Handle empty input case
  if len(inputs) == 0:
    return []
  
  model, tokenizer = load_local_model_and_tokenizer(model_name)
  
  # Collect all possible stop token IDs for instruction-tuned models
  # This handles Qwen, Llama, Mistral, etc. which have special stop tokens
  stop_token_ids = []
  
  # Add the standard eos token
  if tokenizer.eos_token_id is not None:
    stop_token_ids.append(tokenizer.eos_token_id)
  
  # Check for common instruction model stop tokens
  special_stop_tokens = [
      "<|im_end|>",      # Qwen, ChatML format
      "<|eot_id|>",      # Llama 3
      "<|end|>",         # Some models
      "</s>",            # Common
      "<|endoftext|>",   # GPT-style
  ]
  
  for token in special_stop_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    # convert_tokens_to_ids returns the unk_token_id if token not found
    if token_id != tokenizer.unk_token_id and token_id not in stop_token_ids:
      stop_token_ids.append(token_id)
  
  # Also check if the model has generation_config with eos_token_id list
  if hasattr(model, 'generation_config') and model.generation_config is not None:
    gen_config_eos = model.generation_config.eos_token_id
    if gen_config_eos is not None:
      if isinstance(gen_config_eos, list):
        for eid in gen_config_eos:
          if eid not in stop_token_ids:
            stop_token_ids.append(eid)
      elif gen_config_eos not in stop_token_ids:
        stop_token_ids.append(gen_config_eos)
  
  all_outputs = []
  
  for batch_start in tqdm(
      range(0, len(inputs), batch_size),
      desc="Generating batches",
      unit="batch",
  ):
    batch_end = min(batch_start + batch_size, len(inputs))
    batch_inputs = inputs[batch_start:batch_end]
    
    # Apply chat template if the model is instruction-tuned
    # This properly formats the prompt with special tokens
    if tokenizer.chat_template is not None:
      formatted_inputs = []
      for text in batch_inputs:
        # Wrap in chat format
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_inputs.append(formatted)
      batch_inputs = formatted_inputs
    
    # Tokenize the batch
    encodings = tokenizer(
        batch_inputs,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    
    # Move to device
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    
    # Set up generation config
    gen_kwargs = {
        "max_new_tokens": max_decode_steps,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": stop_token_ids if len(stop_token_ids) > 1 else stop_token_ids[0],
    }
    
    if temperature > 0:
      gen_kwargs.update({
          "do_sample": True,
          "temperature": temperature,
          "top_p": 0.95,
      })
    else:
      gen_kwargs["do_sample"] = False
    
    if num_decodes > 1:
      gen_kwargs["num_return_sequences"] = num_decodes
    
    with torch.no_grad():
      # Generate
      output_ids = model.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          **gen_kwargs,
      )
    
    # Decode only the new tokens (exclude the input)
    for i, output in enumerate(output_ids):
      # For num_decodes > 1, outputs are interleaved
      input_idx = i // num_decodes if num_decodes > 1 else i
      input_length = attention_mask[input_idx].sum().item()
      
      # Extract only the generated tokens
      generated_tokens = output[input_length:]
      generated_text = tokenizer.decode(
          generated_tokens,
          skip_special_tokens=True,
      ).strip()
      all_outputs.append(generated_text)
  
  return all_outputs


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with an input string."""
  try:
    completion = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        max_tokens=max_decode_steps,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

  except openai.error.Timeout as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.RateLimitError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.APIConnectionError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"API connection error occurred. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.ServiceUnavailableError as e:
    retry_time = e.retry_after if hasattr(e, "retry_after") else 30
    print(f"Service unavailable. Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except OSError as e:
    retry_time = 5  # Adjust the retry time as needed
    print(
        f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
    )
    time.sleep(retry_time)
    return call_openai_server_single_prompt(
        prompt, max_decode_steps=max_decode_steps, temperature=temperature
    )

  except openai.error.InvalidRequestError as e:
    # Content filter / moderation policy error - don't retry, log the prompt
    print(f"\n{'='*60}")
    print(f"CONTENT FILTER TRIGGERED!")
    print(f"Error: {e}")
    print(f"{'='*60}")
    print(f"PROMPT THAT TRIGGERED THE FILTER:")
    print(f"{'-'*60}")
    print(prompt)
    print(f"{'-'*60}\n")
    return None


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  if isinstance(inputs, str):
    inputs = [inputs]
  
  # Handle empty input case
  if len(inputs) == 0:
    return []
  
  outputs = []
  for input_str in inputs:
    output = call_openai_server_single_prompt(
        input_str,
        model=model,
        max_decode_steps=max_decode_steps,
        temperature=temperature,
    )
    outputs.append(output)
  return outputs
