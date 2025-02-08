import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import random
from transformers import TrainerCallback, TrainingArguments, TrainerControl, TrainerState


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "AI-MO/NuminaMath-CoT"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Helper functions
########################

def safe_extract_content(text: str, start_tag: str, end_tag: str) -> str:
    try:
        if start_tag not in text or end_tag not in text:
            return ""
        content = text.split(start_tag)[-1]
        content = content.split(end_tag)[0]
        return content.strip()
    except Exception:
        return ""

def extract_thought(text: str) -> str:
    return safe_extract_content(text, "<|begin_of_thought|>", "<|end_of_thought|>")

def extract_solution(text: str) -> str:
    return safe_extract_content(text, "<|begin_of_solution|>", "<|end_of_solution|>")

import re
def format_compliance_reward(completions, **kwargs):
    try:
        format_reward = kwargs.get("format_reward", 0.5)
        pattern = r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>\s*<\|begin_of_solution\|>.*?<\|end_of_solution\|>"
        rewards = []

        for completion in completions:
            try:
                # Handle the nested structure correctly
                if isinstance(completion, dict) and 'generated_text' in completion:
                    # If it's the new format with generated_text
                    messages = completion['generated_text']
                    # Find the assistant's message
                    for message in messages:
                        if isinstance(message, dict) and message.get('role') == 'assistant':
                            text = message.get('content', '')
                            break
                    else:
                        # If no assistant message found, use the whole completion
                        text = str(messages) if isinstance(messages, str) else str(completion['generated_text'])
                elif isinstance(completion, str):
                    # If it's already a string, try to extract assistant part or use whole text
                    parts = completion.split('assistant')
                    text = parts[-1].strip() if len(parts) > 1 else completion
                else:
                    # Fallback to string representation
                    text = str(completion)

                has_format = bool(re.search(pattern, text, re.DOTALL))
                rewards.append(format_reward if has_format else 0.0)

            except Exception as e:
                print(f"Error processing completion: {e}")
                rewards.append(0.0)

        return rewards
    except Exception as e:
        print(f"Global error in reward function: {e}")
        return [0.0] * len(completions)

def math_reasoning_reward(completions, **kwargs):
    try:
        thought_reward = kwargs.get("thought_reward", 0.3)
        steps_reward = kwargs.get("steps_reward", 0.2)
        equation_reward = kwargs.get("equation_reward", 0.2)

        rewards = []
        for completion in completions:
            try:
                # Extract text from completion
                if isinstance(completion, dict) and 'generated_text' in completion:
                    messages = completion['generated_text']
                    for message in messages:
                        if isinstance(message, dict) and message.get('role') == 'assistant':
                            text = message.get('content', '')
                            break
                    else:
                        text = str(messages) if isinstance(messages, str) else str(completion['generated_text'])
                elif isinstance(completion, str):
                    parts = completion.split('assistant')
                    text = parts[-1].strip() if len(parts) > 1 else completion
                else:
                    text = str(completion)

                thought = extract_thought(text)
                if not thought:
                    # print("No thought content found")
                    rewards.append(0.0)
                    continue

                reward = 0.0

                # Count steps (paragraphs and numbered items)
                steps = thought.count("\n\n") + len(re.findall(r'\n\d+\.', thought))
                step_contribution = min(steps * steps_reward, 0.6)
                print(f"Steps found: {steps}, contribution: {step_contribution}")
                reward += step_contribution

                # Check for mathematical notation
                math_patterns = [
                    r'[=<>+\-]',  # Basic operators
                    r'\\frac',    # LaTeX fractions
                    r'\\left',    # LaTeX parentheses
                    r'\\right',   # LaTeX parentheses
                    r'\$',        # Inline math
                    r'\\boxed',   # LaTeX boxes
                    r'[0-9]+/[0-9]+',  # Fractions like 1/2
                ]

                for pattern in math_patterns:
                    if re.search(pattern, thought):
                        print(f"Mathematical notation found (pattern: {pattern})")
                        reward += thought_reward
                        break

                # Count equations
                equation_patterns = [
                    r'=',  # Regular equals signs
                    r'\\begin\{equation\}',  # LaTeX equation environments
                    r'\\end\{equation\}',    # LaTeX equation environments
                    r'\\\[',                 # LaTeX display math
                    r'\\\]',                 # LaTeX display math
                ]

                equation_count = 0
                for pattern in equation_patterns:
                    equation_count += len(re.findall(pattern, thought))

                equation_contribution = min(equation_count * equation_reward, 0.4)
                print(f"Equations found: {equation_count}, contribution: {equation_contribution}")
                reward += equation_contribution

                print(f"Total reward: {reward}")
                rewards.append(reward)

            except Exception as e:
                print(f"Error processing completion: {str(e)}")
                rewards.append(0.0)

        return rewards
    except Exception as e:
        print(f"Global error in reward function: {str(e)}")
        return [0.0] * len(completions)

def solution_quality_reward(completions, prompts=None, **kwargs):
    try:
        quality_reward = kwargs.get("quality_reward", 0.4)
        math_notation_reward = kwargs.get("math_notation_reward", 0.2)
        formatting_reward = kwargs.get("latex_reward", 0.2)

        rewards = []
        for completion in completions:
            try:
                # Extract text from completion
                if isinstance(completion, dict) and 'generated_text' in completion:
                    messages = completion['generated_text']
                    for message in messages:
                        if isinstance(message, dict) and message.get('role') == 'assistant':
                            text = message.get('content', '')
                            break
                    else:
                        text = str(messages) if isinstance(messages, str) else str(completion['generated_text'])
                elif isinstance(completion, str):
                    parts = completion.split('assistant')
                    text = parts[-1].strip() if len(parts) > 1 else completion
                else:
                    text = str(completion)

                solution = extract_solution(text)
                if not solution:
                    rewards.append(0.0)
                    continue

                reward = 0.0

                # Check for any kind of answer
                answer_pattern = (
                    r'\d+(?:\.\d+)?|'           # Numbers
                    r'[a-z]=|'                  # Variable assignments
                    r'-?\d+/\d+|'               # Fractions
                    r'\\frac\{[^}]+\}\{[^}]+\}|'  # LaTeX fractions
                    r'\\boxed\{[^}]+\}|'        # Boxed answers
                    r'\b(?:yes|no|true|false)\b' # Boolean answers
                )
                if re.search(answer_pattern, solution, re.IGNORECASE):
                    reward += quality_reward

                # Check for mathematical notation
                math_pattern = (
                    r'[=<>+\-×÷∙∗⋅]|'          # Basic operators
                    r'\\[a-zA-Z]+|'             # LaTeX commands
                    r'\$.*?\$|'                 # Inline math
                    r'.∗?.*?|'                 # Display math
                    r'\b(?:sin|cos|tan|log|lim|inf|sqrt)\b'  # Math functions
                )
                if re.search(math_pattern, solution):
                    reward += math_notation_reward

                # Check for professional formatting
                format_pattern = (
                    r'\\boxed|'                # LaTeX boxing
                    r'\\begin\{.*?\}|'         # LaTeX environments
                    r'\\text|'                 # LaTeX text
                    r'\$\$|\$'                 # Math delimiters
                )
                if re.search(format_pattern, solution):
                    reward += formatting_reward

                rewards.append(reward)

            except Exception as e:
                print(f"Error processing completion: {str(e)}")
                rewards.append(0.0)

        return rewards

    except Exception as e:
        print(f"Global error in solution quality reward: {str(e)}")
        return [0.0] * len(completions)
    
def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub
    dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))

    #####################
    # Prepare and format dataset
    #####################

    # gemerate r1 prompt with a prefix for the model to already start with the thinking process
    def generate_r1_prompt(problem, target):
        r1_prefix = [{
        "role": "system",
        "content": """You are a mathematical reasoning assistant. Your approach involves:
                1. Understanding the problem thoroughly
                2. Breaking down complex problems into simpler steps
                3. Using precise mathematical notation
                4. Providing clear logical justification for each step
                5. Verifying the solution

                Format your response with clear reasoning followed by the solution."""
                    },
        {
        "role": "user",
        "content": f"""Solve this mathematical problem using systematic reasoning: {problem}
                Structure your response as:
                <|begin_of_thought|>
                [Your step-by-step reasoning]
                <|end_of_thought|>

                <|begin_of_solution|>
                [Your complete solution with justification]
                <|end_of_solution|>"""
    },
    {
        "role": "assistant",
        "content": "<|begin_of_thought|>\nLet me solve this step by step:\n\n"
    }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

    # convert our dataset to the r1 prompt
    original_columns = dataset.column_names
    dataset = dataset.map(
        lambda x: generate_r1_prompt(x["problem"], x["solution"]),
        remove_columns= original_columns
    )

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    #########################
    # Instantiate DPO trainer
    #########################

    class OutputLoggerCallback(TrainerCallback):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset  # Store dataset
            self.tokenizer = tokenizer  # Store tokenizer
    
        def on_step_end(self, args, state, control, model=None, **kwargs):
            """Logs sample outputs after every second training step using a dynamic input."""
            if state.global_step % 5 == 0:  # Log every 2 steps
                # Select a random input from dataset
                sample_index = random.randint(0, len(self.dataset) - 1)
                prompt = self.dataset[sample_index]["prompt"]
    
                # Tokenize and generate output
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                with torch.no_grad():
                    try:
                        output_ids = model.generate(input_ids)
                        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                        logger.info(f"Step {state.global_step}: \nGenerated: {generated_text}\n")
                    except Exception as e:
                        logger.error(f"Error generating text: {e}")
    
            return control


    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_compliance_reward, math_reasoning_reward, solution_quality_reward],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )
    
    trainer.add_callback(OutputLoggerCallback(train_dataset, tokenizer))

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()