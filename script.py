import os
import re
import json
import torch
import xml.etree.ElementTree as ET
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, DatasetDict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset (modify based on your dataset path or source)
dataset = load_dataset("your_dataset")
logger.info("Dataset loaded successfully.")

# Define model and tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
logger.info("Model and tokenizer loaded successfully.")

# XML Parsing for STIG Rules
def parse_stig_xml(xml_file):
    namespaces = {
        '': 'http://checklists.nist.gov/xccdf/1.2',  # Default namespace
        'xhtml': 'http://www.w3.org/1999/xhtml'  # XHTML namespace
    }
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    failed_rules = {}
    
    for rule in root.findall('.//Rule', namespaces):
        rule_id = rule.get('id')
        title_element = rule.find('title', namespaces)
        fix_element = rule.find('fix', namespaces)
        
        title = title_element.text.strip() if title_element is not None else "No Title"
        fix = fix_element.text.strip() if fix_element is not None else "No Fix Script"
        
        failed_rules[f"{title} + {rule_id}"] = fix
    
    return failed_rules

# Load and process STIG rules
failed_stig_rules = parse_stig_xml("stig_report.xml")
logger.info(f"Extracted {len(failed_stig_rules)} failed STIG rules.")

# Prepare data for fine-tuning the LLM
formatted_scripts = []
for key, value in failed_stig_rules.items():
    formatted_script = {
        'prompt': (
            f'<> You specialize in analyzing the STIG report and then writing remediation scripts if the test has failed. '
            f'For this particular scan: "{key}", the following data was extracted: {value} <> '
            f'Extracted Data: {value} '
        ),
    }
    formatted_scripts.append(formatted_script)

# Save the formatted scripts to a .jsonl file
output_file = 'formatted_scripts.jsonl'
with open(output_file, 'w') as file:
    for script in formatted_scripts:
        file.write(json.dumps(script) + '\n')

logger.info("Formatted scripts saved as JSONL.")

# Create a dataset from the formatted scripts
dataset = Dataset.from_dict({"text": [script["prompt"] for script in formatted_scripts]})

# Create a dataset dictionary with the "train" split
dataset_dict = DatasetDict({"train": dataset})
logger.info("Dataset formatted and saved successfully.")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
logger.info("Dataset tokenized successfully.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./custom_finetuned_model",
    per_device_train_batch_size=4,  # Increased batch size
    gradient_accumulation_steps=2,  # Adjusted accumulation steps
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,  # Save model every 500 steps
    logging_steps=100,
    learning_rate=1e-4,  # Adjusted learning rate
    weight_decay=0.05,  # Increased weight decay for regularization
    warmup_steps=100,  # Increased warmup steps
    max_steps=1200,  # Adjusted training steps
    fp16=True,
    save_total_limit=2,  # Keep only the latest 2 checkpoints
    logging_dir="./logs",
    load_best_model_at_end=True,  # Load best model after training
)
logger.info("Training arguments set.")

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=None,  # No per-epoch evaluation
    data_collator=data_collator,
)
logger.info("Trainer initialized.")

# Train the model
logger.info("Starting training...")
trainer.train()
logger.info("Training completed successfully.")

# Save final model
model.save_pretrained("./custom_finetuned_model")
tokenizer.save_pretrained("./custom_finetuned_model")
logger.info("Model and tokenizer saved.")
