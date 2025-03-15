# Fine-Tuning-an-LLM-for-STIG-Remediation

# STIG Remediation Script Generator

## Overview
This project fine-tunes a large language model (specifically Llama-2-7b) to generate remediation scripts for failed Security Technical Implementation Guide (STIG) tests. The system parses XML STIG reports, extracts failed rules, and trains the model to generate appropriate remediation scripts based on the failure data.

## Features
- Parses STIG XML reports to extract failed rules
- Formats extracted data for fine-tuning a large language model
- Trains a custom model to generate remediation scripts
- Provides a structured approach to automate STIG compliance remediation

## Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- Hugging Face account (for model access)
- STIG report XML files

### Custom Dataset
To use your own dataset instead of the default:
```python
# Replace "your_dataset" with your actual dataset name or path
dataset = load_dataset("path/to/your/dataset")
```

## Configuration
You can modify the following parameters in the script:

- `MODEL_NAME`: Base model to use (default: "meta-llama/Llama-2-7b-hf")
- Training parameters in `TrainingArguments`:
  - `per_device_train_batch_size`: Batch size per device
  - `gradient_accumulation_steps`: Number of update steps to accumulate before performing a backward/update pass
  - `learning_rate`: Learning rate for the optimizer
  - `max_steps`: Maximum number of training steps

## Project Structure
```
.
├── main.py                  # Main script
├── stig_report.xml          # Input STIG report
├── formatted_scripts.jsonl  # Generated dataset
├── custom_finetuned_model/  # Output directory for fine-tuned model
└── logs/                    # Training logs
```

## Considerations
- The script uses half-precision (fp16) training to reduce memory usage
- Consider using a GPU for faster training
- For large STIG reports, you may need to adjust memory settings

