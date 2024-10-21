Here's the updated README file with a section on the model's capabilities added:

# MathWizard: Fine-tuning Llama-3 Model for Math Problem Solving

## Overview

This repository contains the code and instructions for fine-tuning the Llama-3 model using the Unsloth framework to solve math problems. The model is trained on the OpenMath dataset and can be deployed using a Streamlit application.

## Table of Contents

- [MathWizard: Fine-tuning Llama-3 Model for Math Problem Solving](#mathwizard-fine-tuning-llama-3-model-for-math-problem-solving)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Capabilities](#capabilities)
  - [Fine-tuning the Model](#fine-tuning-the-model)
  - [Running Inference](#running-inference)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [License](#license)

## Prerequisites

- Python 3.8 or higher
- Access to a GPU (recommended) or use Google Colab
- Basic understanding of Python and machine learning concepts

## Installation

To install the necessary libraries, run:

```bash
pip install torch transformers unsloth datasets
```

Alternatively, you can use Google Colab for easier setup. Simply open a new notebook and run:

```python
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Capabilities

The fine-tuned Llama-3 model possesses the following capabilities:

- **Mathematical Problem Solving:** Efficiently solves a variety of mathematical problems, including algebra, calculus, and discrete mathematics.
- **Natural Language Understanding:** Understands instructions and context provided in natural language, making it user-friendly for non-experts.
- **Contextual Responses:** Generates relevant and coherent responses based on given inputs and previous interactions.
- **Flexible Input Handling:** Can accept a range of input formats, including questions, equations, and scenarios requiring mathematical reasoning.
- **Inference Speed:** Enhanced inference speed due to the use of 4-bit quantization and efficient model architecture.

## Fine-tuning the Model

1. **Import Necessary Libraries:**
   Use the following imports in your notebook:

   ```python
   from unsloth import FastLanguageModel
   from datasets import load_dataset
   from transformers import TrainingArguments
   from trl import SFTTrainer
   import torch
   ```

2. **Set Hyperparameters:**
   Customize the following parameters as needed:

   ```python
   max_seq_length = 2048
   load_in_4bit = True
   ```

3. **Load the Pre-trained Model:**
   Load the Llama-3 model and tokenizer:

   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Meta-Llama-3.1-8B",
       max_seq_length=max_seq_length,
       load_in_4bit=load_in_4bit,
   )
   ```

4. **Prepare the Dataset:**
   Use the OpenMath dataset. You can load and format the dataset as follows:

   ```python
   dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train[:90000]")
   ```

5. **Set Up the Trainer:**
   Configure the `SFTTrainer`:

   ```python
   trainer = SFTTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=dataset,
       args=TrainingArguments(
           per_device_train_batch_size=2,
           num_train_epochs=1,
           learning_rate=2e-4,
           output_dir="outputs",
       ),
   )
   ```

6. **Train the Model:**
   Start the training process:
   ```python
   trainer.train()
   ```

## Running Inference

Once the model is fine-tuned, you can run inference as follows:

```python
inputs = tokenizer("Your math problem here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Saving and Loading Models

To save your fine-tuned model, use the following commands:

```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

To push the model to the Hugging Face Hub, run:

```python
model.push_to_hub("your-model-name",token="")
tokenizer.push_to_hub("your-model-name",token="")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
