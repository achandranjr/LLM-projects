#!/usr/bin/env python3
"""
All-in-one script for local LLM fine-tuning and serving
Includes training, inference, and Gradio UI
"""

import os
import json
import torch
import gradio as gr
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse
from typing import List, Dict

class LocalLLM:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """Initialize with a small model suitable for local use"""
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model will be loaded when needed
        self.model = None
        self.tokenizer = None
        self.is_fine_tuned = False
    
    def prepare_data(self, data_path: str = "data/train.jsonl"):
        """Load and prepare training data"""
        print(f"Loading data from {data_path}")
        
        # Load JSONL file
        with open(data_path, 'r', encoding='utf-8') as f:
            examples = [json.loads(line) for line in f]
        
        # Format data
        formatted_examples = []
        for ex in examples:
            if "instruction" in ex:
                text = f"### Instruction: {ex['instruction']}\n"
                if ex.get("input"):
                    text += f"### Input: {ex['input']}\n"
                text += f"### Response: {ex['output']}"
            else:
                text = f"User: {ex['input']}\nAssistant: {ex['output']}"
            
            formatted_examples.append({"text": text})
        
        return Dataset.from_list(formatted_examples)
    
    def fine_tune(
        self,
        train_data_path: str = "data/train.jsonl",
        output_dir: str = "./models/fine-tuned",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        use_lora: bool = True
    ):
        """Fine-tune the model on local data"""
        print(f"\nStarting fine-tuning of {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        # Apply LoRA for efficiency
        if use_lora and self.device.type == 'cuda':
            print("Applying LoRA configuration...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,  # Lower rank for faster training
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # Fewer modules for speed
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Prepare dataset
        dataset = self.prepare_data(train_data_path)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments optimized for local use
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            warmup_steps=50,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",  # Skip eval for speed
            save_strategy="steps",
            fp16=self.device.type == 'cuda',
            push_to_hub=False,
            report_to="none",  # No external logging
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        print("\nTraining started...")
        trainer.train()
        
        # Save
        print(f"\nSaving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        self.is_fine_tuned = True
        print("\nFine-tuning complete!")
    
    def load_model(self, model_path: str = None):
        """Load a model (base or fine-tuned)"""
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Check if it's a LoRA model
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config_path):
                # Load base model + LoRA
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                base_model_name = config.get("base_model_name_or_path", self.model_name)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None,
                    trust_remote_code=True
                )
                self.model = PeftModel.from_pretrained(self.model, model_path)
            else:
                # Load full model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None,
                    trust_remote_code=True
                )
            
            self.is_fine_tuned = True
        else:
            # Load base model
            print(f"Loading base model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Generate text from prompt"""
        if self.model is None:
            raise ValueError("No model loaded! Call load_model() first.")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the new text
        return generated_text[len(prompt):].strip()

def create_gradio_interface(llm: LocalLLM):
    """Create a Gradio interface for the model"""
    
    def chat(message, history, temperature, max_tokens):
        # Format conversation history
        prompt = ""
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        prompt += f"User: {message}\nAssistant:"
        
        # Generate response
        response = llm.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def instruction_mode(instruction, input_text, temperature, max_tokens):
        # Format as instruction
        prompt = f"### Instruction: {instruction}\n"
        if input_text:
            prompt += f"### Input: {input_text}\n"
        prompt += "### Response:"
        
        # Generate response
        response = llm.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    # Create interface with tabs
    with gr.Blocks(title="Local LLM") as interface:
        gr.Markdown("# Local LLM Interface")
        gr.Markdown(f"Model: {llm.model_name} {'(Fine-tuned)' if llm.is_fine_tuned else '(Base)'}")
        
        with gr.Tab("Chat"):
            chatbot = gr.ChatInterface(
                fn=chat,
                additional_inputs=[
                    gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
                    gr.Slider(50, 500, value=256, label="Max Tokens", step=50)
                ],
                examples=[
                    "What is machine learning?",
                    "Write a Python function to sort a list",
                    "Explain quantum computing in simple terms"
                ]
            )
        
        with gr.Tab("Instruction Mode"):
            with gr.Row():
                with gr.Column():
                    instruction = gr.Textbox(
                        label="Instruction",
                        placeholder="E.g., 'Translate to French'",
                        lines=2
                    )
                    input_text = gr.Textbox(
                        label="Input",
                        placeholder="Your input text here...",
                        lines=4
                    )
                    with gr.Row():
                        temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
                        max_tokens = gr.Slider(50, 500, value=256, label="Max Tokens", step=50)
                    submit_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output = gr.Textbox(label="Output", lines=8)
            
            submit_btn.click(
                fn=instruction_mode,
                inputs=[instruction, input_text, temperature, max_tokens],
                outputs=output
            )
            
            gr.Examples(
                examples=[
                    ["Translate to Spanish", "Hello, how are you?"],
                    ["Summarize this text", "Machine learning is a subset of artificial intelligence..."],
                    ["Write a haiku about", "artificial intelligence"]
                ],
                inputs=[instruction, input_text]
            )
    
    return interface

def create_sample_data():
    """Create sample training data"""
    samples = [
        {"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"},
        {"instruction": "Translate to French", "input": "Good morning", "output": "Bonjour"},
        {"instruction": "Translate to French", "input": "Thank you", "output": "Merci"},
        {"instruction": "Summarize", "input": "Machine learning is a method of data analysis that automates analytical model building.", "output": "ML automates building analytical models from data."},
        {"input": "What is Python?", "output": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
        {"input": "How do I install pip?", "output": "Pip usually comes pre-installed with Python. You can check by running 'pip --version' in your terminal."},
    ]
    
    os.makedirs("data", exist_ok=True)
    with open("data/train.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print("Created sample data in data/train.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Local LLM Fine-tuning and Serving")
    parser.add_argument("--mode", choices=["train", "serve", "demo"], default="demo",
                        help="Mode: train model, serve with UI, or run demo")
    parser.add_argument("--model", default="microsoft/phi-2", help="Base model name")
    parser.add_argument("--model-path", default="./models/fine-tuned", help="Path to fine-tuned model")
    parser.add_argument("--data", default="data/train.jsonl", help="Training data path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--create-sample-data", action="store_true", help="Create sample training data")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data()
        return
    
    # Initialize LLM
    llm = LocalLLM(model_name=args.model)
    
    if args.mode == "train":
        # Fine-tune mode
        if not os.path.exists(args.data):
            print(f"Training data not found at {args.data}")
            print("Run with --create-sample-data to create sample data")
            return
        
        llm.fine_tune(
            train_data_path=args.data,
            output_dir=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    elif args.mode == "serve":
        # Serve mode - load model and start UI
        if os.path.exists(args.model_path):
            llm.load_model(args.model_path)
        else:
            print("No fine-tuned model found, loading base model...")
            llm.load_model()
        
        # Create and launch Gradio interface
        interface = create_gradio_interface(llm)
        interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
    elif args.mode == "demo":
        # Demo mode - interactive command line
        print("\n=== Local LLM Demo Mode ===")
        
        # Check if fine-tuned model exists
        if os.path.exists(args.model_path):
            llm.load_model(args.model_path)
        else:
            print("No fine-tuned model found, loading base model...")
            print("Tip: Run with --mode train to fine-tune first!")
            llm.load_model()
        
        print("\nType 'quit' to exit, 'web' to launch web interface")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'web':
                interface = create_gradio_interface(llm)
                interface.launch(share=False)
                break
            elif user_input:
                prompt = f"User: {user_input}\nAssistant:"
                response = llm.generate(prompt, temperature=0.7, max_new_tokens=256)
                print(f"Assistant: {response}")

if __name__ == "__main__":
    main()