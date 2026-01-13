"""
FRODO Training Script
Demonstrates how to train the complete FRODO framework
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
import copy

from frodo import (
    FRODO,
    FRODOConfig,
    DPODataset,
    ReasoningDataset,
)


def create_example_dpo_data():
    """Create example data for DPO training of inference module"""
    return [
        {
            'question': 'Does a banana have more protein than an apple?',
            'preferred_reasoning': 'Step 1: A banana contains approximately 1.3g of protein per 100g. '
                                  'Step 2: An apple contains approximately 0.3g of protein per 100g. '
                                  'Step 3: Since 1.3g > 0.3g, a banana has more protein.',
            'dispreferred_reasoning': 'Step 1: Bananas are yellow. '
                                     'Step 2: Apples are often red. '
                                     'Step 3: Yellow fruits have more protein.',
        },
        {
            'question': 'Would a candle last longer than a battery-powered flashlight?',
            'preferred_reasoning': 'Step 1: A typical candle burns for 4-8 hours. '
                                  'Step 2: A battery-powered flashlight can run for 20-100 hours depending on the battery. '
                                  'Step 3: Therefore, a flashlight generally lasts longer.',
            'dispreferred_reasoning': 'Step 1: Candles use fire. '
                                     'Step 2: Fire is natural energy. '
                                     'Step 3: Natural energy lasts longer.',
        },
        {
            'question': 'Is the sun larger than Earth?',
            'preferred_reasoning': 'Step 1: The sun has a diameter of approximately 1,391,000 km. '
                                  'Step 2: Earth has a diameter of approximately 12,742 km. '
                                  'Step 3: Since 1,391,000 >> 12,742, the sun is much larger than Earth.',
            'dispreferred_reasoning': 'Step 1: The sun appears small in the sky. '
                                     'Step 2: Earth feels big when we walk on it. '
                                     'Step 3: Therefore Earth is larger.',
        },
    ]


def create_example_reasoning_data():
    """Create example data for training the reasoning module"""
    return [
        {
            'question': 'Does a banana have more protein than an apple?',
            'reasoning': 'Step 1: A banana contains approximately 1.3g of protein per 100g. '
                        'Step 2: An apple contains approximately 0.3g of protein per 100g. '
                        'Step 3: Since 1.3g > 0.3g, a banana has more protein.',
            'answer': 'Yes',
            'counterfactual_reasoning': 'Step 1: Bananas are yellow. '
                                       'Step 2: Apples are often red. '
                                       'Step 3: Yellow fruits have more protein.',
        },
        {
            'question': 'Would a candle last longer than a battery-powered flashlight?',
            'reasoning': 'Step 1: A typical candle burns for 4-8 hours. '
                        'Step 2: A battery-powered flashlight can run for 20-100 hours depending on the battery. '
                        'Step 3: Therefore, a flashlight generally lasts longer.',
            'answer': 'No',
            'counterfactual_reasoning': 'Step 1: Candles use fire. '
                                       'Step 2: Fire is natural energy. '
                                       'Step 3: Natural energy lasts longer.',
        },
        {
            'question': 'Is the sun larger than Earth?',
            'reasoning': 'Step 1: The sun has a diameter of approximately 1,391,000 km. '
                        'Step 2: Earth has a diameter of approximately 12,742 km. '
                        'Step 3: Since 1,391,000 >> 12,742, the sun is much larger than Earth.',
            'answer': 'Yes',
            'counterfactual_reasoning': 'Step 1: The sun appears small in the sky. '
                                       'Step 2: Earth feels big when we walk on it. '
                                       'Step 3: Therefore Earth is larger.',
        },
    ]


def train_frodo():
    """Complete training pipeline for FRODO"""
    
    print("=" * 70)
    print("FRODO Training Pipeline")
    print("=" * 70)
    
    # Configuration
    config = FRODOConfig(
        model_name="google/flan-t5-base",  # Using smaller model for demo
        max_length=256,
        learning_rate=5e-5,
        batch_size=2,
        num_epochs=3,
        beta=0.1,
        lambda_lm=1.0,
        lambda_ie=1.0,
        lambda_margin=1.0,
        margin=1.0,
    )
    
    print("\n1. Loading models and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load models for inference and reasoning modules
    inference_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    reasoning_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    # Create FRODO framework
    frodo = FRODO(inference_model, reasoning_model, config)
    frodo = frodo.to(config.device)
    
    # Create reference model for DPO (frozen copy) - AFTER moving main model to device
    print("2. Creating reference model for DPO...")
    reference_model = copy.deepcopy(frodo.inference_module.model)
    frodo.inference_module.set_reference_model(reference_model)
    
    print(f"   Models loaded on device: {config.device}")
    
    # ========== PHASE 1: Train Inference Module with DPO ==========
    print("\n" + "=" * 70)
    print("PHASE 1: Training Inference Module with DPO")
    print("=" * 70)
    
    # Create DPO dataset
    dpo_data = create_example_dpo_data()
    dpo_dataset = DPODataset(dpo_data, tokenizer, max_length=config.max_length)
    dpo_dataloader = DataLoader(dpo_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer for inference module
    inference_optimizer = torch.optim.AdamW(
        frodo.inference_module.parameters(),
        lr=config.learning_rate
    )
    
    print(f"\nTraining with {len(dpo_data)} examples...")
    frodo.train_inference_module(
        train_dataloader=dpo_dataloader,
        num_epochs=config.num_epochs,
        optimizer=inference_optimizer,
    )
    
    # ========== PHASE 2: Train Reasoning Module ==========
    print("\n" + "=" * 70)
    print("PHASE 2: Training Reasoning Module")
    print("=" * 70)
    
    # Create reasoning dataset
    reasoning_data = create_example_reasoning_data()
    reasoning_dataset = ReasoningDataset(reasoning_data, tokenizer, max_length=config.max_length)
    reasoning_dataloader = DataLoader(reasoning_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer for reasoning module
    reasoning_optimizer = torch.optim.AdamW(
        frodo.reasoning_module.parameters(),
        lr=config.learning_rate
    )
    
    print(f"\nTraining with {len(reasoning_data)} examples...")
    print("Loss components:")
    print("  - Language Model Loss (L_LM)")
    print("  - Indirect Effect Loss (L_IE)")
    print("  - Margin Ranking Loss (L_MR)")
    print()
    
    frodo.train_reasoning_module(
        train_dataloader=reasoning_dataloader,
        num_epochs=config.num_epochs,
        optimizer=reasoning_optimizer,
    )
    
    # ========== PHASE 3: Inference ==========
    print("\n" + "=" * 70)
    print("PHASE 3: Testing Inference")
    print("=" * 70)
    
    test_questions = [
        "Does a banana have more protein than an apple?",
        "Is the sun larger than Earth?",
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        reasoning, answer = frodo.generate_reasoning_and_answer(question, tokenizer)
        print(f"Reasoning: {reasoning}")
        print(f"Answer: {answer}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return frodo


if __name__ == "__main__":
    # Run training
    trained_frodo = train_frodo()
    
    print("\n" + "=" * 70)
    print("FRODO Loss Functions Summary")
    print("=" * 70)
    
    print("\n1. DPO Loss (Inference Module):")
    print("   L_DPO = -log(σ(β * (log π_θ(r_w|x) - log π_ref(r_w|x)")
    print("                      - log π_θ(r_l|x) + log π_ref(r_l|x))))")
    print("   where:")
    print("   - r_w: preferred (correct) reasoning chain")
    print("   - r_l: dispreferred (counterfactual) reasoning chain")
    print("   - β: temperature parameter")
    
    print("\n2. Language Model Loss (Reasoning Module):")
    print("   L_LM = -log P_θ(y | x, r)")
    print("   - Standard cross-entropy loss")
    
    print("\n3. Indirect Effect Loss (Reasoning Module):")
    print("   L_IE = -log P_θ(y | x, r)")
    print("   - Encourages model to condition on reasoning steps")
    
    print("\n4. Margin Ranking Loss (Reasoning Module):")
    print("   L_MR = max(0, margin - (h(x, r_w, y_w) - h(x, r_l, y_w)))")
    print("   where:")
    print("   - h(): scoring function (log probability)")
    print("   - r_w: correct reasoning")
    print("   - r_l: counterfactual reasoning")
    
    print("\n5. Combined Reasoning Loss:")
    print("   L_PREF = λ_LM * L_LM + λ_IE * L_IE + λ_MR * L_MR")
    
    print("\n" + "=" * 70)
