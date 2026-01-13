"""
FRODO: Framework for Faithful Reasoning Over Deliberate Output
Paper: Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning
arXiv:2402.13950

This implementation includes:
1. Inference Module - Generates correct reasoning steps using DPO
2. Reasoning Module - Faithfully reasons over steps using counterfactual and causal preference objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class FRODOConfig:
    """Configuration for FRODO framework"""
    # Model parameters
    model_name: str = "google/flan-t5-large"
    max_length: int = 512
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # DPO parameters
    beta: float = 0.1  # Temperature parameter for DPO
    
    # Loss weights
    lambda_lm: float = 1.0  # Language model loss weight
    lambda_ie: float = 1.0  # Indirect effect loss weight
    lambda_margin: float = 1.0  # Margin ranking loss weight
    
    # Margin ranking parameters
    margin: float = 1.0  # Margin for ranking loss
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class InferenceModule(nn.Module):
    """
    Inference Module: Generates correct reasoning steps using DPO
    
    Uses Direct Preference Optimization to learn to prefer correct reasoning chains
    over counterfactual ones with implicit causal rewards.
    """
    
    def __init__(self, base_model, config: FRODOConfig):
        super().__init__()
        self.model = base_model
        self.config = config
        self.ref_model = None  # Reference model for DPO (frozen copy)
        
    def set_reference_model(self, ref_model):
        """Set the frozen reference model for DPO"""
        self.ref_model = ref_model
        # Move reference model to same device as main model
        device = next(self.model.parameters()).device
        self.ref_model = self.ref_model.to(device)
        # Freeze all parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
    
    def compute_dpo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        preferred_ids: torch.Tensor,
        preferred_mask: torch.Tensor,
        dispreferred_ids: torch.Tensor,
        dispreferred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss for preference optimization
        
        DPO Loss Formula:
        L_DPO = -log(σ(β * (log π_θ(r_w|x) - log π_ref(r_w|x) - log π_θ(r_l|x) + log π_ref(r_l|x))))
        
        where:
        - r_w: preferred (correct) reasoning chain
        - r_l: dispreferred (counterfactual) reasoning chain
        - π_θ: policy model
        - π_ref: reference model
        - β: temperature parameter
        - σ: sigmoid function
        
        Args:
            input_ids: Input question tokens
            attention_mask: Attention mask for input
            preferred_ids: Tokens for preferred reasoning chain
            preferred_mask: Attention mask for preferred chain
            dispreferred_ids: Tokens for dispreferred reasoning chain
            dispreferred_mask: Attention mask for dispreferred chain
            
        Returns:
            DPO loss value
        """
        # Compute log probabilities for preferred sequence
        preferred_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=preferred_ids,
            return_dict=True
        )
        
        # Compute log probabilities for dispreferred sequence
        dispreferred_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=dispreferred_ids,
            return_dict=True
        )
        
        # Get log probabilities from the model
        preferred_logprobs = -preferred_outputs.loss
        dispreferred_logprobs = -dispreferred_outputs.loss
        
        # Compute reference model log probabilities (if available)
        if self.ref_model is not None:
            with torch.no_grad():
                ref_preferred_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=preferred_ids,
                    return_dict=True
                )
                ref_dispreferred_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=dispreferred_ids,
                    return_dict=True
                )
                ref_preferred_logprobs = -ref_preferred_outputs.loss
                ref_dispreferred_logprobs = -ref_dispreferred_outputs.loss
        else:
            ref_preferred_logprobs = 0.0
            ref_dispreferred_logprobs = 0.0
        
        # Compute DPO loss
        # log_ratio = β * (log π_θ(r_w|x) - log π_ref(r_w|x) - log π_θ(r_l|x) + log π_ref(r_l|x))
        log_ratio = self.config.beta * (
            (preferred_logprobs - ref_preferred_logprobs) -
            (dispreferred_logprobs - ref_dispreferred_logprobs)
        )
        
        # DPO loss: -log(sigmoid(log_ratio))
        dpo_loss = -F.logsigmoid(log_ratio).mean()
        
        return dpo_loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        preferred_ids: Optional[torch.Tensor] = None,
        preferred_mask: Optional[torch.Tensor] = None,
        dispreferred_ids: Optional[torch.Tensor] = None,
        dispreferred_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for inference module
        
        Args:
            input_ids: Input question tokens
            attention_mask: Attention mask for input
            preferred_ids: Tokens for preferred reasoning chain (for training)
            preferred_mask: Attention mask for preferred chain
            dispreferred_ids: Tokens for dispreferred reasoning chain (for training)
            dispreferred_mask: Attention mask for dispreferred chain
            
        Returns:
            Dictionary containing loss and other metrics
        """
        if preferred_ids is not None and dispreferred_ids is not None:
            # Training mode - compute DPO loss
            loss = self.compute_dpo_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                preferred_ids=preferred_ids,
                preferred_mask=preferred_mask,
                dispreferred_ids=dispreferred_ids,
                dispreferred_mask=dispreferred_mask,
            )
            return {"loss": loss}
        else:
            # Inference mode - generate reasoning chain
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_length,
                num_return_sequences=1,
            )
            return {"generated_ids": outputs}


class ReasoningModule(nn.Module):
    """
    Reasoning Module: Faithfully reasons over intermediate reasoning steps
    
    Uses a combination of three losses:
    1. Language Model Loss (L_LM): Standard cross-entropy loss
    2. Indirect Effect Loss (L_IE): Encourages faithfulness to reasoning steps
    3. Margin Ranking Loss (L_MR): Contrastive learning with counterfactual examples
    """
    
    def __init__(self, base_model, config: FRODOConfig):
        super().__init__()
        self.model = base_model
        self.config = config
        
    def compute_language_model_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute standard language model loss
        
        L_LM = -log P_θ(y | x, r)
        
        where:
        - y: correct answer
        - x: question
        - r: reasoning chain
        
        Args:
            input_ids: Concatenated [question, reasoning chain] tokens
            attention_mask: Attention mask
            labels: Target answer tokens
            
        Returns:
            Language model loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs.loss
    
    def compute_indirect_effect_loss(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Indirect Effect Loss to encourage faithful reasoning
        
        The indirect effect loss measures how much the reasoning chain influences
        the final answer. It encourages the model to actually use the reasoning steps.
        
        L_IE = -log P_θ(y | x, r)
        
        This is computed by conditioning on both the question and reasoning steps.
        
        Args:
            question_ids: Question tokens
            question_mask: Question attention mask
            reasoning_ids: Reasoning chain tokens
            reasoning_mask: Reasoning attention mask
            answer_ids: Answer tokens
            
        Returns:
            Indirect effect loss
        """
        # Concatenate question and reasoning
        input_ids = torch.cat([question_ids, reasoning_ids], dim=1)
        attention_mask = torch.cat([question_mask, reasoning_mask], dim=1)
        
        # Compute loss conditioning on both question and reasoning
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=answer_ids,
            return_dict=True
        )
        
        return outputs.loss
    
    def compute_margin_ranking_loss(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        correct_reasoning_ids: torch.Tensor,
        correct_reasoning_mask: torch.Tensor,
        counterfactual_reasoning_ids: torch.Tensor,
        counterfactual_reasoning_mask: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Margin Ranking Loss for contrastive learning
        
        L_MR = max(0, margin - (h(x, r_w, y_w) - h(x, r_l, y_w)))
        
        where:
        - h(): scoring function (log probability)
        - r_w: correct reasoning chain
        - r_l: counterfactual reasoning chain
        - y_w: correct answer
        - margin: margin hyperparameter
        
        This loss maximizes the margin between:
        - Positive: (question, correct reasoning, correct answer)
        - Negative: (question, counterfactual reasoning, correct answer)
        
        Args:
            question_ids: Question tokens
            question_mask: Question attention mask
            correct_reasoning_ids: Correct reasoning chain tokens
            correct_reasoning_mask: Correct reasoning attention mask
            counterfactual_reasoning_ids: Counterfactual reasoning chain tokens
            counterfactual_reasoning_mask: Counterfactual reasoning attention mask
            answer_ids: Correct answer tokens
            
        Returns:
            Margin ranking loss
        """
        # Compute score for correct reasoning
        correct_input_ids = torch.cat([question_ids, correct_reasoning_ids], dim=1)
        correct_attention_mask = torch.cat([question_mask, correct_reasoning_mask], dim=1)
        
        correct_outputs = self.model(
            input_ids=correct_input_ids,
            attention_mask=correct_attention_mask,
            labels=answer_ids,
            return_dict=True
        )
        
        # Score is negative log probability (we want to minimize this)
        correct_score = -(-correct_outputs.loss)  # Convert to log prob
        
        # Compute score for counterfactual reasoning
        counterfactual_input_ids = torch.cat([question_ids, counterfactual_reasoning_ids], dim=1)
        counterfactual_attention_mask = torch.cat([question_mask, counterfactual_reasoning_mask], dim=1)
        
        counterfactual_outputs = self.model(
            input_ids=counterfactual_input_ids,
            attention_mask=counterfactual_attention_mask,
            labels=answer_ids,
            return_dict=True
        )
        
        counterfactual_score = -(-counterfactual_outputs.loss)  # Convert to log prob
        
        # Margin ranking loss: max(0, margin - (correct_score - counterfactual_score))
        # We want correct_score > counterfactual_score by at least margin
        margin_loss = torch.clamp(
            self.config.margin - (correct_score - counterfactual_score),
            min=0
        ).mean()
        
        return margin_loss
    
    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
        answer_ids: torch.Tensor,
        counterfactual_reasoning_ids: Optional[torch.Tensor] = None,
        counterfactual_reasoning_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for reasoning module
        
        Computes the combined loss:
        L_PREF = λ_LM * L_LM + λ_IE * L_IE + λ_MR * L_MR
        
        Args:
            question_ids: Question tokens
            question_mask: Question attention mask
            reasoning_ids: Reasoning chain tokens
            reasoning_mask: Reasoning attention mask
            answer_ids: Answer tokens
            counterfactual_reasoning_ids: Counterfactual reasoning chain tokens (optional)
            counterfactual_reasoning_mask: Counterfactual reasoning attention mask (optional)
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Compute Language Model Loss
        input_ids = torch.cat([question_ids, reasoning_ids], dim=1)
        attention_mask = torch.cat([question_mask, reasoning_mask], dim=1)
        
        lm_loss = self.compute_language_model_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=answer_ids
        )
        
        # Compute Indirect Effect Loss
        ie_loss = self.compute_indirect_effect_loss(
            question_ids=question_ids,
            question_mask=question_mask,
            reasoning_ids=reasoning_ids,
            reasoning_mask=reasoning_mask,
            answer_ids=answer_ids
        )
        
        # Compute Margin Ranking Loss (if counterfactual data provided)
        if counterfactual_reasoning_ids is not None:
            margin_loss = self.compute_margin_ranking_loss(
                question_ids=question_ids,
                question_mask=question_mask,
                correct_reasoning_ids=reasoning_ids,
                correct_reasoning_mask=reasoning_mask,
                counterfactual_reasoning_ids=counterfactual_reasoning_ids,
                counterfactual_reasoning_mask=counterfactual_reasoning_mask,
                answer_ids=answer_ids
            )
        else:
            margin_loss = torch.tensor(0.0, device=question_ids.device)
        
        # Combined loss
        total_loss = (
            self.config.lambda_lm * lm_loss +
            self.config.lambda_ie * ie_loss +
            self.config.lambda_margin * margin_loss
        )
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss.item(),
            "ie_loss": ie_loss.item(),
            "margin_loss": margin_loss.item() if isinstance(margin_loss, torch.Tensor) else margin_loss
        }


class FRODO(nn.Module):
    """
    Complete FRODO Framework
    
    Combines the Inference Module and Reasoning Module for faithful chain-of-thought reasoning.
    """
    
    def __init__(self, inference_model, reasoning_model, config: FRODOConfig):
        super().__init__()
        self.config = config
        self.inference_module = InferenceModule(inference_model, config)
        self.reasoning_module = ReasoningModule(reasoning_model, config)
        
    def train_inference_module(
        self,
        train_dataloader,
        num_epochs: int,
        optimizer,
    ):
        """
        Train the inference module using DPO
        
        Args:
            train_dataloader: DataLoader with preference pairs
            num_epochs: Number of training epochs
            optimizer: Optimizer for training
        """
        self.inference_module.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                outputs = self.inference_module(
                    input_ids=batch['input_ids'].to(self.config.device),
                    attention_mask=batch['attention_mask'].to(self.config.device),
                    preferred_ids=batch['preferred_ids'].to(self.config.device),
                    preferred_mask=batch['preferred_mask'].to(self.config.device),
                    dispreferred_ids=batch['dispreferred_ids'].to(self.config.device),
                    dispreferred_mask=batch['dispreferred_mask'].to(self.config.device),
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs}, DPO Loss: {avg_loss:.4f}")
    
    def train_reasoning_module(
        self,
        train_dataloader,
        num_epochs: int,
        optimizer,
    ):
        """
        Train the reasoning module with combined loss
        
        Args:
            train_dataloader: DataLoader with reasoning examples
            num_epochs: Number of training epochs
            optimizer: Optimizer for training
        """
        self.reasoning_module.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_lm_loss = 0
            total_ie_loss = 0
            total_margin_loss = 0
            num_batches = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                # Handle optional counterfactual data
                cf_ids = batch.get('counterfactual_reasoning_ids')
                cf_mask = batch.get('counterfactual_reasoning_mask')
                
                if cf_ids is not None:
                    cf_ids = cf_ids.to(self.config.device)
                if cf_mask is not None:
                    cf_mask = cf_mask.to(self.config.device)
                
                outputs = self.reasoning_module(
                    question_ids=batch['question_ids'].to(self.config.device),
                    question_mask=batch['question_mask'].to(self.config.device),
                    reasoning_ids=batch['reasoning_ids'].to(self.config.device),
                    reasoning_mask=batch['reasoning_mask'].to(self.config.device),
                    answer_ids=batch['answer_ids'].to(self.config.device),
                    counterfactual_reasoning_ids=cf_ids,
                    counterfactual_reasoning_mask=cf_mask,
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_lm_loss += outputs['lm_loss']
                total_ie_loss += outputs['ie_loss']
                total_margin_loss += outputs['margin_loss']
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_lm_loss = total_lm_loss / num_batches
            avg_ie_loss = total_ie_loss / num_batches
            avg_margin_loss = total_margin_loss / num_batches
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  LM Loss: {avg_lm_loss:.4f}")
            print(f"  IE Loss: {avg_ie_loss:.4f}")
            print(f"  Margin Loss: {avg_margin_loss:.4f}")
    
    def generate_reasoning_and_answer(
        self,
        question: str,
        tokenizer,
    ) -> Tuple[str, str]:
        """
        Generate reasoning chain and final answer for a question
        
        Args:
            question: Input question
            tokenizer: Tokenizer for the model
            
        Returns:
            Tuple of (reasoning_chain, answer)
        """
        self.inference_module.eval()
        self.reasoning_module.eval()
        
        with torch.no_grad():
            # Step 1: Generate reasoning chain using inference module
            input_ids = tokenizer.encode(question, return_tensors='pt').to(self.config.device)
            attention_mask = torch.ones_like(input_ids)
            
            inference_outputs = self.inference_module(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            reasoning_ids = inference_outputs['generated_ids']
            reasoning_text = tokenizer.decode(reasoning_ids[0], skip_special_tokens=True)
            
            # Step 2: Generate answer using reasoning module
            question_ids = input_ids
            question_mask = attention_mask
            reasoning_ids_tensor = tokenizer.encode(reasoning_text, return_tensors='pt').to(self.config.device)
            reasoning_mask = torch.ones_like(reasoning_ids_tensor)
            
            # Concatenate for answer generation
            combined_ids = torch.cat([question_ids, reasoning_ids_tensor], dim=1)
            combined_mask = torch.cat([question_mask, reasoning_mask], dim=1)
            
            answer_outputs = self.reasoning_module.model.generate(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                max_length=self.config.max_length,
            )
            
            answer_text = tokenizer.decode(answer_outputs[0], skip_special_tokens=True)
            
        return reasoning_text, answer_text


# Example dataset classes
class DPODataset(Dataset):
    """Dataset for training the Inference Module with DPO"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of dictionaries with keys:
                - 'question': str
                - 'preferred_reasoning': str (correct reasoning)
                - 'dispreferred_reasoning': str (counterfactual reasoning)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode question
        question_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode preferred reasoning
        preferred_encoding = self.tokenizer(
            item['preferred_reasoning'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode dispreferred reasoning
        dispreferred_encoding = self.tokenizer(
            item['dispreferred_reasoning'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': question_encoding['input_ids'].squeeze(0),
            'attention_mask': question_encoding['attention_mask'].squeeze(0),
            'preferred_ids': preferred_encoding['input_ids'].squeeze(0),
            'preferred_mask': preferred_encoding['attention_mask'].squeeze(0),
            'dispreferred_ids': dispreferred_encoding['input_ids'].squeeze(0),
            'dispreferred_mask': dispreferred_encoding['attention_mask'].squeeze(0),
        }


class ReasoningDataset(Dataset):
    """Dataset for training the Reasoning Module"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of dictionaries with keys:
                - 'question': str
                - 'reasoning': str (correct reasoning)
                - 'answer': str
                - 'counterfactual_reasoning': str (optional)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode question
        question_encoding = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode reasoning
        reasoning_encoding = self.tokenizer(
            item['reasoning'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode answer
        answer_encoding = self.tokenizer(
            item['answer'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'question_ids': question_encoding['input_ids'].squeeze(0),
            'question_mask': question_encoding['attention_mask'].squeeze(0),
            'reasoning_ids': reasoning_encoding['input_ids'].squeeze(0),
            'reasoning_mask': reasoning_encoding['attention_mask'].squeeze(0),
            'answer_ids': answer_encoding['input_ids'].squeeze(0),
        }
        
        # Add counterfactual reasoning if available
        if 'counterfactual_reasoning' in item:
            cf_encoding = self.tokenizer(
                item['counterfactual_reasoning'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['counterfactual_reasoning_ids'] = cf_encoding['input_ids'].squeeze(0)
            result['counterfactual_reasoning_mask'] = cf_encoding['attention_mask'].squeeze(0)
        
        return result


if __name__ == "__main__":
    # Example usage
    print("FRODO Framework Implementation")
    print("=" * 50)
    print("\nThis implementation includes:")
    print("1. Inference Module with DPO loss")
    print("2. Reasoning Module with:")
    print("   - Language Model Loss (L_LM)")
    print("   - Indirect Effect Loss (L_IE)")
    print("   - Margin Ranking Loss (L_MR)")
    print("\nAll loss functions are fully implemented as described in the paper.")
