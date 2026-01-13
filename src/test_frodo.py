"""
Test script to verify FRODO implementation
Tests all loss functions and components
"""

import torch
import numpy as np
from frodo import (
    FRODOConfig,
    InferenceModule,
    ReasoningModule,
    FRODO,
)


class DummyModel(torch.nn.Module):
    """Dummy model for testing"""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        x = self.embedding(input_ids)
        logits = self.output(x.mean(dim=1, keepdim=True).expand(-1, input_ids.size(1), -1))
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        if return_dict:
            return type('obj', (object,), {'loss': loss, 'logits': logits})()
        return loss
    
    def generate(self, input_ids, attention_mask=None, max_length=50, num_return_sequences=1):
        # Simple generation: just return input with some random tokens
        batch_size = input_ids.size(0)
        generated = torch.randint(0, 100, (batch_size, max_length), device=input_ids.device)
        return generated


def test_config():
    """Test configuration"""
    print("Testing FRODOConfig...")
    config = FRODOConfig()
    assert config.beta == 0.1
    assert config.lambda_lm == 1.0
    assert config.lambda_ie == 1.0
    assert config.lambda_margin == 1.0
    print("✓ Config test passed")


def test_inference_module():
    """Test Inference Module and DPO loss"""
    print("\nTesting Inference Module...")
    
    config = FRODOConfig(device='cpu')
    model = DummyModel()
    ref_model = DummyModel()
    
    inference_module = InferenceModule(model, config)
    inference_module.set_reference_model(ref_model)
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    preferred_ids = torch.randint(0, 100, (batch_size, seq_len))
    preferred_mask = torch.ones_like(preferred_ids)
    dispreferred_ids = torch.randint(0, 100, (batch_size, seq_len))
    dispreferred_mask = torch.ones_like(dispreferred_ids)
    
    # Test DPO loss computation
    outputs = inference_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        preferred_ids=preferred_ids,
        preferred_mask=preferred_mask,
        dispreferred_ids=dispreferred_ids,
        dispreferred_mask=dispreferred_mask,
    )
    
    assert 'loss' in outputs
    assert outputs['loss'].requires_grad
    print(f"  DPO Loss: {outputs['loss'].item():.4f}")
    print("✓ Inference Module test passed")


def test_reasoning_module():
    """Test Reasoning Module and all its losses"""
    print("\nTesting Reasoning Module...")
    
    config = FRODOConfig(device='cpu')
    model = DummyModel()
    
    reasoning_module = ReasoningModule(model, config)
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    
    question_ids = torch.randint(0, 100, (batch_size, seq_len))
    question_mask = torch.ones_like(question_ids)
    reasoning_ids = torch.randint(0, 100, (batch_size, seq_len))
    reasoning_mask = torch.ones_like(reasoning_ids)
    answer_ids = torch.randint(0, 100, (batch_size, seq_len))
    counterfactual_ids = torch.randint(0, 100, (batch_size, seq_len))
    counterfactual_mask = torch.ones_like(counterfactual_ids)
    
    # Test all losses
    print("\n  Testing individual loss components:")
    
    # 1. Language Model Loss
    lm_loss = reasoning_module.compute_language_model_loss(
        input_ids=torch.cat([question_ids, reasoning_ids], dim=1),
        attention_mask=torch.cat([question_mask, reasoning_mask], dim=1),
        labels=answer_ids,
    )
    print(f"    L_LM (Language Model Loss): {lm_loss.item():.4f}")
    assert lm_loss.requires_grad
    
    # 2. Indirect Effect Loss
    ie_loss = reasoning_module.compute_indirect_effect_loss(
        question_ids=question_ids,
        question_mask=question_mask,
        reasoning_ids=reasoning_ids,
        reasoning_mask=reasoning_mask,
        answer_ids=answer_ids,
    )
    print(f"    L_IE (Indirect Effect Loss): {ie_loss.item():.4f}")
    assert ie_loss.requires_grad
    
    # 3. Margin Ranking Loss
    margin_loss = reasoning_module.compute_margin_ranking_loss(
        question_ids=question_ids,
        question_mask=question_mask,
        correct_reasoning_ids=reasoning_ids,
        correct_reasoning_mask=reasoning_mask,
        counterfactual_reasoning_ids=counterfactual_ids,
        counterfactual_reasoning_mask=counterfactual_mask,
        answer_ids=answer_ids,
    )
    print(f"    L_MR (Margin Ranking Loss): {margin_loss.item():.4f}")
    assert margin_loss.requires_grad
    
    # Test combined forward pass
    outputs = reasoning_module(
        question_ids=question_ids,
        question_mask=question_mask,
        reasoning_ids=reasoning_ids,
        reasoning_mask=reasoning_mask,
        answer_ids=answer_ids,
        counterfactual_reasoning_ids=counterfactual_ids,
        counterfactual_reasoning_mask=counterfactual_mask,
    )
    
    assert 'loss' in outputs
    assert 'lm_loss' in outputs
    assert 'ie_loss' in outputs
    assert 'margin_loss' in outputs
    
    print(f"\n  Combined Loss: {outputs['loss'].item():.4f}")
    print(f"    = {config.lambda_lm} * {outputs['lm_loss']:.4f} (L_LM)")
    print(f"    + {config.lambda_ie} * {outputs['ie_loss']:.4f} (L_IE)")
    print(f"    + {config.lambda_margin} * {outputs['margin_loss']:.4f} (L_MR)")
    
    # Verify combined loss calculation
    expected_loss = (
        config.lambda_lm * outputs['lm_loss'] +
        config.lambda_ie * outputs['ie_loss'] +
        config.lambda_margin * outputs['margin_loss']
    )
    assert abs(outputs['loss'].item() - expected_loss) < 1e-5
    
    print("✓ Reasoning Module test passed")


def test_frodo_framework():
    """Test complete FRODO framework"""
    print("\nTesting FRODO Framework...")
    
    config = FRODOConfig(device='cpu')
    inference_model = DummyModel()
    reasoning_model = DummyModel()
    
    frodo = FRODO(inference_model, reasoning_model, config)
    
    assert frodo.inference_module is not None
    assert frodo.reasoning_module is not None
    
    print("✓ FRODO Framework test passed")


def test_gradient_flow():
    """Test that gradients flow properly through all losses"""
    print("\nTesting gradient flow...")
    
    config = FRODOConfig(device='cpu')
    
    # Test Inference Module gradients
    model = DummyModel()
    ref_model = DummyModel()
    inference_module = InferenceModule(model, config)
    inference_module.set_reference_model(ref_model)
    
    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    preferred_ids = torch.randint(0, 100, (batch_size, seq_len))
    preferred_mask = torch.ones_like(preferred_ids)
    dispreferred_ids = torch.randint(0, 100, (batch_size, seq_len))
    dispreferred_mask = torch.ones_like(dispreferred_ids)
    
    outputs = inference_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        preferred_ids=preferred_ids,
        preferred_mask=preferred_mask,
        dispreferred_ids=dispreferred_ids,
        dispreferred_mask=dispreferred_mask,
    )
    
    outputs['loss'].backward()
    
    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients found in inference module"
    print("  ✓ Inference module gradients OK")
    
    # Test Reasoning Module gradients
    model = DummyModel()
    reasoning_module = ReasoningModule(model, config)
    
    question_ids = torch.randint(0, 100, (batch_size, seq_len))
    question_mask = torch.ones_like(question_ids)
    reasoning_ids = torch.randint(0, 100, (batch_size, seq_len))
    reasoning_mask = torch.ones_like(reasoning_ids)
    answer_ids = torch.randint(0, 100, (batch_size, seq_len))
    counterfactual_ids = torch.randint(0, 100, (batch_size, seq_len))
    counterfactual_mask = torch.ones_like(counterfactual_ids)
    
    outputs = reasoning_module(
        question_ids=question_ids,
        question_mask=question_mask,
        reasoning_ids=reasoning_ids,
        reasoning_mask=reasoning_mask,
        answer_ids=answer_ids,
        counterfactual_reasoning_ids=counterfactual_ids,
        counterfactual_reasoning_mask=counterfactual_mask,
    )
    
    outputs['loss'].backward()
    
    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients found in reasoning module"
    print("  ✓ Reasoning module gradients OK")
    
    print("✓ Gradient flow test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("FRODO Implementation Tests")
    print("=" * 70)
    
    try:
        test_config()
        test_inference_module()
        test_reasoning_module()
        test_frodo_framework()
        test_gradient_flow()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        
        print("\nImplementation Summary:")
        print("  ✓ DPO Loss (Inference Module)")
        print("  ✓ Language Model Loss (Reasoning Module)")
        print("  ✓ Indirect Effect Loss (Reasoning Module)")
        print("  ✓ Margin Ranking Loss (Reasoning Module)")
        print("  ✓ Combined Preference Loss")
        print("  ✓ Gradient flow through all components")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
