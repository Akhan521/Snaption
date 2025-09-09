'''
Test script for Snaption package.
Run all tests to verify the functionality of the package.
'''

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add the snaption package to the path (for local testing)
sys.path.insert(0, '.')

try:
    import snaption
    print("\n✅ Successfully imported snaption package!")
    print(f"Version: {snaption.__version__}\n")
except ImportError as e:
    print(f"\n❌ Failed to import snaption package: {e}\n")
    exit(1)

def test_model_creation():
    '''
    Test creating a model without loading weights.
    '''
    print("\nTesting model creation...")

    # Create a dummy vocabulary mapper.
    dummy_text = ['hello world test']
    tokenizer, vocab_mapper = snaption.build_vocab(dummy_text)

    try:
        model = snaption.ImageCaptioner(
            context_length = 20,
            vocab_size = len(vocab_mapper),
            num_blocks = 2, # Smaller for testing
            model_dim = 64, # Smaller for testing
            num_heads = 4,  # Smaller for testing
            dropout_prob = 0.1
        )
        print("✅ Model created successfully!")
        print(f"   - Vocab size: {len(vocab_mapper)} tokens")
        print(f"   - Model parameters: {sum(p.numel() for p in model.parameters())}")
        return True
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False

def run_all_tests():
    '''Run all tests for the snaption package.'''
    print("Running all tests for the snaption package...")
    print("=" * 50)

    # Our tests:
    tests = [
        ("Model Creation", test_model_creation),
    ]

    results = []
    for test_name, test_function in tests:
        try:
            result = test_function()
            results.append(result is not False)
        except Exception as e:
            print(f"❌ '{test_name}' test failed with exception: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("\nTest Summary:")
    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ Pass" if results[i] else "❌ Fail"
        print(f"    {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed.")

    if passed == total:
        print("🎉 All tests passed successfully! Your package is ready for use.\n")
    else:
        print("⚠️ Some tests failed. Please check the details above.\n")

if __name__ == "__main__":
    run_all_tests()