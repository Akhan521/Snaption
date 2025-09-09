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


def test_vocab_mapper():
    '''Test vocabulary mapper functionality.'''
    print("\nTesting vocabulary mapper...")

    try:
        sample_captions = [
            "a dog running in the park",
            "a cat sitting on a chair",
            "a bird flying in the sky"
        ]

        tokenizer, vocab_mapper = snaption.build_vocab(sample_captions)

        # Test encoding and decoding.
        test_caption = "a dog sitting"
        tokens = tokenizer(test_caption)
        encoded = vocab_mapper(tokens)
        decoded = vocab_mapper.decode(encoded)

        print("✅ Vocabulary mapper created successfully!")
        print(f"   - Original caption: {test_caption}")
        print(f"   - Tokens: {tokens}")
        print(f"   - Encoded: {encoded}")
        print(f"   - Decoded: {decoded}")
        print(f"   - Vocab size: {len(vocab_mapper)} tokens")
        print(f"   - Special tokens: {vocab_mapper.get_special_tokens()}")
        return True
    except Exception as e:
        print(f"❌ Vocabulary mapper test failed: {e}")
        return False


def test_model_creation(show_header: bool = True):
    '''
    Test creating a model without loading weights.
    '''
    if show_header:
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
        return model, vocab_mapper
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None
    

def test_image_preprocessing():
    '''Test image preprocessing pipeline.'''
    print("\nTesting image preprocessing...")

    try:
        # Create a dummy SnaptionModel (without loading weights).
        snaption_model = snaption.SnaptionModel()

        # Test with a dummy image.
        dummy_image = np.random.randint(low=0, high=255, size=(224, 224, 3), dtype=np.uint8)

        # Test preprocessing.
        processed = snaption_model._preprocess_image(dummy_image)

        print("✅ Image preprocessing successful!")
        print(f"   - Input shape: {dummy_image.shape}")
        print(f"   - Output shape: {processed.shape}")
        print(f"   - Output dtype: {processed.dtype}")
        print(f"   - Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        return True
    except Exception as e:
        print(f"❌ Image preprocessing test failed: {e}")
        return False


def test_model_forward_pass():
    '''Test a forward pass through the model with dummy data.'''
    print("\nTesting model forward pass...")

    # Get a test model and vocab mapper.
    model, vocab_mapper = test_model_creation(show_header = False)
    if model is None:
        return False
    
    try:
        # Create a dummy batch.
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_captions = torch.randint(low=0, high=len(vocab_mapper), size=(batch_size, 10))

        # Forward pass.
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images, dummy_captions)

        print("✅ Model forward pass successful!")
        print(f"   - Input images shape: {dummy_images.shape}")
        print(f"   - Input captions shape: {dummy_captions.shape}")
        print(f"   - Output logits shape: {outputs.shape}")
        print(f"   - Expected output shape: ({batch_size}, 10, {len(vocab_mapper)})")
        return True
    except Exception as e:
        print(f"❌ Model forward pass test failed: {e}")
        return False


def test_generation():
    '''Test caption generation (without trained weights).'''
    print("\nTesting caption generation...")

    # Get a test model and vocab mapper.
    model, vocab_mapper = test_model_creation(show_header = False)
    if model is None:
        return False
    
    try:
        # Create a dummy image.
        dummy_image = torch.randn(1, 3, 224, 224)

        # Generate caption (will be random without trained weights).
        model.eval()
        generated = model.generate(dummy_image, vocab_mapper, max_length = 10)
        caption = vocab_mapper.decode(generated[0])

        print("✅ Caption generation successful!")
        print(f"   - Generated tokens: {generated[0].tolist()}")
        print(f"   - Generated caption: {caption}")
        print("    - Note: Caption is random since model isn't trained.")
        return True
    except Exception as e:
        print(f"❌ Caption generation test failed: {e}")
        return False


def run_all_tests():
    '''Run all tests for the snaption package.'''
    print("Running all tests for the snaption package...")
    print("=" * 50)

    # Our tests:
    tests = [
        ("Vocabulary Mapper", test_vocab_mapper),
        ("Model Creation", test_model_creation),
        ("Image Preprocessing", test_image_preprocessing),
        ("Model Forward Pass", test_model_forward_pass),
        ("Caption Generation", test_generation)
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
        print("\n🎉 All tests passed successfully! Your package is ready for use.\n")
    else:
        print("\n⚠️ Some tests failed. Please check the details above.\n")

if __name__ == "__main__":
    run_all_tests()