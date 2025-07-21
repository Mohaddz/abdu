#!/usr/bin/env python3

import sys

def test_dependencies():
    """Test if all required dependencies are available"""
    dependencies = [
        ('torch', 'PyTorch'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy')
    ]
    
    print("Testing core dependencies...")
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} is available")
        except ImportError as e:
            print(f"✗ {name} is not available: {e}")
    
    # Test optional dependencies
    optional_deps = [
        ('sentence_transformers', 'Sentence Transformers'),
        ('lightgbm', 'LightGBM')
    ]
    
    print("\nTesting optional dependencies...")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✓ {name} is available")
        except ImportError as e:
            print(f"⚠ {name} is not available: {e}")
            print(f"   Install with: pip install {module}")
    
    # Test GPU availability
    try:
        import torch
        print(f"\nGPU Status:")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU device: {torch.cuda.get_device_name()}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"✗ Could not check GPU status: {e}")

if __name__ == "__main__":
    test_dependencies()