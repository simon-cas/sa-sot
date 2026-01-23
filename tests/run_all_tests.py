#!/usr/bin/env python3
"""Run all unit tests"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import importlib.util

def run_test_file(test_file):
    """Run a test file and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}...")
    print('='*60)
    
    try:
        # Execute test file directly (it will print its own output)
        spec = importlib.util.spec_from_file_location("test_module", test_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    test_dir = os.path.dirname(__file__)
    test_files = [
        'test_sasot_model.py',
        'test_decoder_forward.py',
        'test_cif.py',
        'test_sat.py',
        'test_speaker_amsoftmax.py',
        'test_speaker_loss.py',
        'test_speaker_pooling.py',
        'test_self_attention.py',
        'test_t_sot.py',
    ]
    
    results = []
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            success = run_test_file(test_path)
            results.append((test_file, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_file, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_file, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_file}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

