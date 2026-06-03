"""
Quick test script for comprehensive blockage analysis (processes only 3 samples for testing).
"""
import sys
import os

# Import the main analysis class
from comprehensive_blockage_analysis import ComprehensiveBlockageAnalysis

def main():
    """Quick test with limited samples."""
    TEST_DIR = "E:/Thesis Dataset 2/testing"
    TARGET_SHAPE = (128, 128, 64)
    NUM_SAMPLES = 3  # Just 3 samples for quick test
    
    print("="*80)
    print("QUICK TEST: COMPREHENSIVE BLOCKAGE ANALYSIS")
    print("="*80)
    print(f"Processing {NUM_SAMPLES} samples for testing...")
    print("="*80)
    
    # Initialize analysis pipeline
    analyzer = ComprehensiveBlockageAnalysis(
        test_dir=TEST_DIR,
        device='auto',
        target_shape=TARGET_SHAPE
    )
    
    # Run analysis on limited samples
    analyzer.analyze_dataset(num_samples=NUM_SAMPLES, save_individual=True)
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
