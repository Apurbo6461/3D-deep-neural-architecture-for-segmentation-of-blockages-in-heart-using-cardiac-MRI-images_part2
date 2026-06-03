import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Heart Blockage Detection with 3D Models')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], 
                       default='both', help='Mode: train, evaluate, or both')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("="*60)
        print("TRAINING MODELS")
        print("="*60)
        from train_models import main as train_main
        train_main()
    
    if args.mode in ['evaluate', 'both']:
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        from evaluate_models import main as eval_main
        eval_main()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nResults saved in:")
    print("  - evaluation_results.csv")
    print("  - evaluation_results.json")
    print("  - model_comparison.png")
    print("  - training_comparison.png")

if __name__ == "__main__":
    main()
