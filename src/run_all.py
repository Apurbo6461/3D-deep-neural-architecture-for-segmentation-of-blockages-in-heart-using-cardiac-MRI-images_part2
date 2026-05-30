"""
Master script that runs everything and displays all results.
Uses 20 samples for evaluation and comprehensive analysis.
"""
import os
import sys

# Run from script directory so all outputs go to src/
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_script_dir)
sys.path.insert(0, _script_dir)

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def main():
    print_header("HEART BLOCKAGE DETECTION - COMPLETE PIPELINE (20 SAMPLES)")
    
    # Step 1: Comprehensive blockage analysis (20 samples)
    print_header("STEP 1: COMPREHENSIVE BLOCKAGE ANALYSIS (20 samples)")
    try:
        from comprehensive_blockage_analysis import main as comprehensive_main
        comprehensive_main()
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Evaluation (20 samples) - updates evaluation_results and model_comparison
    print_header("STEP 2: EVALUATION (20 samples)")
    try:
        from evaluate_models import main as evaluate_main
        evaluate_main()
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Bar charts from evaluation results
    print_header("STEP 3: BAR CHARTS (accuracy_comparison, blockage_rate_comparison)")
    try:
        from quick_run import main as quick_main
        quick_main()
    except Exception as e:
        print(f"Error in quick_run: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Create 3D visualizations
    print_header("STEP 4: CREATING 3D VISUALIZATIONS")
    try:
        from create_3d_visualizations import main as viz3d_main
        viz3d_main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Blockage detection demo (3 models × 5 samples each)
    print_header("STEP 5: BLOCKAGE DETECTION (3 models × 5 samples each)")
    try:
        from blockage_detection_demo import run_blockage_detection_demo
        run_blockage_detection_demo()
    except Exception as e:
        print(f"Error in blockage detection demo: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    # List all generated files
    files_to_check = [
        "comprehensive_analysis_results.json",
        "comprehensive_analysis_results.csv",
        "accuracy_comparison_comprehensive.png",
        "blockage_region_analysis.png",
        "evaluation_results.csv",
        "evaluation_results.json",
        "model_comparison.png",
        "accuracy_comparison.png",
        "blockage_rate_comparison.png",
        "3d_blockage_detection_3d_u_net_sample1.png",
        "3d_blockage_detection_v_net_sample2.png",
        "3d_blockage_detection_resatt_3d_u_net_sample3.png",
        "blockage_detection_results.json",
        "blockage_detection_results.csv",
        "blockage_detection_comparison.png",
        "blockage_detection_3d_u_net_sample1.png",
        "blockage_detection_v_net_sample1.png",
        "blockage_detection_resatt_3d_u_net_sample1.png"
    ]
    
    print("\nGenerated Files:")
    print("-" * 70)
    
    for filename in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename) / 1024  # KB
            print(f"  ✓ {filename:<50} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename:<50} (not found)")
    
    # Display results summary
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    
    try:
        import json
        with open("evaluation_results.json", "r") as f:
            results = json.load(f)
        
        print(f"\n{'Model':<25} {'Dice':<8} {'IoU':<8} {'Accuracy':<10} {'Blockage Rate':<15} {'Blockage F1':<12}")
        print("-" * 70)
        for r in results:
            print(f"{r['model_name']:<25} {r['dice_score']:<8.3f} {r['iou_score']:<8.3f} "
                  f"{r['accuracy']:<10.3f} {r['mean_blockage_rate']:<15.2f}% {r['blockage_detection_f1']:<12.3f}")
        
        # Find best model
        best_dice = max(results, key=lambda x: x['dice_score'])
        best_blockage = max(results, key=lambda x: x['blockage_detection_f1'])
        
        print("\n" + "-" * 70)
        print("BEST PERFORMING MODELS:")
        print("-" * 70)
        print(f"  Best Dice Score: {best_dice['model_name']} ({best_dice['dice_score']:.3f})")
        print(f"  Best Blockage Detection: {best_blockage['model_name']} (F1: {best_blockage['blockage_detection_f1']:.3f})")
        
    except Exception as e:
        print(f"Could not load results: {e}")
    
    print_header("ALL VISUALIZATIONS COMPLETE!")
    print("\nYou can now view:")
    print("  1. accuracy_comparison.png - Bar charts comparing Dice, IoU, and Accuracy")
    print("  2. blockage_rate_comparison.png - Bar charts for blockage detection metrics")
    print("  3. 3d_blockage_detection_*.png - 3D visualizations showing detected blockages")
    print("  4. blockage_detection_<model>_sample<N>.png - Blockage detection (3 models × 5 samples)")
    print("  5. blockage_detection_comparison.png - Blockage metrics comparison across models")
    print("\nAll files are saved in the current directory.")

if __name__ == "__main__":
    main()
