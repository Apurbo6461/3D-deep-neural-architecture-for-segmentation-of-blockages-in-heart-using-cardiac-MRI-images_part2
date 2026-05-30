import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_report_image():
    # Data from the analysis
    data = {
        'Model': ['ResAtt-3D-U-Net', '3D U-Net', 'V-Net'],
        'Dice Score': [1.18, 1.25, 0.33],
        'Sensitivity': [0.86, 0.91, 0.99],
        'Specificity': [0.997, 0.997, 0.698],
        'Accuracy': [0.97, 0.97, 0.69]
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4a4a4a')
        elif row == 1: # ResAtt-3D-U-Net row
             cell.set_facecolor('#e6f3ff') # Light blue highlight
             cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f5f5f5')
            
    # Add Title
    plt.title('3D Model Comparison Report (20 Samples)', fontsize=16, weight='bold', pad=20)
    
    # Add Footnote
    plt.figtext(0.5, 0.1, "Comparison of ResNet Attention 3D U-Net against Standard 3D U-Net and V-Net", 
                ha="center", fontsize=10, style='italic')
    
    # Save
    output_path = 'src/model_comparison_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved report image to {output_path}")

if __name__ == "__main__":
    generate_report_image()
