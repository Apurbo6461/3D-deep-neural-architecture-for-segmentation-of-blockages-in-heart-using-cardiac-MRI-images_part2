"""
Efficiency Analysis Module
Measures and reports efficiency metrics for the 3D deep neural architectures.
This justifies the "efficient" part of the thesis title.
"""
import torch
import numpy as np
import time
import os
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠ psutil not available. Memory monitoring will be limited.")

from data.dataset import MedicalDataset
from models.unet3d_standard import UNet3D
from models.vnet import VNet3D
from models.unet3d import ResAttUNet3D

class EfficiencyAnalyzer:
    """
    Analyzes efficiency metrics for 3D deep neural architectures.
    """
    
    def __init__(self, device='auto'):
        """Initialize efficiency analyzer."""
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Efficiency Analysis using device: {self.device}")
    
    def count_parameters(self, model):
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_parameters_millions': total_params / 1e6,
            'trainable_parameters_millions': trainable_params / 1e6
        }
    
    def estimate_model_size(self, model):
        """Estimate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return {
            'model_size_mb': size_all_mb,
            'model_size_gb': size_all_mb / 1024
        }
    
    def measure_inference_time(self, model, test_loader, num_samples=10, warmup=3):
        """Measure average inference time per sample."""
        model = model.to(self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                if i >= warmup:
                    break
                imgs = imgs.to(self.device)
                _ = model(imgs)
        
        # Synchronize if using GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elif self.device == 'mps':
            torch.mps.synchronize()
        
        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                imgs = imgs.to(self.device)
                
                # Measure time
                start_time = time.time()
                _ = model(imgs)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elif self.device == 'mps':
                    torch.mps.synchronize()
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'samples_per_second': 1.0 / np.mean(inference_times),
            'inference_times': inference_times
        }
    
    def measure_memory_usage(self, model, test_loader, num_samples=5):
        """Measure peak memory usage during inference."""
        model = model.to(self.device)
        model.eval()
        
        # Get initial memory
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024**2  # MB
        else:
            initial_memory = 0
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        elif self.device == 'mps':
            # MPS doesn't have detailed memory tracking
            initial_gpu_memory = 0
        else:
            initial_gpu_memory = 0
        
        peak_memory = initial_memory
        peak_gpu_memory = initial_gpu_memory
        
        with torch.no_grad():
            for i, (imgs, _) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                imgs = imgs.to(self.device)
                _ = model(imgs)
                
                # Check memory
                if PSUTIL_AVAILABLE:
                    current_memory = process.memory_info().rss / 1024**2
                    peak_memory = max(peak_memory, current_memory)
                
                if self.device == 'cuda':
                    current_gpu = torch.cuda.memory_allocated() / 1024**2
                    peak_gpu_memory = max(peak_gpu_memory, current_gpu)
        
        if not PSUTIL_AVAILABLE:
            peak_memory = 0
            initial_memory = 0
        
        return {
            'peak_cpu_memory_mb': peak_memory,
            'peak_gpu_memory_mb': peak_gpu_memory if self.device != 'cpu' else 0,
            'memory_overhead_mb': peak_memory - initial_memory if PSUTIL_AVAILABLE else 0
        }
    
    def calculate_flops(self, model, input_shape=(1, 1, 128, 128, 64)):
        """Estimate FLOPs (Floating Point Operations) for the model."""
        # Simple estimation based on architecture
        # This is a rough estimate - for exact FLOPs, use tools like thop or fvcore
        
        def count_conv3d_flops(layer, input_shape):
            """Count FLOPs for a 3D convolution."""
            if not isinstance(layer, torch.nn.Conv3d):
                return 0
            
            kernel_size = layer.kernel_size
            out_channels = layer.out_channels
            in_channels = layer.in_channels
            
            # Output shape
            output_size = np.prod(input_shape[2:])  # H * W * D
            
            # FLOPs = kernel_ops * output_elements
            kernel_ops = np.prod(kernel_size) * in_channels
            flops = kernel_ops * out_channels * output_size
            
            return flops
        
        # Rough estimation: count major operations
        total_flops = 0
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # This is a simplified estimation
        # For accurate FLOPs, consider using thop or fvcore library
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Estimate based on model architecture
        # This is a placeholder - actual FLOP counting requires more sophisticated tools
        # For now, we'll provide a relative estimate
        
        return {
            'estimated_flops_g': 'N/A - Use thop/fvcore for exact count',
            'note': 'FLOP counting requires specialized tools. This is a placeholder.'
        }
    
    def analyze_model_efficiency(self, model, model_name, test_loader):
        """Comprehensive efficiency analysis for a model."""
        print(f"\nAnalyzing efficiency for {model_name}...")
        
        # Parameter count
        param_info = self.count_parameters(model)
        print(f"  Parameters: {param_info['total_parameters_millions']:.2f}M")
        
        # Model size
        size_info = self.estimate_model_size(model)
        print(f"  Model Size: {size_info['model_size_mb']:.2f} MB")
        
        # Inference time
        inference_info = self.measure_inference_time(model, test_loader, num_samples=10)
        print(f"  Inference Time: {inference_info['mean_inference_time']*1000:.2f} ms ± {inference_info['std_inference_time']*1000:.2f} ms")
        print(f"  Throughput: {inference_info['samples_per_second']:.2f} samples/sec")
        
        # Memory usage
        memory_info = self.measure_memory_usage(model, test_loader, num_samples=5)
        print(f"  Peak CPU Memory: {memory_info['peak_cpu_memory_mb']:.2f} MB")
        if memory_info['peak_gpu_memory_mb'] > 0:
            print(f"  Peak GPU Memory: {memory_info['peak_gpu_memory_mb']:.2f} MB")
        
        # Combine all metrics
        efficiency_metrics = {
            'model_name': model_name,
            **param_info,
            **size_info,
            **inference_info,
            **memory_info
        }
        
        return efficiency_metrics
    
    def compare_models_efficiency(self, models_config, test_dir, target_shape=(128, 128, 64)):
        """Compare efficiency across all models."""
        print("="*80)
        print("EFFICIENCY ANALYSIS FOR 3D DEEP NEURAL ARCHITECTURES")
        print("="*80)
        
        # Load test dataset
        test_dataset = MedicalDataset(root_dir=test_dir, target_shape=target_shape)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        all_metrics = []
        
        for model_name, config in models_config.items():
            try:
                model = config['model']
                model_path = config.get('path', None)
                
                # Load weights if available
                if model_path and os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                # Analyze efficiency
                metrics = self.analyze_model_efficiency(model, model_name, test_loader)
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"  ⚠ Error analyzing {model_name}: {e}")
                continue
        
        return all_metrics
    
    def generate_efficiency_report(self, efficiency_metrics, output_dir='.'):
        """Generate efficiency comparison report and visualizations."""
        if not efficiency_metrics:
            print("No efficiency metrics to report.")
            return
        
        df = pd.DataFrame(efficiency_metrics)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'efficiency_analysis_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved efficiency results to {csv_path}")
        
        # Save JSON
        json_path = os.path.join(output_dir, 'efficiency_analysis_results.json')
        with open(json_path, 'w') as f:
            json.dump(efficiency_metrics, f, indent=2)
        print(f"✓ Saved efficiency results to {json_path}")
        
        # Create visualizations
        self._create_efficiency_visualizations(efficiency_metrics, output_dir)
    
    def _create_efficiency_visualizations(self, efficiency_metrics, output_dir):
        """Create efficiency comparison visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = [m['model_name'] for m in efficiency_metrics]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 1. Model Size
        model_sizes = [m['model_size_mb'] for m in efficiency_metrics]
        axes[0, 0].bar(model_names, model_sizes, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Model Size Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Model Size (MB)', fontsize=11)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, size in enumerate(model_sizes):
            axes[0, 0].text(i, size + 0.5, f'{size:.1f} MB', ha='center', fontweight='bold')
        
        # 2. Parameter Count
        param_counts = [m['total_parameters_millions'] for m in efficiency_metrics]
        axes[0, 1].bar(model_names, param_counts, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Parameter Count Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Parameters (Millions)', fontsize=11)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, count in enumerate(param_counts):
            axes[0, 1].text(i, count + 0.1, f'{count:.2f}M', ha='center', fontweight='bold')
        
        # 3. Inference Time
        inference_times = [m['mean_inference_time'] * 1000 for m in efficiency_metrics]  # Convert to ms
        inference_stds = [m['std_inference_time'] * 1000 for m in efficiency_metrics]
        axes[0, 2].bar(model_names, inference_times, yerr=inference_stds, capsize=5,
                      color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Inference Time (ms)', fontsize=11)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        for i, (time, std) in enumerate(zip(inference_times, inference_stds)):
            axes[0, 2].text(i, time + std + 2, f'{time:.1f} ms', ha='center', fontweight='bold')
        
        # 4. Throughput
        throughputs = [m['samples_per_second'] for m in efficiency_metrics]
        axes[1, 0].bar(model_names, throughputs, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Throughput Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Samples per Second', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, throughput in enumerate(throughputs):
            axes[1, 0].text(i, throughput + 0.01, f'{throughput:.2f}', ha='center', fontweight='bold')
        
        # 5. CPU Memory
        cpu_memories = [m['peak_cpu_memory_mb'] for m in efficiency_metrics]
        axes[1, 1].bar(model_names, cpu_memories, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Peak CPU Memory Usage', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Memory (MB)', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, mem in enumerate(cpu_memories):
            axes[1, 1].text(i, mem + 5, f'{mem:.1f} MB', ha='center', fontweight='bold')
        
        # 6. GPU Memory (if available)
        gpu_memories = [m.get('peak_gpu_memory_mb', 0) for m in efficiency_metrics]
        if any(gpu_memories):
            axes[1, 2].bar(model_names, gpu_memories, color=colors[:len(model_names)], alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('Peak GPU Memory Usage', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Memory (MB)', fontsize=11)
            axes[1, 2].grid(True, alpha=0.3, axis='y')
            for i, mem in enumerate(gpu_memories):
                if mem > 0:
                    axes[1, 2].text(i, mem + 5, f'{mem:.1f} MB', ha='center', fontweight='bold')
        else:
            axes[1, 2].text(0.5, 0.5, 'GPU Memory\nNot Available', 
                           ha='center', va='center', fontsize=12,
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Peak GPU Memory Usage', fontsize=12, fontweight='bold')
        
        plt.suptitle('Efficiency Analysis: 3D Deep Neural Architectures\n'
                     'Model Size, Inference Time, and Memory Usage Comparison',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'efficiency_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved efficiency comparison graph to efficiency_comparison.png")


def main():
    """Main function to run efficiency analysis."""
    TEST_DIR = "E:\Thesis Dataset 2\testing"
    TARGET_SHAPE = (128, 128, 64)
    
    analyzer = EfficiencyAnalyzer(device='auto')
    
    models_config = {
        '3D U-Net': {
            'model': UNet3D(in_channels=1, out_channels=1, base_filters=32),
            'path': 'best_3d_u-net.pth'
        },
        'V-Net': {
            'model': VNet3D(in_channels=1, out_channels=1, base_filters=16),
            'path': 'best_v-net.pth'
        },
        'ResAtt-3D-U-Net': {
            'model': ResAttUNet3D(in_channels=1, out_channels=1, base_filters=16),
            'path': 'best_resatt_3d_u_net.pth'
        }
    }
    
    # Compare efficiency
    efficiency_metrics = analyzer.compare_models_efficiency(
        models_config, TEST_DIR, TARGET_SHAPE
    )
    
    # Generate report
    analyzer.generate_efficiency_report(efficiency_metrics, output_dir='.')
    
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
