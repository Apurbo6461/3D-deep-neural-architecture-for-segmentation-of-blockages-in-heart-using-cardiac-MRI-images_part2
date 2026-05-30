"""
Anatomical Region Identifier for Cardiac MRI
Identifies which part of the heart (LV, RV, Myocardial) contains blockages
based on spatial, morphological, and intensity features.
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, center_of_mass

class AnatomicalRegionIdentifier:
    """
    Identifies anatomical regions in cardiac MRI based on spatial and morphological features.
    """
    
    def __init__(self):
        """Initialize the region identifier."""
        self.region_names = {
            'LV': 'Left Ventricle',
            'RV': 'Right Ventricle', 
            'MYOCARDIAL': 'Myocardium',
            'UNKNOWN': 'Unknown Region'
        }
    
    def identify_regions(self, segmentation_mask, original_image=None):
        """
        Identify anatomical regions in the segmentation mask.
        
        Args:
            segmentation_mask: Binary segmentation mask (H, W, D)
            original_image: Original intensity image (H, W, D) for intensity-based analysis
        
        Returns:
            dict: Region masks and labels
        """
        # Convert to numpy if needed
        if hasattr(segmentation_mask, 'cpu'):
            mask = segmentation_mask.cpu().numpy()
        else:
            mask = segmentation_mask.copy()
        
        # Ensure 3D
        if len(mask.shape) == 4:
            mask = mask[0] if mask.shape[0] == 1 else mask[0]
        if len(mask.shape) == 3:
            pass  # Good
        else:
            raise ValueError(f"Expected 3D mask, got shape {mask.shape}")
        
        # Binary mask
        binary_mask = (mask > 0).astype(np.float32)
        
        # Get image if available
        img = None
        if original_image is not None:
            if hasattr(original_image, 'cpu'):
                img = original_image.cpu().numpy()
            else:
                img = original_image.copy()
            if len(img.shape) == 4:
                img = img[0] if img.shape[0] == 1 else img[0]
        
        # Identify regions using multiple methods
        regions = self._identify_heart_chambers(binary_mask, img)
        
        return regions
    
    def _identify_heart_chambers(self, binary_mask, original_image=None):
        """
        Identify LV, RV, and Myocardial regions using spatial and morphological analysis.
        """
        h, w, d = binary_mask.shape
        
        # Check if mask is empty
        if np.sum(binary_mask) == 0:
            regions = {
                'LV': np.zeros_like(binary_mask),
                'RV': np.zeros_like(binary_mask),
                'MYOCARDIAL': np.zeros_like(binary_mask),
                'UNKNOWN': np.zeros_like(binary_mask)
            }
            return regions
        
        # Get center of mass of entire heart
        com = center_of_mass(binary_mask)
        # Check for NaN values (can occur with empty regions)
        if np.isnan(com[0]) or np.isnan(com[1]) or np.isnan(com[2]):
            # Use center of volume as fallback
            center_x, center_y, center_z = h // 2, w // 2, d // 2
        else:
            center_x, center_y, center_z = int(com[0]), int(com[1]), int(com[2])
        
        # Split into regions based on spatial location and morphology
        regions = {
            'LV': np.zeros_like(binary_mask),
            'RV': np.zeros_like(binary_mask),
            'MYOCARDIAL': np.zeros_like(binary_mask),
            'UNKNOWN': np.zeros_like(binary_mask)
        }
        
        # Method 1: Spatial partitioning based on anatomical knowledge
        # In cardiac MRI, LV is typically left and central, RV is right and anterior
        # Myocardium surrounds the LV
        
        # Find connected components
        labeled_mask, num_components = label(binary_mask > 0)
        
        if num_components == 0:
            return regions
        
        # Analyze each component
        component_regions = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_mask == comp_id).astype(np.float32)
            comp_size = np.sum(comp_mask)
            
            # Get component center of mass
            comp_com = center_of_mass(comp_mask)
            # Check for NaN values
            if np.isnan(comp_com[0]) or np.isnan(comp_com[1]) or np.isnan(comp_com[2]):
                # Skip this component if center of mass is invalid
                continue
            comp_x, comp_y, comp_z = comp_com[0], comp_com[1], comp_com[2]
            
            # Calculate spatial features
            # Relative position to heart center
            rel_x = (comp_x - center_x) / (h + 1e-8)
            rel_y = (comp_y - center_y) / (w + 1e-8)
            rel_z = (comp_z - center_z) / (d + 1e-8)
            
            # Morphological features
            # Compactness (how round/compact)
            volume = comp_size
            surface_area = self._calculate_surface_area(comp_mask)
            compactness = (36 * np.pi * volume**2) / (surface_area**3 + 1e-8)
            
            # Elongation (aspect ratio)
            bbox = self._get_bounding_box(comp_mask)
            bbox_size = [bbox[i].stop - bbox[i].start for i in range(3)]
            max_dim = max(bbox_size)
            min_dim = min(bbox_size)
            elongation = max_dim / (min_dim + 1e-8)
            
            # Distance transform features
            distance_map = ndimage.distance_transform_edt(comp_mask)
            max_distance = distance_map.max()
            mean_distance = distance_map[comp_mask > 0].mean()
            
            # Intensity features (if image available)
            mean_intensity = 0.5  # Default
            if original_image is not None:
                mean_intensity = original_image[comp_mask > 0].mean()
            
            # Classify component
            region_type = self._classify_component(
                comp_size, rel_x, rel_y, rel_z, compactness, 
                elongation, max_distance, mean_distance, mean_intensity,
                h, w, d
            )
            
            regions[region_type] = np.maximum(regions[region_type], comp_mask)
            component_regions.append({
                'mask': comp_mask,
                'type': region_type,
                'size': comp_size,
                'center': (comp_x, comp_y, comp_z)
            })
        
        # Post-processing: Refine regions
        regions = self._refine_regions(regions, binary_mask, original_image)
        
        return regions
    
    def _classify_component(self, size, rel_x, rel_y, rel_z, compactness, 
                           elongation, max_distance, mean_distance, mean_intensity,
                           h, w, d):
        """
        Classify a component into LV, RV, or Myocardial.
        """
        # Size thresholds (relative to volume)
        total_volume = h * w * d
        size_ratio = size / total_volume
        
        # Large chambers: LV and RV
        if size_ratio > 0.05:
            # LV: Typically left side (negative rel_x in some conventions, or positive)
            # RV: Typically right side
            # In standard cardiac views, LV is often more central/left
            if rel_x < 0.4:  # Left side
                return 'LV'
            elif rel_x > 0.6:  # Right side
                return 'RV'
            else:
                # Central region - could be either, use compactness
                if compactness > 0.3:
                    return 'LV'  # LV is typically more compact
                else:
                    return 'RV'
        
        # Medium-sized regions: Myocardium
        if size_ratio > 0.01 and size_ratio < 0.05:
            # Myocardium surrounds LV, has medium compactness
            if 0.1 < compactness < 0.5:
                return 'MYOCARDIAL'
        
        # Default to UNKNOWN
        return 'UNKNOWN'
    
    def _calculate_surface_area(self, mask):
        """Calculate surface area of a 3D mask."""
        # Use morphological gradient to find surface
        kernel = np.ones((3, 3, 3))
        dilated = ndimage.binary_dilation(mask > 0, structure=kernel)
        surface = dilated & (~(mask > 0))
        return np.sum(surface)
    
    def _get_bounding_box(self, mask):
        """Get bounding box of a mask."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return [slice(0, 1), slice(0, 1), slice(0, 1)]
        return [slice(coords[i].min(), coords[i].max() + 1) for i in range(3)]
    
    def _refine_regions(self, regions, binary_mask, original_image=None):
        """Refine region assignments using additional heuristics."""
        # Ensure all foreground is assigned
        assigned = sum([regions[k] for k in regions.keys()])
        unassigned = binary_mask - assigned
        unassigned = np.clip(unassigned, 0, 1)
        
        # Assign unassigned regions to nearest known region
        if np.sum(unassigned) > 0:
            # Find nearest region for each unassigned voxel
            for region_type in ['LV', 'RV', 'MYOCARDIAL']:
                if np.sum(regions[region_type]) > 0:
                    # Use distance transform to assign unassigned voxels
                    distance_to_region = ndimage.distance_transform_edt(1 - regions[region_type])
                    # Assign if close enough
                    close_voxels = (distance_to_region < 5) & (unassigned > 0)
                    regions[region_type] = np.maximum(regions[region_type], close_voxels.astype(np.float32))
                    unassigned = unassigned - close_voxels.astype(np.float32)
                    unassigned = np.clip(unassigned, 0, 1)
        
        # Remaining unassigned goes to UNKNOWN
        regions['UNKNOWN'] = np.maximum(regions['UNKNOWN'], unassigned)
        
        return regions
    
    def identify_blockage_regions(self, blockage_mask, anatomical_regions):
        """
        Identify which anatomical region(s) contain blockages.
        
        Args:
            blockage_mask: Binary mask of detected blockages (H, W, D)
            anatomical_regions: Dict of region masks from identify_regions()
        
        Returns:
            dict: Blockage information per region
        """
        blockage_info = {}
        
        for region_name, region_mask in anatomical_regions.items():
            if region_name == 'UNKNOWN':
                continue
            
            # Find overlap between blockages and this region
            overlap = (blockage_mask > 0) & (region_mask > 0)
            blockage_in_region = np.sum(overlap)
            total_blockage = np.sum(blockage_mask > 0)
            region_size = np.sum(region_mask > 0)
            
            if blockage_in_region > 0:
                blockage_rate = (blockage_in_region / region_size * 100) if region_size > 0 else 0
                blockage_percentage = (blockage_in_region / total_blockage * 100) if total_blockage > 0 else 0
                
                blockage_info[region_name] = {
                    'has_blockage': True,
                    'blockage_voxels': int(blockage_in_region),
                    'blockage_rate': blockage_rate,
                    'blockage_percentage': blockage_percentage,
                    'region_size': int(region_size)
                }
            else:
                blockage_info[region_name] = {
                    'has_blockage': False,
                    'blockage_voxels': 0,
                    'blockage_rate': 0.0,
                    'blockage_percentage': 0.0,
                    'region_size': int(region_size)
                }
        
        return blockage_info
