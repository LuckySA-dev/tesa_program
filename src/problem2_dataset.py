"""
Problem 2: Create Training Dataset for Regression Models
Prepare features and labels for range, azimuth, elevation prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from hw import sprint


class RegressionDatasetBuilder:
    """Build training dataset for regression models"""
    
    def __init__(self):
        """Initialize dataset builder"""
        self.features_list = []
        self.labels_list = []
        
    def add_detection(self, frame_id, cx, cy, bbox_w, bbox_h, 
                     range_m, azimuth_deg, elevation_deg,
                     image_width=2048, image_height=1364):
        """
        Add a detection with ground truth labels
        
        Args:
            frame_id: Frame number
            cx, cy: Center coordinates (pixels)
            bbox_w, bbox_h: Bounding box dimensions (pixels)
            range_m: Distance from camera (meters) - GROUND TRUTH
            azimuth_deg: Horizontal angle (degrees) - GROUND TRUTH
            elevation_deg: Vertical angle (degrees) - GROUND TRUTH
            image_width, image_height: Image dimensions
        """
        # Calculate normalized features
        cx_norm = cx / image_width
        cy_norm = cy / image_height
        bbox_w_norm = bbox_w / image_width
        bbox_h_norm = bbox_h / image_height
        
        # Additional features
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0
        area = bbox_w * bbox_h
        area_norm = area / (image_width * image_height)
        
        # Distance from center (normalized)
        center_x = image_width / 2
        center_y = image_height / 2
        dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        dist_from_center_norm = dist_from_center / np.sqrt(center_x**2 + center_y**2)
        
        # Angle from center
        angle_from_center = np.degrees(np.arctan2(cy - center_y, cx - center_x))
        
        # Features
        features = {
            'frame_id': frame_id,
            'cx': cx,
            'cy': cy,
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'cx_norm': cx_norm,
            'cy_norm': cy_norm,
            'bbox_w_norm': bbox_w_norm,
            'bbox_h_norm': bbox_h_norm,
            'aspect_ratio': aspect_ratio,
            'area': area,
            'area_norm': area_norm,
            'dist_from_center': dist_from_center,
            'dist_from_center_norm': dist_from_center_norm,
            'angle_from_center': angle_from_center
        }
        
        # Labels
        labels = {
            'range_m': range_m,
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg
        }
        
        self.features_list.append(features)
        self.labels_list.append(labels)
    
    def build_dataset(self, output_path=None):
        """
        Build complete dataset
        
        Args:
            output_path: Path to save CSV (optional)
            
        Returns:
            DataFrame with features and labels
        """
        if not self.features_list:
            raise ValueError("No data added yet!")
        
        # Merge features and labels
        features_df = pd.DataFrame(self.features_list)
        labels_df = pd.DataFrame(self.labels_list)
        
        dataset = pd.concat([features_df, labels_df], axis=1)
        
        if output_path:
            dataset.to_csv(output_path, index=False)
            sprint(f"💾 Saved dataset: {output_path}")
            print(f"   • Total samples: {len(dataset)}")
            print(f"   • Features: {len(features_df.columns)}")
            print(f"   • Labels: {len(labels_df.columns)}")
        
        return dataset
    
    def load_from_ground_truth(self, ground_truth_csv, detections_csv):
        """
        Load dataset from ground truth CSV
        
        Ground truth format:
        video_id, frame, cx, cy, range_m, azimuth_deg, elevation_deg
        
        Detections format:
        frame_id, object_id, bbox_x, bbox_y, bbox_w, bbox_h
        """
        gt_df = pd.read_csv(ground_truth_csv)
        det_df = pd.read_csv(detections_csv)
        
        sprint(f"📂 Loading ground truth: {ground_truth_csv}")
        print(f"   • GT samples: {len(gt_df)}")
        print(f"   • Detection samples: {len(det_df)}")
        
        matched = 0
        unmatched = 0
        
        # Match detections to ground truth
        for _, gt_row in gt_df.iterrows():
            frame = gt_row['frame']
            gt_cx = gt_row['cx']
            gt_cy = gt_row['cy']
            
            # Find closest detection in same frame
            frame_dets = det_df[det_df['frame_id'] == frame]
            
            if len(frame_dets) == 0:
                unmatched += 1
                continue
            
            # Calculate distances
            distances = []
            for _, det_row in frame_dets.iterrows():
                det_cx = det_row['bbox_x'] + det_row['bbox_w'] / 2
                det_cy = det_row['bbox_y'] + det_row['bbox_h'] / 2
                dist = np.sqrt((det_cx - gt_cx)**2 + (det_cy - gt_cy)**2)
                distances.append((dist, det_row))
            
            # Match if within 20 pixels (competition threshold)
            min_dist, matched_det = min(distances, key=lambda x: x[0])
            
            if min_dist <= 20:
                # Add matched detection
                self.add_detection(
                    frame_id=int(matched_det['frame_id']),
                    cx=int(matched_det['bbox_x'] + matched_det['bbox_w'] / 2),
                    cy=int(matched_det['bbox_y'] + matched_det['bbox_h'] / 2),
                    bbox_w=int(matched_det['bbox_w']),
                    bbox_h=int(matched_det['bbox_h']),
                    range_m=float(gt_row['range_m']),
                    azimuth_deg=float(gt_row['azimuth_deg']),
                    elevation_deg=float(gt_row['elevation_deg'])
                )
                matched += 1
            else:
                unmatched += 1
        
        sprint(f"✅ Matched: {matched} samples")
        sprint(f"⚠️  Unmatched: {unmatched} samples")
        
        return matched, unmatched


def create_mock_ground_truth(detections_csv, output_csv, video_name='video_01.mp4'):
    """
    Create mock ground truth for testing
    (In real competition, this will be provided)
    """
    sprint("🔧 Creating mock ground truth for testing...")
    
    # Load detections
    det_df = pd.read_csv(detections_csv)
    
    gt_data = []
    
    for _, row in det_df.iterrows():
        frame = row['frame_id']
        
        # Calculate center from bbox
        cx = int(row['bbox_x'] + row['bbox_w'] / 2)
        cy = int(row['bbox_y'] + row['bbox_h'] / 2)
        
        # Mock range based on bbox size (larger = closer)
        area = row['bbox_w'] * row['bbox_h']
        range_m = 100 - (area / 500)  # Mock: 50-100m
        range_m = max(50, min(100, range_m))  # Clamp
        
        # Mock azimuth based on horizontal position
        # Center = 0°, left = negative, right = positive
        image_center_x = 1024  # 2048/2
        azimuth_deg = ((cx - image_center_x) / image_center_x) * 45  # ±45°
        
        # Mock elevation based on vertical position
        # Center = 0°, top = positive, bottom = negative
        image_center_y = 682  # 1364/2
        elevation_deg = -((cy - image_center_y) / image_center_y) * 30  # ±30°
        
        gt_data.append({
            'video_id': video_name,
            'frame': frame,
            'cx': cx,
            'cy': cy,
            'range_m': round(range_m, 1),
            'azimuth_deg': round(azimuth_deg, 1),
            'elevation_deg': round(elevation_deg, 1)
        })
    
    gt_df = pd.DataFrame(gt_data)
    gt_df.to_csv(output_csv, index=False)
    
    sprint(f"💾 Saved mock ground truth: {output_csv}")
    print(f"   • Samples: {len(gt_df)}")
    print(f"   • Range: {gt_df['range_m'].min():.1f} - {gt_df['range_m'].max():.1f} m")
    print(f"   • Azimuth: {gt_df['azimuth_deg'].min():.1f} - {gt_df['azimuth_deg'].max():.1f}°")
    print(f"   • Elevation: {gt_df['elevation_deg'].min():.1f} - {gt_df['elevation_deg'].max():.1f}°")
    
    return gt_df


# Main execution
if __name__ == '__main__':
    print("="*70)
    sprint("📊 PROBLEM 2: CREATE TRAINING DATASET")
    print("="*70)
    
    # Step 1: Create mock ground truth (in real competition, this is provided)
    sprint("\n🔧 Step 1: Create Mock Ground Truth")
    mock_gt = create_mock_ground_truth(
        detections_csv='problem1_bytetrack.csv',
        output_csv='mock_ground_truth.csv'
    )
    
    # Step 2: Build training dataset
    sprint("\n📦 Step 2: Build Training Dataset")
    builder = RegressionDatasetBuilder()
    
    matched, unmatched = builder.load_from_ground_truth(
        ground_truth_csv='mock_ground_truth.csv',
        detections_csv='problem1_bytetrack.csv'
    )
    
    # Step 3: Save dataset
    sprint("\n💾 Step 3: Save Training Dataset")
    dataset = builder.build_dataset(output_path='training_dataset.csv')
    
    # Step 4: Summary
    print("\n" + "="*70)
    sprint("✅ DATASET CREATION COMPLETE")
    print("="*70)
    sprint(f"📊 Dataset Statistics:")
    print(f"   • Total samples: {len(dataset)}")
    print(f"   • Features: {dataset.shape[1] - 3} (excluding labels)")
    print(f"   • Labels: 3 (range_m, azimuth_deg, elevation_deg)")
    sprint(f"\n📈 Feature Columns:")
    feature_cols = [c for c in dataset.columns if c not in ['range_m', 'azimuth_deg', 'elevation_deg']]
    for col in feature_cols:
        print(f"   • {col}")
    sprint(f"\n🎯 Label Columns:")
    print(f"   • range_m (distance)")
    print(f"   • azimuth_deg (horizontal angle)")
    print(f"   • elevation_deg (vertical angle)")
    print("="*70)
