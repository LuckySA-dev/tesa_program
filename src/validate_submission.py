"""
Validation Script for TESA Defence Competition
Calculates Detection Accuracy and MAE for predictions vs ground truth
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from hw import sprint


class SubmissionValidator:
    """Validator for competition submission"""
    
    def __init__(self, pixel_threshold=20):
        """
        Initialize validator
        
        Args:
            pixel_threshold: Detection accuracy threshold in pixels (default: 20)
        """
        self.pixel_threshold = pixel_threshold
        self.results = {}
        
    def load_data(self, predictions_path, ground_truth_path):
        """
        Load predictions and ground truth
        
        Args:
            predictions_path: Path to predictions CSV
            ground_truth_path: Path to ground truth CSV
            
        Returns:
            Tuple of (predictions_df, ground_truth_df)
        """
        sprint(f"📥 Loading data...")
        
        # Load predictions
        pred_df = pd.read_csv(predictions_path)
        print(f"   • Predictions: {len(pred_df)} rows")
        print(f"   • Columns: {list(pred_df.columns)}")
        
        # Load ground truth
        gt_df = pd.read_csv(ground_truth_path)
        print(f"   • Ground truth: {len(gt_df)} rows")
        print(f"   • Columns: {list(gt_df.columns)}")
        
        return pred_df, gt_df
    
    def match_detections(self, pred_df, gt_df):
        """
        Match predictions to ground truth based on frame and proximity
        
        Args:
            pred_df: Predictions DataFrame
            gt_df: Ground truth DataFrame
            
        Returns:
            DataFrame with matched pairs
        """
        sprint(f"\n🔍 Matching detections (threshold: ±{self.pixel_threshold} pixels)...")
        
        matched = []
        unmatched_pred = 0
        unmatched_gt = 0
        
        # Group by video and frame
        for video_id in pred_df['video_id'].unique():
            pred_video = pred_df[pred_df['video_id'] == video_id]
            gt_video = gt_df[gt_df['video_id'] == video_id]
            
            for frame in pred_video['frame'].unique():
                pred_frame = pred_video[pred_video['frame'] == frame]
                gt_frame = gt_video[gt_video['frame'] == frame]
                
                # Match each prediction to closest ground truth
                used_gt_indices = set()
                
                for _, pred_row in pred_frame.iterrows():
                    best_match = None
                    min_dist = float('inf')
                    best_gt_idx = None
                    
                    for gt_idx, gt_row in gt_frame.iterrows():
                        if gt_idx in used_gt_indices:
                            continue
                        
                        # Calculate Euclidean distance
                        dist = np.sqrt(
                            (pred_row['cx'] - gt_row['cx'])**2 +
                            (pred_row['cy'] - gt_row['cy'])**2
                        )
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_match = gt_row
                            best_gt_idx = gt_idx
                    
                    # Check if within threshold
                    if best_match is not None and min_dist <= self.pixel_threshold:
                        used_gt_indices.add(best_gt_idx)
                        matched.append({
                            'video_id': video_id,
                            'frame': frame,
                            'pred_cx': pred_row['cx'],
                            'pred_cy': pred_row['cy'],
                            'gt_cx': best_match['cx'],
                            'gt_cy': best_match['cy'],
                            'pixel_error': min_dist,
                            'pred_range_m': pred_row['range_m_pred'],
                            'gt_range_m': best_match['range_m'],
                            'pred_azimuth_deg': pred_row['azimuth_deg_pred'],
                            'gt_azimuth_deg': best_match['azimuth_deg'],
                            'pred_elevation_deg': pred_row['elevation_deg_pred'],
                            'gt_elevation_deg': best_match['elevation_deg']
                        })
                    else:
                        unmatched_pred += 1
                
                # Count unmatched ground truth
                unmatched_gt += len(gt_frame) - len(used_gt_indices)
        
        matched_df = pd.DataFrame(matched)
        
        sprint(f"   ✅ Matched: {len(matched_df)}")
        sprint(f"   ❌ Unmatched predictions: {unmatched_pred}")
        sprint(f"   ❌ Unmatched ground truth: {unmatched_gt}")
        
        return matched_df, unmatched_pred, unmatched_gt
    
    def calculate_detection_accuracy(self, matched_df, total_pred, total_gt):
        """
        Calculate detection accuracy metrics
        
        Args:
            matched_df: DataFrame with matched detections
            total_pred: Total number of predictions
            total_gt: Total number of ground truth
            
        Returns:
            Dict with accuracy metrics
        """
        sprint(f"\n📊 Detection Accuracy (±{self.pixel_threshold} pixels):")
        
        tp = len(matched_df)  # True Positives
        fp = total_pred - tp   # False Positives
        fn = total_gt - tp     # False Negatives
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Average pixel error
        avg_pixel_error = matched_df['pixel_error'].mean() if len(matched_df) > 0 else 0
        
        accuracy_metrics = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_pixel_error': avg_pixel_error
        }
        
        print(f"   • True Positives: {tp}")
        print(f"   • False Positives: {fp}")
        print(f"   • False Negatives: {fn}")
        print(f"   • Precision: {precision:.3f}")
        print(f"   • Recall: {recall:.3f}")
        print(f"   • F1 Score: {f1:.3f}")
        print(f"   • Avg Pixel Error: {avg_pixel_error:.2f} px")
        
        self.results['detection'] = accuracy_metrics
        return accuracy_metrics
    
    def calculate_mae(self, matched_df):
        """
        Calculate Mean Absolute Error for range, azimuth, elevation
        
        Args:
            matched_df: DataFrame with matched detections
            
        Returns:
            Dict with MAE metrics
        """
        sprint(f"\n📐 Mean Absolute Error (MAE):")
        
        if len(matched_df) == 0:
            sprint("   ⚠️ No matched detections to calculate MAE")
            return {}
        
        # Calculate MAE for each target
        mae_range = np.abs(matched_df['pred_range_m'] - matched_df['gt_range_m']).mean()
        mae_azimuth = np.abs(matched_df['pred_azimuth_deg'] - matched_df['gt_azimuth_deg']).mean()
        mae_elevation = np.abs(matched_df['pred_elevation_deg'] - matched_df['gt_elevation_deg']).mean()
        
        # Calculate RMSE as well
        rmse_range = np.sqrt(((matched_df['pred_range_m'] - matched_df['gt_range_m'])**2).mean())
        rmse_azimuth = np.sqrt(((matched_df['pred_azimuth_deg'] - matched_df['gt_azimuth_deg'])**2).mean())
        rmse_elevation = np.sqrt(((matched_df['pred_elevation_deg'] - matched_df['gt_elevation_deg'])**2).mean())
        
        mae_metrics = {
            'range_mae': mae_range,
            'azimuth_mae': mae_azimuth,
            'elevation_mae': mae_elevation,
            'range_rmse': rmse_range,
            'azimuth_rmse': rmse_azimuth,
            'elevation_rmse': rmse_elevation
        }
        
        sprint(f"   📏 Range:")
        print(f"      • MAE: {mae_range:.3f} m")
        print(f"      • RMSE: {rmse_range:.3f} m")
        
        sprint(f"   🧭 Azimuth:")
        print(f"      • MAE: {mae_azimuth:.3f}°")
        print(f"      • RMSE: {rmse_azimuth:.3f}°")
        
        sprint(f"   📐 Elevation:")
        print(f"      • MAE: {mae_elevation:.3f}°")
        print(f"      • RMSE: {rmse_elevation:.3f}°")
        
        self.results['regression'] = mae_metrics
        return mae_metrics
    
    def calculate_per_video_metrics(self, matched_df):
        """Calculate metrics per video"""
        sprint(f"\n📹 Per-Video Metrics:")
        
        if len(matched_df) == 0:
            return {}
        
        per_video = {}
        
        for video_id in matched_df['video_id'].unique():
            video_data = matched_df[matched_df['video_id'] == video_id]
            
            per_video[video_id] = {
                'detections': len(video_data),
                'avg_pixel_error': video_data['pixel_error'].mean(),
                'range_mae': np.abs(video_data['pred_range_m'] - video_data['gt_range_m']).mean(),
                'azimuth_mae': np.abs(video_data['pred_azimuth_deg'] - video_data['gt_azimuth_deg']).mean(),
                'elevation_mae': np.abs(video_data['pred_elevation_deg'] - video_data['gt_elevation_deg']).mean()
            }
            
            print(f"   {video_id}:")
            print(f"      • Detections: {per_video[video_id]['detections']}")
            print(f"      • Avg Pixel Error: {per_video[video_id]['avg_pixel_error']:.2f} px")
            print(f"      • Range MAE: {per_video[video_id]['range_mae']:.3f} m")
            print(f"      • Azimuth MAE: {per_video[video_id]['azimuth_mae']:.3f}°")
            print(f"      • Elevation MAE: {per_video[video_id]['elevation_mae']:.3f}°")
        
        self.results['per_video'] = per_video
        return per_video
    
    def validate(self, predictions_path, ground_truth_path):
        """
        Run complete validation
        
        Args:
            predictions_path: Path to predictions CSV
            ground_truth_path: Path to ground truth CSV
            
        Returns:
            Dict with all validation results
        """
        print("="*70)
        sprint("🔍 TESA DEFENCE - SUBMISSION VALIDATION")
        print("="*70)
        
        # Load data
        pred_df, gt_df = self.load_data(predictions_path, ground_truth_path)
        
        # Match detections
        matched_df, unmatched_pred, unmatched_gt = self.match_detections(pred_df, gt_df)
        
        # Calculate metrics
        self.calculate_detection_accuracy(matched_df, len(pred_df), len(gt_df))
        self.calculate_mae(matched_df)
        self.calculate_per_video_metrics(matched_df)
        
        # Overall score (example weighted combination)
        if self.results.get('detection') and self.results.get('regression'):
            f1 = self.results['detection']['f1_score']
            mae_avg = (
                self.results['regression']['range_mae'] / 100 +  # Normalize by typical range
                self.results['regression']['azimuth_mae'] / 180 +  # Normalize by max angle
                self.results['regression']['elevation_mae'] / 90   # Normalize by max angle
            ) / 3
            
            overall_score = 0.5 * f1 + 0.5 * (1 - mae_avg)  # Higher is better
            
            print(f"\n{'='*70}")
            sprint(f"🎯 OVERALL SCORE: {overall_score:.3f}")
            print(f"   • Detection F1: {f1:.3f}")
            print(f"   • Normalized MAE: {mae_avg:.3f}")
            print(f"{'='*70}")
            
            self.results['overall_score'] = overall_score
        
        return self.results
    
    def save_report(self, output_path):
        """Save validation report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# TESA Defence - Validation Report\n\n")
            
            # Detection metrics
            if 'detection' in self.results:
                f.write("## Detection Accuracy\n\n")
                for key, value in self.results['detection'].items():
                    f.write(f"- **{key}**: {value:.3f}\n")
                f.write("\n")
            
            # Regression metrics
            if 'regression' in self.results:
                f.write("## Regression MAE\n\n")
                for key, value in self.results['regression'].items():
                    f.write(f"- **{key}**: {value:.3f}\n")
                f.write("\n")
            
            # Per-video metrics
            if 'per_video' in self.results:
                f.write("## Per-Video Metrics\n\n")
                for video_id, metrics in self.results['per_video'].items():
                    f.write(f"### {video_id}\n\n")
                    for key, value in metrics.items():
                        f.write(f"- **{key}**: {value:.3f}\n")
                    f.write("\n")
            
            # Overall score
            if 'overall_score' in self.results:
                f.write(f"## Overall Score: {self.results['overall_score']:.3f}\n")
        
        sprint(f"\n💾 Saved report: {output_path}")


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate submission against ground truth')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth CSV')
    parser.add_argument('--threshold', type=int, default=20,
                       help='Detection accuracy threshold in pixels (default: 20)')
    parser.add_argument('--report', type=str, default=None,
                       help='Save validation report to file')
    
    args = parser.parse_args()
    
    # Validate
    validator = SubmissionValidator(pixel_threshold=args.threshold)
    results = validator.validate(args.predictions, args.ground_truth)
    
    # Save report if requested
    if args.report:
        validator.save_report(args.report)
