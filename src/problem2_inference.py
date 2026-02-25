"""
Problem 2: Regression Pipeline
Load trained models and predict range, azimuth, elevation from detections
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from hw import sprint


def _force_xgb_cpu(model):
    """Force XGBoost model to predict on CPU to avoid device mismatch."""
    try:
        booster = getattr(model, "get_booster", None)
        if booster:
            booster().set_param({"device": "cpu"})
        else:
            model.set_param({"device": "cpu"})
    except Exception:
        pass
    return model


class RegressionPipeline:
    """Inference pipeline for regression models"""
    
    def __init__(self, models_dir=None, model_type='xgboost'):
        """
        Initialize regression pipeline
        
        Args:
            models_dir: Directory containing trained models
            model_type: 'random_forest' or 'xgboost'
        """
        if models_dir is None:
            models_dir = str(Path(__file__).resolve().parent.parent / 'models')
        self.models_dir = Path(models_dir)
        self.model_type = model_type
        self.models = {}
        self.metadata = None
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        sprint(f"Loading models from: {self.models_dir}/")
        
        # Load metadata
        metadata_file = self.models_dir / f'metadata_{self.model_type}.pkl'
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.feature_columns = self.metadata['feature_columns']
        self.target_columns = self.metadata['target_columns']
        
        # Load models
        for target in self.target_columns:
            model_file = self.models_dir / f'{target}_{self.model_type}.pkl'
            with open(model_file, 'rb') as f:
                self.models[target] = _force_xgb_cpu(pickle.load(f))
            sprint(f"   Loaded: {model_file.name}")
        
        print(f"   • Features: {len(self.feature_columns)}")
        print(f"   • Targets: {len(self.target_columns)}")
    
    def extract_features(self, cx, cy, bbox_w, bbox_h,
                        image_width=2048, image_height=1364):
        """
        Extract features from detection
        
        Args:
            cx, cy: Center coordinates
            bbox_w, bbox_h: Bounding box dimensions
            image_width, image_height: Image dimensions
            
        Returns:
            dict: Features
        """
        # Normalized features
        cx_norm = cx / image_width
        cy_norm = cy / image_height
        bbox_w_norm = bbox_w / image_width
        bbox_h_norm = bbox_h / image_height
        
        # Additional features
        aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0
        area = bbox_w * bbox_h
        area_norm = area / (image_width * image_height)
        
        # Distance from center
        center_x = image_width / 2
        center_y = image_height / 2
        dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        dist_from_center_norm = dist_from_center / np.sqrt(center_x**2 + center_y**2)
        
        # Angle from center
        angle_from_center = np.degrees(np.arctan2(cy - center_y, cx - center_x))
        
        features = {
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
        
        return features
    
    def predict(self, cx, cy, bbox_w, bbox_h):
        """
        Predict range, azimuth, elevation for a detection
        
        Args:
            cx, cy: Center coordinates
            bbox_w, bbox_h: Bounding box dimensions
            
        Returns:
            dict: {range_m, azimuth_deg, elevation_deg}
        """
        # Extract features
        features = self.extract_features(cx, cy, bbox_w, bbox_h)
        
        # Prepare feature vector
        X = pd.DataFrame([features])[self.feature_columns]
        
        # Predict each target
        predictions = {}
        for target in self.target_columns:
            pred = self.models[target].predict(X)[0]
            predictions[target] = pred
        
        return predictions
    
    def predict_batch(self, detections_df, img_width=None, img_height=None):
        """
        Predict for batch of detections
        
        Args:
            detections_df: DataFrame with columns [frame_id, object_id, center_x, center_y, w, h] (normalized 0-1)
            img_width: Image width in pixels (if None, will try to auto-detect from metadata)
            img_height: Image height in pixels (if None, will try to auto-detect from metadata)
            
        Returns:
            DataFrame with predictions added
        """
        sprint(f"Predicting for {len(detections_df)} detections...")
        
        # Auto-detect or use provided dimensions
        if img_width is None or img_height is None:
            # Try to extract from metadata or use default
            if 'img_width' in detections_df.columns and 'img_height' in detections_df.columns:
                img_width = int(detections_df['img_width'].iloc[0])
                img_height = int(detections_df['img_height'].iloc[0])
                print(f"   • Auto-detected dimensions: {img_width}x{img_height}")
            else:
                # Default fallback (but warn user)
                img_width = 2048
                img_height = 1364
                sprint(f"   Using default dimensions: {img_width}x{img_height}")
                sprint(f"   For accurate results, provide img_width and img_height parameters!")
        else:
            print(f"   • Using provided dimensions: {img_width}x{img_height}")
        
        predictions_list = []
        
        for _, row in detections_df.iterrows():
            # Convert normalized coordinates to pixels
            cx = int(row['center_x'] * img_width)
            cy = int(row['center_y'] * img_height)
            w = int(row['w'] * img_width)
            h = int(row['h'] * img_height)
            
            # Predict
            pred = self.predict(cx, cy, w, h)
            
            predictions_list.append({
                'frame_id': row['frame_id'],
                'object_id': row['object_id'],
                'cx': cx,
                'cy': cy,
                'range_m_pred': round(pred['range_m'], 1),
                'azimuth_deg_pred': round(pred['azimuth_deg'], 1),
                'elevation_deg_pred': round(pred['elevation_deg'], 1)
            })
        
        predictions_df = pd.DataFrame(predictions_list)
        
        sprint(f"   Predictions complete")
        print(f"   • Range: {predictions_df['range_m_pred'].min():.1f} - {predictions_df['range_m_pred'].max():.1f} m")
        print(f"   • Azimuth: {predictions_df['azimuth_deg_pred'].min():.1f} - {predictions_df['azimuth_deg_pred'].max():.1f}°")
        print(f"   • Elevation: {predictions_df['elevation_deg_pred'].min():.1f} - {predictions_df['elevation_deg_pred'].max():.1f}°")
        
        return predictions_df


# Main execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run regression inference')
    parser.add_argument('--detections', type=str, required=True,
                       help='Detections CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output predictions CSV')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['random_forest', 'xgboost'],
                       help='Model type')
    parser.add_argument('--width', type=int, default=None,
                       help='Video width in pixels (auto-detect from video if not provided)')
    parser.add_argument('--height', type=int, default=None,
                       help='Video height in pixels (auto-detect from video if not provided)')
    parser.add_argument('--video', type=str, default=None,
                       help='Video file to auto-detect dimensions')
    
    args = parser.parse_args()
    
    print("="*70)
    sprint("PROBLEM 2: REGRESSION INFERENCE")
    print("="*70)
    
    # Auto-detect dimensions from video if provided
    if args.video and (args.width is None or args.height is None):
        import cv2
        cap = cv2.VideoCapture(args.video)
        args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        sprint(f"Auto-detected dimensions from video: {args.width}x{args.height}")
    
    # Initialize pipeline
    pipeline = RegressionPipeline(model_type=args.model)
    
    # Load detections
    sprint(f"\nLoading detections: {args.detections}")
    detections_df = pd.read_csv(args.detections)
    print(f"   • Detections: {len(detections_df)}")
    
    # Predict
    predictions_df = pipeline.predict_batch(detections_df, img_width=args.width, img_height=args.height)
    
    # Save
    predictions_df.to_csv(args.output, index=False)
    sprint(f"\nSaved predictions: {args.output}")
    
    # Summary
    print(f"\n{'='*70}")
    sprint("INFERENCE COMPLETE")
    print("="*70)
    sprint(f"Prediction Summary:")
    print(f"   • Total predictions: {len(predictions_df)}")
    print(f"   • Output format: frame_id, object_id, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred")
    print("="*70)
