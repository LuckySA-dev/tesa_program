"""
Problem 2 - Fixed Format Converter
Converts predictions to correct format: frame_id, object_id, direction, distance, height
"""

import pandas as pd
import numpy as np
import math
import argparse
from hw import sprint


def azimuth_to_direction(azimuth_deg):
    """
    Convert azimuth to direction (0-360°)
    
    Azimuth convention:
    - Negative = Left (West)
    - Positive = Right (East)
    - 0 = Forward (North)
    
    Direction convention (compass):
    - 0° = North
    - 90° = East
    - 180° = South
    - 270° = West
    
    Args:
        azimuth_deg: Azimuth angle (-180 to +180 or -90 to +90)
        
    Returns:
        direction: Direction in degrees (0-360)
    """
    # Assume camera faces North (0°)
    # Positive azimuth = turn right (East)
    # Negative azimuth = turn left (West)
    direction = (azimuth_deg + 360) % 360
    return direction


def calculate_height(distance_m, elevation_deg):
    """
    Calculate vertical height from distance and elevation angle
    
    Uses trigonometry: height = distance × sin(elevation)
    
    Args:
        distance_m: Horizontal/slant distance to drone (meters)
        elevation_deg: Elevation angle (degrees)
                      Positive = above horizon
                      Negative = below horizon
                      
    Returns:
        height_m: Vertical height above camera level (meters)
    """
    elevation_rad = math.radians(elevation_deg)
    height_m = distance_m * math.sin(elevation_rad)
    return height_m


def convert_to_problem2_format(input_csv, output_csv):
    """
    Convert inference format to Problem 2 format
    
    Input format:
        frame_id, object_id, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
        
    Output format:
        frame_id, object_id, direction, distance, height
    
    Args:
        input_csv: Path to input CSV (e.g., p2_localization_final.csv)
        output_csv: Path to output CSV (e.g., p2_localization_final.csv)
    """
    sprint(f"📥 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"   • Input records: {len(df)}")
    print(f"   • Input columns: {list(df.columns)}")
    
    # Check required columns
    required = ['frame_id', 'object_id', 'range_m_pred', 'azimuth_deg_pred', 'elevation_deg_pred']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Create output DataFrame
    output_df = pd.DataFrame()
    
    # frame_id: use existing frame_id
    output_df['frame_id'] = df['frame_id'].astype(int)
    
    # object_id: use existing object_id
    output_df['object_id'] = df['object_id'].astype(int)
    
    # direction: convert azimuth to 0-360°
    sprint(f"\n🧭 Converting azimuth → direction...")
    output_df['direction'] = df['azimuth_deg_pred'].apply(azimuth_to_direction)
    
    # distance: rename from range_m
    output_df['distance'] = df['range_m_pred']
    
    # height: calculate from distance and elevation
    sprint(f"📐 Calculating height from elevation angle...")
    output_df['height'] = df.apply(
        lambda row: calculate_height(row['range_m_pred'], row['elevation_deg_pred']),
        axis=1
    )
    
    # Round to reasonable precision
    output_df['direction'] = output_df['direction'].round(1)
    output_df['distance'] = output_df['distance'].round(1)
    output_df['height'] = output_df['height'].round(1)
    
    # Save
    output_df.to_csv(output_csv, index=False)
    
    sprint(f"\n✅ Converted to Problem 2 format")
    print(f"   • Output records: {len(output_df)}")
    print(f"   • Output columns: {list(output_df.columns)}")
    print(f"   • Format: frame_id, object_id, direction, distance, height")
    
    # Summary statistics
    sprint(f"\n📊 Statistics:")
    print(f"   • Direction: {output_df['direction'].min():.1f}° - {output_df['direction'].max():.1f}°")
    print(f"   • Distance: {output_df['distance'].min():.1f} - {output_df['distance'].max():.1f} m")
    print(f"   • Height: {output_df['height'].min():.1f} - {output_df['height'].max():.1f} m")
    
    sprint(f"\n💾 Saved: {output_csv}")
    
    # Show sample
    sprint(f"\n📋 Sample (first 5 rows):")
    print(output_df.head())
    
    return output_df


def convert_with_tracking(detection_csv, prediction_csv, output_csv):
    """
    Convert with proper object tracking IDs
    
    This version uses actual track_id from detection results
    
    Args:
        detection_csv: CSV with frame_id, object_id (from problem1)
        prediction_csv: CSV with predictions (from regression)
        output_csv: Output CSV for Problem 2
    """
    sprint(f"📥 Loading detection tracking: {detection_csv}")
    det_df = pd.read_csv(detection_csv)
    
    sprint(f"📥 Loading predictions: {prediction_csv}")
    pred_df = pd.read_csv(prediction_csv)
    
    # Merge on frame_id and bbox position
    # (Assuming detection and prediction are in same order per frame)
    sprint(f"\n🔗 Merging detection IDs with predictions...")
    
    # For simplicity, assume they're aligned by frame order
    output_df = pd.DataFrame()
    output_df['frame_id'] = pred_df['frame']
    output_df['object_id'] = det_df['object_id'].values[:len(pred_df)]
    
    # Convert predictions
    output_df['direction'] = pred_df['azimuth_deg_pred'].apply(azimuth_to_direction).round(1)
    output_df['distance'] = pred_df['range_m_pred'].round(1)
    output_df['height'] = pred_df.apply(
        lambda row: calculate_height(row['range_m_pred'], row['elevation_deg_pred']),
        axis=1
    ).round(1)
    
    # Save
    output_df.to_csv(output_csv, index=False)
    
    sprint(f"\n✅ Converted with tracking IDs")
    print(f"   • Output: {output_csv}")
    print(f"   • Records: {len(output_df)}")
    
    return output_df


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to Problem 2 format')
    parser.add_argument('--input', type=str, default='submissions/submission.csv',
                       help='Input CSV (integration format)')
    parser.add_argument('--output', type=str, default='submissions/p2_localization_final.csv',
                       help='Output CSV (Problem 2 format)')
    parser.add_argument('--detection', type=str, default=None,
                       help='Optional: Detection CSV with track IDs')
    parser.add_argument('--prediction', type=str, default=None,
                       help='Optional: Prediction CSV (if using --detection)')
    
    args = parser.parse_args()
    
    print("="*70)
    sprint("🔧 PROBLEM 2 FORMAT CONVERTER")
    print("="*70)
    print("Conversion:")
    sprint("  • azimuth_deg → direction (0-360°)")
    sprint("  • range_m → distance (m)")
    sprint("  • elevation_deg + distance → height (m)")
    print("="*70)
    
    if args.detection and args.prediction:
        # Use tracking IDs from detection
        convert_with_tracking(args.detection, args.prediction, args.output)
    else:
        # Simple conversion from integration output
        convert_to_problem2_format(args.input, args.output)
    
    print("\n" + "="*70)
    sprint("✅ CONVERSION COMPLETE")
    print("="*70)
