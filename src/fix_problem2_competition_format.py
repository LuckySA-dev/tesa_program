"""
Fix Problem 2 CSV Format
Convert regression output to competition format: direction, distance, height
"""

import pandas as pd
import numpy as np
from pathlib import Path
from hw import sprint


def convert_to_competition_format(input_csv: str, output_csv: str):
    """
    Convert Problem 2 predictions to competition format
    
    Input format:
        frame_id, object_id, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
    
    Output format:
        frame_id, object_id, direction, distance, height
        - direction: azimuth in degrees (0-360)
        - distance: range in meters
        - height: calculated from range and elevation
    """
    print("="*70)
    sprint("🔧 CONVERTING PROBLEM 2 TO COMPETITION FORMAT")
    print("="*70)
    
    # Read input
    sprint(f"\n📂 Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   • Records: {len(df)}")
    print(f"   • Columns: {list(df.columns)}")
    
    # Check required columns
    required = ['frame_id', 'object_id', 'range_m_pred', 'azimuth_deg_pred', 'elevation_deg_pred']
    missing = [col for col in required if col not in df.columns]
    if missing:
        sprint(f"   ❌ Missing columns: {missing}")
        return
    
    # Convert to competition format
    sprint("\n🔄 Converting format...")
    
    # 1. direction = azimuth (convert to 0-360 range if needed)
    df['direction'] = df['azimuth_deg_pred'].apply(lambda x: x if x >= 0 else x + 360)
    
    # 2. distance = range (already in meters)
    df['distance'] = df['range_m_pred']
    
    # 3. height = range * sin(elevation)
    # Convert elevation to radians
    elevation_rad = np.radians(df['elevation_deg_pred'])
    df['height'] = df['range_m_pred'] * np.sin(elevation_rad)
    
    # Select only required columns
    output_df = df[['frame_id', 'object_id', 'direction', 'distance', 'height']].copy()
    
    # Round to reasonable precision
    output_df['direction'] = output_df['direction'].round(2)
    output_df['distance'] = output_df['distance'].round(2)
    output_df['height'] = output_df['height'].round(2)
    
    # Save
    Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    output_df.to_csv(output_csv, index=False)
    
    sprint(f"\n✅ Conversion complete!")
    print(f"   • Output: {output_csv}")
    print(f"   • Records: {len(output_df)}")
    print(f"   • Columns: {list(output_df.columns)}")
    
    # Statistics
    sprint(f"\n📊 Value ranges:")
    print(f"   • direction: {output_df['direction'].min():.1f}° - {output_df['direction'].max():.1f}°")
    print(f"   • distance: {output_df['distance'].min():.1f}m - {output_df['distance'].max():.1f}m")
    print(f"   • height: {output_df['height'].min():.1f}m - {output_df['height'].max():.1f}m")
    
    # Sample
    sprint(f"\n📋 Sample (first 3 rows):")
    print(output_df.head(3).to_string(index=False))
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert Problem 2 predictions to competition format'
    )
    parser.add_argument('--input', type=str,
                       default='submissions/p2_localization_final.csv',
                       help='Input CSV with predictions')
    parser.add_argument('--output', type=str,
                       default='submissions/p2_localization_competition.csv',
                       help='Output CSV in competition format')
    
    args = parser.parse_args()
    
    convert_to_competition_format(args.input, args.output)
