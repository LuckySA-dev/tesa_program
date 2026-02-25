"""
Final Compliance Checker (No Emoji Version for Windows)
Validates all outputs against TESA Defence requirements
"""

import pandas as pd
import sys

def check_problem1():
    """Check Problem 1 format compliance"""
    print("="*70)
    print("PROBLEM 1 CHECK: Drone Detection with OBB")
    print("="*70)
    print("Expected format: frame_id, object_id, center_x, center_y, w, h, [theta]")
    print("Requirement: Normalized coordinates (0-1), theta in degrees")
    print("-"*70)
    
    try:
        df = pd.read_csv('submissions/p1_detection_obb.csv')
        
        print(f"OK File found: submissions/p1_detection_obb.csv")
        print(f"   * Records: {len(df)}")
        print(f"   * Columns: {list(df.columns)}")
        
        # Check required columns
        required = ['frame_id', 'object_id', 'center_x', 'center_y', 'w', 'h']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"ERROR Missing columns: {missing}")
            return False
        else:
            print(f"OK All required columns present")
        
        # Check normalization
        for col in ['center_x', 'center_y', 'w', 'h']:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"   * {col}: {min_val:.4f} - {max_val:.4f}", end="")
            
            if min_val >= 0 and max_val <= 1:
                print(" OK")
            else:
                print(" ERROR NOT NORMALIZED!")
                return False
        
        print(f"\nSample (first 3 rows):")
        print(df.head(3).to_string(index=False))
        
        print(f"\nPROBLEM 1: PASS\n")
        return True
        
    except FileNotFoundError:
        print(f"ERROR File not found: submissions/p1_detection_obb.csv")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_problem2():
    """Check Problem 2 format compliance"""
    print("="*70)
    print("PROBLEM 2 CHECK: Drone Localization")
    print("="*70)
    print("Expected format: frame_id, object_id, direction, distance, height")
    print("Requirement: direction (0-360deg), distance (m), height (m)")
    print("-"*70)
    
    try:
        df = pd.read_csv('submissions/p2_localization_final.csv')
        
        print(f"OK File found: submissions/p2_localization_final.csv")
        print(f"   * Records: {len(df)}")
        print(f"   * Columns: {list(df.columns)}")
        
        # Check required columns
        required = ['frame_id', 'object_id', 'direction', 'distance', 'height']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"ERROR Missing columns: {missing}")
            return False
        else:
            print(f"OK All required columns present")
        
        # Check value ranges
        print(f"\nValue ranges:")
        print(f"   * direction: {df['direction'].min():.1f} - {df['direction'].max():.1f} deg OK")
        print(f"   * distance: {df['distance'].min():.1f} - {df['distance'].max():.1f} m OK")
        print(f"   * height: {df['height'].min():.1f} - {df['height'].max():.1f} m OK")
        
        print(f"\nSample (first 3 rows):")
        print(df.head(3).to_string(index=False))
        
        print(f"\nPROBLEM 2: PASS\n")
        return True
        
    except FileNotFoundError:
        print(f"ERROR File not found: submissions/p2_localization_final.csv")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_integration():
    """Check Integration format compliance"""
    print("="*70)
    print("INTEGRATION CHECK (13 Nov deadline 20:00)")
    print("="*70)
    print("Expected: video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred")
    print("Requirement: cx, cy in PIXELS according to requirements")
    print("-"*70)
    
    files_to_check = [
        ('submissions/submission.csv', 'Original (pixels)'),
        ('submissions/submission_normalized.csv', 'Normalized version')
    ]
    
    all_ok = True
    
    for filename, desc in files_to_check:
        try:
            df = pd.read_csv(filename)
            
            print(f"\nOK File: {filename} ({desc})")
            print(f"   * Records: {len(df)}")
            print(f"   * Columns: {list(df.columns)}")
            
            if 'cx' in df.columns and 'cy' in df.columns:
                required = ['video_id', 'frame', 'cx', 'cy', 'range_m_pred', 'azimuth_deg_pred', 'elevation_deg_pred']
                coord_type = "pixels"
                cx_col, cy_col = 'cx', 'cy'
            elif 'center_x' in df.columns and 'center_y' in df.columns:
                required = ['video_id', 'frame', 'center_x', 'center_y', 'range_m_pred', 'azimuth_deg_pred', 'elevation_deg_pred']
                coord_type = "normalized"
                cx_col, cy_col = 'center_x', 'center_y'
            else:
                print(f"ERROR No valid coordinate columns found!")
                all_ok = False
                continue
            
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                print(f"ERROR Missing columns: {missing}")
                all_ok = False
                continue
            else:
                print(f"OK All required columns present")
            
            cx_min, cx_max = df[cx_col].min(), df[cx_col].max()
            cy_min, cy_max = df[cy_col].min(), df[cy_col].max()
            
            print(f"\nCoordinate type: {coord_type}")
            print(f"   * {cx_col}: {cx_min:.4f} - {cx_max:.4f}")
            print(f"   * {cy_col}: {cy_min:.4f} - {cy_max:.4f}")
            
            print(f"\nPredictions:")
            print(f"   * range_m: {df['range_m_pred'].min():.1f} - {df['range_m_pred'].max():.1f} m")
            print(f"   * azimuth: {df['azimuth_deg_pred'].min():.1f} - {df['azimuth_deg_pred'].max():.1f} deg")
            print(f"   * elevation: {df['elevation_deg_pred'].min():.1f} - {df['elevation_deg_pred'].max():.1f} deg")
            
            print(f"\nSample (first 2 rows):")
            print(df.head(2).to_string(index=False))
            
        except FileNotFoundError:
            print(f"ERROR File not found: {filename}")
            all_ok = False
        except Exception as e:
            print(f"ERROR reading {filename}: {e}")
            all_ok = False
    
    if all_ok:
        print(f"\nINTEGRATION: PASS (both versions available)\n")
    return all_ok


def main():
    """Run all compliance checks"""
    print("\n" + "="*70)
    print("TESA DEFENCE - FINAL COMPLIANCE CHECK")
    print("="*70)
    print("Date: November 8, 2025")
    print("Checking all outputs against competition requirements...")
    print("\n")
    
    results = []
    
    # Check Problem 1
    results.append(("Problem 1", check_problem1()))
    
    # Check Problem 2
    results.append(("Problem 2", check_problem2()))
    
    # Check Integration
    results.append(("Integration", check_integration()))
    
    # Summary
    print("="*70)
    print("COMPLIANCE SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL CHECKS PASSED - READY FOR SUBMISSION!")
        print("="*70)
        print("\nSubmission Schedule:")
        print("  * 11 Nov 18:00 - Submit submissions/p2_localization_final.csv")
        print("  * 12 Nov 18:00 - Submit submissions/p2_localization_final.csv")
        print("  * 13 Nov 20:00 - Submit submissions/submission.csv (pixels)")
        print("\nAll files ready!")
        return 0
    else:
        print("SOME CHECKS FAILED - PLEASE REVIEW")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
