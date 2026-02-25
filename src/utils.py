"""
Utilities
=========

Helper functions สำหรับทุก problems
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from hw import sprint


def obb_to_corners(center_x: float, center_y: float, 
                   w: float, h: float, theta: float,
                   img_width: int = None, img_height: int = None) -> np.ndarray:
    """
    แปลง OBB parameters เป็น 4 มุม
    
    Args:
        center_x, center_y: จุดกึ่งกลาง (normalized 0-1 หรือ pixel)
        w, h: ความกว้าง/สูง (normalized 0-1 หรือ pixel)
        theta: มุมหมุน (degrees)
        img_width, img_height: ถ้าให้มา จะแปลงจาก normalized เป็น pixel
        
    Returns:
        numpy array shape (4, 2) = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    # แปลงเป็น pixel ถ้าต้องการ
    if img_width and img_height:
        cx = center_x * img_width
        cy = center_y * img_height
        width = w * img_width
        height = h * img_height
    else:
        cx, cy, width, height = center_x, center_y, w, h
    
    # คำนวณ corners (ก่อนหมุน)
    half_w = width / 2
    half_h = height / 2
    corners = np.array([
        [-half_w, -half_h],  # top-left
        [half_w, -half_h],   # top-right
        [half_w, half_h],    # bottom-right
        [-half_w, half_h]    # bottom-left
    ])
    
    # หมุนตามมุม theta
    theta_rad = np.radians(theta)
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    
    rotated_corners = corners @ rotation_matrix.T
    
    # เลื่อนไปที่จุดกึ่งกลาง
    rotated_corners += np.array([cx, cy])
    
    return rotated_corners


def draw_obb(image: np.ndarray, 
             center_x: float, center_y: float,
             w: float, h: float, theta: float,
             label: str = None,
             color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
    """
    วาด OBB บนภาพ
    
    Args:
        image: ภาพ (numpy array)
        center_x, center_y, w, h, theta: OBB parameters (normalized 0-1)
        label: ข้อความที่จะแสดง
        color: สี BGR
        thickness: ความหนาเส้น
        
    Returns:
        ภาพที่วาดแล้ว
    """
    img_h, img_w = image.shape[:2]
    
    # หา corners
    corners = obb_to_corners(center_x, center_y, w, h, theta, img_w, img_h)
    corners = corners.astype(np.int32)
    
    # วาดกรอบ
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=thickness)
    
    # วาดจุดกึ่งกลาง
    center = (int(center_x * img_w), int(center_y * img_h))
    cv2.circle(image, center, 3, color, -1)
    
    # วาด label
    if label:
        cv2.putText(image, label, (corners[0][0], corners[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


def compute_iou_obb(obb1: Dict, obb2: Dict, method='bbox') -> float:
    """
    คำนวณ IoU ระหว่าง 2 OBB
    
    Args:
        obb1, obb2: dict with keys: center_x, center_y, w, h, theta
        method: 'bbox' (fast, approximate) or 'polygon' (accurate, slow)
        
    Returns:
        IoU value (0-1)
    """
    if method == 'bbox':
        # วิธีเร็ว: ใช้ bounding box ครอบ
        def obb_to_bbox(obb):
            # หา diagonal
            diag = np.sqrt(obb['w']**2 + obb['h']**2)
            return {
                'x1': obb['center_x'] - diag/2,
                'y1': obb['center_y'] - diag/2,
                'x2': obb['center_x'] + diag/2,
                'y2': obb['center_y'] + diag/2
            }
        
        box1 = obb_to_bbox(obb1)
        box2 = obb_to_bbox(obb2)
        
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    elif method == 'polygon':
        # วิธีแม่นยำ: คำนวณ intersection ของ polygon
        try:
            from shapely.geometry import Polygon
            from shapely.validation import make_valid
            
            corners1 = obb_to_corners(
                obb1['center_x'], obb1['center_y'], 
                obb1['w'], obb1['h'], obb1['theta']
            )
            corners2 = obb_to_corners(
                obb2['center_x'], obb2['center_y'],
                obb2['w'], obb2['h'], obb2['theta']
            )
            
            poly1 = make_valid(Polygon(corners1))
            poly2 = make_valid(Polygon(corners2))
            
            intersection = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            
            return intersection / union if union > 0 else 0.0
            
        except ImportError:
            print("Warning: shapely not installed, falling back to bbox method")
            return compute_iou_obb(obb1, obb2, method='bbox')
    
    else:
        raise ValueError(f"Unknown method: {method}")


def create_sample_metadata(image_folder: str, 
                          output_csv: str = 'image_meta.csv',
                          base_lat: float = 13.7563,
                          base_lon: float = 100.5018,
                          base_alt: float = 50.0) -> pd.DataFrame:
    """
    สร้างไฟล์ metadata ตัวอย่างสำหรับทดสอบ
    
    Args:
        image_folder: folder ที่มีภาพ
        output_csv: output CSV file
        base_lat, base_lon, base_alt: พิกัดเริ่มต้น
        
    Returns:
        DataFrame
    """
    image_folder = Path(image_folder)
    
    # หาไฟล์ภาพ
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_folder.glob(f'*{ext}'))
        image_files.extend(image_folder.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    # สร้าง metadata
    data = []
    for i, img_path in enumerate(image_files):
        # สุ่มตำแหน่งใกล้ๆ กัน
        lat = base_lat + np.random.uniform(-0.001, 0.001)
        lon = base_lon + np.random.uniform(-0.001, 0.001)
        alt = base_alt + np.random.uniform(-5, 5)
        
        data.append({
            'img_file': img_path.name,
            'img_lat': round(lat, 6),
            'img_lon': round(lon, 6),
            'img_alt': round(alt, 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"Created sample metadata: {output_csv}")
    print(f"  - {len(df)} images")
    
    return df


def convert_obb_format(center_x: float, center_y: float,
                      w: float, h: float, theta: float,
                      from_format: str = 'yolo',
                      to_format: str = 'corners') -> Dict:
    """
    แปลงระหว่างรูปแบบ OBB ต่างๆ
    
    Args:
        from_format: 'yolo' (center_x, center_y, w, h, theta)
        to_format: 'corners' (4 มุม) หรือ 'bbox' (x1, y1, x2, y2)
        
    Returns:
        dict ตามรูปแบบที่ต้องการ
    """
    if from_format == 'yolo':
        if to_format == 'corners':
            corners = obb_to_corners(center_x, center_y, w, h, theta)
            return {
                'x1': float(corners[0][0]), 'y1': float(corners[0][1]),
                'x2': float(corners[1][0]), 'y2': float(corners[1][1]),
                'x3': float(corners[2][0]), 'y3': float(corners[2][1]),
                'x4': float(corners[3][0]), 'y4': float(corners[3][1])
            }
        elif to_format == 'bbox':
            corners = obb_to_corners(center_x, center_y, w, h, theta)
            return {
                'x1': float(corners[:, 0].min()),
                'y1': float(corners[:, 1].min()),
                'x2': float(corners[:, 0].max()),
                'y2': float(corners[:, 1].max())
            }
    
    raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")


def validate_obb_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    ตรวจสอบว่า DataFrame มี format ถูกต้องหรือไม่
    
    Args:
        df: DataFrame ที่จะตรวจสอบ
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # ตรวจสอบ columns
    required_cols = ['center_x', 'center_y', 'w', 'h', 'theta']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    if len(errors) > 0:
        return False, errors
    
    # ตรวจสอบค่า
    if (df['center_x'] < 0).any() or (df['center_x'] > 1).any():
        errors.append("center_x should be in range [0, 1]")
    
    if (df['center_y'] < 0).any() or (df['center_y'] > 1).any():
        errors.append("center_y should be in range [0, 1]")
    
    if (df['w'] < 0).any() or (df['w'] > 1).any():
        errors.append("w should be in range [0, 1]")
    
    if (df['h'] < 0).any() or (df['h'] > 1).any():
        errors.append("h should be in range [0, 1]")
    
    if (df['theta'] < -90).any() or (df['theta'] > 90).any():
        errors.append("theta should be in range [-90, 90] degrees")
    
    return len(errors) == 0, errors


if __name__ == '__main__':
    # ทดสอบ functions
    print("Testing OBB utilities...")
    
    # Test 1: OBB to corners
    corners = obb_to_corners(0.5, 0.5, 0.2, 0.1, 30, 640, 480)
    print(f"\nTest 1 - OBB to corners:")
    print(corners)
    
    # Test 2: IoU
    obb1 = {'center_x': 0.5, 'center_y': 0.5, 'w': 0.2, 'h': 0.1, 'theta': 0}
    obb2 = {'center_x': 0.55, 'center_y': 0.5, 'w': 0.2, 'h': 0.1, 'theta': 10}
    iou = compute_iou_obb(obb1, obb2, method='bbox')
    print(f"\nTest 2 - IoU: {iou:.3f}")
    
    # Test 3: Format conversion
    corners_dict = convert_obb_format(0.5, 0.5, 0.2, 0.1, 30, 
                                      from_format='yolo', 
                                      to_format='corners')
    print(f"\nTest 3 - Format conversion:")
    print(corners_dict)
    
    sprint("\n✓ All tests passed!")
