"""
Visualization Tools
===================

แสดงผลลัพธ์จากแต่ละ problem
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from utils import draw_obb, obb_to_corners
from hw import sprint


def visualize_detection(image_path: str, 
                        detection_csv: str,
                        output_path: str = None,
                        show: bool = True):
    """
    แสดงผลลัพธ์ Problem 1: Detection
    
    Args:
        image_path: path to image
        detection_csv: CSV file from problem 1
        output_path: save output image (optional)
        show: แสดงภาพหรือไม่
    """
    # อ่านภาพ
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    img_name = Path(image_path).name
    
    # อ่าน CSV
    df = pd.read_csv(detection_csv)
    detections = df[df['img_file'] == img_name]
    
    print(f"Image: {img_name}")
    print(f"Detections: {len(detections)}")
    
    # วาดแต่ละ detection
    for i, row in detections.iterrows():
        label = f"Drone {i+1}"
        image = draw_obb(
            image,
            row['center_x'], row['center_y'],
            row['w'], row['h'], row['theta'],
            label=label,
            color=(0, 255, 0),
            thickness=2
        )
    
    # บันทึกหรือแสดง
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved to: {output_path}")
    
    if show:
        cv2.imshow('Detection Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image


def visualize_localization(image_path: str,
                          localization_csv: str,
                          output_path: str = None,
                          show: bool = True):
    """
    แสดงผลลัพธ์ Problem 2: Localization
    
    Args:
        image_path: path to image
        localization_csv: CSV file from problem 2
        output_path: save output image (optional)
        show: แสดงภาพหรือไม่
    """
    # อ่านภาพ
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    img_name = Path(image_path).name
    
    # อ่าน CSV
    df = pd.read_csv(localization_csv)
    detections = df[df['img_file'] == img_name]
    
    print(f"Image: {img_name}")
    print(f"Localizations: {len(detections)}")
    
    # วาดแต่ละ detection พร้อมพิกัด
    for i, row in detections.iterrows():
        label = f"Drone {i+1}\n"
        label += f"Lat: {row['drone_lat']:.6f}\n"
        label += f"Lon: {row['drone_lon']:.6f}\n"
        label += f"Alt: {row['drone_alt']:.1f}m"
        
        image = draw_obb(
            image,
            row['center_x'], row['center_y'],
            row['w'], row['h'], row['theta'],
            label=f"D{i+1}",
            color=(255, 0, 0),
            thickness=2
        )
        
        # แสดงข้อมูลพิกัดด้านข้าง
        y_offset = 30 + i * 80
        cv2.putText(image, f"Drone {i+1}:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"Lat: {row['drone_lat']:.6f}", (10, y_offset+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Lon: {row['drone_lon']:.6f}", (10, y_offset+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f"Alt: {row['drone_alt']:.1f}m", (10, y_offset+60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # บันทึกหรือแสดง
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved to: {output_path}")
    
    if show:
        cv2.imshow('Localization Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return image


def visualize_tracking_video(video_path: str,
                            tracking_csv: str,
                            output_path: str = None,
                            show: bool = True,
                            skip_frames: int = 0):
    """
    แสดงผลลัพธ์ Problem 3: Tracking
    
    Args:
        video_path: path to video
        tracking_csv: CSV file from problem 3
        output_path: save output video (optional)
        show: แสดงวิดีโอหรือไม่
        skip_frames: skip ทุกๆ n frames
    """
    # อ่านวิดีโอ
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    video_id = Path(video_path).stem
    
    # อ่าน CSV
    df = pd.read_csv(tracking_csv)
    df = df[df['video_id'] == video_id]
    
    print(f"Video: {video_id}")
    print(f"Total tracks: {len(df)}")
    print(f"Unique track IDs: {df['track_id'].nunique()}")
    
    # ตั้งค่าวิดีโอ output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if output_path:
        from utils import VideoWriterSafe
        writer = VideoWriterSafe(output_path, fps, (width, height))
    
    # สร้างสีแต่ละ track
    track_ids = df['track_id'].unique()
    colors = {}
    np.random.seed(42)
    for tid in track_ids:
        colors[tid] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ดึง tracks ใน frame นี้
        tracks = df[df['frame_id'] == frame_count]
        
        # วาดแต่ละ track
        for _, row in tracks.iterrows():
            tid = row['track_id']
            color = colors.get(tid, (0, 255, 0))
            
            label = f"ID:{tid}"
            frame = draw_obb(
                frame,
                row['center_x'], row['center_y'],
                row['w'], row['h'], row['theta'],
                label=label,
                color=color,
                thickness=2
            )
        
        # แสดงข้อมูล frame
        info_text = f"Frame: {frame_count} | Tracks: {len(tracks)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # บันทึก
        if writer:
            writer.write(frame)
        
        # แสดง
        if show:
            cv2.imshow('Tracking Result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        # Skip frames
        if skip_frames > 0:
            for _ in range(skip_frames):
                cap.read()
                frame_count += 1
    
    cap.release()
    if writer:
        writer.release()
        print(f"Saved video to: {output_path}")
    
    cv2.destroyAllWindows()


def create_summary_images(image_folder: str,
                         detection_csv: str,
                         output_folder: str = 'visualizations',
                         max_images: int = 10):
    """
    สร้างภาพสรุปสำหรับหลายภาพ
    
    Args:
        image_folder: folder ที่มีภาพ
        detection_csv: CSV file from problem 1
        output_folder: output folder
        max_images: จำนวนภาพสูงสุดที่จะสร้าง
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # อ่าน CSV
    df = pd.read_csv(detection_csv)
    
    # หาภาพที่มี detection
    images_with_detections = df['img_file'].unique()[:max_images]
    
    print(f"Creating visualizations for {len(images_with_detections)} images...")
    
    for i, img_name in enumerate(images_with_detections, 1):
        img_path = image_folder / img_name
        if img_path.exists():
            output_path = output_folder / f"vis_{img_name}"
            print(f"[{i}/{len(images_with_detections)}] {img_name}")
            visualize_detection(str(img_path), detection_csv, 
                              str(output_path), show=False)
    
    sprint(f"\n✓ Saved visualizations to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--problem', type=int, required=True, choices=[1, 2, 3],
                       help='Problem number (1, 2, or 3)')
    parser.add_argument('--image', type=str, help='Image file (for problem 1 or 2)')
    parser.add_argument('--video', type=str, help='Video file (for problem 3)')
    parser.add_argument('--csv', type=str, required=True, help='CSV result file')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--no-show', action='store_true', help='Do not display')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip frames in video (problem 3)')
    
    args = parser.parse_args()
    
    if args.problem == 1:
        if not args.image:
            print("Error: --image required for problem 1")
            return
        visualize_detection(args.image, args.csv, args.output, not args.no_show)
    
    elif args.problem == 2:
        if not args.image:
            print("Error: --image required for problem 2")
            return
        visualize_localization(args.image, args.csv, args.output, not args.no_show)
    
    elif args.problem == 3:
        if not args.video:
            print("Error: --video required for problem 3")
            return
        visualize_tracking_video(args.video, args.csv, args.output, 
                               not args.no_show, args.skip_frames)


if __name__ == '__main__':
    main()
