import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from problem_3_pipeline import Problem3Pipeline

def test_pipeline():
    print("Testing Pipeline...")
    pipeline = Problem3Pipeline(
        model_path=str(PROJECT_ROOT / 'runs' / 'detect' / 'drone_detect_v21_max_data' / 'weights' / 'best.pt'),
        conf_threshold=0.10,
        iou_threshold=0.3,
        use_track_merging=True
    )
    
    # Run on just 10 frames
    pipeline.process_video(
        video_path=str(PROJECT_ROOT / 'videos' / 'P3_VIDEO.mp4'),
        output_path=str(PROJECT_ROOT / 'outputs' / 'problem_3' / 'test' / 'test_output.mp4'),
        start_frame=0,
        end_frame=10
    )
    print("Test Complete")

if __name__ == "__main__":
    test_pipeline()
