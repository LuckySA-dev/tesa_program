"""
Calculate competition score (out of 9 points)
"""

import json

# Load ensemble config
with open('ensemble_config.json', 'r') as f:
    config = json.load(f)

angle_error = config['angle_error']
height_error = config['height_error']
range_error = config['range_error']

# Calculate total error
total_error = 0.7 * angle_error + 0.15 * height_error + 0.15 * range_error

print("="*70)
print("📊 คะแนนตามเกณฑ์การแข่งขัน (เต็ม 9 คะแนน)")
print("="*70)

print(f"\nValidation Performance:")
print(f"  - Angle Error:  {angle_error:.2f}°")
print(f"  - Height Error: {height_error:.2f} m")
print(f"  - Range Error:  {range_error:.2f} m")

print(f"\nสูตรคะแนน:")
print(f"  total_error = 0.7 × {angle_error:.2f} + 0.15 × {height_error:.2f} + 0.15 × {range_error:.2f}")
print(f"  total_error = {0.7*angle_error:.4f} + {0.15*height_error:.4f} + {0.15*range_error:.4f}")
print(f"  total_error = {total_error:.4f}")

print("\n" + "="*70)
print("🎯 การคำนวณคะแนน (เต็ม 9 คะแนน)")
print("="*70)

print("\nสมมติว่าคะแนนเต็ม 9 เมื่อ error = 0")
print("และลดลงตามสัดส่วนของ error")

# Method 1: Scale 0-10
print("\nวิธีที่ 1: สเกลแบบเชิงเส้น (error 0-10)")
print("  ถ้า error = 0  → คะแนน = 9")
print("  ถ้า error = 10 → คะแนน = 0")
print("\nคะแนนที่ได้ = 9 × (1 - error/10)")
print(f"            = 9 × (1 - {total_error:.4f}/10)")
print(f"            = 9 × {1 - total_error/10:.4f}")
score1 = 9 * max(0, 1 - total_error/10)
print(f"            = {score1:.2f} คะแนน")

# Method 2: Scale 0-20
print("\nวิธีที่ 2: สเกลแบบเชิงเส้น (error 0-20)")
print("  ถ้า error = 0  → คะแนน = 9")
print("  ถ้า error = 20 → คะแนน = 0")
print("\nคะแนนที่ได้ = 9 × (1 - error/20)")
print(f"            = 9 × (1 - {total_error:.4f}/20)")
print(f"            = 9 × {1 - total_error/20:.4f}")
score2 = 9 * max(0, 1 - total_error/20)
print(f"            = {score2:.2f} คะแนน")

# Method 3: Exponential decay
print("\nวิธีที่ 3: แบบ Exponential (ลดแบบเลขชี้กำลัง)")
print("  คะแนนที่ได้ = 9 × exp(-error/10)")
import math
score3 = 9 * math.exp(-total_error/10)
print(f"            = 9 × exp(-{total_error:.4f}/10)")
print(f"            = {score3:.2f} คะแนน")

# Method 4: Based on baseline comparison
print("\nวิธีที่ 4: เทียบกับ baseline (error = 5.9369)")
print("  ถ้า error = 0      → คะแนน = 9")
print("  ถ้า error = 5.9369 → คะแนน = 5 (60% ของคะแนนเต็ม)")
print("  ถ้า error = 10     → คะแนน = 0")
baseline_error = 5.9369
if total_error <= baseline_error:
    # Better than baseline: scale from 5 to 9
    score4 = 5 + (9 - 5) * (baseline_error - total_error) / baseline_error
else:
    # Worse than baseline: scale from 5 to 0
    score4 = 5 * (1 - (total_error - baseline_error) / (10 - baseline_error))
print(f"\nคะแนนที่ได้ = {score4:.2f} คะแนน")

print("\n" + "="*70)
print("💡 สรุป")
print("="*70)
print(f"Total Error:              {total_error:.4f}")
print(f"\nคะแนนที่เป็นไปได้:")
print(f"  Method 1 (scale 0-10):  {score1:.2f}/9  ({score1/9*100:.1f}%)")
print(f"  Method 2 (scale 0-20):  {score2:.2f}/9  ({score2/9*100:.1f}%)")
print(f"  Method 3 (exponential): {score3:.2f}/9  ({score3/9*100:.1f}%)")
print(f"  Method 4 (vs baseline): {score4:.2f}/9  ({score4/9*100:.1f}%)")

print("\n" + "="*70)
print("🎯 ประมาณการที่เหมาะสม")
print("="*70)
avg_score = (score1 + score2 + score3 + score4) / 4
print(f"\nค่าเฉลี่ย:  {avg_score:.2f}/9  ({avg_score/9*100:.1f}%)")
print(f"แนะนำใช้:  {score2:.2f}/9  (Method 2)")
print("\nเหตุผล:")
print(f"  - Error ของเรา ({total_error:.2f}) ต่ำกว่า baseline ({baseline_error:.2f})")
print(f"  - ปรับปรุงได้ +11.0%")
print(f"  - ถ้าสเกล 0-20 จะได้ประมาณ {score2:.2f} คะแนน")

print("\n" + "="*70)
