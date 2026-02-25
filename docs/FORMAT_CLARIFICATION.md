# 🚨 IMPORTANT CLARIFICATION - Integration Format

**Date:** November 8, 2025

## ⚠️ Format Confusion in Requirements

### The Issue:
ในเอกสารโจทย์มี **2 ส่วนที่ขัดแย้งกัน**:

#### 1. ส่วนบน (AI-Base Competition):
```
11 พ.ย.: frame_id, object_id, direction, distance, height
12 พ.ย.: frame_id, object_id, direction, distance, height
```

#### 2. ส่วนล่าง (Integration 13 พ.ย.):
```
video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred
```
- **Ground Truth:** `cx, cy = จุดกึ่งกลางของโดรนในภาพ (พิกเซล)`
- **Submission:** `cx, cy = จุดกึ่งกลางโดรนที่ทำนาย (พิกเซล)`

---

## 📊 Current Status

### ✅ Files We Have:

| File | Format | Purpose |
|------|--------|---------|
| `p1_detection_obb.csv` | `frame_id, object_id, center_x, center_y, w, h` (normalized 0-1) | Problem 1: Detection with OBB |
| `p2_localization_final.csv` | `frame_id, object_id, direction, distance, height` | Problem 2 & AI Competition (11-12 พ.ย.) |
| `submission.csv` | `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred` (pixels) | Integration (13 พ.ย.) - MATCHES EXAMPLE |
| `submission_normalized.csv` | Same but normalized coordinates | Alternative version |

---

## 🎯 Which Format to Submit?

### For Integration (13 พ.ย. 20:00):

**ตามตัวอย่างในโจทย์:**
```csv
video_id,frame,cx,cy,range_m_pred,azimuth_deg_pred,elevation_deg_pred
clip_drone.mp4,0,642,278,53.1,-4.9,3.2
```

✅ **ใช้ `submission.csv`** (พิกเซล) เพราะ:
1. ตัวอย่างใช้ `cx, cy` (ไม่ใช่ `center_x, center_y`)
2. Ground truth ระบุชัด: "พิกเซล"
3. ตัวเลขในตัวอย่าง (642, 278) เป็นพิกเซล ไม่ใช่ 0-1

---

## 🔍 Evidence from Requirements:

### 1. Ground Truth Format (train):
```
video_id,frame,cx,cy,range_m,azimuth_deg,elevation_deg

cx, cy = จุดกึ่งกลางของโดรนในภาพ (พิกเซล)  ← ชัดเจน!
```

### 2. Submission Format:
```
video_id,frame,cx,cy,range_m_pred,azimuth_deg_pred,elevation_deg_pred

cx, cy = จุดกึ่งกลางโดรนที่ทำนาย (พิกเซล)  ← ชัดเจน!
```

### 3. Example:
```
clip_drone.mp4,0,642,278,53.1,-4.9,3.2  ← 642, 278 = pixels
```

---

## ✅ Final Answer:

### Submission Files:

| Date | Time | File | Format |
|------|------|------|--------|
| **11 พ.ย.** | 18:00 | `p2_localization_final.csv` | `frame_id, object_id, direction, distance, height` |
| **12 พ.ย.** | 18:00 | `p2_localization_final.csv` | `frame_id, object_id, direction, distance, height` |
| **13 พ.ย.** | 20:00 | `submission.csv` | `video_id, frame, cx, cy, range_m_pred, azimuth_deg_pred, elevation_deg_pred` **(PIXELS)** |

---

## 📝 About YOLO OBB Format:

**YOLO OBB Normalization (0-1) applies to:**
- ✅ Problem 1 output only (`p1_detection_obb.csv`)
- ❌ NOT for Integration submission

**Why the confusion?**
- YOLO OBB training uses normalized format
- But competition submission uses pixels (easier to evaluate)
- Different purposes: Training format ≠ Submission format

---

## 🎉 Summary:

### All Files Compliant:
1. ✅ `p1_detection_obb.csv` - Normalized (0-1) for YOLO OBB
2. ✅ `p2_localization_final.csv` - Converted format (direction, distance, height)
3. ✅ `submission.csv` - **Pixels** matching competition example

### Ready to Submit:
- 11-12 พ.ย.: Use Problem 2 format (direction/distance/height)
- 13 พ.ย.: Use Integration format with **PIXELS** (matches example exactly)

**No further changes needed!** 🎯
