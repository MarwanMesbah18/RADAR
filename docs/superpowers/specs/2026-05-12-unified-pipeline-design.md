# Unified Pipeline Design — Photo & Video Tabs

## Context

The current photo and video tabs have different flows. The video tab is fully automatic (process all frames, no user interaction). The photo tab has a stepped flow but only one mode. The user wants:

1. Both tabs to follow a similar interactive pattern
2. Two modes in photo tab: direct plate detection OR cars-then-plates
3. Video tab to be interactive: scan → select car → auto/manual plate → enhance + OCR
4. Consistent 3-column enhancement+OCR comparison (Original, LapSRN, Real-ESRGAN) at the end of every flow
5. Auto modes will be improved later — for now, basic "best crop" logic

## Shared Component: Enhancement+OCR Comparison

A shared display function used by both tabs at the end of their flow.

**Input**: plate crop (BGR numpy array)

**Display**: 3 columns side-by-side:
| Original | LapSRN (AI Light) | Real-ESRGAN (AI Heavy) |
|----------|-------------------|------------------------|
| Plate image (no enhancement) | LapSRN-enhanced image | Real-ESRGAN-enhanced image |
| OCR text + confidence | OCR text + confidence | OCR text + confidence |

Each column shows the image on top, then the OCR result (plate text, numbers, letters) and average confidence below.

**Implementation**: New function in `ui/display.py` — `show_enhancement_comparison(plate_crop: np.ndarray)`. It calls `enhance_lapsrn()`, `enhance_realesrgan()`, `ocr_plate()` on each, and renders the 3-column Streamlit layout.

---

## Photo Tab Redesign

**File**: `ui/photo_tab.py`

### Flow

1. Upload image (unchanged)
2. Two buttons always visible below the image:
   - **"Detect Plates Directly"** — calls `detect_plates()` on the full image
   - **"Detect Cars First"** — calls `track_cars()` then shows car selection
3. Depending on which button was clicked:

#### Path A: Detect Plates Directly
- Run `detect_plates()` on the full uploaded image
- If plates found: show plate count, then for each plate show the 3-column enhancement+OCR comparison
- If no plates: show "No plates detected" warning

#### Path B: Detect Cars First
- Run `track_cars()` on the image
- Show all detected cars in a grid with car crops
- User selects a car (selectbox)
- Run `detect_plates()` on the selected car's crop
- If plate found: show car crop + plate crop, then the 3-column enhancement+OCR comparison
- If no plate: show "No plate detected on this car"

### State Management

Session state keys:
- `photo_result`: cached pipeline result (for cars-first path)
- `photo_direct_plates`: cached plate detections (for direct path)
- `photo_mode`: "direct" or "cars" — which path is active
- `selected_car`: index of selected car (cars path)
- `tmp_file`: temp file path for uploaded image

---

## Video Tab Redesign

**File**: `ui/video_tab.py`, `pipelines/video_pipeline.py`

### Flow

#### Step 1: Upload + Scan
- Upload video (unchanged)
- Click "Scan Video" button
- System runs the full video through ByteTrack car tracking (reuses existing `process_video` logic but simplified)
- Extract unique cars by track ID, save best crop for each
- Show progress bar during scan
- Display all unique cars in a grid (4 per row), each with best crop image and label "Car #1 (sedan)"

#### Step 2: Select Car + Choose Mode
- User selects a car from the grid (radio/selectbox)
- Two buttons appear:
  - **"Auto Mode"**: Scans all frames for that car's track ID, picks the plate crop with the largest bounding box area (assumed clearest/closest). Runs plate detection on that frame's car crop.
  - **"Manual Mode"**: Shows top 5 plate candidates from different frames for that car (sorted by crop size). User picks the clearest one. Each candidate shows the frame number and a thumbnail.

#### Step 3: Enhancement + OCR
- Once a plate crop is selected (auto or manual), show the 3-column enhancement+OCR comparison (same shared component as photo tab)

### Pipeline Changes

The video pipeline needs a new function to support the interactive flow:

**New function**: `scan_video_cars(video_path, progress_callback)` — processes all frames, tracks cars, returns a list of `ScannedCar` objects with:
- `track_id`, `car_class`, `best_crop`, `best_frame_num`, `first_seen_frame`

**New function**: `find_plate_crops(video_path, track_id, top_n=5)` — seeks through the video for a specific track ID's frames, runs plate detection, returns ranked list of plate crops with frame numbers.

The existing `process_video()` function remains for future auto-processing improvements.

### New Data Classes

```python
@dataclass
class ScannedCar:
    track_id: int
    car_class: str
    best_crop: np.ndarray        # best car crop across all frames
    best_frame_num: int
    first_seen_frame: int

@dataclass
class PlateCandidate:
    frame_num: int
    plate_crop: np.ndarray       # cropped plate image
    plate_bbox: tuple            # in frame coordinates
    car_crop: np.ndarray         # car crop from that frame
    crop_area: float             # plate crop area (for ranking)
```

---

## Files Changed

| File | Change |
|------|--------|
| `ui/photo_tab.py` | Full rewrite — two buttons, direct/cars paths, shared comparison |
| `ui/video_tab.py` | Full rewrite — scan → select → auto/manual → comparison |
| `ui/display.py` | Add `show_enhancement_comparison()` shared function |
| `pipelines/video_pipeline.py` | Add `scan_video_cars()` and `find_plate_crops()` |
| `pipelines/photo_pipeline.py` | May need minor changes to support direct plate path |

Files NOT changed:
- `core/` modules — all core ML functions stay the same
- `config.py` — no new config needed
- `app.py` — sidebar and tab structure unchanged

---

## Verification

1. **Photo tab — Direct mode**: Upload image → click "Detect Plates Directly" → see plate count → see 3-column comparison with Original, LapSRN, Real-ESRGAN OCR results
2. **Photo tab — Cars mode**: Upload image → click "Detect Cars First" → select car → see plate → see 3-column comparison
3. **Video tab — Auto**: Upload video → scan → select car → auto mode → see 3-column comparison
4. **Video tab — Manual**: Upload video → scan → select car → manual mode → pick from candidates → see 3-column comparison
5. Both tabs show the same comparison format at the end
