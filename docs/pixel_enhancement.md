# Pixel Enhancement Techniques for Vehicle Detection Under Adverse Conditions

## 1. Introduction

Standard vehicle detection pipelines using classical computer vision techniques such as frame differencing and grid-based occupancy analysis perform well under ideal lighting conditions. However, their accuracy degrades significantly under adverse environmental conditions — fog, rain, and low-light (night/dusk) — due to reduced contrast, noise, and visual artefacts in the captured frames.

To address this, the **Enhanced Pipeline** (`enhanced_pipeline.py`) implements a suite of pixel-level image enhancement techniques that are conditionally applied based on real-time scene classification. These enhancements operate at the pixel level before the frame differencing and detection stages, ensuring that the vehicle detection module receives clean, contrast-normalised inputs regardless of weather or lighting.

The pipeline supports two operational modes:
- **Normal Mode** — all enhancements applied at full quality
- **Fast Mode** (`--fast`) — lightweight alternatives used for near real-time performance

---

## 2. Scene Classification

Before applying any enhancement, the system first classifies the current scene to determine which techniques are necessary. This avoids unnecessary processing during clear daylight conditions and enables targeted enhancements for specific degradations.

### 2.1 Lighting Classification

The average pixel brightness of the grayscale frame is used to classify the lighting condition:

| Condition | Brightness Range | Action |
|-----------|-----------------|--------|
| **Night** | brightness < 50 | Aggressive gamma correction (γ=2.2), high-clip CLAHE, bilateral denoising |
| **Dusk** | 50 ≤ brightness < 120 | Moderate gamma correction (γ=1.5), medium-clip CLAHE, bilateral denoising |
| **Day** | brightness ≥ 120 | Standard histogram equalisation only |

```python
brightness = np.mean(frame_gray)
```

### 2.2 Fog Detection using Dark Channel Prior

Fog density is estimated using the **Dark Channel Prior** (He et al., 2009). The dark channel of an image is the minimum intensity value across all colour channels within a local patch. In fog-free outdoor images, this value tends toward zero due to shadows, dark objects, or colourful surfaces. In foggy images, atmospheric scattering raises the dark channel values.

The fog index is computed as:

```
fog_index = 1.0 - mean(dark_channel) / 255.0
```

A fog index exceeding the threshold of **0.6** triggers fog compensation.

```python
dark_channel = cv2.erode(frame_gray, np.ones((15,15), np.uint8))
fog_index = 1.0 - float(np.mean(dark_channel)) / 255.0
is_foggy = fog_index > 0.6
```

### 2.3 Rain Detection using Laplacian Variance and Edge Directionality

Rain streaks introduce high-frequency vertical patterns in the image. Two metrics are combined for rain detection:

1. **Laplacian Standard Deviation** — Rain streaks increase the overall high-frequency content. A Laplacian is applied, and its standard deviation is measured. A value exceeding **35.0** indicates high-frequency noise consistent with rain.

2. **Vertical-to-Horizontal Edge Ratio** — Rain streaks are predominantly vertical. Sobel filters are applied in both directions, and the mean absolute gradient ratio is computed. A ratio exceeding **1.5** indicates strong vertical edge dominance.

Both conditions must be met simultaneously to classify the scene as rainy:

```python
lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
lap_std = np.std(lap)

sobel_v = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_h = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
vert_ratio = np.mean(np.abs(sobel_v)) / np.mean(np.abs(sobel_h))

is_rainy = (lap_std > 35.0) and (vert_ratio > 1.5)
```

---

## 3. Fog Compensation

### 3.1 Dark Channel Prior Dehazing (Normal Mode)

The primary dehazing technique implements the **Dark Channel Prior** (DCP) method, which models atmospheric scattering as:

```
I(x) = J(x) · t(x) + A · (1 - t(x))
```

Where:
- `I(x)` — observed hazy image
- `J(x)` — scene radiance (desired output)
- `t(x)` — transmission map (how much light penetrates the haze)
- `A` — global atmospheric light

**Implementation steps:**

1. **Dark Channel Computation** — Minimum intensity across all colour channels, followed by morphological erosion with a 15×15 kernel.

2. **Atmospheric Light Estimation** — The top 0.1% brightest pixels in the dark channel are identified. The corresponding pixels in the original image are averaged to estimate the atmospheric light vector `A`.

3. **Transmission Map** — Estimated using the normalised dark channel:
   ```
   t(x) = 1 - ω · dark_channel_normalised(x)
   ```
   Where ω = 0.95 (a small amount of haze is preserved for depth perception).

4. **Transmission Refinement** — The raw transmission map is refined using a **guided filter** (`cv2.ximgproc.guidedFilter`) with the grayscale image as the guide, maintaining edge alignment. Falls back to Gaussian blur if the `ximgproc` module is unavailable.

5. **Scene Radiance Recovery** — The haze-free image is recovered per-channel:
   ```
   J(x) = (I(x) - A) / max(t(x), t_min) + A
   ```
   Where `t_min = 0.1` prevents division instability.

### 3.2 Fast CLAHE Dehazing (Fast Mode)

For real-time applications, a lightweight alternative applies **CLAHE (Contrast Limited Adaptive Histogram Equalisation)** directly on the **V (Value/Brightness) channel** of the HSV colour space:

```python
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
v = clahe.apply(v)
result = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
```

This achieves approximately 80% of the DCP's contrast improvement at roughly 1/50th of the computational cost. Colour information is fully preserved since only the brightness channel is modified.

---

## 4. Rain Streak Removal

Rain streaks are mitigated using a two-step spatial filtering approach:

### 4.1 Median Filtering

A **5×5 median blur** is applied to the entire frame. Median filtering is particularly effective for rain removal because rain streaks are thin, high-intensity impulse-like features — exactly the type of noise that median filters are optimised to suppress while preserving edges.

### 4.2 Directional Morphological Closing

A **horizontal morphological close** operation (15×1 rectangular kernel) fills any residual vertical gaps left by the rain streaks. Since rain streaks are predominantly vertical, a horizontal kernel closes the gaps without disrupting horizontal edge structures (road markings, vehicle edges, etc.).

```python
derained = cv2.medianBlur(frame_bgr, 5)
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
derained = cv2.morphologyEx(derained, cv2.MORPH_CLOSE, h_kernel)
```

### 4.3 Rain-Specific Detection Parameters

When rain is detected, the detection parameters are adapted to account for the remaining noise:

| Parameter | Normal | Rain-Adapted |
|-----------|--------|-------------|
| Blur kernel | (5,5) | **(9,9)** — stronger smoothing to suppress residual streaks |
| Binary threshold | 75 | **50** — lower threshold since rain reduces overall contrast |
| Dilation iterations | 3 | **4** — more aggressive morphology to merge fragmented detections |

---

## 5. Low-Light Enhancement

### 5.1 Gamma Correction

Dark frames from night and dusk conditions are brightened using **gamma correction**, which applies a non-linear intensity transformation:

```
Output = 255 × (Input / 255) ^ (1/γ)
```

| Condition | Gamma (γ) | Effect |
|-----------|-----------|--------|
| Night | 2.2 | Strong brightening — dark regions are significantly lifted |
| Dusk | 1.5 | Moderate brightening — subtle lift without over-exposure |

For performance, the gamma mapping is **pre-computed as a 256-entry Look-Up Table (LUT)** at module load time. Applying the correction then reduces to a single O(1) table lookup per pixel via `cv2.LUT()`:

```python
GAMMA_LUT_NIGHT = np.array([((i / 255.0) ** (1.0 / 2.2)) * 255
                             for i in range(256)]).astype("uint8")

corrected = cv2.LUT(gray, GAMMA_LUT_NIGHT)
```

### 5.2 CLAHE — Contrast Limited Adaptive Histogram Equalisation

Standard histogram equalisation (`cv2.equalizeHist`) redistributes pixel intensities globally, which can amplify noise in dark regions. **CLAHE** solves this by:

1. Dividing the image into **8×8 tiles**
2. Applying histogram equalisation independently to each tile
3. **Clipping the histogram** at a limit to prevent noise amplification
4. Bilinear interpolation at tile boundaries to eliminate block artefacts

The clip limit controls the aggressiveness of contrast enhancement:

| Condition | Clip Limit | Rationale |
|-----------|-----------|-----------|
| Night | 4.0 | Higher clip — aggressive enhancement needed for very dark frames |
| Dusk | 2.5 | Moderate clip — balanced contrast without over-enhancement |
| Day | N/A | Standard `equalizeHist` used (no clipping needed) |

```python
CLAHE_NIGHT = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
CLAHE_DUSK  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
```

### 5.3 Noise Reduction

Low-light frames inherently contain higher sensor noise. Two denoising strategies are implemented:

**Bilateral Filter (Normal Mode):**
An edge-preserving filter that smooths noise while maintaining sharp edges. It considers both **spatial proximity** and **intensity similarity** when averaging pixels:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| d (diameter) | 5 | Neighbourhood size |
| σ_color | 50 | Intensity similarity range — pixels with similar brightness are averaged |
| σ_space | 50 | Spatial proximity range — closer pixels contribute more |

**Gaussian Blur (Fast Mode):**
A simple 5×5 Gaussian blur replaces the bilateral filter. It is approximately 10× faster but does not preserve edges. This trade-off is acceptable because the downstream frame differencing operation is inherently robust to mild uniform blurring.

---

## 6. Colour-Preserving Display Enhancement

For the output video visualisation, enhancements are applied to the display frame while **preserving the original colour information**. This is achieved by operating exclusively on the **V (Value) channel** of the HSV colour space:

```python
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

v = apply_gamma(v, lighting)          # Brighten
v = adaptive_equalize(v, lighting)    # Contrast + denoise

frame[ry:ry+rh, rx:rx+rw] = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
```

By modifying only the V channel, the hue (colour) and saturation (colour intensity) remain untouched. This produces an output video where enhanced regions retain their natural colours — vehicles, road markings, traffic lights, and trees appear in their original colours with improved brightness and contrast.

---

## 7. Adaptive Detection Parameters

The detection stage (frame differencing → thresholding → morphology → grid analysis) uses **condition-specific parameters** to optimise accuracy under each scene type:

| Condition | Blur Kernel | Binary Threshold | Dilation Iterations | Additional |
|-----------|-------------|------------------|--------------------|----|
| Day | (5,5) | 75 | 3 | — |
| Dusk | (5,5) | 40 | 3 | — |
| Night | (5,5) | 30 | 4 | Morphological close |
| Fog | (7,7) | 30–40 | 4 | Morphological close |
| Rain | (9,9) | 50 | 4 | Morphological close |

**Rationale:**
- Lower thresholds for night/fog compensate for reduced contrast between vehicles and background
- Larger blur kernels for fog/rain suppress residual atmospheric noise
- Additional dilation and morphological closing merge fragmented vehicle detections caused by uneven illumination

---

## 8. Processing Pipeline Flow

The complete enhancement pipeline operates in the following order for each frame pair:

```
┌──────────────────────────────────────────────────────────────┐
│                    FRAME INPUT                               │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  1. SCENE CLASSIFICATION                                     │
│     • Brightness analysis → Night / Dusk / Day               │
│     • Dark Channel fog estimation → Foggy / Clear            │
│     • Laplacian + Sobel rain detection → Rainy / Dry         │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2. FOG COMPENSATION (if foggy)                              │
│     • Normal: Dark Channel Prior dehazing                    │
│     • Fast: CLAHE on V channel                               │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  3. RAIN REMOVAL (if rainy)                                  │
│     • Median blur (5×5)                                      │
│     • Horizontal morphological close (15×1)                  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  4. COLOUR → GRAYSCALE CONVERSION                            │
│     • BGR → Grayscale for detection processing               │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  5. GAMMA CORRECTION (if night/dusk)                         │
│     • Pre-computed LUT application                           │
│     • γ=2.2 (night) or γ=1.5 (dusk)                         │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  6. ADAPTIVE HISTOGRAM EQUALISATION                          │
│     • CLAHE with condition-specific clip limits              │
│     • Bilateral / Gaussian denoising for low-light           │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  7. FRAME DIFFERENCING + DETECTION                           │
│     • absdiff between consecutive enhanced frames            │
│     • Adaptive threshold + morphology                        │
│     • Grid-based occupancy analysis                          │
│     • Connected component clustering                         │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Performance Optimisation

### 9.1 Fast Mode

The `--fast` flag enables a performance-optimised mode that substitutes computationally expensive operations:

| Operation | Normal Mode | Fast Mode | Speedup |
|-----------|------------|-----------|---------|
| Fog compensation | Dark Channel Prior (~50ms/ROI) | CLAHE on V channel (~1ms/ROI) | ~50× |
| Noise reduction | Bilateral filter (~5ms/ROI) | Gaussian blur (~0.5ms/ROI) | ~10× |
| Display enhancement | Full HSV-based pipeline | Skipped entirely | ∞ |

**Measured performance (1659 frames, DUSK + FOG scene):**

| Metric | Normal Mode | Fast Mode | Improvement |
|--------|------------|-----------|-------------|
| Execution time | 135.63s | 5.94s | ~23× faster |
| Processing speed | 12.23 FPS | 279.15 FPS | ~23× faster |
| Memory usage | 54.62 MB | 53.37 MB | Similar |
| Vehicle count (L1) | 10 | 10 | Identical |
| Vehicle count (L2) | 20 | 20 | Identical |
| Combined count | 30 | 30 | Identical |
| Avg speed L1 | 46.8 km/h | 51.2 km/h | ~9% variance |
| Avg speed L2 | 40.3 km/h | 44.4 km/h | ~10% variance |
| Detection accuracy | Full | Preserved | — |
| Output video quality | Enhanced (dehazed, colour-corrected) | Raw frames with overlays | — |

### 9.2 Pre-Computed Resources

Several resources are computed once at module load time to avoid redundant computation:

- **Gamma LUTs** — 256-entry lookup tables for night (γ=2.2) and dusk (γ=1.5) gamma correction
- **CLAHE objects** — Pre-initialised with condition-specific clip limits
- **Morphological kernels** — Structuring elements for dilation and closing operations
- **Channel mapping** — Pre-resolved colour channel indices

### 9.3 Parallel Processing

Enhancement functions are executed within a **multiprocessing Pool** across all available CPU cores. Each worker process independently applies the full enhancement chain to its assigned ROI pair, enabling parallel fog compensation, rain removal, gamma correction, and CLAHE processing.

The `FAST_MODE` flag is propagated to worker processes via a **pool initializer function**, ensuring consistent behaviour across all workers.

---

## 10. Configuration Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BRIGHTNESS_NIGHT` | 50 | Mean brightness threshold for night classification |
| `BRIGHTNESS_DUSK` | 120 | Mean brightness threshold for dusk classification |
| `FOG_THRESHOLD` | 0.6 | Fog index threshold for triggering dehazing |
| `RAIN_STD_THRESHOLD` | 35.0 | Laplacian std threshold for rain detection |
| `RAIN_VERT_RATIO` | 1.5 | Vertical-to-horizontal edge ratio for rain confirmation |
| `GAMMA_NIGHT` | 2.2 | Gamma value for night correction |
| `GAMMA_DUSK` | 1.5 | Gamma value for dusk correction |
| `CLAHE_CLIP_NIGHT` | 4.0 | CLAHE clip limit for night |
| `CLAHE_CLIP_DUSK` | 2.5 | CLAHE clip limit for dusk |
| `CLAHE_TILE` | (8, 8) | CLAHE tile grid size |
| `BILATERAL_D` | 5 | Bilateral filter diameter |
| `BILATERAL_SIGMA_COLOR` | 50 | Bilateral filter colour sigma |
| `BILATERAL_SIGMA_SPACE` | 50 | Bilateral filter spatial sigma |
| `BLUR_KERNEL_RAIN` | (9, 9) | Gaussian blur kernel for rain conditions |
| `BINARY_THRESH_RAIN` | 50 | Binary threshold for rain conditions |
| `DILATE_ITER_RAIN` | 4 | Dilation iterations for rain conditions |

---

## 11. Usage

### Normal Mode (full enhancement quality)
```bash
python run.py --pipeline enhanced
```

### Fast Mode (optimised for speed)
```bash
python run.py --pipeline enhanced --fast
```

### Output
- **Video**: `output/videos/output_enhanced_cv.mp4`
- **Report (JSON)**: `output/reports/enhanced_<timestamp>.json`
- **Report (Markdown)**: `output/reports/enhanced_<timestamp>.md`
