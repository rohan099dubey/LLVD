import cv2
import numpy as np
import psutil
import time
import os
import json
import threading
from functools import wraps
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import atexit
from pathlib import Path
from queue import Queue
import math

# Setup logging directory
LOG_DIR = Path("output/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Detection parameters
PIXEL_THRESHOLD = 100
BLUR_KERNEL_DAY = (5, 5)
BLUR_KERNEL_FOG = (7, 7)
BINARY_THRESH_DAY = 75
BINARY_THRESH_NIGHT = 30
DILATE_ITER_DAY = 3
DILATE_ITER_NIGHT = 4

# Scene classification thresholds
BRIGHTNESS_NIGHT = 50
BRIGHTNESS_DUSK = 120
FOG_THRESHOLD = 0.6

# Rain detection thresholds
RAIN_STD_THRESHOLD = 35.0
RAIN_VERT_RATIO = 1.5

# Gamma correction for night/dusk
GAMMA_NIGHT = 2.2
GAMMA_DUSK = 1.5
GAMMA_LUT_NIGHT = np.array([((i / 255.0) ** (1.0 / GAMMA_NIGHT)) * 255
                             for i in range(256)]).astype("uint8")
GAMMA_LUT_DUSK = np.array([((i / 255.0) ** (1.0 / GAMMA_DUSK)) * 255
                            for i in range(256)]).astype("uint8")

# Bilateral filter params (edge-preserving denoise)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 50
BILATERAL_SIGMA_SPACE = 50

# Rain-specific adaptive detection params
BLUR_KERNEL_RAIN = (9, 9)
BINARY_THRESH_RAIN = 50
DILATE_ITER_RAIN = 4

# CLAHE settings for adaptive contrast
CLAHE_CLIP_NIGHT = 4.0
CLAHE_CLIP_DUSK = 2.5
CLAHE_TILE = (8, 8)

# Tracking settings
MAX_DISAPPEARED = 5
MAX_DISTANCE = 60
MIN_CLUSTER_SIZE = 2

# Speed estimation constants
PIXELS_PER_METER = 15.0
SPEED_EMA_ALPHA = 0.3
ASSUMED_FPS = 25.0

# Visualization settings
COUNTING_LINE_FRAC = 0.5
COLOR_OCCUPIED = (0, 255, 0)
COLOR_EMPTY = (0, 0, 255)
COLOR_TRACK = (255, 255, 0)
COLOR_COUNT_LINE = (0, 165, 255)
RECT_THICKNESS = 2

# Processing configs
WRITER_QUEUE_SIZE = 128
PREFETCH_BATCHES = 2
DEFAULT_BATCH_SIZE = 64

# Pre-computed kernels
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
MORPH_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Initialize CLAHE objects
CLAHE_NIGHT = cv2.createCLAHE(clipLimit=CLAHE_CLIP_NIGHT, tileGridSize=CLAHE_TILE)
CLAHE_DUSK = cv2.createCLAHE(clipLimit=CLAHE_CLIP_DUSK, tileGridSize=CLAHE_TILE)

# Thread-local storage for profiling
_timing_stack = threading.local()

def _current_stack():
    if not hasattr(_timing_stack, 'frames'):
        _timing_stack.frames = []
    return _timing_stack.frames

timing_stats = defaultdict(lambda: {
    'total_time': 0.0, 'call_count': 0,
    'min_time': float('inf'), 'max_time': 0.0
})

# Decorator to measure execution time of functions
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stack = _current_stack()
        frame = {'start': time.perf_counter(), 'child_time': 0.0}
        stack.append(frame)
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            f = stack.pop()
            elapsed = end - f['start']
            exclusive = max(0.0, elapsed - f['child_time'])
            s = timing_stats[func.__name__]
            s['total_time'] += exclusive
            s['call_count'] += 1
            s['min_time'] = min(s['min_time'], exclusive)
            s['max_time'] = max(s['max_time'], exclusive)
            if stack:
                stack[-1]['child_time'] += elapsed
    return wrapper

def _safe_serializable_stats(stats_dict):
    out = {}
    for name, s in stats_dict.items():
        mt = s['min_time']
        out[name] = {
            'total_time': float(s['total_time']),
            'call_count': int(s['call_count']),
            'min_time': float(mt) if mt != float('inf') else None,
            'max_time': float(s['max_time'])
        }
    return out

def dump_timing_stats_to_file_on_exit():
    try:
        with open(LOG_DIR / f"prof_{os.getpid()}.json", "w") as f:
            json.dump(_safe_serializable_stats(timing_stats), f)
    except Exception:
        pass

atexit.register(dump_timing_stats_to_file_on_exit)

# Load user configuration
with open("user_input_data.json", "r") as _f:
    _config = json.load(_f)

VIDEO_PATH = _config["video"]
COLOR_CHANNEL = _config["color_channel"]
NUM_ROWS = _config["grids"]["rows"]
NUM_COLS = _config["grids"]["cols"]
TOTAL_CELLS = NUM_ROWS * NUM_COLS

ROI1 = (545, 159, 284, 140)
ROI2 = (238, 161, 284, 140)

GRID_W1 = ROI1[2] // NUM_COLS
GRID_H1 = ROI1[3] // NUM_ROWS
GRID_W2 = ROI2[2] // NUM_COLS
GRID_H2 = ROI2[3] // NUM_ROWS

CHANNEL_MAP = {
    'H': [0], 'S': [1], 'V': [2],
    'H+S': [0, 1], 'H+V': [0, 2], 'S+V': [1, 2],
    'H+S+V': [0, 1, 2], 'gray': 'gray'
}
CHANNELS = CHANNEL_MAP[COLOR_CHANNEL]
_USE_GRAYSCALE_FASTPATH = (CHANNELS == 'gray' or CHANNELS == [2])

# classify scene based on brightness, fog density, and rain
@time_function
def classify_scene(frame_gray):
    brightness = float(np.mean(frame_gray))

    if brightness < BRIGHTNESS_NIGHT:
        lighting = 'night'
    elif brightness < BRIGHTNESS_DUSK:
        lighting = 'dusk'
    else:
        lighting = 'day'

    # Estimate fog using Dark Channel Prior
    dark_channel = cv2.erode(frame_gray, np.ones((15, 15), np.uint8))
    fog_index = 1.0 - float(np.mean(dark_channel)) / 255.0
    is_foggy = fog_index > FOG_THRESHOLD

    # Detect rain using Laplacian variance + vertical edge dominance
    lap = cv2.Laplacian(frame_gray, cv2.CV_64F)
    lap_std = float(np.std(lap))
    sobel_v = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_h = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    vert_ratio = (np.mean(np.abs(sobel_v)) + 1e-6) / (np.mean(np.abs(sobel_h)) + 1e-6)
    is_rainy = (lap_std > RAIN_STD_THRESHOLD) and (vert_ratio > RAIN_VERT_RATIO)

    return lighting, is_foggy, is_rainy, brightness, fog_index

# Remove haze using Dark Channel Prior and guided filter
@time_function
def dehaze_dcp(img_bgr, omega=0.95, t_min=0.1, radius=40):
    img = img_bgr.astype(np.float64) / 255.0
    
    # Calculate dark channel
    dark = np.min(img, axis=2)
    kernel = np.ones((15, 15), np.uint8)
    dark = cv2.erode(dark.astype(np.float32), kernel).astype(np.float64)

    # Estimate atmospheric light
    num_pixels = dark.size
    num_top = max(int(num_pixels * 0.001), 1)
    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_top)[-num_top:]
    flat_img = img.reshape(-1, 3)
    A = flat_img[indices].mean(axis=0)
    A = np.clip(A, 0.01, 1.0)

    # Transmission map
    normalized = img / A[np.newaxis, np.newaxis, :]
    dark_norm = np.min(normalized, axis=2)
    dark_norm = cv2.erode(dark_norm.astype(np.float32), kernel).astype(np.float64)
    t_map = 1.0 - omega * dark_norm

    # Refine transmission map
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, 'ximgproc'):
        t_refined = cv2.ximgproc.guidedFilter(guide, t_map.astype(np.float32), radius, 1e-3)
    else:
        t_refined = cv2.GaussianBlur(t_map.astype(np.float32), (radius * 2 + 1, radius * 2 + 1), 0)
    t_refined = np.clip(t_refined.astype(np.float64), t_min, 1.0)

    # Recover scene radiance
    J = np.empty_like(img)
    for c in range(3):
        J[:, :, c] = (img[:, :, c] - A[c]) / t_refined + A[c]
    
    return np.clip(J * 255, 0, 255).astype(np.uint8)

# Remove rain streaks using median blur and directional morphology
@time_function
def remove_rain_streaks(frame_bgr):
    """Remove rain streaks using channel-wise median + directional morph."""
    derained = cv2.medianBlur(frame_bgr, 5)
    # Horizontal morphological close to fill vertical streak gaps
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    derained = cv2.morphologyEx(derained, cv2.MORPH_CLOSE, h_kernel)
    return derained

# Apply gamma correction for dark frames
@time_function
def apply_gamma(gray, lighting):
    """Brighten dark frames using pre-computed gamma LUTs."""
    if lighting == 'night':
        return cv2.LUT(gray, GAMMA_LUT_NIGHT)
    elif lighting == 'dusk':
        return cv2.LUT(gray, GAMMA_LUT_DUSK)
    return gray

# Apply histogram equalization based on lighting condition
@time_function
def adaptive_equalize(roi_gray, lighting):
    if lighting == 'night':
        eq = CLAHE_NIGHT.apply(roi_gray)
    elif lighting == 'dusk':
        eq = CLAHE_DUSK.apply(roi_gray)
    else:
        eq = cv2.equalizeHist(roi_gray)
    # Edge-preserving denoise for low-light conditions
    if lighting in ('night', 'dusk'):
        eq = cv2.bilateralFilter(eq, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    return eq

# Get detection parameters suited for current scene
def get_adaptive_params(lighting, is_foggy, is_rainy=False):
    if is_rainy:
        return {'blur_kernel': BLUR_KERNEL_RAIN, 'binary_thresh': BINARY_THRESH_RAIN, 'dilate_iter': DILATE_ITER_RAIN}
    elif is_foggy:
        return {'blur_kernel': BLUR_KERNEL_FOG, 'binary_thresh': BINARY_THRESH_NIGHT if lighting == 'night' else 40, 'dilate_iter': DILATE_ITER_NIGHT}
    elif lighting == 'night':
        return {'blur_kernel': BLUR_KERNEL_DAY, 'binary_thresh': BINARY_THRESH_NIGHT, 'dilate_iter': DILATE_ITER_NIGHT}
    elif lighting == 'dusk':
        return {'blur_kernel': BLUR_KERNEL_DAY, 'binary_thresh': 40, 'dilate_iter': DILATE_ITER_DAY}
    else:
        return {'blur_kernel': BLUR_KERNEL_DAY, 'binary_thresh': BINARY_THRESH_DAY, 'dilate_iter': DILATE_ITER_DAY}

# Find connected components in binary mask to identify vehicles
@time_function
def find_connected_components(binary_matrix):
    if binary_matrix.sum() == 0:
        return []

    mat_u8 = binary_matrix.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mat_u8, connectivity=8)

    components = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < MIN_CLUSTER_SIZE:
            continue

        cells = list(zip(*np.where(labels == label_id)))
        cy, cx = centroids[label_id]
        
        min_row = int(stats[label_id, cv2.CC_STAT_TOP])
        min_col = int(stats[label_id, cv2.CC_STAT_LEFT])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])

        components.append({
            'cells': cells,
            'centroid': (float(cy), float(cx)),
            'area': area,
            'bbox': (min_row, min_col, min_row + h - 1, min_col + w - 1)
        })

    return components

# Tracks objects using centroid matching and handles ID assignment
class CentroidTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = {}
        self.speeds = {}
        self.prev_centroids = {}
        self.counted = set()
        self.total_count = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.speeds[self.next_id] = 0.0
        self.prev_centroids[self.next_id] = centroid
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.speeds[oid]
        del self.prev_centroids[oid]

    def update(self, components, grid_w, grid_h, roi_x, roi_y):
        input_centroids = []
        for comp in components:
            cx_grid, cy_grid = comp['centroid']
            px = roi_x + cx_grid * grid_w
            py = roi_y + cy_grid * grid_h
            input_centroids.append((px, py))

        if not input_centroids:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(input_centroids, dtype=np.float32)

        if not self.objects:
            for c in input_centroids:
                self.register(tuple(c))
            return self.objects

        # Match existing objects to new centroids based on distance
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid] for oid in object_ids], dtype=np.float32)

        diff = object_centroids[:, np.newaxis, :] - input_centroids[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            new_centroid = tuple(input_centroids[col])
            
            # Update speed
            old_centroid = self.objects[oid]
            displacement = math.sqrt((new_centroid[0] - old_centroid[0]) ** 2 + (new_centroid[1] - old_centroid[1]) ** 2)
            speed_kmh = (displacement / PIXELS_PER_METER) * ASSUMED_FPS * 3.6
            self.speeds[oid] = SPEED_EMA_ALPHA * speed_kmh + (1 - SPEED_EMA_ALPHA) * self.speeds[oid]

            self.prev_centroids[oid] = old_centroid
            self.objects[oid] = new_centroid
            self.disappeared[oid] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle disappeared and new objects
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in set(range(len(input_centroids))) - used_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects

    # Count vehicles crossing the predefined line
    def check_counting_line(self, roi_y, roi_h):
        line_y = roi_y + roi_h * COUNTING_LINE_FRAC
        new_counts = 0

        for oid, centroid in self.objects.items():
            if oid in self.counted:
                continue
            prev = self.prev_centroids.get(oid)
            if prev is None:
                continue

            prev_y, curr_y = prev[1], centroid[1]
            if (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y):
                self.counted.add(oid)
                self.total_count += 1
                new_counts += 1

        return new_counts

# Process grid efficiently using vectorized operations
@time_function
def process_grid_vectorized(channel_data, grid_w, grid_h, params):
    blur_k = params['blur_kernel']
    thresh = params['binary_thresh']
    dilate_i = params['dilate_iter']

    masks = []
    for ch in channel_data:
        blur = cv2.GaussianBlur(ch, blur_k, 0)
        _, th = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, DILATE_KERNEL, iterations=dilate_i)
        if dilate_i > DILATE_ITER_DAY:
            dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, MORPH_CLOSE_KERNEL)
        masks.append(dilated)

    if len(masks) == 1:
        combined = masks[0]
    else:
        combined = masks[0]
        for m in masks[1:]:
            cv2.bitwise_and(combined, m, dst=combined)

    usable_h = NUM_ROWS * grid_h
    usable_w = NUM_COLS * grid_w
    grid_region = combined[:usable_h, :usable_w]
    reshaped = grid_region.reshape(NUM_ROWS, grid_h, NUM_COLS, grid_w)
    cell_counts = reshaped.astype(np.uint32).sum(axis=(1, 3)) // 255

    return (cell_counts >= PIXEL_THRESHOLD).astype(np.int8)

# Process a single ROI with adaptive settings
@time_function
def process_single_roi(roi_f1, roi_f2, grid_w, grid_h, lighting, is_foggy, is_rainy=False):
    # Dehaze foggy ROIs
    if is_foggy:
        roi_f1 = dehaze_dcp(roi_f1)
        roi_f2 = dehaze_dcp(roi_f2)
    # Remove rain streaks
    if is_rainy:
        roi_f1 = remove_rain_streaks(roi_f1)
        roi_f2 = remove_rain_streaks(roi_f2)

    g1 = cv2.cvtColor(roi_f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(roi_f2, cv2.COLOR_BGR2GRAY)

    # Gamma correction for dark frames
    g1 = apply_gamma(g1, lighting)
    g2 = apply_gamma(g2, lighting)

    eq1 = adaptive_equalize(g1, lighting)
    eq2 = adaptive_equalize(g2, lighting)

    if _USE_GRAYSCALE_FASTPATH:
        ch_data = [cv2.absdiff(eq1, eq2)]
    else:
        eq1_bgr = cv2.cvtColor(eq1, cv2.COLOR_GRAY2BGR)
        eq2_bgr = cv2.cvtColor(eq2, cv2.COLOR_GRAY2BGR)
        diff_bgr = cv2.absdiff(eq1_bgr, eq2_bgr)
        hsv = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2HSV)
        ch_data = [cv2.split(hsv)[i] for i in CHANNELS]

    params = get_adaptive_params(lighting, is_foggy, is_rainy)
    return process_grid_vectorized(ch_data, grid_w, grid_h, params)

# Worker function for parallel processing
def worker_task(args):
    frame_idx, r1f1, r1f2, r2f1, r2f2, lighting, is_foggy, is_rainy = args

    mat1 = process_single_roi(r1f1, r1f2, GRID_W1, GRID_H1, lighting, is_foggy, is_rainy)
    mat2 = process_single_roi(r2f1, r2f2, GRID_W2, GRID_H2, lighting, is_foggy, is_rainy)

    d1 = float(np.count_nonzero(mat1)) / TOTAL_CELLS
    d2 = float(np.count_nonzero(mat2)) / TOTAL_CELLS

    comps1 = find_connected_components(mat1)
    comps2 = find_connected_components(mat2)

    comps1_data = [{'centroid': c['centroid'], 'area': c['area'], 'bbox': c['bbox']} for c in comps1]
    comps2_data = [{'centroid': c['centroid'], 'area': c['area'], 'bbox': c['bbox']} for c in comps2]

    return frame_idx, mat1, mat2, d1, d2, comps1_data, comps2_data

class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self._queue = Queue(maxsize=WRITER_QUEUE_SIZE)
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame):
        self._queue.put(frame)

    def release(self):
        self._queue.put(None)
        self._thread.join()
        self._writer.release()

def _load_image_sequence(path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = sorted(f for f in os.listdir(path) if Path(f).suffix.lower() in exts)
    return [os.path.join(path, f) for f in files]

def _crop_rois(frame):
    r1x, r1y, r1w, r1h = ROI1
    r2x, r2y, r2w, r2h = ROI2
    roi1 = np.ascontiguousarray(frame[r1y:r1y + r1h, r1x:r1x + r1w])
    roi2 = np.ascontiguousarray(frame[r2y:r2y + r2h, r2x:r2x + r2w])
    return roi1, roi2

@time_function
def _enhance_frame_for_display(frame, lighting, is_foggy, is_rainy):
    """Apply all visual enhancements to the output frame for display."""
    # Full-frame dehazing for fog
    if is_foggy:
        frame[:] = dehaze_dcp(frame)
    # Full-frame rain-streak removal
    if is_rainy:
        frame[:] = remove_rain_streaks(frame)

    # ROI-level equalization + gamma + bilateral
    for (rx, ry, rw, rh) in [ROI1, ROI2]:
        roi = frame[ry:ry + rh, rx:rx + rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = apply_gamma(gray, lighting)
        eq = adaptive_equalize(gray, lighting)
        frame[ry:ry + rh, rx:rx + rw] = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# Yields batches of ROI crops for processing
def batch_roi_generator(video_source, batch_size):
    """Yield batches with pre-cropped ROIs and scene condition flags."""
    is_seq = os.path.isdir(video_source)
    
    if is_seq:
        files = _load_image_sequence(video_source)
        if len(files) < 2:
            raise ValueError("Need at least 2 images in sequence")

        prev = cv2.imread(files[0])
        prev_rois = _crop_rois(prev)
        batch = []
        gray_scene = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        lighting, is_foggy, is_rainy, _, _ = classify_scene(gray_scene)

        for i in range(1, len(files)):
            curr = cv2.imread(files[i])
            if curr is None: continue
            curr_rois = _crop_rois(curr)

            batch.append((i, prev_rois[0], curr_rois[0], prev_rois[1], curr_rois[1], prev, lighting, is_foggy, is_rainy))

            prev = curr
            prev_rois = curr_rois

            if len(batch) >= batch_size:
                yield batch
                batch = []
                gray_scene = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                lighting, is_foggy, is_rainy, _, _ = classify_scene(gray_scene)

        if batch: yield batch
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_source}")

        ret, prev = cap.read()
        if not ret: return

        prev_rois = _crop_rois(prev)
        batch = []
        idx = 1
        gray_scene = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        lighting, is_foggy, is_rainy, _, _ = classify_scene(gray_scene)

        while True:
            ret, curr = cap.read()
            if not ret: break
            curr_rois = _crop_rois(curr)

            batch.append((idx, prev_rois[0], curr_rois[0], prev_rois[1], curr_rois[1], prev, lighting, is_foggy, is_rainy))

            prev = curr
            prev_rois = curr_rois
            idx += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []
                gray_scene = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                lighting, is_foggy, is_rainy, _, _ = classify_scene(gray_scene)

        if batch: yield batch
        cap.release()

def prefetch_batches(gen, count=PREFETCH_BATCHES):
    q = Queue(maxsize=count)
    def _reader():
        for batch in gen:
            q.put(batch)
        q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    while True:
        batch = q.get()
        if batch is None: break
        yield batch

@time_function
def draw_grid_overlay(frame, roi_x, roi_y, grid_w, grid_h, matrix):
    for row in range(NUM_ROWS):
        y1 = roi_y + row * grid_h
        y2 = y1 + grid_h
        for col in range(NUM_COLS):
            x1 = roi_x + col * grid_w
            x2 = x1 + grid_w
            color = COLOR_OCCUPIED if matrix[row, col] else COLOR_EMPTY
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, RECT_THICKNESS)

@time_function
def draw_tracking_info(frame, tracker, roi_x, roi_y, roi_h, label=""):
    line_y = int(roi_y + roi_h * COUNTING_LINE_FRAC)
    cv2.line(frame, (roi_x, line_y), (roi_x + ROI1[2], line_y), COLOR_COUNT_LINE, 2)

    for oid, centroid in tracker.objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        speed = tracker.speeds.get(oid, 0.0)
        cv2.circle(frame, (cx, cy), 4, COLOR_TRACK, -1)
        label_text = f"#{oid}" + (f" {speed:.0f}km/h" if speed > 1.0 else "")
        cv2.putText(frame, label_text, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TRACK, 1)

    cv2.putText(frame, f"{label} Count: {tracker.total_count}", (roi_x, roi_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

@time_function
def draw_scene_info(frame, lighting, is_foggy, is_rainy, brightness, fog_idx):
    condition = lighting.upper()
    if is_foggy:
        condition += " + FOG"
    if is_rainy:
        condition += " + RAIN"
    color = {'day': (0, 255, 0), 'dusk': (0, 200, 255), 'night': (150, 100, 255)}.get(lighting, (255, 255, 255))
    cv2.putText(frame, f"Scene: {condition}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Bright: {brightness:.0f} Fog: {fog_idx:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def _merge_into(merged, name, total, calls, mn, mx):
    if name not in merged:
        merged[name] = {
            'total_time': total, 'call_count': calls,
            'min_time': mn, 'max_time': mx
        }
    else:
        m = merged[name]
        m['total_time'] += total
        m['call_count'] += calls
        m['min_time'] = min(m['min_time'], mn)
        m['max_time'] = max(m['max_time'], mx)


def merge_profile_logs(main_stats=None):
    merged = {}

    # 1) Merge worker process JSON logs
    main_pid = os.getpid()
    for fp in sorted(LOG_DIR.glob("prof_*.json")):
        # Skip the main process log to avoid double-counting
        try:
            pid_str = fp.stem.split('_')[1]
            if int(pid_str) == main_pid:
                continue
        except (IndexError, ValueError):
            pass
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        for name, s in data.items():
            total = float(s.get('total_time', 0))
            calls = int(s.get('call_count', 0))
            mn = s.get('min_time')
            mx = float(s.get('max_time', 0))
            mn = float(mn) if mn is not None else float('inf')
            _merge_into(merged, name, total, calls, mn, mx)

    # 2) Merge main process stats directly from memory
    if main_stats:
        for name, s in main_stats.items():
            total = float(s['total_time'])
            calls = int(s['call_count'])
            mn = float(s['min_time']) if s['min_time'] != float('inf') else float('inf')
            mx = float(s['max_time'])
            _merge_into(merged, name, total, calls, mn, mx)

    for s in merged.values():
        if s['min_time'] == float('inf'):
            s['min_time'] = None
    return merged

def print_profiling_results(exec_time, stats):
    print(f"\n{'=' * 90}\nFUNCTION EXCLUSIVE WALL-CLOCK TIME ANALYSIS\n{'=' * 90}")
    print(f"{'Function':<40} {'Calls':<10} {'Excl(s)':<12} {'Avg(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12}")
    print("-" * 98)

    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    total_profiled = sum(s['total_time'] for _, s in sorted_stats)

    for name, s in sorted_stats:
        if s['call_count'] > 0:
            avg = (s['total_time'] / s['call_count']) * 1000
            mn = (s['min_time'] * 1000) if s['min_time'] is not None else 0
            mx = s['max_time'] * 1000
            pct = (s['total_time'] / exec_time * 100) if exec_time > 0 else 0
            print(f"{name:<40} {s['call_count']:<10} {s['total_time']:<12.4f} {avg:<12.4f} {mn:<12.4f} {mx:<12.4f} ({pct:>5.1f}%)")

    overhead = ((exec_time - total_profiled) / exec_time * 100 if exec_time > 0 else 0)
    print(f"\nProfiled Functions: {total_profiled:.2f}s\nTotal Execution:   {exec_time:.2f}s\nOverhead (IPC/IO): {overhead:.1f}%\n{'=' * 90}")

def main():
    start = time.time()
    
    # Clean up old logs
    for fp in LOG_DIR.glob("prof_*.json"):
        try: fp.unlink()
        except: pass

    is_seq = os.path.isdir(VIDEO_PATH)
    if is_seq:
        files = _load_image_sequence(VIDEO_PATH)
        first = cv2.imread(files[0])
        h, w = first.shape[:2]
        total_frames = len(files) - 1
        fps = ASSUMED_FPS
        del first
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        fps = cap.get(cv2.CAP_PROP_FPS) or ASSUMED_FPS
        cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs('output/videos', exist_ok=True)
    writer = AsyncVideoWriter('output/videos/output_enhanced_cv.mp4', fourcc, fps, (w, h))

    # Initial scene check
    if is_seq:
        first = cv2.imread(_load_image_sequence(VIDEO_PATH)[0])
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        _, first = cap.read()
        cap.release()
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    init_lighting, init_foggy, init_rainy, init_bright, init_fog_idx = classify_scene(first_gray)
    del first, first_gray

    tracker1 = CentroidTracker()
    tracker2 = CentroidTracker()
    batch_size = DEFAULT_BATCH_SIZE
    n_procs = max(1, cpu_count() - 1)

    print(f"Pipeline: {n_procs} workers | batch={batch_size}")
    print(f"Scene: {init_lighting.upper()}"
          f"{' + FOG' if init_foggy else ''}"
          f"{' + RAIN' if init_rainy else ''}"
          f" (bright={init_bright:.0f}, fog_idx={init_fog_idx:.2f})")
    print(f"Features: Tracking | Counting | Speed | Night/Fog/Rain Adapt")
    print(f"Grid: {NUM_ROWS}x{NUM_COLS} | "
          f"Fast-path: {'grayscale' if _USE_GRAYSCALE_FASTPATH else 'HSV'}")

    d_lane1, d_lane2 = [], []
    speed_samples_l1, speed_samples_l2 = [], []
    scene_log = []
    frame_count = 0

    with Pool(processes=n_procs) as pool:
        pbar = tqdm(total=total_frames, desc="Processing frames")
        gen = batch_roi_generator(VIDEO_PATH, batch_size)
        
        for batch in prefetch_batches(gen):
            batch_lighting, batch_foggy, batch_rainy = batch[0][6], batch[0][7], batch[0][8]
            scene_log.append((batch_lighting, batch_foggy, batch_rainy))

            worker_args = [(idx, r1f1, r1f2, r2f1, r2f2, lighting, foggy, rainy) for idx, r1f1, r1f2, r2f1, r2f2, _, lighting, foggy, rainy in batch]
            original_frames = {idx: frame for idx, _, _, _, _, frame, _, _, _ in batch}

            # Parallel ROI processing (timed to capture IPC overhead)
            ipc_start = time.perf_counter()
            results = pool.map(worker_task, worker_args)
            ipc_elapsed = time.perf_counter() - ipc_start
            s = timing_stats['pool_map_ipc']
            s['total_time'] += ipc_elapsed
            s['call_count'] += 1
            s['min_time'] = min(s['min_time'], ipc_elapsed)
            s['max_time'] = max(s['max_time'], ipc_elapsed)
            results.sort(key=lambda x: x[0])

            for fidx, mat1, mat2, d1, d2, comps1, comps2 in results:
                frame = original_frames[fidx]
                _enhance_frame_for_display(frame, batch_lighting, batch_foggy, batch_rainy)

                draw_grid_overlay(frame, ROI1[0], ROI1[1], GRID_W1, GRID_H1, mat1)
                draw_grid_overlay(frame, ROI2[0], ROI2[1], GRID_W2, GRID_H2, mat2)

                tracker1.update(comps1, GRID_W1, GRID_H1, ROI1[0], ROI1[1])
                tracker2.update(comps2, GRID_W2, GRID_H2, ROI2[0], ROI2[1])

                tracker1.check_counting_line(ROI1[1], ROI1[3])
                tracker2.check_counting_line(ROI2[1], ROI2[3])

                draw_tracking_info(frame, tracker1, ROI1[0], ROI1[1], ROI1[3], "L1")
                draw_tracking_info(frame, tracker2, ROI2[0], ROI2[1], ROI2[3], "L2")

                draw_scene_info(frame, batch_lighting, batch_foggy, batch_rainy, init_bright, init_fog_idx)
                cv2.putText(frame, f"Frame: {fidx}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                d_lane1.append(d1)
                d_lane2.append(d2)
                for s in tracker1.speeds.values():
                    if s > 1.0: speed_samples_l1.append(s)
                for s in tracker2.speeds.values():
                    if s > 1.0: speed_samples_l2.append(s)

                writer.write(frame)
                frame_count += 1

            pbar.update(len(results))
            del batch, worker_args, original_frames, results

        pbar.close()

    writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    elapsed = time.time() - start
    mem_mb = psutil.Process().memory_info().rss / (1024 ** 2)
    
    avg_speed_l1 = np.mean(speed_samples_l1) if speed_samples_l1 else 0.0
    avg_speed_l2 = np.mean(speed_samples_l2) if speed_samples_l2 else 0.0

    # Scene condition summary
    scene_counts = defaultdict(int)
    for lit, fog, rain in scene_log:
        key = lit
        if fog:
            key += '+fog'
        if rain:
            key += '+rain'
        scene_counts[key] += 1

    print(f"\n{'=' * 70}")
    print("EXECUTION RESULTS (ADVANCED CLASSICAL CV)")
    print(f"{'=' * 70}")
    print(f"Execution Time:       {elapsed:.2f}s")
    print(f"Memory Usage:         {mem_mb:.2f} MB")
    print(f"Frames Processed:     {frame_count}")
    print(f"Processing Speed:     {frame_count / elapsed:.2f} FPS")
    print(f"{'─' * 70}")
    print(f"DENSITY:")
    print(f"  Avg Lane 1:         {np.mean(d_lane1):.4f}")
    print(f"  Avg Lane 2:         {np.mean(d_lane2):.4f}")
    print(f"{'─' * 70}")
    print(f"VEHICLE COUNTING:")
    print(f"  Lane 1 Total:       {tracker1.total_count}")
    print(f"  Lane 2 Total:       {tracker2.total_count}")
    print(f"  Combined:           {tracker1.total_count + tracker2.total_count}")
    print(f"{'─' * 70}")
    print(f"SPEED ESTIMATION:")
    print(f"  Lane 1 Avg:         {avg_speed_l1:.1f} km/h")
    print(f"  Lane 2 Avg:         {avg_speed_l2:.1f} km/h")
    print(f"  Lane 1 Samples:     {len(speed_samples_l1)}")
    print(f"  Lane 2 Samples:     {len(speed_samples_l2)}")
    print(f"{'─' * 70}")
    print(f"SCENE CONDITIONS:")
    for cond, cnt in sorted(scene_counts.items()):
        print(f"  {cond:<20} {cnt} batches")
    print(f"{'─' * 70}")
    print(f"TRACKING:")
    print(f"  Total IDs assigned: {tracker1.next_id + tracker2.next_id}")
    print(f"  Active L1 tracks:   {len(tracker1.objects)}")
    print(f"  Active L2 tracks:   {len(tracker2.objects)}")
    print(f"{'=' * 70}")

    return elapsed

if __name__ == '__main__':
    print(f"\n{'=' * 70}")
    print("ADVANCED CLASSICAL CV TRAFFIC ANALYSIS")
    print(f"{'=' * 70}\n")

    execution_time = main()

    # Merge worker JSON logs + main process in-memory stats directly
    merged_stats = merge_profile_logs(main_stats=timing_stats)
    if not merged_stats:
        print("\n[Warning] No profiling logs found.")
    print_profiling_results(execution_time, merged_stats)

    print(f"\n{'=' * 90}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 90}")
