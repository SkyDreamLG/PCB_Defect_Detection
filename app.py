# coding:utf-8
from collections import deque

from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
from flask import Flask, render_template, request, jsonify, session, send_file
import tempfile
import uuid
from threading import Thread, Lock
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta
import psutil
import gc
import zipfile
import csv
import sqlite3
import json
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

DEFECT_MAP = {
    'missing_hole': '缺失孔',
    'mouse_bite': '鼠咬缺陷',
    'short_circuit': '短路'
}

COLOR_MAP = {
    'missing_hole': (255, 0, 0),
    'mouse_bite': (0, 255, 0),
    'short_circuit': (0, 0, 255)
}

DEFECT_DESCRIPTIONS = {
    'missing_hole': 'PCB板上缺失的钻孔',
    'mouse_bite': '类似老鼠咬过的弧形缺陷',
    'short_circuit': '线路之间的异常连接'
}

processing_status = {}
status_lock = Lock()
detector_instance = None
detector_lock = Lock()
task_manager = None
db_manager = None


class DatabaseManager:
    def __init__(self, db_path='pcb_defect_history.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_path TEXT,
                        detected_file_path TEXT,
                        detection_time TEXT NOT NULL,
                        has_defect BOOLEAN DEFAULT 0,
                        defect_count INTEGER DEFAULT 0,
                        defects_json TEXT,
                        conf_threshold REAL,
                        iou_threshold REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_records(detection_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detection_records(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_has_defect ON detection_records(has_defect)')
                conn.commit()
                logger.info("✅ 数据库初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def save_record(self, record):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                tz = timezone(timedelta(hours=8))
                detection_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    INSERT INTO detection_records 
                    (task_id, filename, file_path, detected_file_path, detection_time, 
                     has_defect, defect_count, defects_json, conf_threshold, iou_threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['task_id'],
                    record['filename'],
                    record.get('file_path', ''),
                    record.get('detected_file_path', ''),
                    detection_time,
                    record.get('has_defect', False),
                    record.get('defect_count', 0),
                    json.dumps(record.get('defects', []), ensure_ascii=False),
                    record.get('conf_threshold', 0.25),
                    record.get('iou_threshold', 0.45)
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"保存记录失败: {e}")
            return None

    def query_records(self, start_date=None, end_date=None, has_defect=None, filename=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT * FROM detection_records WHERE 1=1"
                params = []
                if start_date:
                    query += " AND detection_time >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND detection_time <= ?"
                    params.append(end_date + " 23:59:59")
                if has_defect is not None:
                    query += " AND has_defect = ?"
                    params.append(1 if has_defect else 0)
                if filename:
                    query += " AND filename LIKE ?"
                    params.append(f"%{filename}%")
                query += " ORDER BY detection_time DESC"
                cursor.execute(query, params)
                records = cursor.fetchall()
                result = []
                for record in records:
                    record_dict = dict(record)
                    if record_dict['defects_json']:
                        record_dict['defects'] = json.loads(record_dict['defects_json'])
                    else:
                        record_dict['defects'] = []
                    result.append(record_dict)
                return result
        except Exception as e:
            logger.error(f"查询记录失败: {e}")
            return []

    def get_record_by_id(self, record_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM detection_records WHERE id = ?", (record_id,))
                record = cursor.fetchone()
                if record:
                    record_dict = dict(record)
                    if record_dict['defects_json']:
                        record_dict['defects'] = json.loads(record_dict['defects_json'])
                    else:
                        record_dict['defects'] = []
                    return record_dict
                return None
        except Exception as e:
            logger.error(f"获取记录失败: {e}")
            return None


class TaskManager:
    def __init__(self, max_workers=2):
        self.tasks = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cleanup_thread = Thread(target=self._cleanup_old_tasks, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"任务管理器初始化完成，最大并发: {max_workers}")

    def create_task(self, task_id, total_files):
        with self.lock:
            self.tasks[task_id] = {
                'id': task_id,
                'status': 'processing',
                'total': total_files,
                'completed': 0,
                'progress': 0,
                'results': [None] * total_files,
                'errors': [],
                'created_at': time.time(),
                'last_updated': time.time()
            }
        return self.tasks[task_id]

    def update_task(self, task_id, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['last_updated'] = time.time()
                if 'completed' in kwargs and self.tasks[task_id]['total'] > 0:
                    progress = (kwargs['completed'] / self.tasks[task_id]['total']) * 100
                    self.tasks[task_id]['progress'] = progress
                    if kwargs['completed'] >= self.tasks[task_id]['total']:
                        self.tasks[task_id]['status'] = 'completed'
                        self.tasks[task_id]['progress'] = 100

    def get_task(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                return task.copy()
            return None

    def _cleanup_old_tasks(self):
        while True:
            time.sleep(300)
            try:
                current_time = time.time()
                with self.lock:
                    to_delete = []
                    for task_id, task in self.tasks.items():
                        if task['status'] in ['completed', 'error']:
                            if current_time - task['last_updated'] > 1800:
                                to_delete.append(task_id)
                    for task_id in to_delete:
                        del self.tasks[task_id]
                    if to_delete:
                        logger.info(f"清理了 {len(to_delete)} 个旧任务")
            except Exception as e:
                logger.error(f"清理任务时出错: {e}")


class ModelManager:
    def __init__(self, model_path='PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'):
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.lock = Lock()
        self.load_model()

    def _get_device(self):
        if torch.cuda.is_available():
            device = 0
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("使用CPU")
        return device

    def load_model(self):
        try:
            with self.lock:
                if self.model is None:
                    logger.info(f"加载模型: {self.model_path}")
                    torch_version = torch.__version__
                    logger.info(f"PyTorch版本: {torch_version}")
                    major_version = int(torch_version.split('.')[0])
                    minor_version = int(torch_version.split('.')[1])
                    if major_version >= 2 and minor_version >= 6:
                        logger.info("检测到PyTorch 2.6+，使用兼容模式加载")
                        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
                        try:
                            import ultralytics.nn.tasks
                            torch.serialization.add_safe_globals([ultralytics.nn.tasks.YOLOv10DetectionModel])
                            self.model = YOLO(self.model_path)
                        except Exception as e:
                            logger.warning(f"安全加载失败，尝试备用方法: {e}")
                            import warnings
                            warnings.filterwarnings("ignore", category=UserWarning)
                            original_load = torch.load
                            def patched_load(f, *args, **kwargs):
                                kwargs['weights_only'] = False
                                return original_load(f, *args, **kwargs)
                            torch.load = patched_load
                            try:
                                self.model = YOLO(self.model_path)
                                logger.info("使用猴子补丁加载成功")
                            except Exception as e2:
                                logger.error(f"猴子补丁加载失败: {e2}")
                                torch.load = original_load
                                raise
                            finally:
                                torch.load = original_load
                    else:
                        self.model = YOLO(self.model_path)
                    logger.info("✅ 模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error("请尝试以下解决方案：")
            logger.error("1. 升级ultralytics包: pip install -U ultralytics")
            logger.error("2. 或者降级PyTorch: pip install torch==2.5.1")
            logger.error("3. 或者手动设置环境变量后运行: export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 && python app.py")
            raise

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        try:
            with self.lock:
                results = self.model(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False
                )
            return results
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

    def draw_detections(self, img, results):
        img_with_boxes = img.copy()
        detection_list = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = results[0].names[cls_id]
                    cls_name_cn = DEFECT_MAP.get(cls_name, cls_name)
                    color = COLOR_MAP.get(cls_name, (255, 255, 255))
                    description = DEFECT_DESCRIPTIONS.get(cls_name, '')
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    cv2.rectangle(
                        img_with_boxes,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width + 5, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        img_with_boxes,
                        label,
                        (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )
                    detection_list.append({
                        'defect_type': cls_name_cn,
                        'defect_type_en': cls_name,
                        'description': description,
                        'confidence': conf,
                        'coordinates': [int(x1), int(y1), int(x2), int(y2)],
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    })
        return img_with_boxes, detection_list


import warnings


class CircuitBoardDetector:
    def __init__(self):
        self.tracked_boards = []
        self.frame_count = 0
        self.last_stable_detection = []
        self.stable_counter = 0
        self.detection_history = deque(maxlen=5)
        self.fps = 0
        self.last_time = time.time()
        self.fps_history = deque(maxlen=30)
        self.min_area_ratio = 0.005
        self.max_area_ratio = 0.6
        self.square_threshold = 0.6
        self.morph_kernel_size = 5
        self.gaussian_kernel_size = 5
        self.denoise_strength = 10
        self.denoise_template_size = 7
        self.denoise_search_size = 21

    def preprocess_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
        return blurred

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def denoise_image(self, image):
        if image is None or image.size == 0:
            return image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None,
            self.denoise_strength, self.denoise_strength,
            self.denoise_template_size, self.denoise_search_size
        )
        return denoised

    def morphological_edge_detection(self, img):
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient_h = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_h)
        gradient_v = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_v)
        gradient_cross = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_cross)
        gradient = cv2.bitwise_or(gradient_h, gradient_v)
        gradient = cv2.bitwise_or(gradient, gradient_cross)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_rect)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_rect)
        combined = cv2.bitwise_or(gradient, tophat)
        combined = cv2.bitwise_or(combined, blackhat)
        return combined

    def detect_multiple_quadrilaterals(self, frame, max_count=5):
        processed = self.preprocess_image(frame)
        edges1 = cv2.Canny(processed, 30, 90)
        edges2 = cv2.Canny(processed, 50, 150)
        edges3 = cv2.Canny(processed, 70, 200)
        edges = cv2.bitwise_or(edges1, edges2)
        edges = cv2.bitwise_or(edges, edges3)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quadrilaterals = []
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * 0.005
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if 3 <= len(approx) <= 6:
                if len(approx) != 4:
                    epsilon2 = 0.01 * cv2.arcLength(contour, True)
                    approx2 = cv2.approxPolyDP(contour, epsilon2, True)
                    if len(approx2) == 4:
                        approx = approx2
                if len(approx) == 4:
                    if cv2.isContourConvex(approx):
                        x, y, w, h = cv2.boundingRect(approx)
                        rect_area = w * h
                        rectangularity = area / rect_area if rect_area > 0 else 0
                        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                        angles = []
                        for i in range(4):
                            p1 = approx[i][0]
                            p2 = approx[(i + 1) % 4][0]
                            p3 = approx[(i + 2) % 4][0]
                            v1 = p1 - p2
                            v2 = p3 - p2
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                            angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                            angles.append(angle)
                        angle_score = 1 - np.mean([abs(a - 90) for a in angles]) / 90
                        confidence = (
                                rectangularity * 0.4 +
                                angle_score * 0.3 +
                                (1 / max(aspect_ratio, 0.1)) * 0.3
                        )
                        quadrilaterals.append({
                            'contour': approx,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2),
                            'rectangularity': rectangularity,
                            'aspect_ratio': aspect_ratio,
                            'confidence': confidence,
                            'angles': angles
                        })
        quadrilaterals.sort(key=lambda x: x['area'], reverse=True)
        return quadrilaterals[:max_count]

    def temporal_smoothing(self, current_detections):
        self.detection_history.append(current_detections)
        if len(self.detection_history) < 3:
            return current_detections
        smoothed_detections = []
        for det in current_detections:
            match_count = 0
            for hist_dets in self.detection_history:
                for hist_det in hist_dets:
                    dist = np.sqrt((det['center'][0] - hist_det['center'][0]) ** 2 +
                                   (det['center'][1] - hist_det['center'][1]) ** 2)
                    size_diff = abs(det['bbox'][2] - hist_det['bbox'][2]) / max(det['bbox'][2], 1)
                    if dist < 50 and size_diff < 0.3:
                        match_count += 1
                        break
            if match_count >= len(self.detection_history) // 2:
                smoothed_detections.append(det)
        return smoothed_detections

    def detect_quad_in_roi(self, roi):
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_quad = None
        max_area = 0
        roi_area = roi.shape[0] * roi.shape[1]
        min_area = roi_area * 0.3
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0
                if rectangularity > 0.7:
                    if area > max_area:
                        max_area = area
                        best_quad = approx.reshape(-1, 2)
        return best_quad

    def crop_background(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(roi.shape[1] - x, w + 2 * margin)
            h = min(roi.shape[0] - y, h + 2 * margin)
            return roi[y:y + h, x:x + w]
        return roi


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED_FOLDER'] = 'detected'

from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1
)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

task_manager = TaskManager(max_workers=2)
db_manager = DatabaseManager()


def get_detector():
    global detector_instance
    with detector_lock:
        if detector_instance is None:
            model_path = 'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__),
                                          'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt')
            detector_instance = ModelManager(model_path)
    return detector_instance


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_batch_detect', methods=['POST'])
def start_batch_detect():
    try:
        files = request.files.getlist('files')
        conf_threshold = float(request.form.get('conf_threshold', 0.25))
        iou_threshold = float(request.form.get('iou_threshold', 0.45))
        if not files:
            return jsonify({'error': '请上传图片文件'})
        if len(files) > 1000:
            return jsonify({'error': '为了系统性能，单次最多只能上传1000张图片'})
        file_contents = []
        for file in files:
            file.seek(0)
            content = file.read()
            file_contents.append({
                'filename': file.filename,
                'content': content
            })
        task_id = str(uuid.uuid4())
        task_manager.create_task(task_id, len(file_contents))
        thread = Thread(
            target=async_process_batch,
            args=(task_id, file_contents, conf_threshold, iou_threshold)
        )
        thread.daemon = True
        thread.start()
        logger.info(f"任务创建成功: {task_id}, 文件数: {len(file_contents)}")
        return jsonify({
            'task_id': task_id,
            'message': '任务已创建',
            'total_files': len(file_contents)
        })
    except Exception as e:
        logger.error(f"创建任务失败: {e}")
        return jsonify({'error': str(e)}), 500


def async_process_batch(task_id, file_contents, conf_threshold, iou_threshold):
    try:
        detector = get_detector()
        total_files = len(file_contents)
        logger.info(f"开始处理任务 {task_id}, 共 {total_files} 张图片")
        for idx, file_data in enumerate(file_contents):
            try:
                task_manager.update_task(task_id, status='processing')
                img = Image.open(BytesIO(file_data['content']))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                results = detector.predict(
                    img_bgr,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                img_with_boxes, detections = detector.draw_detections(img_bgr, results)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{file_data['filename']}"
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                cv2.imwrite(original_path, img_bgr)
                detected_path = None
                if detections:
                    detected_filename = f"detected_{safe_filename}"
                    detected_path = os.path.join(app.config['DETECTED_FOLDER'], detected_filename)
                    cv2.imwrite(detected_path, img_with_boxes)
                original_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                detected_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
                original_buffer = BytesIO()
                detected_buffer = BytesIO()
                original_pil.save(original_buffer, format='JPEG', quality=85, optimize=True)
                detected_pil.save(detected_buffer, format='JPEG', quality=85, optimize=True)
                result_item = {
                    'filename': file_data['filename'],
                    'original_image': f"data:image/jpeg;base64,{base64.b64encode(original_buffer.getvalue()).decode()}",
                    'detected_image': f"data:image/jpeg;base64,{base64.b64encode(detected_buffer.getvalue()).decode()}",
                    'detections': detections,
                    'status': 'completed',
                    'index': idx,
                    'file_path': original_path,
                    'detected_file_path': detected_path
                }
                db_record = {
                    'task_id': task_id,
                    'filename': file_data['filename'],
                    'file_path': original_path,
                    'detected_file_path': detected_path,
                    'has_defect': len(detections) > 0,
                    'defect_count': len(detections),
                    'defects': detections,
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold
                }
                db_manager.save_record(db_record)
                with status_lock:
                    task = task_manager.get_task(task_id)
                    if task:
                        task['results'][idx] = result_item
                        completed = task['completed'] + 1
                        task_manager.update_task(task_id, completed=completed)
                logger.info(f"任务 {task_id} 进度: {idx + 1}/{total_files}")
                if idx % 5 == 4:
                    gc.collect()
            except Exception as e:
                logger.error(f"处理文件 {file_data['filename']} 失败: {e}")
                error_item = {
                    'filename': file_data['filename'],
                    'original_image': '',
                    'detected_image': '',
                    'detections': [],
                    'error': str(e),
                    'status': 'error',
                    'index': idx
                }
                with status_lock:
                    task = task_manager.get_task(task_id)
                    if task:
                        task['results'][idx] = error_item
                        completed = task['completed'] + 1
                        task_manager.update_task(task_id, completed=completed)
        logger.info(f"任务 {task_id} 处理完成")
    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {e}")
        task_manager.update_task(task_id, status='error', error=str(e))


@app.route('/get_detection_result/<task_id>')
def get_detection_result(task_id):
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({'status': 'not_found'})
        response = {
            'status': task['status'],
            'progress': task.get('progress', 0),
            'results': []
        }
        for result in task.get('results', []):
            if result and result.get('status') in ['completed', 'error']:
                response['results'].append(result)
        return jsonify(response)
    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/download_results', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        results = data.get('results', [])
        if not results:
            return jsonify({'error': '没有结果数据'}), 400
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('original_images/', '')
            zip_file.writestr('detected_images/', '')
            for result in results:
                filename = result['filename']
                if result['original_image']:
                    base64_data = result['original_image'].split(',')[1]
                    img_bytes = base64.b64decode(base64_data)
                    zip_file.writestr(f'original_images/{filename}', img_bytes)
                if result['detected_image']:
                    base64_data = result['detected_image'].split(',')[1]
                    img_bytes = base64.b64decode(base64_data)
                    detected_filename = f'{os.path.splitext(filename)[0]}_detected{os.path.splitext(filename)[1]}'
                    zip_file.writestr(f'detected_images/{detected_filename}', img_bytes)
            csv_buffer = StringIO()
            fieldnames = ['filename', 'defect_type', 'defect_description', 'confidence', 'x1', 'y1', 'x2', 'y2',
                          'width', 'height']
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                if result['detections']:
                    for detection in result['detections']:
                        writer.writerow({
                            'filename': result['filename'],
                            'defect_type': detection['defect_type'],
                            'defect_description': detection['description'],
                            'confidence': detection['confidence'],
                            'x1': detection['coordinates'][0],
                            'y1': detection['coordinates'][1],
                            'x2': detection['coordinates'][2],
                            'y2': detection['coordinates'][3],
                            'width': detection['width'],
                            'height': detection['height']
                        })
                else:
                    writer.writerow({
                        'filename': result['filename'],
                        'defect_type': '正常',
                        'defect_description': '无缺陷',
                        'confidence': '',
                        'x1': '',
                        'y1': '',
                        'x2': '',
                        'y2': '',
                        'width': '',
                        'height': ''
                    })
            zip_file.writestr('detection_report.csv', csv_buffer.getvalue())
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'pcb_defect_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        logger.error(f"下载结果失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query_history')
def query_history():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filename = request.args.get('filename')
        has_defect = request.args.get('has_defect')
        has_defect_bool = None
        if has_defect is not None and has_defect != '':
            has_defect_bool = has_defect == '1'
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date,
            filename=filename,
            has_defect=has_defect_bool
        )
        return jsonify({
            'records': records,
            'total': len(records)
        })
    except Exception as e:
        logger.error(f"查询历史记录失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_history_image/<int:record_id>')
def get_history_image(record_id):
    try:
        image_type = request.args.get('type', 'detected')
        record = db_manager.get_record_by_id(record_id)
        if not record:
            return '记录不存在', 404
        file_path = record['detected_file_path'] if image_type == 'detected' else record['file_path']
        if not file_path or not os.path.exists(file_path):
            return '图片不存在', 404
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"获取历史图片失败: {e}")
        return str(e), 500


@app.route('/download_history')
def download_history():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filename = request.args.get('filename')
        has_defect = request.args.get('has_defect')
        has_defect_bool = None
        if has_defect is not None and has_defect != '':
            has_defect_bool = has_defect == '1'
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date,
            filename=filename,
            has_defect=has_defect_bool
        )
        if not records:
            return jsonify({'error': '没有符合条件的数据'}), 400
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            csv_buffer = StringIO()
            fieldnames = ['id', 'filename', 'detection_time', 'has_defect', 'defect_count', 'defects_detail']
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                if record['file_path'] and os.path.exists(record['file_path']):
                    zip_file.write(record['file_path'], f"original/{record['filename']}")
                if record['detected_file_path'] and os.path.exists(record['detected_file_path']):
                    detected_filename = f"detected_{record['filename']}"
                    zip_file.write(record['detected_file_path'], f"detected/{detected_filename}")
                defects_detail = ''
                if record.get('defects'):
                    defects_detail = '; '.join(
                        [f"{d['defect_type']}({d['confidence']:.2f})" for d in record['defects']])
                writer.writerow({
                    'id': record['id'],
                    'filename': record['filename'],
                    'detection_time': record['detection_time'],
                    'has_defect': '是' if record['has_defect'] else '否',
                    'defect_count': record['defect_count'],
                    'defects_detail': defects_detail
                })
            zip_file.writestr('history_report.csv', csv_buffer.getvalue())
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        logger.error(f"下载历史记录失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download_history_item/<int:record_id>')
def download_history_item(record_id):
    try:
        record = db_manager.get_record_by_id(record_id)
        if not record:
            return jsonify({'error': '记录不存在'}), 404
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if record['file_path'] and os.path.exists(record['file_path']):
                zip_file.write(record['file_path'], f"original/{record['filename']}")
            if record['detected_file_path'] and os.path.exists(record['detected_file_path']):
                detected_filename = f"detected_{record['filename']}"
                zip_file.write(record['detected_file_path'], f"detected/{detected_filename}")
            info_buffer = StringIO()
            info_buffer.write(f"文件名: {record['filename']}\n")
            info_buffer.write(f"检测时间: {record['detection_time']}\n")
            info_buffer.write(f"是否有缺陷: {'是' if record['has_defect'] else '否'}\n")
            info_buffer.write(f"缺陷数量: {record['defect_count']}\n\n")
            if record.get('defects'):
                info_buffer.write("缺陷详情:\n")
                for i, defect in enumerate(record['defects'], 1):
                    info_buffer.write(f"  {i}. 类型: {defect['defect_type']}\n")
                    info_buffer.write(f"     置信度: {defect['confidence']:.2f}\n")
                    info_buffer.write(f"     位置: {defect['coordinates']}\n")
                    info_buffer.write(f"     尺寸: {defect['width']}x{defect['height']}\n\n")
            zip_file.writestr('details.txt', info_buffer.getvalue())
        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'history_item_{record_id}.zip'
        )
    except Exception as e:
        logger.error(f"下载历史记录项失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/history_detail/<int:record_id>')
def history_detail(record_id):
    record = db_manager.get_record_by_id(record_id)
    if not record:
        return '记录不存在', 404
    return render_template('history_detail.html', record=record)


@app.route('/system_status')
def system_status():
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        pending_tasks = 0
        with status_lock:
            for task_id, task in task_manager.tasks.items():
                if task['status'] == 'processing':
                    pending_tasks += 1
        return jsonify({
            'model_loaded': detector_instance is not None,
            'pending_tasks': pending_tasks,
            'memory_usage': f"{memory_mb:.1f} MB",
            'timestamp': time.time()
        })
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    return render_template('stats.html')


@app.route('/api/stats/overview')
def get_stats_overview():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date
        )
        total_count = len(records)
        defect_count = sum(1 for r in records if r['has_defect'])
        normal_count = total_count - defect_count
        defect_ratio = (defect_count / total_count * 100) if total_count > 0 else 0
        return jsonify({
            'total_count': total_count,
            'defect_count': defect_count,
            'normal_count': normal_count,
            'defect_ratio': round(defect_ratio, 2)
        })
    except Exception as e:
        logger.error(f"获取统计概览失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/defect_types')
def get_defect_types_stats():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date,
            has_defect=True
        )
        defect_counts = {
            'missing_hole': 0,
            'mouse_bite': 0,
            'short_circuit': 0
        }
        defect_names = {
            'missing_hole': '缺失孔',
            'mouse_bite': '鼠咬缺陷',
            'short_circuit': '短路'
        }
        for record in records:
            if record.get('defects'):
                for defect in record['defects']:
                    defect_type_en = defect.get('defect_type_en', '')
                    if defect_type_en in defect_counts:
                        defect_counts[defect_type_en] += 1
        result = []
        colors = {
            'missing_hole': '#f05454',
            'mouse_bite': '#8ecf8e',
            'short_circuit': '#5f9df3'
        }
        for defect_type_en, count in defect_counts.items():
            result.append({
                'name': defect_names.get(defect_type_en, defect_type_en),
                'name_en': defect_type_en,
                'count': count,
                'color': colors.get(defect_type_en, '#999999')
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取缺陷类型统计失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/daily')
def get_daily_stats_api():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        if not start_date or not end_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')
        daily_stats = get_daily_stats(start_date, end_date)
        return jsonify(daily_stats)
    except Exception as e:
        logger.error(f"获取每日统计失败: {e}")
        return jsonify({'error': str(e)}), 500


def get_daily_stats(start_date, end_date):
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    DATE(detection_time) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN has_defect = 1 THEN 1 ELSE 0 END) as defect_count
                FROM detection_records
                WHERE DATE(detection_time) BETWEEN ? AND ?
                GROUP BY DATE(detection_time)
                ORDER BY date
            ''', (start_date, end_date))
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    'date': row['date'],
                    'total': row['total'],
                    'defect': row['defect_count'] or 0,
                    'normal': row['total'] - (row['defect_count'] or 0)
                })
            return result
    except Exception as e:
        logger.error(f"获取每日统计失败: {e}")
        return []


@app.route('/api/stats/confidence_distribution')
def get_confidence_distribution():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date,
            has_defect=True
        )
        confidence_ranges = [
            {'range': '0-0.5', 'min': 0, 'max': 0.5, 'count': 0},
            {'range': '0.5-0.7', 'min': 0.5, 'max': 0.7, 'count': 0},
            {'range': '0.7-0.8', 'min': 0.7, 'max': 0.8, 'count': 0},
            {'range': '0.8-0.9', 'min': 0.8, 'max': 0.9, 'count': 0},
            {'range': '0.9-1.0', 'min': 0.9, 'max': 1.0, 'count': 0}
        ]
        for record in records:
            if record.get('defects'):
                for defect in record['defects']:
                    conf = defect.get('confidence', 0)
                    for range_item in confidence_ranges:
                        if range_item['min'] <= conf < range_item['max']:
                            range_item['count'] += 1
                            break
        return jsonify(confidence_ranges)
    except Exception as e:
        logger.error(f"获取置信度分布失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/export')
def export_stats():
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        defect_types = get_defect_types_stats().json
        daily_stats = get_daily_stats(start_date, end_date) if start_date and end_date else []
        output = StringIO()
        output.write("=== 缺陷类型统计 ===\n")
        writer = csv.writer(output)
        writer.writerow(['缺陷类型', '数量'])
        for item in defect_types:
            writer.writerow([item['name'], item['count']])
        output.write("\n\n=== 每日检测统计 ===\n")
        writer = csv.writer(output)
        writer.writerow(['日期', '总数', '缺陷数', '正常数'])
        for item in daily_stats:
            writer.writerow([item['date'], item['total'], item['defect'], item['normal']])
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode('utf-8-sig')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logger.error(f"导出统计失败: {e}")
        return jsonify({'error': str(e)}), 500


video_sessions = {}
video_session_lock = Lock()


class VideoDetectionSession:
    def __init__(self):
        self.camera_active = False
        self.current_frame = None
        self.detected_board = None
        self.extracted_pcb = None
        self.yolo_result = None
        self.last_update = time.time()
        self.defect_info = []
        self.save_count = 0
        self.detected_frame = None
        self.detected_board_at_moment = None
        self.extracted_pcb_info_at_moment = None


def get_video_session(session_id):
    with video_session_lock:
        if session_id not in video_sessions:
            video_sessions[session_id] = VideoDetectionSession()
        return video_sessions[session_id]


@app.route('/video_detect')
def video_detect():
    return render_template('video_detect.html')


@app.route('/api/video/start_camera', methods=['POST'])
def start_camera():
    try:
        session_id = request.json.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '摄像头启动成功'
        })
    except Exception as e:
        logger.error(f"启动摄像头失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        image_data = data.get('image')
        if not session_id or not image_data:
            return jsonify({'error': '参数错误'}), 400
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': '图像解码失败'}), 400
        morph_detector = CircuitBoardDetector()
        detected_quads = morph_detector.detect_multiple_quadrilaterals(frame, max_count=1)
        processed_frame = frame.copy()
        has_board = False
        board_info = None
        extracted_pcb_info = None
        if detected_quads:
            has_board = True
            quad = detected_quads[0]
            if 'contour' in quad:
                cv2.drawContours(processed_frame, [quad['contour']], -1, (0, 255, 0), 3)
                for point in quad['contour']:
                    x, y = point[0]
                    cv2.circle(processed_frame, (x, y), 6, (255, 255, 255), -1)
                    cv2.circle(processed_frame, (x, y), 3, (0, 255, 0), -1)
            x, y, w, h = quad['bbox']
            board_info = {
                'bbox': [int(x), int(y), int(w), int(h)],
                'center': [int(x + w // 2), int(y + h // 2)],
                'confidence': float(quad.get('confidence', 0.5)),
                'contour': quad['contour'].tolist() if 'contour' in quad else None
            }
            if 'contour' in quad and len(quad['contour']) >= 4:
                contour_points = quad['contour'].reshape(-1, 2)
                epsilon = 0.02 * cv2.arcLength(quad['contour'], True)
                approx = cv2.approxPolyDP(quad['contour'], epsilon, True)
                if len(approx) == 4:
                    corners = approx.reshape(-1, 2)
                else:
                    hull = cv2.convexHull(contour_points.astype(np.float32))
                    hull = hull.reshape(-1, 2)
                    if len(hull) >= 4:
                        rect = cv2.minAreaRect(hull.astype(np.float32))
                        corners = cv2.boxPoints(rect)
                    else:
                        corners = np.array([
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ], dtype=np.float32)
                ordered_corners = morph_detector.order_points(corners)
                edge_lengths = [
                    np.linalg.norm(ordered_corners[1] - ordered_corners[0]),
                    np.linalg.norm(ordered_corners[2] - ordered_corners[1]),
                    np.linalg.norm(ordered_corners[3] - ordered_corners[2]),
                    np.linalg.norm(ordered_corners[0] - ordered_corners[3])
                ]
                width = int((edge_lengths[0] + edge_lengths[2]) / 2)
                height = int((edge_lengths[1] + edge_lengths[3]) / 2)
                width = max(width, 100)
                height = max(height, 100)
                dst_points = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                M = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), dst_points)
                warped = cv2.warpPerspective(frame, M, (width, height))
                pcb_roi = warped
                target_size = 640
                h_roi, w_roi = pcb_roi.shape[:2]
                scale = min(target_size / w_roi, target_size / h_roi)
                new_w = int(w_roi * scale)
                new_h = int(h_roi * scale)
                resized_roi = cv2.resize(pcb_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                square_roi = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2
                square_roi[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
                extracted_pcb_info = {
                    'original_roi': pcb_roi,
                    'processed_roi': square_roi,
                    'bbox': board_info['bbox'],
                    'corners': ordered_corners.tolist(),
                    'width': width,
                    'height': height
                }
                for i, corner in enumerate(ordered_corners):
                    cx, cy = int(corner[0]), int(corner[1])
                    cv2.putText(processed_frame, str(i + 1), (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                pcb_roi = frame[y:y + h, x:x + w]
                quad_in_roi = morph_detector.detect_quad_in_roi(pcb_roi)
                if quad_in_roi is not None:
                    quad_points = quad_in_roi + np.array([x, y])
                    ordered_corners = morph_detector.order_points(quad_points)
                    width = max(
                        np.linalg.norm(ordered_corners[1] - ordered_corners[0]),
                        np.linalg.norm(ordered_corners[2] - ordered_corners[3])
                    )
                    height = max(
                        np.linalg.norm(ordered_corners[3] - ordered_corners[0]),
                        np.linalg.norm(ordered_corners[2] - ordered_corners[1])
                    )
                    width = max(int(width), 100)
                    height = max(int(height), 100)
                    dst_points = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), dst_points)
                    warped = cv2.warpPerspective(frame, M, (width, height))
                    target_size = 640
                    h_roi, w_roi = warped.shape[:2]
                    scale = min(target_size / w_roi, target_size / h_roi)
                    new_w = int(w_roi * scale)
                    new_h = int(h_roi * scale)
                    resized_roi = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    square_roi = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    y_offset = (target_size - new_h) // 2
                    x_offset = (target_size - new_w) // 2
                    square_roi[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
                    extracted_pcb_info = {
                        'original_roi': warped,
                        'processed_roi': square_roi,
                        'bbox': board_info['bbox'],
                        'corners': ordered_corners.tolist()
                    }
                else:
                    pcb_roi = frame[y:y + h, x:x + w]
                    target_size = 640
                    h_roi, w_roi = pcb_roi.shape[:2]
                    scale = min(target_size / w_roi, target_size / h_roi)
                    new_w = int(w_roi * scale)
                    new_h = int(h_roi * scale)
                    resized_roi = cv2.resize(pcb_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    square_roi = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    y_offset = (target_size - new_h) // 2
                    x_offset = (target_size - new_w) // 2
                    square_roi[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
                    extracted_pcb_info = {
                        'original_roi': pcb_roi,
                        'processed_roi': square_roi,
                        'bbox': board_info['bbox']
                    }
            cv2.putText(processed_frame, f"PCB Detected (Conf: {quad.get('confidence', 0):.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame,
                        f"Size: {extracted_pcb_info['width'] if extracted_pcb_info and 'width' in extracted_pcb_info else w}x{extracted_pcb_info['height'] if extracted_pcb_info and 'height' in extracted_pcb_info else h}",
                        (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = base64.b64encode(buffer).decode()
        session = get_video_session(session_id)
        session.current_frame = frame
        session.detected_board = board_info
        session.extracted_pcb_info = extracted_pcb_info
        return jsonify({
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'has_board': has_board,
            'board_info': board_info
        })
    except Exception as e:
        logger.error(f"处理视频帧失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/detect_defect', methods=['POST'])
def detect_video_defect():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': '参数错误'}), 400
        session = get_video_session(session_id)
        if session.current_frame is None or not session.detected_board:
            return jsonify({'error': '没有检测到电路板'}), 400
        session.detected_frame = session.current_frame.copy()
        session.detected_board_at_moment = session.detected_board.copy() if session.detected_board else None
        if hasattr(session, 'extracted_pcb_info') and session.extracted_pcb_info:
            import copy
            session.extracted_pcb_info_at_moment = {
                'original_roi': session.extracted_pcb_info[
                    'original_roi'].copy() if 'original_roi' in session.extracted_pcb_info else None,
                'processed_roi': session.extracted_pcb_info[
                    'processed_roi'].copy() if 'processed_roi' in session.extracted_pcb_info else None,
                'bbox': copy.deepcopy(session.extracted_pcb_info.get('bbox')),
                'corners': copy.deepcopy(session.extracted_pcb_info.get('corners')),
                'width': session.extracted_pcb_info.get('width'),
                'height': session.extracted_pcb_info.get('height')
            }
        else:
            session.extracted_pcb_info_at_moment = None
        detector = get_detector()
        if hasattr(session, 'extracted_pcb_info') and session.extracted_pcb_info:
            processed_roi = session.extracted_pcb_info['processed_roi']
            results = detector.predict(processed_roi, conf_threshold=0.25, iou_threshold=0.45)
            pcb_with_boxes, detections = detector.draw_detections(processed_roi, results)
            _, buffer = cv2.imencode('.jpg', pcb_with_boxes)
            yolo_image = base64.b64encode(buffer).decode()
            original_pcb_img = session.extracted_pcb_info['original_roi']
            _, original_buffer = cv2.imencode('.jpg', original_pcb_img)
            original_pcb = base64.b64encode(original_buffer).decode()
        else:
            frame = session.current_frame
            board_info = session.detected_board
            x, y, w, h = board_info['bbox']
            pcb_roi = frame[y:y + h, x:x + w]
            if pcb_roi.size == 0:
                return jsonify({'error': '抠图失败'}), 400
            target_size = 640
            h_roi, w_roi = pcb_roi.shape[:2]
            scale = min(target_size / w_roi, target_size / h_roi)
            new_w = int(w_roi * scale)
            new_h = int(h_roi * scale)
            resized_roi = cv2.resize(pcb_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            square_roi = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            square_roi[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_roi
            results = detector.predict(square_roi, conf_threshold=0.25, iou_threshold=0.45)
            pcb_with_boxes, detections = detector.draw_detections(square_roi, results)
            _, buffer = cv2.imencode('.jpg', pcb_with_boxes)
            yolo_image = base64.b64encode(buffer).decode()
            _, original_buffer = cv2.imencode('.jpg', pcb_roi)
            original_pcb = base64.b64encode(original_buffer).decode()
        session.yolo_result = {
            'image': yolo_image,
            'original_pcb': original_pcb,
            'detections': detections
        }
        session.defect_info = detections
        return jsonify({
            'success': True,
            'yolo_image': f"data:image/jpeg;base64,{yolo_image}",
            'original_pcb': f"data:image/jpeg;base64,{original_pcb}",
            'detections': detections,
            'has_defect': len(detections) > 0
        })
    except Exception as e:
        logger.error(f"缺陷检测失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/save_result', methods=['POST'])
def save_video_result():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': '参数错误'}), 400
        session = get_video_session(session_id)
        if session.detected_frame is None:
            return jsonify({'error': '没有可保存的图像，请先点击"识别缺陷"按钮'}), 400
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"video_{timestamp}_{session.save_count}.jpg"
        if session.detected_board_at_moment:
            frame_with_box = session.detected_frame.copy()
            if session.detected_board_at_moment.get('contour'):
                contour_np = np.array(session.detected_board_at_moment['contour'], dtype=np.int32)
                cv2.drawContours(frame_with_box, [contour_np], -1, (0, 255, 0), 3)
                for point in contour_np:
                    x, y = point[0]
                    cv2.circle(frame_with_box, (x, y), 6, (255, 255, 255), -1)
                    cv2.circle(frame_with_box, (x, y), 3, (0, 255, 0), -1)
                if session.extracted_pcb_info_at_moment and 'corners' in session.extracted_pcb_info_at_moment:
                    corners = session.extracted_pcb_info_at_moment['corners']
                    for i, corner in enumerate(corners):
                        cx, cy = int(corner[0]), int(corner[1])
                        cv2.putText(frame_with_box, str(i + 1), (cx - 20, cy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                x, y, w, h = session.detected_board_at_moment['bbox']
                cv2.rectangle(frame_with_box, (x, y), (x + w, y + h), (0, 255, 0), 3)
            x, y, w, h = session.detected_board_at_moment['bbox']
            cv2.putText(frame_with_box, f"PCB Detected (Conf: {session.detected_board_at_moment.get('confidence', 0):.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{safe_filename}")
            cv2.imwrite(original_path, frame_with_box)
        else:
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{safe_filename}")
            cv2.imwrite(original_path, session.detected_frame)
        detected_path = None
        if session.yolo_result and session.yolo_result.get('image'):
            try:
                image_data = session.yolo_result['image']
                if ',' in image_data:
                    yolo_image_data = image_data.split(',')[1]
                else:
                    yolo_image_data = image_data
                yolo_bytes = base64.b64decode(yolo_image_data)
                nparr = np.frombuffer(yolo_bytes, np.uint8)
                yolo_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if yolo_img is not None:
                    detected_path = os.path.join(app.config['DETECTED_FOLDER'], f"detected_{safe_filename}")
                    cv2.imwrite(detected_path, yolo_img)
                else:
                    logger.warning("YOLO图像解码失败")
            except Exception as e:
                logger.error(f"解码YOLO图像失败: {e}")
        original_pcb_path = None
        if session.yolo_result and session.yolo_result.get('original_pcb'):
            try:
                original_pcb_data = session.yolo_result['original_pcb']
                if ',' in original_pcb_data:
                    original_pcb_data = original_pcb_data.split(',')[1]
                pcb_bytes = base64.b64decode(original_pcb_data)
                nparr = np.frombuffer(pcb_bytes, np.uint8)
                pcb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if pcb_img is not None:
                    original_pcb_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pcb_{safe_filename}")
                    cv2.imwrite(original_pcb_path, pcb_img)
            except Exception as e:
                logger.error(f"解码PCB图像失败: {e}")
        db_record = {
            'task_id': f"video_{session_id}",
            'filename': safe_filename,
            'file_path': original_path,
            'detected_file_path': detected_path,
            'has_defect': len(session.defect_info) > 0,
            'defect_count': len(session.defect_info),
            'defects': session.defect_info,
            'conf_threshold': 0.25,
            'iou_threshold': 0.45
        }
        db_manager.save_record(db_record)
        session.save_count += 1
        return jsonify({
            'success': True,
            'message': '保存成功',
            'save_count': session.save_count
        })
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/clear_session', methods=['POST'])
def clear_video_session():
    try:
        session_id = request.json.get('session_id')
        if session_id and session_id in video_sessions:
            with video_session_lock:
                del video_sessions[session_id]
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"清除会话失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        data = request.get_json()
        mode = data.get('mode', 'filtered')
        conn = None
        cursor = None
        deleted_count = 0
        file_paths_to_delete = []
        try:
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if mode == 'all':
                    cursor.execute("SELECT file_path, detected_file_path FROM detection_records")
                    rows = cursor.fetchall()
                    for row in rows:
                        if row['file_path'] and os.path.exists(row['file_path']):
                            file_paths_to_delete.append(row['file_path'])
                        if row['detected_file_path'] and os.path.exists(row['detected_file_path']):
                            file_paths_to_delete.append(row['detected_file_path'])
                    cursor.execute("DELETE FROM detection_records")
                    deleted_count = cursor.rowcount
                elif mode == 'single' or mode == 'filtered':
                    record_ids = data.get('record_ids', [])
                    if not record_ids:
                        filters = data.get('filters', {})
                        query = "SELECT id, file_path, detected_file_path FROM detection_records WHERE 1=1"
                        params = []
                        if filters.get('start_date'):
                            query += " AND detection_time >= ?"
                            params.append(filters['start_date'])
                        if filters.get('end_date'):
                            query += " AND detection_time <= ?"
                            params.append(filters['end_date'] + " 23:59:59")
                        if filters.get('filename'):
                            query += " AND filename LIKE ?"
                            params.append(f"%{filters['filename']}%")
                        has_defect = filters.get('has_defect')
                        if has_defect is not None and has_defect != '':
                            query += " AND has_defect = ?"
                            params.append(1 if has_defect == '1' else 0)
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
                        record_ids = [row['id'] for row in rows]
                        for row in rows:
                            if row['file_path'] and os.path.exists(row['file_path']):
                                file_paths_to_delete.append(row['file_path'])
                            if row['detected_file_path'] and os.path.exists(row['detected_file_path']):
                                file_paths_to_delete.append(row['detected_file_path'])
                    if record_ids:
                        if not file_paths_to_delete:
                            placeholders = ','.join(['?'] * len(record_ids))
                            cursor.execute(
                                f"SELECT file_path, detected_file_path FROM detection_records WHERE id IN ({placeholders})",
                                record_ids)
                            rows = cursor.fetchall()
                            for row in rows:
                                if row['file_path'] and os.path.exists(row['file_path']):
                                    file_paths_to_delete.append(row['file_path'])
                                if row['detected_file_path'] and os.path.exists(row['detected_file_path']):
                                    file_paths_to_delete.append(row['detected_file_path'])
                        placeholders = ','.join(['?'] * len(record_ids))
                        cursor.execute(f"DELETE FROM detection_records WHERE id IN ({placeholders})", record_ids)
                        deleted_count = cursor.rowcount
                conn.commit()
                deleted_files = 0
                for file_path in file_paths_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            deleted_files += 1
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
                logger.info(f"清除历史记录: 删除了 {deleted_count} 条记录, {deleted_files} 个文件")
                return jsonify({
                    'success': True,
                    'deleted_count': deleted_count,
                    'deleted_files': deleted_files,
                    'message': f'成功删除 {deleted_count} 条记录'
                })
        except sqlite3.Error as e:
            logger.error(f"数据库操作失败: {e}")
            return jsonify({'success': False, 'error': f'数据库操作失败: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"清除历史记录失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def main():
    print("=" * 60)
    print("🚀 PCB缺陷检测系统")
    print("=" * 60)
    print("🌐 访问地址: http://localhost:5000")
    print("📝 按 Ctrl+C 退出")
    print("=" * 60)
    model_path = 'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'
    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(alt_path):
            print(f"✅ 找到模型文件: {alt_path}")
        else:
            print(f"⚠️ 警告: 模型文件不存在: {model_path}")
            print("   首次检测时会自动加载模型")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


if __name__ == '__main__':
    main()