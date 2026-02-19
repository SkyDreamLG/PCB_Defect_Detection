# coding:utf-8
from ultralytics import YOLOv10
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置中文字体 ==========
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# PCB缺陷类别映射
DEFECT_MAP = {
    'missing_hole': '缺失孔',
    'mouse_bite': '鼠咬缺陷',
    'short_circuit': '短路'
}

# 颜色映射
COLOR_MAP = {
    'missing_hole': (255, 0, 0),  # 红色
    'mouse_bite': (0, 255, 0),  # 绿色
    'short_circuit': (0, 0, 255)  # 蓝色
}

# 缺陷描述
DEFECT_DESCRIPTIONS = {
    'missing_hole': 'PCB板上缺失的钻孔',
    'mouse_bite': '类似老鼠咬过的弧形缺陷',
    'short_circuit': '线路之间的异常连接'
}

# 全局变量
processing_status = {}
status_lock = Lock()
detector_instance = None
detector_lock = Lock()
task_manager = None
db_manager = None


# ========== 数据库管理器 ==========
class DatabaseManager:
    """SQLite数据库管理器，负责存储检测历史"""

    def __init__(self, db_path='pcb_defect_history.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 创建检测记录表
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

                # 创建索引
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_records(detection_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detection_records(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_has_defect ON detection_records(has_defect)')

                conn.commit()
                logger.info("✅ 数据库初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def save_record(self, record):
        """保存检测记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 获取东八区时间
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
        """查询检测记录"""
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

                # 转换为字典列表
                result = []
                for record in records:
                    record_dict = dict(record)
                    # 解析缺陷JSON
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
        """根据ID获取记录"""
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
    """任务管理器，负责管理和监控所有检测任务"""

    def __init__(self, max_workers=2):
        self.tasks = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cleanup_thread = Thread(target=self._cleanup_old_tasks, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"任务管理器初始化完成，最大并发: {max_workers}")

    def create_task(self, task_id, total_files):
        """创建新任务"""
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
        """更新任务状态"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['last_updated'] = time.time()

                # 计算进度
                if 'completed' in kwargs and self.tasks[task_id]['total'] > 0:
                    progress = (kwargs['completed'] / self.tasks[task_id]['total']) * 100
                    self.tasks[task_id]['progress'] = progress

                    # 如果完成数量等于总数，标记为完成
                    if kwargs['completed'] >= self.tasks[task_id]['total']:
                        self.tasks[task_id]['status'] = 'completed'
                        self.tasks[task_id]['progress'] = 100

    def get_task(self, task_id):
        """获取任务信息"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                # 返回副本，避免修改
                return task.copy()
            return None

    def _cleanup_old_tasks(self):
        """清理30分钟前的旧任务"""
        while True:
            time.sleep(300)  # 每5分钟检查一次
            try:
                current_time = time.time()
                with self.lock:
                    to_delete = []
                    for task_id, task in self.tasks.items():
                        # 清理30分钟前的已完成或错误任务
                        if task['status'] in ['completed', 'error']:
                            if current_time - task['last_updated'] > 1800:  # 30分钟
                                to_delete.append(task_id)

                    for task_id in to_delete:
                        del self.tasks[task_id]

                    if to_delete:
                        logger.info(f"清理了 {len(to_delete)} 个旧任务")
            except Exception as e:
                logger.error(f"清理任务时出错: {e}")


class ModelManager:
    """模型管理器，负责模型的加载和推理"""

    def __init__(self, model_path='PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'):
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.lock = Lock()
        self.load_model()

    def _get_device(self):
        """获取可用设备"""
        if torch.cuda.is_available():
            device = 0
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("使用CPU")
        return device

    def load_model(self):
        """加载模型（兼容PyTorch 2.6+）"""
        try:
            with self.lock:
                if self.model is None:
                    logger.info(f"加载模型: {self.model_path}")

                    # 检查PyTorch版本
                    torch_version = torch.__version__
                    logger.info(f"PyTorch版本: {torch_version}")

                    # 获取主版本号
                    major_version = int(torch_version.split('.')[0])
                    minor_version = int(torch_version.split('.')[1])

                    # 对于PyTorch 2.6+，需要特殊处理
                    if major_version >= 2 and minor_version >= 6:
                        logger.info("检测到PyTorch 2.6+，使用兼容模式加载")

                        # 方法1：使用环境变量（最简单）
                        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

                        try:
                            # 方法2：使用safe_globals上下文管理器
                            import ultralytics.nn.tasks

                            # 添加安全全局类
                            torch.serialization.add_safe_globals([ultralytics.nn.tasks.YOLOv10DetectionModel])

                            # 加载模型
                            self.model = YOLOv10(self.model_path)

                        except Exception as e:
                            logger.warning(f"安全加载失败，尝试备用方法: {e}")

                            # 方法3：猴子补丁torch.load
                            import warnings
                            warnings.filterwarnings("ignore", category=UserWarning)

                            # 保存原始函数
                            original_load = torch.load

                            def patched_load(f, *args, **kwargs):
                                # 强制设置weights_only=False
                                kwargs['weights_only'] = False
                                return original_load(f, *args, **kwargs)

                            # 应用补丁
                            torch.load = patched_load

                            try:
                                self.model = YOLOv10(self.model_path)
                                logger.info("使用猴子补丁加载成功")
                            except Exception as e2:
                                logger.error(f"猴子补丁加载失败: {e2}")
                                # 恢复原始函数
                                torch.load = original_load
                                raise
                            finally:
                                # 恢复原始函数
                                torch.load = original_load
                    else:
                        # 旧版本PyTorch直接加载
                        self.model = YOLOv10(self.model_path)

                    logger.info("✅ 模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error("请尝试以下解决方案：")
            logger.error("1. 升级ultralytics包: pip install -U ultralytics")
            logger.error("2. 或者降级PyTorch: pip install torch==2.5.1")
            logger.error("3. 或者手动设置环境变量后运行: export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 && python app.py")
            raise

    def predict(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """执行预测"""
        try:
            with self.lock:  # 确保模型推理是线程安全的
                results = self.model(
                    image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False  # 减少日志输出
                )
            return results
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

    def draw_detections(self, img, results):
        """绘制检测结果"""
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

                    # 绘制边界框
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{cls_name} {conf:.2f}"
                    font_scale = 0.5
                    thickness = 1
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )

                    # 标签背景
                    cv2.rectangle(
                        img_with_boxes,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width + 5, y1),
                        color,
                        -1
                    )

                    # 标签文字
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


# Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTED_FOLDER'] = 'detected'

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# 初始化管理器
task_manager = TaskManager(max_workers=2)  # 限制并发数为2
db_manager = DatabaseManager()


# 模型懒加载
def get_detector():
    """获取检测器实例（懒加载）"""
    global detector_instance
    with detector_lock:
        if detector_instance is None:
            model_path = 'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'
            if not os.path.exists(model_path):
                # 尝试相对路径
                model_path = os.path.join(os.path.dirname(__file__),
                                          'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt')
            detector_instance = ModelManager(model_path)
    return detector_instance


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/start_batch_detect', methods=['POST'])
def start_batch_detect():
    """启动批量检测任务（异步处理）"""
    try:
        files = request.files.getlist('files')
        conf_threshold = float(request.form.get('conf_threshold', 0.25))
        iou_threshold = float(request.form.get('iou_threshold', 0.45))

        if not files:
            return jsonify({'error': '请上传图片文件'})

        # 解除20个文件的限制，改为更宽松的限制（例如1000个文件）
        if len(files) > 1000:
            return jsonify({'error': '为了系统性能，单次最多只能上传1000张图片'})

        # 读取文件内容
        file_contents = []
        for file in files:
            file.seek(0)
            content = file.read()
            file_contents.append({
                'filename': file.filename,
                'content': content
            })

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务
        task_manager.create_task(task_id, len(file_contents))

        # 启动异步处理
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
    """异步批量处理"""
    try:
        detector = get_detector()
        total_files = len(file_contents)

        logger.info(f"开始处理任务 {task_id}, 共 {total_files} 张图片")

        # 逐个处理文件
        for idx, file_data in enumerate(file_contents):
            try:
                # 更新状态
                task_manager.update_task(task_id, status='processing')

                # 处理单张图片
                img = Image.open(BytesIO(file_data['content']))

                # 转换为OpenCV格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # 预测
                results = detector.predict(
                    img_bgr,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )

                # 绘制检测结果
                img_with_boxes, detections = detector.draw_detections(img_bgr, results)

                # 保存原始图片到服务器
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{file_data['filename']}"

                # 保存原始图片
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                cv2.imwrite(original_path, img_bgr)

                # 保存检测图片（如果有缺陷）
                detected_path = None
                if detections:
                    detected_filename = f"detected_{safe_filename}"
                    detected_path = os.path.join(app.config['DETECTED_FOLDER'], detected_filename)
                    cv2.imwrite(detected_path, img_with_boxes)

                # 转换为base64用于前端显示
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

                # 保存到数据库
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

                # 更新结果
                with status_lock:
                    task = task_manager.get_task(task_id)
                    if task:
                        task['results'][idx] = result_item
                        completed = task['completed'] + 1
                        task_manager.update_task(task_id, completed=completed)

                logger.info(f"任务 {task_id} 进度: {idx + 1}/{total_files}")

                # 手动触发垃圾回收
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
    """获取检测结果（轻量级接口）"""
    try:
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify({'status': 'not_found'})

        # 只返回必要的字段，减少数据传输
        response = {
            'status': task['status'],
            'progress': task.get('progress', 0),
            'results': []
        }

        # 只返回已完成的结果
        for result in task.get('results', []):
            if result and result.get('status') in ['completed', 'error']:
                response['results'].append(result)

        return jsonify(response)

    except Exception as e:
        logger.error(f"获取结果失败: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/download_results', methods=['POST'])
def download_results():
    """下载检测结果（包含原始图片、检测图片和CSV报告）"""
    try:
        data = request.get_json()
        results = data.get('results', [])

        if not results:
            return jsonify({'error': '没有结果数据'}), 400

        # 创建ZIP文件
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 创建子目录
            zip_file.writestr('original_images/', '')  # 原始图片目录
            zip_file.writestr('detected_images/', '')  # 检测图片目录

            # 写入图片和CSV报告
            for result in results:
                filename = result['filename']

                # 处理原始图片
                if result['original_image']:
                    # 提取base64数据
                    base64_data = result['original_image'].split(',')[1]
                    img_bytes = base64.b64decode(base64_data)
                    zip_file.writestr(f'original_images/{filename}', img_bytes)

                # 处理检测图片
                if result['detected_image']:
                    # 提取base64数据
                    base64_data = result['detected_image'].split(',')[1]
                    img_bytes = base64.b64decode(base64_data)
                    # 使用相同文件名但添加_detected后缀
                    detected_filename = f'{os.path.splitext(filename)[0]}_detected{os.path.splitext(filename)[1]}'
                    zip_file.writestr(f'detected_images/{detected_filename}', img_bytes)

            # 创建CSV报告
            csv_buffer = StringIO()
            fieldnames = ['filename', 'defect_type', 'defect_description', 'confidence', 'x1', 'y1', 'x2', 'y2',
                          'width', 'height']
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                if result['detections']:  # 如果有缺陷
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
                else:  # 正常图片
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

        # 返回ZIP文件
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
    """查询历史记录"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filename = request.args.get('filename')
        has_defect = request.args.get('has_defect')

        # 转换has_defect参数
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
    """获取历史记录的图片"""
    try:
        image_type = request.args.get('type', 'detected')  # 'original' 或 'detected'

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
    """下载历史记录（根据查询条件）"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filename = request.args.get('filename')
        has_defect = request.args.get('has_defect')

        # 转换has_defect参数
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

        # 创建ZIP文件
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 创建CSV报告
            csv_buffer = StringIO()
            fieldnames = ['id', 'filename', 'detection_time', 'has_defect', 'defect_count', 'defects_detail']
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                # 添加图片到ZIP
                if record['file_path'] and os.path.exists(record['file_path']):
                    zip_file.write(record['file_path'], f"original/{record['filename']}")

                if record['detected_file_path'] and os.path.exists(record['detected_file_path']):
                    detected_filename = f"detected_{record['filename']}"
                    zip_file.write(record['detected_file_path'], f"detected/{detected_filename}")

                # 写入CSV记录
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
    """下载单个历史记录"""
    try:
        record = db_manager.get_record_by_id(record_id)
        if not record:
            return jsonify({'error': '记录不存在'}), 404

        # 创建ZIP文件
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 添加原始图片
            if record['file_path'] and os.path.exists(record['file_path']):
                zip_file.write(record['file_path'], f"original/{record['filename']}")

            # 添加检测图片
            if record['detected_file_path'] and os.path.exists(record['detected_file_path']):
                detected_filename = f"detected_{record['filename']}"
                zip_file.write(record['detected_file_path'], f"detected/{detected_filename}")

            # 创建详细信息文件
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
    """显示历史记录详情页面"""
    record = db_manager.get_record_by_id(record_id)
    if not record:
        return '记录不存在', 404

    return render_template('history_detail.html', record=record)


@app.route('/system_status')
def system_status():
    """获取系统状态"""
    try:
        # 获取内存使用情况
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        # 获取待处理任务数
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


# ========== 统计页面相关路由 ==========

@app.route('/stats')
def stats():
    """统计页面"""
    return render_template('stats.html')


@app.route('/api/stats/overview')
def get_stats_overview():
    """获取统计概览数据"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # 查询记录（带日期筛选）
        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date
        )

        total_count = len(records)
        defect_count = sum(1 for r in records if r['has_defect'])
        normal_count = total_count - defect_count

        # 缺陷比例
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
    """获取各类缺陷数量统计"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        records = db_manager.query_records(
            start_date=start_date,
            end_date=end_date,
            has_defect=True  # 只查询有缺陷的记录
        )

        # 统计各类缺陷
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

        # 格式化返回数据
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
    """获取每日检测数量统计"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        if not start_date or not end_date:
            # 默认最近30天
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
    """获取每日统计数据的辅助函数"""
    try:
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 查询每日总数和有缺陷的数量
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
    """获取缺陷置信度分布"""
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
    """导出统计数据为CSV"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # 获取数据
        defect_types = get_defect_types_stats().json
        daily_stats = get_daily_stats(start_date, end_date) if start_date and end_date else []

        # 创建CSV
        output = StringIO()

        # 写入缺陷类型统计
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

        # 返回CSV文件
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


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 PCB缺陷检测系统")
    print("=" * 60)
    print("🌐 访问地址: http://localhost:5000")
    print("📝 按 Ctrl+C 退出")
    print("=" * 60)

    # 检查模型文件
    model_path = 'PCB_Defect_Detection/yolov10s_ep300_bs24/weights/best.pt'
    if not os.path.exists(model_path):
        alt_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(alt_path):
            print(f"✅ 找到模型文件: {alt_path}")
        else:
            print(f"⚠️ 警告: 模型文件不存在: {model_path}")
            print("   首次检测时会自动加载模型")

    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


if __name__ == '__main__':
    main()