import threading
import queue

import cv2
from paddleocr import PaddleOCR
from config import AppConfig
from utils import PlateUtils

class AsyncOCRManager:
    
    def __init__(self, config: AppConfig):
        self.config = config
        
        print("Initializing PaddleOCR Engine...")
        self.ocr_engine = PaddleOCR(
            use_angle_cls=False, 
            use_gpu=True, 
            lang='en',
            det_db_thresh=0.3, 
            det_db_box_thresh=0.5,
            rec_batch_num=6, 
            show_log=False
        )
        
        self.queue = queue.Queue(maxsize=config.OCR_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        
        self.active = True
        
        self.stats = {
            "attempts": 0,
            "successes": 0
        }
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("✓ Async OCR Manager initialized.")

    def _preprocess_image(self, img_crop):
        """
        Hàm tiền xử lý ảnh trước khi đưa vào OCR:
        1. Chuyển Gray
        2. Resize x2 hoặc x3
        3. Tăng tương phản
        """
        try:
            # 1. Chuyển sang ảnh xám (Grayscale)
            if len(img_crop.shape) == 3:
                gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_crop

            # 2. Phóng to ảnh (Upscaling)
            scale_factor = 2.0 
            h, w = gray.shape[:2]
            
            if w < 100:
                scale_factor = 3.0
                
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # 3. Tăng độ tương phản (Contrast Normalization)
            processed = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


            return processed
            
        except Exception as e:
            print(f"[Preprocess Error] {e}")
            return img_crop

    def _worker(self):
      
        while self.active:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None: 
                    break
                
                tracker_id, lp_crop, frame_index, class_id = task
                self.stats["attempts"] += 1
                
                try:
                    processed_input = self._preprocess_image(lp_crop)
                    # Run PaddleOCR
                    ocr_result = self.ocr_engine.ocr(processed_input, cls=False)
                    
                    text, conf = PlateUtils.clean_ocr_text(
                        ocr_result, 
                        class_id, 
                        self.config.OCR_CONFIDENCE_THRESHOLD
                    )
                    
               
                    self.result_queue.put((tracker_id, text, conf, frame_index))
                    
                except Exception as e:
                    print(f"[OCR Worker] Error processing #{tracker_id}: {e}")
                    
            except queue.Empty:
                continue

    def add_task(self, tracker_id, lp_crop, frame_index, class_id):

        try:
            self.queue.put_nowait((tracker_id, lp_crop, frame_index, class_id))
        except queue.Full:
            pass # Drop task if queue is full (real-time priority)

    def get_results(self):
 
        results = []
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                results.append(res)
                if res[1]:
                    self.stats["successes"] += 1
            except queue.Empty:
                break
        return results

    def stop(self):
 
        self.active = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)