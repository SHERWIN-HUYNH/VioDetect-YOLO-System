import re
from typing import Tuple, Optional, List

class PlateUtils:
    """
    Class chứa các hàm tiện ích xử lý văn bản biển số.
    """
    
    CHAR_TO_INT = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'D': '0', 'G': '6'}
    INT_TO_CHAR = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '4': 'A', '6': 'G'}
    
    CAR_IDS = [2, 5, 7]
    MOTOR_IDS = [3]

    @staticmethod
    def clean_ocr_text(ocr_result, class_id: Optional[int], conf_threshold: float) -> Tuple[str, float]:
        """
        Xử lý kết quả thô từ PaddleOCR, áp dụng heuristic cho biển số VN.
        """
        if not ocr_result:
            return "", 0.0
        try:
            text_parts = []
            conf_parts = []
            
            # Extract text from PaddleOCR structure
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                    for item in ocr_result[0]:
                        if len(item) >= 2:
                            text, confidence = item[1][0], item[1][1]
                            if text:
                                text_parts.append(str(text))
                                conf_parts.append(confidence)
            
            if not text_parts:
                return "", 0.0
            
            full_text = ''.join(text_parts)
            cleaned = re.sub(r'[^A-Z0-9]', '', full_text.upper())
            
            if len(cleaned) < 6: 
                return "", 0.0

            text_list = list(cleaned)
            
            # 2 ký tự đầu mã tỉnh             
            for i in [0, 1]:
                if i < len(text_list) and text_list[i] in PlateUtils.CHAR_TO_INT:
                    text_list[i] = PlateUtils.CHAR_TO_INT[text_list[i]]
            # 4 ký tự cuối là số 
            for i in range(len(text_list) - 4, len(text_list)):
                if i >= 0 and text_list[i] in PlateUtils.CHAR_TO_INT:
                    text_list[i] = PlateUtils.CHAR_TO_INT[text_list[i]]
            
            # Xe tải: ký tự thứ 3 là chữ, 4 là số 
            if class_id in PlateUtils.CAR_IDS:
                if len(text_list) > 2 and text_list[2] in PlateUtils.INT_TO_CHAR:
                    text_list[2] = PlateUtils.INT_TO_CHAR[text_list[2]]
                if 3 < len(text_list) and text_list[3] in PlateUtils.CHAR_TO_INT:
                    text_list[3] = PlateUtils.CHAR_TO_INT[text_list[3]]

            # Xe máy: ký tự thứ 3 thường là chữ (ví dụ 29-H1), thứ 4 có thể chữ hoặc số 
            elif class_id in PlateUtils.MOTOR_IDS:
                if len(text_list) > 2 and text_list[2] in PlateUtils.INT_TO_CHAR:
                    text_list[2] = PlateUtils.INT_TO_CHAR[text_list[2]]

            final_text = "".join(text_list)
            avg_conf = sum(conf_parts) / len(conf_parts) if conf_parts else 0.0
            
            if avg_conf < conf_threshold:
                return "", 0.0

            return final_text, avg_conf
        
        except Exception as e:
            print(f"[Text Clean Error] {e}")
            return "", 0.0