import cv2
import numpy as np
from scipy.spatial import distance as dist
import io
import importlib

try:
    import tensorflow as tf
    _TFLITE_INTERPRETER = tf.lite.Interpreter
except ImportError:
    tflite = importlib.import_module("tflite_runtime.interpreter")
    _TFLITE_INTERPRETER = tflite.Interpreter


class ExamScanner:
    def __init__(self, model_path):
        # 1. 初始化加载模型
        self.interpreter = _TFLITE_INTERPRETER(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 配置信息
        self.fixed_answers = []
        self.expected_cols = 7
        self.single_score = 5

    def _my_warp_perspective(self, image, pts):
        """ 内部私有方法：透视变换 """
        xSorted = pts[np.argsort(pts[:, 0]), :]
        left = xSorted[:2, :]
        right = xSorted[2:, :]
        left = left[np.argsort(left[:, 1]), :]
        (tl, bl) = left
        D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
        (br, tr) = right[np.argsort(D)[::-1], :]
        src = np.array([tl, tr, br, bl], dtype="float32")

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _predict_letter(self, roi):
        """ 内部方法：识别单个 ROI 区域的字母 """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            pad = 5
            roi_char = binary[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        else:
            roi_char = binary

        resized = cv2.resize(roi_char, (28, 28))
        input_data = resized.reshape(1, 28, 28, 1).astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        abcd_probs = output_data[0][0:4]
        return chr(ord('A') + np.argmax(abcd_probs))

    def _get_lines_position(self, binary_img, orientation='horizontal'):
        # 逻辑保持不变，但增加一个容错处理
        h, w = binary_img.shape
        min_scale = 8 if orientation == 'horizontal' else 2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (w // min_scale, 1) if orientation == 'horizontal' else (1, h // min_scale))

        temp = cv2.erode(binary_img, kernel, iterations=1)
        lines_img = cv2.dilate(temp, kernel, iterations=1)
        cnts, _ = cv2.findContours(lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coords = sorted([(y + h_c // 2) if orientation == 'horizontal' else (x + w_c // 2)
                         for (x, y, w_c, h_c) in [cv2.boundingRect(c) for c in cnts]])

        if not coords: return []
        merged = [coords[0]]
        for val in coords[1:]:
            if val - merged[-1] > 15: merged.append(val)
        return merged

    def process_and_score(self, image_bytes, user_answers=None, user_cols=None, score_per_item=None):
        """ 主入口：接收图片字节，返回识别得分和字母列表 """

        current_answers = user_answers if user_answers else self.fixed_answers
        current_cols = user_cols if user_cols else self.expected_cols
        current_score = score_per_item if score_per_item is not None else self.single_score
        
        # 将字节转为 OpenCV 图像
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- 1. 查找试卷轮廓并矫正 (简化逻辑) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 200)
        cnts, _ = cv2.findContours(cv2.dilate(edged, None, iterations=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        paper = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                paper = self._my_warp_perspective(img, approx.reshape(4, 2))
                break

        if paper is None: 
            return 0, []  # 返回空列表而不是 None

        # --- 2. 行列检测与识别 ---
        paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
        paper_bin = cv2.adaptiveThreshold(paper_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        h_lines = self._get_lines_position(paper_bin, 'horizontal')
        v_lines = self._get_lines_position(paper_bin, 'vertical')

        detected_letters = []
        correct_count = 0
        current_ans_idx = 0
        has_answers = current_answers and len(current_answers) > 0  # 检查是否有标准答案

        # 3. 遍历逻辑 - 修复：根据行列线遍历，而不是根据答案数量
        for row_idx in range(len(h_lines) - 1):
            if row_idx % 2 == 1:  # 答案行（索引1, 3, 5...）
                y_t, y_b = h_lines[row_idx], h_lines[row_idx + 1]

                # 使用传入的列数 current_cols，或者根据竖线数量自动计算
                actual_cols = min(len(v_lines) - 1, current_cols) if current_cols else len(v_lines) - 1
                
                for col_idx in range(actual_cols):
                    # 修复：如果有标准答案才检查索引，否则一直识别
                    if has_answers and current_ans_idx >= len(current_answers):
                        break

                    x_l, x_r = v_lines[col_idx], v_lines[col_idx + 1]

                    # ROI 提取逻辑保持不变
                    roi = paper[y_t + int((y_b - y_t) * 0.15): y_b - int((y_b - y_t) * 0.1),
                    x_l + int((x_r - x_l) * 0.1): x_r - int((x_r - x_l) * 0.1)]

                    if roi.size == 0: 
                        continue

                    letter = self._predict_letter(roi)
                    detected_letters.append(letter)

                    # 对比用户传来的标准答案（如果有）
                    if has_answers and current_ans_idx < len(current_answers):
                        if letter == current_answers[current_ans_idx]:
                            correct_count += 1
                        current_ans_idx += 1
                    elif not has_answers:
                        # 没有标准答案时，只识别不计分
                        current_ans_idx += 1

        total_score = correct_count * current_score if has_answers else 0
        return total_score, detected_letters
