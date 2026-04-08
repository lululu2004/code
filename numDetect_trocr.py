import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_trocr")
class TrOCRDigitReader:
    """
    TrOCR 数字识别类：专门针对学号场景进行了优化
    """
    def __init__(self, model_name=None, device=None):
        # 初始化处理器和模型
        # 如果没有传入路径，默认使用本地的 model_path
        target_path = model_name if model_name else model_path
        
        print(f"正在从本地加载模型: {target_path}")
        self.processor = TrOCRProcessor.from_pretrained(target_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(target_path)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.expected_len = 10 # 设定学号预期长度
        
        # 提取数字 Token ID 列表，用于约束模型输出
        tokenizer = self.processor.tokenizer
        self.allowed_digit_token_ids = set()
        for d in "0123456789":
            ids = tokenizer.encode(d, add_special_tokens=False)
            for tid in ids: self.allowed_digit_token_ids.add(int(tid))
        # 加上起始和结束符
        for sid in (self.model.config.decoder_start_token_id, self.model.config.eos_token_id):
            if sid is not None: self.allowed_digit_token_ids.add(int(sid))

    def readtext(self, image_np):
        """
        核心识别函数：接收 OpenCV 图片，输出学号字符串和置信度
        """
        # 将图片转为 RGB 格式以适配 TrOCR
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pixel_values = self.processor(images=image_rgb, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            # 使用 prefix_allowed_tokens_fn 强制模型只生成数字
            def prefix_fn(batch_id, input_ids): return list(self.allowed_digit_token_ids)
            
            outputs = self.model.generate(
                pixel_values,
                min_new_tokens=self.expected_len,
                max_new_tokens=self.expected_len,
                prefix_allowed_tokens_fn=prefix_fn,
                output_scores=True,
                return_dict_in_generate=True
            )

        # 解码并去除空格
        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        text = text.replace(" ", "")
        
        # 通过 Softmax 计算平均置信度
        probs = [torch.softmax(s[0], dim=-1).max().item() for s in outputs.scores]
        confidence = sum(probs) / len(probs) if probs else 0.0
        
        return text, confidence

def preprocess_for_ocr_with_steps(roi_bgr):
    """
    预处理流水线：包含放大、增强对比度、去噪及抹除横线
    """
    # 0. 添加白边防止文字贴边
    roi_padded = cv2.copyMakeBorder(roi_bgr, 8, 8, 18, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    # 1. 放大 3 倍提高清晰度
    h, w = roi_padded.shape[:2]
    resized = cv2.resize(roi_padded, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    # 2. 灰度化与 CLAHE 局部增强
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 提取横线并从灰度图中抹除
    bin_map = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    lines = cv2.morphologyEx(bin_map, cv2.MORPH_OPEN, h_kernel, iterations=1)
    
    no_lines = enhanced.copy()
    no_lines[lines > 0] = 255 # 将检测到的横线区域设为纯白
    
    # 4. 轻锐化处理
    blur = cv2.GaussianBlur(no_lines, (0, 0), 1.0)
    final_roi = cv2.addWeighted(no_lines, 1.4, blur, -0.4, 0)
    
    return final_roi