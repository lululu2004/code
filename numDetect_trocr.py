import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCRDigitReader:
    """
    TrOCR 数字识别类：还原为联网加载模式
    """
    def __init__(self, model_name="microsoft/trocr-base-printed", device=None):
        # 1. 直接使用官方 ID 进行联网加载
        print(f"正在从 Hugging Face 联网加载/校验模型: {model_name}")
        
        # 这里会检查本地 .cache 是否有更新，如果有网络会尝试同步
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.expected_len = 10 
        
        # 提取数字 Token ID 列表 (保持不变...)
        tokenizer = self.processor.tokenizer
        self.allowed_digit_token_ids = set()
        for d in "0123456789":
            ids = tokenizer.encode(d, add_special_tokens=False)
            for tid in ids: self.allowed_digit_token_ids.add(int(tid))
        
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