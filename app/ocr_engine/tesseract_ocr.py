"""
Tesseract OCR引擎實作
"""

import re
import subprocess
import time
from typing import Dict, List, Optional

import pytesseract
from PIL import Image

from .base_ocr import BaseOCREngine, OCRResult


class TesseractOCREngine(BaseOCREngine):
    """
    Tesseract OCR引擎實作
    """

    # 語言代碼映射
    LANGUAGE_MAPPING = {
        "english": "eng",
        "chinese": "chi_sim+chi_tra",
        "japanese": "jpn",
        "korean": "kor",
        "eng": "eng",
        "chi_sim": "chi_sim",
        "chi_tra": "chi_tra",
        "jpn": "jpn",
        "kor": "kor",
    }

    def __init__(self, tesseract_cmd: str = "tesseract"):
        """
        初始化Tesseract引擎

        Args:
            tesseract_cmd: Tesseract命令路徑
        """
        super().__init__("Tesseract")
        self.tesseract_cmd = tesseract_cmd

        # 設置pytesseract的tesseract命令路徑
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # 檢查Tesseract是否可用
        self._check_tesseract_installation()

    def _get_version(self) -> str:
        """
        獲取Tesseract版本

        Returns:
            版本字串
        """
        try:
            result = subprocess.run(
                [self.tesseract_cmd, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                # 從輸出中提取版本號
                version_match = re.search(r"tesseract\s+(\d+\.\d+)", result.stdout)
                if version_match:
                    return version_match.group(1)
            return "Unknown"
        except Exception:
            return "Unknown"

    def _check_tesseract_installation(self):
        """
        檢查Tesseract安裝狀態
        """
        try:
            result = subprocess.run(
                [self.tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"Tesseract version: {self.version}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"Tesseract is not properly installed or not in PATH: {e}"
            )

    def recognize_text(
        self, image_path: str, languages: Optional[List[str]] = None, **kwargs
    ) -> OCRResult:
        """
        使用Tesseract識別圖片文字

        Args:
            image_path: 圖片檔案路徑
            languages: 語言列表
            **kwargs: 額外的參數
                - config: Tesseract配置字串
                - oem: OCR引擎模式 (0-3)
                - psm: 頁面分割模式 (0-13)

        Returns:
            OCRResult 對象
        """
        start_time = time.time()

        try:
            # 處理語言參數
            lang_str = self._process_languages(languages)

            # 處理額外參數
            config = kwargs.get("config", "")
            oem = kwargs.get("oem", 3)  # 默認使用默認OCR引擎模式
            psm = kwargs.get("psm", 6)  # 默認使用統一區塊的文字

            # 構建配置字串
            if not config:
                config = f"--oem {oem} --psm {psm}"

            # 使用pytesseract進行OCR
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image, lang=lang_str, config=config, output_type=pytesseract.Output.DICT
            )

            # 提取文字、信心度和bounding box
            text_parts = []
            confidences = []
            bounding_boxes = []

            for i, confidence in enumerate(data["conf"]):
                if (
                    confidence != "-1" and data["text"][i].strip()
                ):  # -1表示非文字區域，且文字不為空
                    text_parts.append(data["text"][i])
                    confidences.append(float(confidence))

                    # 收集bounding box資訊
                    bounding_boxes.append(
                        {
                            "text": data["text"][i],
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "w": data["width"][i],
                            "h": data["height"][i],
                            "confidence": float(confidence),
                        }
                    )

            # 合併文字
            # 智能拼接：根據字符性質決定是否加空格
            def is_chinese_char(text):
                """檢查文字是否為中文字符"""
                return bool(
                    text and any("\u4e00" <= char <= "\u9fff" for char in text.strip())
                )

            def is_english_word(text):
                """檢查文字是否為英文單詞（包含字母和常見標點）"""
                if not text or not text.strip():
                    return False
                text = text.strip()
                # 英文單詞應該主要包含字母、數字和常見標點
                return any(c.isalpha() for c in text) and not any(
                    "\u4e00" <= c <= "\u9fff" for c in text
                )

            # 根據字符性質智能拼接
            if not text_parts:
                recognized_text = ""
            else:
                result_parts = [text_parts[0]]  # 第一個部分總是保留

                for i in range(1, len(text_parts)):
                    prev_part = text_parts[i - 1].strip()
                    curr_part = text_parts[i].strip()

                    # 如果前一個或當前部分為空，跳過
                    if not prev_part or not curr_part:
                        result_parts.append(text_parts[i])
                        continue

                    # 判斷是否需要加空格
                    prev_is_chinese = is_chinese_char(prev_part)
                    curr_is_chinese = is_chinese_char(curr_part)
                    prev_is_english = is_english_word(prev_part)
                    curr_is_english = is_english_word(curr_part)

                    # 規則：
                    # 1. 中文字符間不加空格
                    # 2. 英文單詞間加空格
                    # 3. 其他情況（標點、數字等）根據上下文判斷
                    if (prev_is_chinese and curr_is_chinese) or (
                        prev_is_english and curr_is_english
                    ):
                        # 同類型字符間的連接規則
                        if prev_is_chinese and curr_is_chinese:
                            # 中文字符間直接連接
                            result_parts.append(text_parts[i])
                        elif prev_is_english and curr_is_english:
                            # 英文單詞間加空格
                            result_parts.append(" " + text_parts[i])
                        else:
                            # 其他情況，加空格
                            result_parts.append(" " + text_parts[i])
                    else:
                        # 不同類型或不明確的情況，加空格
                        result_parts.append(" " + text_parts[i])

                recognized_text = "".join(result_parts).strip()

            # 計算平均信心度
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            processing_time = time.time() - start_time

            return OCRResult(
                text=recognized_text,
                confidence=avg_confidence,
                processing_time=processing_time,
                engine_name=self.name,
                engine_version=self.version,
                bounding_boxes=bounding_boxes,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                engine_version=self.version,
                bounding_boxes=[],
            )

    def _process_languages(self, languages: Optional[List[str]]) -> str:
        """
        處理語言參數

        Args:
            languages: 語言列表

        Returns:
            Tesseract語言字串
        """
        if not languages:
            return "eng"  # 默認英語

        # 轉換語言代碼
        tesseract_langs = []
        for lang in languages:
            if lang in self.LANGUAGE_MAPPING:
                tesseract_langs.append(self.LANGUAGE_MAPPING[lang])
            else:
                # 如果找不到映射，直接使用
                tesseract_langs.append(lang)

        return "+".join(tesseract_langs)

    def get_supported_languages(self) -> List[str]:
        """
        獲取Tesseract支持的語言列表

        Returns:
            語言代碼列表
        """
        try:
            result = subprocess.run(
                [self.tesseract_cmd, "--list-langs"],
                capture_output=True,
                text=True,
                check=True,
            )

            # 解析輸出，跳過第一行（通常是"Available languages:"）
            lines = result.stdout.strip().split("\n")[1:]
            return [lang.strip() for lang in lines if lang.strip()]

        except subprocess.CalledProcessError:
            # 如果命令失敗，返回常見的語言
            return ["eng", "chi_sim", "chi_tra", "jpn", "kor"]

    def get_detailed_data(
        self, image_path: str, languages: Optional[List[str]] = None, **kwargs
    ) -> Dict:
        """
        獲取詳細的OCR數據，包括每個文字的邊界框和信心度

        Args:
            image_path: 圖片檔案路徑
            languages: 語言列表
            **kwargs: 額外參數

        Returns:
            詳細數據字典
        """
        lang_str = self._process_languages(languages)
        config = kwargs.get("config", "--oem 3 --psm 6")

        image = Image.open(image_path)
        data = pytesseract.image_to_data(
            image, lang=lang_str, config=config, output_type=pytesseract.Output.DICT
        )

        # 過濾掉空文字和信心度為-1的項目
        filtered_data = {
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
        }

        for i in range(len(data["text"])):
            if data["conf"][i] != "-1" and data["text"][i].strip():
                filtered_data["text"].append(data["text"][i])
                filtered_data["conf"].append(data["conf"][i])
                filtered_data["left"].append(data["left"][i])
                filtered_data["top"].append(data["top"][i])
                filtered_data["width"].append(data["width"][i])
                filtered_data["height"].append(data["height"][i])

        return filtered_data
