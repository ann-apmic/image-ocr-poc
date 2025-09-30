"""
OCR引擎基類
定義OCR引擎的統一接口
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class OCRResult:
    """
    OCR識別結果
    """

    def __init__(
        self,
        text: str,
        confidence: float,
        processing_time: float,
        engine_name: str,
        engine_version: Optional[str] = None,
        bounding_boxes: Optional[List[Dict]] = None,
    ):
        """
        初始化OCR結果

        Args:
            text: 識別出的文字
            confidence: 信心度 (0-100)
            processing_time: 處理時間(秒)
            engine_name: 引擎名稱
            engine_version: 引擎版本
            bounding_boxes: 文字區域的bounding box列表，每個包含:
                - text: 文字內容
                - x, y, w, h: 座標和尺寸
                - confidence: 該區域的信心度
        """
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.engine_name = engine_name
        self.engine_version = engine_version
        self.bounding_boxes = bounding_boxes or []
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        """
        轉換為字典格式

        Returns:
            結果字典
        """
        return {
            "text": self.text,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "bounding_boxes": self.bounding_boxes,
        }


class BaseOCREngine(ABC):
    """
    OCR引擎抽象基類
    """

    def __init__(self, name: str, version: Optional[str] = None):
        """
        初始化OCR引擎

        Args:
            name: 引擎名稱
            version: 引擎版本
        """
        self.name = name
        self.version = version or self._get_version()

    @abstractmethod
    def _get_version(self) -> str:
        """
        獲取引擎版本

        Returns:
            版本字串
        """
        pass

    @abstractmethod
    def recognize_text(
        self, image_path: str, languages: Optional[List[str]] = None, **kwargs
    ) -> OCRResult:
        """
        識別圖片中的文字

        Args:
            image_path: 圖片檔案路徑
            languages: 語言列表 (例如 ['eng', 'chi_sim'])
            **kwargs: 額外的引擎特定參數

        Returns:
            OCRResult 對象
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        獲取支持的語言列表

        Returns:
            語言代碼列表
        """
        pass

    def is_language_supported(self, language: str) -> bool:
        """
        檢查是否支持指定語言

        Args:
            language: 語言代碼

        Returns:
            是否支持
        """
        return language in self.get_supported_languages()

    def preprocess_image(self, image_path: str) -> str:
        """
        預處理圖片 (可選實現)

        Args:
            image_path: 原始圖片路徑

        Returns:
            處理後的圖片路徑 (默認返回原路徑)
        """
        return image_path
