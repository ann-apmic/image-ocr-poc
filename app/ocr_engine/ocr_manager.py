"""
OCR引擎管理器
負責管理和切換不同的OCR引擎實作
"""

import importlib
from typing import Dict, List, Optional, Type

from .base_ocr import BaseOCREngine, OCRResult


class OCREngineManager:
    """
    OCR引擎管理器
    支持動態載入和切換不同的OCR引擎
    """

    def __init__(self):
        """
        初始化管理器
        """
        self.engines: Dict[str, BaseOCREngine] = {}
        self.default_engine: Optional[str] = None

        # 自動載入內建引擎
        self._load_builtin_engines()

    def _load_builtin_engines(self):
        """
        載入內建的OCR引擎
        """
        try:
            from .tesseract_ocr import TesseractOCREngine

            self.register_engine("tesseract", TesseractOCREngine)
            self.default_engine = "tesseract"
        except ImportError as e:
            print(f"無法載入Tesseract引擎: {e}")

    def register_engine(self, name: str, engine_class: Type[BaseOCREngine], **kwargs):
        """
        註冊OCR引擎

        Args:
            name: 引擎名稱
            engine_class: 引擎類別
            **kwargs: 引擎初始化參數
        """
        try:
            engine_instance = engine_class(**kwargs)
            self.engines[name] = engine_instance
            print(
                f"成功註冊OCR引擎: {name} ({engine_instance.name} v{engine_instance.version})"
            )
        except Exception as e:
            print(f"註冊OCR引擎 {name} 失敗: {e}")

    def get_engine(self, name: Optional[str] = None) -> BaseOCREngine:
        """
        獲取OCR引擎實例

        Args:
            name: 引擎名稱，如果為None則返回默認引擎

        Returns:
            OCR引擎實例

        Raises:
            ValueError: 如果引擎不存在
        """
        engine_name = name or self.default_engine

        if engine_name not in self.engines:
            available = list(self.engines.keys())
            raise ValueError(f"OCR引擎 '{engine_name}' 不存在。可用的引擎: {available}")

        return self.engines[engine_name]

    def list_engines(self) -> List[str]:
        """
        列出所有已註冊的引擎

        Returns:
            引擎名稱列表
        """
        return list(self.engines.keys())

    def recognize_text(
        self,
        image_path: str,
        engine: Optional[str] = None,
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> OCRResult:
        """
        使用指定引擎識別圖片文字

        Args:
            image_path: 圖片檔案路徑
            engine: 引擎名稱
            languages: 語言列表
            **kwargs: 引擎特定參數

        Returns:
            OCRResult 對象
        """
        ocr_engine = self.get_engine(engine)
        return ocr_engine.recognize_text(image_path, languages, **kwargs)

    def batch_recognize(
        self,
        image_paths: List[str],
        engine: Optional[str] = None,
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> List[OCRResult]:
        """
        批量識別多張圖片的文字

        Args:
            image_paths: 圖片檔案路徑列表
            engine: 引擎名稱
            languages: 語言列表
            **kwargs: 引擎特定參數

        Returns:
            OCRResult 對象列表
        """
        ocr_engine = self.get_engine(engine)
        results = []

        for image_path in image_paths:
            result = ocr_engine.recognize_text(image_path, languages, **kwargs)
            results.append(result)

        return results

    def get_engine_info(self, name: Optional[str] = None) -> Dict:
        """
        獲取引擎資訊

        Args:
            name: 引擎名稱

        Returns:
            引擎資訊字典
        """
        engine = self.get_engine(name)
        return {
            "name": engine.name,
            "version": engine.version,
            "supported_languages": engine.get_supported_languages(),
        }

    def load_external_engine(
        self, module_path: str, class_name: str, engine_name: str, **kwargs
    ):
        """
        載入外部OCR引擎

        Args:
            module_path: 模塊路徑 (例如 'my_engines.custom_ocr')
            class_name: 類別名稱
            engine_name: 註冊名稱
            **kwargs: 初始化參數
        """
        try:
            module = importlib.import_module(module_path)
            engine_class = getattr(module, class_name)

            # 檢查是否為BaseOCREngine的子類
            if not issubclass(engine_class, BaseOCREngine):
                raise TypeError(f"{class_name} 必須是 BaseOCREngine 的子類")

            self.register_engine(engine_name, engine_class, **kwargs)

        except Exception as e:
            print(f"載入外部引擎失敗: {e}")


# 全域管理器實例
ocr_manager = OCREngineManager()
