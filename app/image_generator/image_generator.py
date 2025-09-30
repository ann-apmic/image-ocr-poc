"""
圖片生成模塊
用於生成測試用的OCR圖片，包含多語言文字和各種視覺效果
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 嘗試匯入OpenCV，如果失敗則使用替代方案
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("警告: OpenCV不可用，將使用PIL-only模式。某些效果可能無法應用。")


class ImageGenerator:
    """
    OCR測試圖片生成器

    支持功能：
    - 多語言文字生成（英文、中文、日文、韓文）
    - 字體大小調整
    - 模糊效果
    - 噪點效果
    - 旋轉效果
    - 設定記錄和儲存
    """

    # 預設設定
    DEFAULT_CONFIG = {
        "width": 800,
        "height": 600,
        "background_color": (255, 255, 255),  # 白色背景
        "font_size_range": (20, 100),
        "blur_range": (0, 3),
        "noise_level_range": (0, 50),
        "rotation_range": (-5, 5),
        "text_color": (0, 0, 0),  # 黑色文字
    }

    # 測試文字樣本
    TEST_TEXTS = {
        "english": [
            "Hello World",
            "OpenAI GPT-4",
            "Machine Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Deep Learning",
            "Artificial Intelligence",
            "The quick brown fox jumps over the lazy dog",
        ],
        "chinese": [
            "你好世界",
            "人工智能",
            "機器學習",
            "計算機視覺",
            "自然語言處理",
            "深度學習",
            "圖像識別",
            "歡迎使用OCR測試系統",
        ],
        "japanese": [
            "こんにちは世界",
            "人工知能",
            "機械学習",
            "コンピュータビジョン",
            "自然言語処理",
            "深層学習",
            "画像認識",
            "OCRテストシステムへようこそ",
        ],
        "korean": [
            "안녕하세요 세계",
            "인공지능",
            "기계학습",
            "컴퓨터비전",
            "자연어처리",
            "딥러닝",
            "이미지인식",
            "OCR 테스트 시스템에 오신 것을 환영합니다",
        ],
    }

    def __init__(self, output_dir: str = "_testing_images"):
        """
        初始化圖片生成器

        Args:
            output_dir: 輸出目錄路徑
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.configs_dir = os.path.join(output_dir, "configs")

        # 創建輸出目錄
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)

        # 載入系統字體
        self.fonts = self._load_system_fonts()

    def _load_system_fonts(self) -> Dict[str, List[str]]:
        """
        載入系統中可用的字體

        Returns:
            按語言分類的字體路徑列表
        """
        fonts = {"english": [], "chinese": [], "japanese": [], "korean": []}

        # 使用 fc-list 獲取系統中所有字體
        try:
            import subprocess

            result = subprocess.run(
                ["fc-list"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                font_lines = result.stdout.strip().split("\n")
                for line in font_lines:
                    if ":" in line:
                        font_path = line.split(":")[0].strip()
                        font_name = (
                            line.split(":")[1].strip()
                            if len(line.split(":")) > 1
                            else ""
                        )

                        # 分類字體
                        font_name_lower = font_name.lower()
                        font_path_lower = font_path.lower()

                        # 中文字體
                        if (
                            any(
                                keyword in font_name_lower
                                for keyword in [
                                    "noto sans cjk",
                                    "noto serif cjk",
                                    "ukai",
                                    "uming",
                                    "wenquanyi",
                                    "wqy",
                                    "ar pl",
                                    "microsoft yahei",
                                    "simsun",
                                ]
                            )
                            or "cjk" in font_path_lower
                        ):
                            fonts["chinese"].append(font_path)
                            fonts["japanese"].append(font_path)
                            fonts["korean"].append(font_path)

                        # 日文字體
                        elif any(
                            keyword in font_name_lower for keyword in ["jp", "japan"]
                        ):
                            fonts["japanese"].append(font_path)

                        # 韓文字體
                        elif any(
                            keyword in font_name_lower for keyword in ["kr", "korea"]
                        ):
                            fonts["korean"].append(font_path)

                        # 英文字體（作為備用）
                        elif any(
                            keyword in font_name_lower
                            for keyword in [
                                "dejavu",
                                "liberation",
                                "ubuntu",
                                "arial",
                                "times",
                            ]
                        ):
                            fonts["english"].append(font_path)
                            # 一些英文字體也支持基本的中日韓字符
                            fonts["chinese"].append(font_path)
                            fonts["japanese"].append(font_path)
                            fonts["korean"].append(font_path)

        except Exception as e:
            print(f"使用 fc-list 載入字體失敗: {e}，改用檔案系統掃描")

        # 如果 fc-list 失敗，備用檔案系統掃描
        if not any(fonts.values()):
            font_paths = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                "/System/Library/Fonts",  # macOS
                "/Library/Fonts",  # macOS
                "C:/Windows/Fonts",  # Windows
            ]

            for base_path in font_paths:
                if os.path.exists(base_path):
                    for root, dirs, files in os.walk(base_path):
                        for file in files:
                            if file.endswith((".ttf", ".ttc", ".otf")):
                                font_path = os.path.join(root, file)
                                file_lower = file.lower()

                                # 基於檔案名分類
                                if any(
                                    keyword in file_lower
                                    for keyword in [
                                        "noto",
                                        "cjk",
                                        "ukai",
                                        "uming",
                                        "wqy",
                                        "wenquanyi",
                                        "arpl",
                                    ]
                                ):
                                    fonts["chinese"].append(font_path)
                                    fonts["japanese"].append(font_path)
                                    fonts["korean"].append(font_path)
                                elif any(
                                    keyword in file_lower
                                    for keyword in ["dejavu", "liberation"]
                                ):
                                    fonts["english"].append(font_path)
                                    fonts["chinese"].append(font_path)  # 備用

        # 如果還是沒有找到字體，使用PIL默認字體
        for lang in fonts:
            if not fonts[lang]:
                fonts[lang] = [None]  # 使用PIL默認字體

        # 輸出載入的字體資訊
        for lang, font_list in fonts.items():
            valid_fonts = [f for f in font_list if f is not None]
            print(f"載入 {lang} 字體: {len(valid_fonts)} 個")

        return fonts

    def generate_image(
        self, text: str, language: str = "english", config: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        生成單張測試圖片

        Args:
            text: 要添加的文字
            language: 文字語言 ('english', 'chinese', 'japanese', 'korean')
            config: 自定義設定，如果為None則使用隨機設定

        Returns:
            (圖片檔案路徑, 使用的設定字典)
        """
        # 使用默認設定或自定義設定
        if config is None:
            config = self._generate_random_config()
        else:
            # 合併默認設定和用戶提供的設定
            default_config = self.DEFAULT_CONFIG.copy()
            default_config.update(config)
            config = default_config

        # 創建背景圖片
        image = Image.new(
            "RGB", (config["width"], config["height"]), config["background_color"]
        )
        draw = ImageDraw.Draw(image)

        # 選擇字體
        font_path = random.choice(self.fonts.get(language, [None]))
        try:
            font = (
                ImageFont.truetype(font_path, config["font_size"])
                if font_path
                else ImageFont.load_default()
            )
        except:
            font = ImageFont.load_default()

        # 計算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (config["width"] - text_width) // 2
        y = (config["height"] - text_height) // 2

        # 繪製文字
        draw.text((x, y), text, font=font, fill=config["text_color"])

        # 應用視覺效果
        image = self._apply_effects(image, config)

        # 生成檔案名稱
        import uuid

        image_id = str(uuid.uuid4())[:8]
        image_filename = f"{language}_{image_id}.png"
        image_path = os.path.join(self.images_dir, image_filename)

        # 儲存圖片
        image.save(image_path)

        # 儲存設定
        config_filename = f"{language}_{image_id}.json"
        config_path = os.path.join(self.configs_dir, config_filename)
        config_data = {
            "image_id": image_id,
            "language": language,
            "text": text,
            "image_path": image_path,
            "config_path": config_path,
            **config,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return image_path, config_data

    def _generate_random_config(self) -> Dict:
        """
        生成隨機設定

        Returns:
            隨機設定字典
        """
        config = self.DEFAULT_CONFIG.copy()

        # 隨機字體大小
        config["font_size"] = random.randint(*config["font_size_range"])

        # 隨機模糊程度
        config["blur_radius"] = random.uniform(*config["blur_range"])

        # 隨機噪點等級
        config["noise_level"] = random.randint(*config["noise_level_range"])

        # 隨機旋轉角度
        config["rotation_angle"] = random.uniform(*config["rotation_range"])

        return config

    def _apply_effects(self, image: Image.Image, config: Dict) -> Image.Image:
        """
        應用視覺效果到圖片

        Args:
            image: PIL圖片對象
            config: 設定字典

        Returns:
            處理後的圖片
        """
        # 使用PIL-only模式如果OpenCV不可用
        if not OPENCV_AVAILABLE:
            # 只應用基本的PIL效果
            # 應用模糊
            if config.get("blur_radius", 0) > 0:
                image = image.filter(ImageFilter.GaussianBlur(config["blur_radius"]))

            # 簡單的噪點效果（使用PIL）
            if config.get("noise_level", 0) > 0:
                # 轉換為numpy數組
                img_array = np.array(image)

                # 添加噪點
                noise = np.random.normal(
                    0, config["noise_level"], img_array.shape
                ).astype(np.uint8)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(
                    np.uint8
                )

                image = Image.fromarray(img_array)

            # PIL旋轉（如果需要的話）
            if config.get("rotation_angle", 0) != 0:
                image = image.rotate(
                    config["rotation_angle"],
                    fillcolor=tuple(config["background_color"]),
                    expand=True,
                )

            return image

        # 使用OpenCV進行完整效果處理
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 應用旋轉
        if config.get("rotation_angle", 0) != 0:
            height, width = cv_image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(
                center, config["rotation_angle"], 1.0
            )
            cv_image = cv2.warpAffine(
                cv_image,
                rotation_matrix,
                (width, height),
                borderValue=config["background_color"][::-1],
            )  # BGR格式

        # 添加噪點
        if config.get("noise_level", 0) > 0:
            noise = np.random.normal(0, config["noise_level"], cv_image.shape).astype(
                np.uint8
            )
            cv_image = cv2.add(cv_image, noise)

        # 轉回PIL格式
        image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # 應用模糊
        if config.get("blur_radius", 0) > 0:
            image = image.filter(ImageFilter.GaussianBlur(config["blur_radius"]))

        return image

    def generate_batch(
        self, count: int = 10, languages: Optional[List[str]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        批量生成測試圖片

        Args:
            count: 生成圖片數量
            languages: 要使用的語言列表，默認為所有語言

        Returns:
            生成的圖片路徑和設定列表
        """
        if languages is None:
            languages = list(self.TEST_TEXTS.keys())

        results = []

        for _ in range(count):
            # 隨機選擇語言和文字
            language = random.choice(languages)
            text = random.choice(self.TEST_TEXTS[language])

            # 生成圖片
            image_path, config = self.generate_image(text, language)
            results.append((image_path, config))

        return results

    def load_config(self, config_path: str) -> Dict:
        """
        載入圖片設定

        Args:
            config_path: 設定檔案路徑

        Returns:
            設定字典
        """
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
