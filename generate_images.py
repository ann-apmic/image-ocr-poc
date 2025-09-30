#!/usr/bin/env python3
"""
圖片生成腳本
生成包含大量文字的測試圖片，用於OCR測試
"""

import os
import random
import sys
from typing import Dict, List, Optional

# 添加專案路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "app")
sys.path.insert(0, app_dir)

from app.image_generator.image_generator import ImageGenerator

# ==================== 生成參數設定 ====================

# 文字內容設定
TEXT_CONTENT = {
    "english": {
        "sentences": [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming our world.",
            "Natural language processing enables computers to understand human text.",
            "Computer vision allows machines to interpret and understand visual information.",
            "Deep learning algorithms can automatically learn features from data.",
        ],
        "paragraphs": [
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data. The field has seen tremendous growth in recent years, with applications ranging from image recognition to natural language processing.",
            "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos, it seeks to automate tasks that the human visual system can do. Applications include object detection, facial recognition, and autonomous vehicles.",
            "Natural language processing combines computational linguistics with statistical and machine learning models. It enables computers to process and understand human language. Key applications include machine translation, sentiment analysis, and conversational agents like chatbots.",
        ],
        "articles": [
            """The Future of Artificial Intelligence

Artificial Intelligence (AI) is revolutionizing industries across the globe. From healthcare to finance, transportation to entertainment, AI technologies are enhancing efficiency, accuracy, and innovation.

In healthcare, AI algorithms can analyze medical images with unprecedented precision, assisting doctors in early disease detection. Machine learning models can predict patient outcomes and recommend personalized treatment plans.

The transportation sector is witnessing a transformation through autonomous vehicles. Self-driving cars use computer vision, sensor fusion, and deep learning to navigate complex road environments safely.

Education is another field benefiting from AI. Intelligent tutoring systems can adapt to individual learning styles, providing personalized instruction and feedback. Language learning apps use natural language processing to offer conversational practice.

As AI continues to evolve, ethical considerations become increasingly important. Ensuring transparency, fairness, and accountability in AI systems is crucial for building public trust and maximizing societal benefits.""",
            """The Impact of Machine Learning on Society

Machine learning has become an integral part of our daily lives, often operating behind the scenes to enhance our experiences and solve complex problems.

Recommendation systems power platforms like Netflix and Spotify, analyzing user preferences to suggest content. These systems use collaborative filtering and deep learning to understand patterns in user behavior.

In finance, machine learning algorithms detect fraudulent transactions with high accuracy. They analyze transaction patterns and identify anomalies that might indicate suspicious activity.

Social media platforms use machine learning for content moderation, automatically detecting and removing harmful content. Natural language processing helps identify hate speech, misinformation, and other problematic content.

Environmental monitoring benefits from machine learning through predictive analytics. Weather forecasting models use historical data and current conditions to predict storms and climate patterns.

The healthcare industry leverages machine learning for drug discovery, analyzing molecular structures to identify potential treatments. Medical imaging analysis assists radiologists in detecting abnormalities.

As machine learning continues to advance, its societal impact will only grow, bringing both opportunities and challenges that require careful consideration.""",
        ],
    },
    "chinese": {
        "sentences": [
            "人工智能正在改變我們的世界。",
            "機器學習是一種強大的技術工具。",
            "自然語言處理讓電腦能夠理解人類語言。",
            "計算機視覺使機器能夠解釋視覺資訊。",
            "深度學習演算法可以自動從數據中學習特徵。",
        ],
        "paragraphs": [
            "機器學習是人工智慧的一個子集，它使計算機能夠在沒有明確程式設計的情況下學習。它使用演算法和統計模型來分析數據中的模式並得出推論。近年來，該領域取得了巨大的發展，應用範圍從圖像識別到自然語言處理。",
            "計算機視覺是人工智慧領域的一個分支，它訓練計算機解釋和理解視覺世界。使用來自相機和視頻的數位圖像，它試圖自動化人類視覺系統所能做到的任務。應用包括物體檢測、面部識別和自動駕駛汽車。",
            "自然語言處理將計算語言學與統計和機器學習模型相結合。它使計算機能夠處理和理解人類語言。關鍵應用包括機器翻譯、情感分析和對話代理如聊天機器人。",
        ],
        "articles": [
            """人工智慧的未來

人工智慧（AI）正在徹底改變全球各行業。從醫療保健到金融、交通運輸到娛樂，AI技術正在提高效率、準確性和創新性。

在醫療保健領域，AI演算法能夠以前所未有的精準度分析醫學圖像，幫助醫生早期發現疾病。機器學習模型可以預測患者預後並推薦個性化治療方案。

交通運輸部門正通過自動駕駛汽車見證轉型。自動駕駛汽車使用計算機視覺、感測器融合和深度學習來安全導航複雜的道路環境。

教育是另一個受益於AI的領域。智慧輔導系統可以適應個別學習風格，提供個性化指導和反饋。語言學習應用使用自然語言處理來提供對話練習。

隨著AI的不斷發展，倫理考量變得越來越重要。確保AI系統的透明度、公平性和問責制對於建立公眾信任和最大化社會利益至關重要。""",
            """機器學習對社會的影響

機器學習已經成為我們日常生活的一部分，往往在幕後運作以增強我們的體驗並解決複雜問題。

推薦系統為Netflix和Spotify等平台提供動力，分析用戶偏好以建議內容。這些系統使用協同過濾和深度學習來理解用戶行為模式。

在金融領域，機器學習演算法以高準確度檢測欺詐交易。它們分析交易模式並識別可能表示可疑活動的異常。

社交媒體平台使用機器學習進行內容審核，自動檢測和移除有害內容。自然語言處理有助於識別仇恨言論、錯誤資訊和其他問題內容。

環境監測通過預測分析受益於機器學習。天氣預報模型使用歷史數據和當前條件來預測風暴和氣候模式。

醫療保健行業利用機器學習進行藥物發現，分析分子結構以識別潛在治療方法。醫學圖像分析幫助放射科醫生檢測異常。

隨著機器學習的持續進步，其社會影響只會增長，帶來需要仔細考慮的機會和挑戰。""",
        ],
    },
}

# 生成設定
GENERATE_SETTINGS = [
    {
        "name": "簡單句子",
        "text_type": "sentences",
        "font_size": 24,
        "max_chars_per_line": 50,
        "line_spacing": 1.2,
        "margin": 40,
        "enabled": True,
    },
    {
        "name": "段落文字",
        "text_type": "paragraphs",
        "font_size": 20,
        "max_chars_per_line": 60,
        "line_spacing": 1.3,
        "margin": 30,
        "enabled": True,
    },
    {
        "name": "文章內容",
        "text_type": "articles",
        "font_size": 18,
        "max_chars_per_line": 70,
        "line_spacing": 1.4,
        "margin": 25,
        "enabled": True,
    },
]

# 圖片尺寸設定
IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 800

# 效果設定
ENABLE_EFFECTS = True
BLUR_RANGE = (0, 1.5)
NOISE_RANGE = (0, 25)
ROTATION_RANGE = (-2, 2)

# 語言設定
LANGUAGES_TO_GENERATE = ["english", "chinese"]  # 要生成哪些語言的圖片

# 生成數量設定
IMAGES_PER_SETTING = 3  # 每個設定生成多少張圖片

# ==================== 參數設定結束 ====================


class AdvancedImageGenerator(ImageGenerator):
    """
    進階圖片生成器，支援多行文字和自動布局
    """

    def generate_text_image(
        self, text: str, language: str, settings: Dict, apply_effects: bool = True
    ) -> tuple:
        """
        生成包含大量文字的圖片

        Args:
            text: 文字內容
            language: 語言
            settings: 生成設定
            apply_effects: 是否應用視覺效果

        Returns:
            (圖片路徑, 設定字典)
        """
        # 準備圖片配置
        config = {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "background_color": (255, 255, 255),
            "text_color": (0, 0, 0),
            "font_size": settings["font_size"],
        }

        # 如果啟用效果，添加隨機效果
        if apply_effects:
            config.update(
                {
                    "blur_radius": random.uniform(*BLUR_RANGE),
                    "noise_level": random.randint(*NOISE_RANGE),
                    "rotation_angle": random.uniform(*ROTATION_RANGE),
                }
            )

        # 創建圖片並繪製文字
        image = self._create_text_image(text, language, config, settings)

        # 生成檔案名稱
        import uuid

        image_id = str(uuid.uuid4())[:8]
        # 只使用語言和ID，不包含設定名稱
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
            "text_type": settings["text_type"],
            "image_path": image_path,
            "config_path": config_path,
            "settings": settings,
            **config,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            import json

            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return image_path, config_data

    def _create_text_image(
        self, text: str, language: str, config: Dict, settings: Dict
    ):
        """
        創建包含文字的圖片，支援自動換行
        """
        from PIL import Image, ImageDraw

        # 創建圖片
        image = Image.new(
            "RGB", (config["width"], config["height"]), config["background_color"]
        )
        draw = ImageDraw.Draw(image)

        # 選擇字體 - 優先選擇支援目標語言的字體
        available_fonts = self.fonts.get(language, [None])
        if not available_fonts or available_fonts == [None]:
            # 如果沒有語言特定的字體，嘗試其他語言的字體
            for lang in ["english", "chinese", "japanese", "korean"]:
                if lang != language and self.fonts.get(lang):
                    available_fonts = self.fonts[lang]
                    break

        if not available_fonts or available_fonts == [None]:
            available_fonts = [None]

        # 隨機選擇字體，但排除None
        valid_fonts = [f for f in available_fonts if f is not None]
        if valid_fonts:
            font_path = random.choice(valid_fonts)
        else:
            font_path = None

        try:
            font = self._get_font(font_path, config["font_size"])
            # 測試字體是否能正確渲染中文
            if language in ["chinese", "japanese", "korean"]:
                test_text = (
                    "測試"
                    if language == "chinese"
                    else ("テスト" if language == "japanese" else "테스트")
                )
                test_bbox = font.getbbox(test_text)
                if not test_bbox or test_bbox[2] <= test_bbox[0]:
                    # 如果字體無法渲染測試文字，嘗試其他字體
                    raise Exception("字體無法渲染目標語言文字")
        except Exception as e:
            print(f"  字體載入失敗 ({font_path or '默認字體'}): {e}")
            # 嘗試備用字體
            font = self._get_font(None, config["font_size"])
            print(
                f"  使用備用字體: {font.getname() if hasattr(font, 'getname') else '默認'}"
            )

        # 分割文字為行
        lines = self._split_text_into_lines(text, font, settings)

        # 計算更好的行高
        margin = settings["margin"]

        # 使用字體度量來計算行高
        # 獲取字體的行距資訊
        try:
            # 嘗試獲取字體的 ascent 和 descent
            ascent, descent = font.getmetrics()
            # 對於中日韓文字，使用更大的行高來避免重疊
            if language in ["chinese", "japanese", "korean"]:
                line_height = int(
                    (ascent + descent) * settings["line_spacing"] * 1.5
                )  # 增加50%的行高
            else:
                line_height = int((ascent + descent) * settings["line_spacing"])
        except:
            # 如果無法獲取度量，使用經驗值
            if language in ["chinese", "japanese", "korean"]:
                line_height = int(
                    config["font_size"] * settings["line_spacing"] * 1.8
                )  # 為CJK文字使用更大的行高
            else:
                line_height = int(config["font_size"] * settings["line_spacing"] * 1.2)

        total_text_height = len(lines) * line_height

        # 確保文字不會超出圖片範圍
        available_height = config["height"] - 2 * margin
        if total_text_height > available_height:
            # 如果文字太長，縮小字體
            scale_factor = available_height / total_text_height
            new_font_size = int(config["font_size"] * scale_factor * 0.9)  # 留一些邊距
            config["font_size"] = max(new_font_size, 12)  # 最小字體大小
            font = self._get_font(font_path, config["font_size"])

            # 重新計算行高（保持CJK文字的行高優勢）
            try:
                ascent, descent = font.getmetrics()
                if language in ["chinese", "japanese", "korean"]:
                    line_height = int(
                        (ascent + descent) * settings["line_spacing"] * 1.5
                    )
                else:
                    line_height = int((ascent + descent) * settings["line_spacing"])
            except:
                if language in ["chinese", "japanese", "korean"]:
                    line_height = int(
                        config["font_size"] * settings["line_spacing"] * 1.8
                    )
                else:
                    line_height = int(
                        config["font_size"] * settings["line_spacing"] * 1.2
                    )

            lines = self._split_text_into_lines(text, font, settings)
            total_text_height = len(lines) * line_height

        # 計算文字區域的垂直起始位置（垂直居中）
        text_block_height = len(lines) * line_height
        start_y = margin + (available_height - text_block_height) // 2
        start_y = max(start_y, margin)  # 確保不小於邊距

        # 繪製文字
        current_y = start_y
        for i, line in enumerate(lines):
            # 移除行首尾空白
            line = line.strip()
            if not line:
                continue

            # 計算這行的寬度並居中
            line_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            x = (config["width"] - line_width) // 2

            # 確保不超出左右邊界
            if x < margin:
                x = margin
            max_x = config["width"] - margin - line_width
            if x > max_x:
                x = max_x

            # 確保x不為負數
            x = max(x, 0)

            # 繪製文字行（使用頂部對齊）
            draw.text((x, current_y), line, font=font, fill=config["text_color"])

            # 移動到下一行（確保行間距）
            current_y += line_height

        return image

    def _get_font(self, font_path, font_size):
        """獲取字體對象"""
        from PIL import ImageFont

        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
        else:
            return ImageFont.load_default()

    def _split_text_into_lines(self, text: str, font, settings: Dict) -> List[str]:
        """
        將文字分割為多行，確保每行長度適中
        """
        max_chars = settings["max_chars_per_line"]

        # 如果是中文或日文韓文，使用字符數限制
        if any(ord(char) > 127 for char in text):
            # 對於中日韓文，按字符數分割
            lines = []
            current_line = ""
            words = list(text)  # 按字符分割

            for word in words:
                if len(current_line + word) <= max_chars:
                    current_line += word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return lines
        else:
            # 對於英文，按單詞分割
            lines = []
            words = text.split()
            current_line = ""

            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += " " + word if current_line else word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            return lines


def generate_test_images():
    """
    生成測試圖片的主函數
    """
    print("=== 圖片生成器 ===\n")

    # 清理舊檔案
    print("正在清理舊的測試檔案...")
    cleanup_old_files()
    print("清理完成\n")

    # 初始化生成器
    generator = AdvancedImageGenerator()

    total_generated = 0

    # 為每個語言生成圖片
    for language in LANGUAGES_TO_GENERATE:
        if language not in TEXT_CONTENT:
            print(f"警告: 語言 '{language}' 的文字內容未定義，跳過")
            continue

        print(f"正在生成 {language} 圖片...")

        lang_content = TEXT_CONTENT[language]

        # 為每個設定生成圖片
        for setting in GENERATE_SETTINGS:
            if not setting.get("enabled", True):
                continue

            text_type = setting["text_type"]
            if text_type not in lang_content:
                print(f"  警告: 文字類型 '{text_type}' 在語言 '{language}' 中不存在")
                continue

            texts = lang_content[text_type]
            if not texts:
                continue

            print(f"  生成設定: {setting['name']} ({len(texts)} 個樣本)")

            # 為每個設定生成多張圖片
            for i in range(IMAGES_PER_SETTING):
                # 隨機選擇文字內容
                text = random.choice(texts)

                try:
                    img_path, config = generator.generate_text_image(
                        text, language, setting, ENABLE_EFFECTS
                    )

                    effects_info = ""
                    if ENABLE_EFFECTS:
                        effects = []
                        if config.get("blur_radius", 0) > 0:
                            effects.append(f"模糊:{config['blur_radius']:.1f}")
                        if config.get("noise_level", 0) > 0:
                            effects.append(f"噪點:{config['noise_level']}")
                        if config.get("rotation_angle", 0) != 0:
                            effects.append(f"旋轉:{config['rotation_angle']:.1f}°")
                        if effects:
                            effects_info = f" [{', '.join(effects)}]"

                    print(f"    {i+1}. {os.path.basename(img_path)}{effects_info}")

                    total_generated += 1

                except Exception as e:
                    print(f"    錯誤: 生成圖片失敗 - {e}")

    print(f"\n總共生成了 {total_generated} 張測試圖片")
    print("圖片和設定已保存到 _testing_images/ 目錄")


def cleanup_old_files():
    """
    清理舊的測試檔案（圖片和設定檔案）
    """
    import os
    import shutil

    images_dir = "_testing_images/images"
    configs_dir = "_testing_images/configs"

    # 刪除圖片檔案
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            if file.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(images_dir, file)
                try:
                    os.remove(file_path)
                    print(f"  刪除圖片: {file}")
                except Exception as e:
                    print(f"  刪除圖片失敗 {file}: {e}")

    # 刪除設定檔案
    if os.path.exists(configs_dir):
        for file in os.listdir(configs_dir):
            if file.endswith(".json"):
                file_path = os.path.join(configs_dir, file)
                try:
                    os.remove(file_path)
                    print(f"  刪除設定: {file}")
                except Exception as e:
                    print(f"  刪除設定失敗 {file}: {e}")

    # 如果目錄為空，創建 .gitkeep 檔案保持目錄結構
    for dir_path in [images_dir, configs_dir]:
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            gitkeep_path = os.path.join(dir_path, ".gitkeep")
            try:
                with open(gitkeep_path, "w") as f:
                    f.write("")
                print(f"  創建 .gitkeep: {dir_path}")
            except Exception as e:
                print(f"  創建 .gitkeep 失敗 {dir_path}: {e}")


if __name__ == "__main__":
    generate_test_images()
