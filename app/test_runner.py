"""
OCR測試運行器
整合圖片生成和OCR測試，提供完整的測試流程
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from image_generator.image_generator import ImageGenerator
from ocr_engine.ocr_manager import ocr_manager


class OCRTestRunner:
    """
    OCR測試運行器
    """

    def __init__(self, output_dir: str = "_testing_images"):
        """
        初始化測試運行器

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = output_dir
        self.reports_dir = os.path.join(output_dir, "reports")
        self.image_generator = ImageGenerator(output_dir)

        os.makedirs(self.reports_dir, exist_ok=True)

    def run_test(
        self,
        image_count: int = 10,
        languages: Optional[List[str]] = None,
        ocr_engine: Optional[str] = None,
        generate_new_images: bool = True,
    ) -> Dict:
        """
        運行完整的OCR測試

        Args:
            image_count: 生成的圖片數量
            languages: 測試語言列表
            ocr_engine: OCR引擎名稱
            generate_new_images: 是否生成新圖片

        Returns:
            測試結果摘要
        """
        print("開始OCR測試...")

        # 1. 生成測試圖片
        if generate_new_images:
            print(f"生成 {image_count} 張測試圖片...")
            test_images = self.image_generator.generate_batch(image_count, languages)
        else:
            print("使用現有的測試圖片...")
            test_images = self._load_existing_images()

        if not test_images:
            raise ValueError("沒有找到測試圖片")

        print(f"共載入 {len(test_images)} 張測試圖片")

        # 2. 運行OCR測試
        print(f"使用 {ocr_engine or '默認'} 引擎進行OCR識別...")
        ocr_results = []
        test_details = []

        for image_path, config in test_images:
            # OCR識別
            ocr_result = ocr_manager.recognize_text(
                image_path, engine=ocr_engine, languages=[config["language"]]
            )

            # 計算準確度
            accuracy = self._calculate_accuracy(config["text"], ocr_result.text)

            # 記錄詳細結果
            detail = {
                "image_id": config["image_id"],
                "image_path": image_path,
                "original_text": config["text"],
                "language": config["language"],
                "recognized_text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "accuracy": accuracy,
                "processing_time": ocr_result.processing_time,
                "font_size": config.get("font_size", "N/A"),
                "blur_radius": config.get("blur_radius", "N/A"),
                "noise_level": config.get("noise_level", "N/A"),
                "rotation_angle": config.get("rotation_angle", "N/A"),
                "ocr_engine": ocr_result.engine_name,
                "engine_version": ocr_result.engine_version,
            }

            test_details.append(detail)
            ocr_results.append(ocr_result)

        # 3. 生成報告
        summary = self._generate_report(test_details, ocr_engine or "default")

        print("測試完成！")
        print(f"平均準確度: {summary['average_accuracy']:.2f}%")
        print(f"平均信心度: {summary['average_confidence']:.2f}%")
        print(f"平均處理時間: {summary['average_processing_time']:.3f}秒")

        return summary

    def _load_existing_images(self) -> List[Tuple[str, Dict]]:
        """
        載入現有的測試圖片和設定

        Returns:
            (圖片路徑, 設定) 列表
        """
        images_dir = os.path.join(self.output_dir, "images")
        configs_dir = os.path.join(self.output_dir, "configs")

        if not os.path.exists(images_dir) or not os.path.exists(configs_dir):
            return []

        test_images = []

        for config_file in os.listdir(configs_dir):
            if config_file.endswith(".json"):
                config_path = os.path.join(configs_dir, config_file)
                try:
                    config = self.image_generator.load_config(config_path)
                    image_path = config["image_path"]

                    if os.path.exists(image_path):
                        test_images.append((image_path, config))
                except Exception as e:
                    print(f"載入設定檔案 {config_file} 失敗: {e}")

        return test_images

    def _calculate_accuracy(self, original_text: str, recognized_text: str) -> float:
        """
        計算文字識別準確度

        Args:
            original_text: 原始文字
            recognized_text: 識別出的文字

        Returns:
            準確度百分比 (0-100)
        """
        if not original_text:
            return 100.0 if not recognized_text else 0.0

        # 簡單的字元級準確度計算
        original_clean = "".join(original_text.split()).lower()
        recognized_clean = "".join(recognized_text.split()).lower()

        if not recognized_clean:
            return 0.0

        # 計算最長公共子序列長度
        def lcs_length(s1: str, s2: str) -> int:
            if not s1 or not s2:
                return 0

            # 使用動態規劃
            dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[-1][-1]

        lcs_len = lcs_length(original_clean, recognized_clean)
        accuracy = (lcs_len / len(original_clean)) * 100

        return min(accuracy, 100.0)  # 確保不超過100%

    def _generate_report(self, test_details: List[Dict], engine_name: str) -> Dict:
        """
        生成測試報告

        Args:
            test_details: 測試詳細結果
            engine_name: 引擎名稱

        Returns:
            摘要統計
        """
        if not test_details:
            return {}

        # 轉換為DataFrame進行統計
        df = pd.DataFrame(test_details)

        # 計算統計數據
        summary = {
            "total_images": len(test_details),
            "average_accuracy": df["accuracy"].mean(),
            "average_confidence": df["confidence"].mean(),
            "average_processing_time": df["processing_time"].mean(),
            "engine_name": engine_name,
            "timestamp": datetime.now().isoformat(),
            "language_breakdown": {},
            "effect_analysis": {},
        }

        # 按語言統計
        for language in df["language"].unique():
            lang_data = df[df["language"] == language]
            summary["language_breakdown"][language] = {
                "count": len(lang_data),
                "average_accuracy": lang_data["accuracy"].mean(),
                "average_confidence": lang_data["confidence"].mean(),
            }

        # 按效果分析
        effect_columns = ["font_size", "blur_radius", "noise_level", "rotation_angle"]
        for effect in effect_columns:
            if effect in df.columns:
                # 將數值分組統計
                try:
                    df_effect = df.copy()
                    df_effect[effect] = pd.to_numeric(
                        df_effect[effect], errors="coerce"
                    )

                    # 按四分位數分組
                    if not df_effect[effect].empty:
                        quartiles = df_effect[effect].quantile([0.25, 0.5, 0.75])
                        summary["effect_analysis"][effect] = {
                            "low": df_effect[df_effect[effect] <= quartiles[0.25]][
                                "accuracy"
                            ].mean(),
                            "medium": df_effect[
                                (df_effect[effect] > quartiles[0.25])
                                & (df_effect[effect] <= quartiles[0.75])
                            ]["accuracy"].mean(),
                            "high": df_effect[df_effect[effect] > quartiles[0.75]][
                                "accuracy"
                            ].mean(),
                        }
                except Exception as e:
                    print(f"分析效果 {effect} 時出錯: {e}")

        # 保存詳細報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"ocr_test_report_{engine_name}_{timestamp}.json"
        report_path = os.path.join(self.reports_dir, report_filename)

        report_data = {"summary": summary, "details": test_details}

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 保存CSV格式的詳細結果
        csv_filename = f"ocr_test_details_{engine_name}_{timestamp}.csv"
        csv_path = os.path.join(self.reports_dir, csv_filename)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        summary["report_path"] = report_path
        summary["csv_path"] = csv_path

        print(f"測試報告已保存到: {report_path}")
        print(f"詳細結果CSV已保存到: {csv_path}")

        return summary

    def list_available_engines(self) -> List[str]:
        """
        列出可用的OCR引擎

        Returns:
            引擎名稱列表
        """
        return ocr_manager.list_engines()


def main():
    """
    主函數 - 命令行接口
    """
    parser = argparse.ArgumentParser(description="OCR測試運行器")
    parser.add_argument("--images", type=int, default=10, help="生成的測試圖片數量")
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["english", "chinese", "japanese", "korean"],
        default=["english", "chinese"],
        help="測試語言",
    )
    parser.add_argument("--engine", type=str, default=None, help="OCR引擎名稱")
    parser.add_argument(
        "--no-generate", action="store_true", help="不生成新圖片，使用現有圖片"
    )
    parser.add_argument("--list-engines", action="store_true", help="列出可用的OCR引擎")

    args = parser.parse_args()

    runner = OCRTestRunner()

    if args.list_engines:
        engines = runner.list_available_engines()
        print("可用的OCR引擎:")
        for engine in engines:
            info = ocr_manager.get_engine_info(engine)
            print(f"  - {engine}: {info['name']} v{info['version']}")
            print(f"    支持語言: {', '.join(info['supported_languages'])}")
        return

    try:
        summary = runner.run_test(
            image_count=args.images,
            languages=args.languages,
            ocr_engine=args.engine,
            generate_new_images=not args.no_generate,
        )

        print("\n測試摘要:")
        print(f"總圖片數: {summary['total_images']}")
        print(f"平均準確度: {summary['average_accuracy']:.2f}%")
        print(f"平均信心度: {summary['average_confidence']:.2f}%")
        print(f"平均處理時間: {summary['average_processing_time']:.3f}秒")

    except Exception as e:
        print(f"測試失敗: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
