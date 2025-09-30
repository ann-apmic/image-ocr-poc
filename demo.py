#!/usr/bin/env python3
"""
OCR測試系統演示腳本
快速展示圖片生成和OCR測試功能
"""

import os
import sys

from app.ocr_engine.ocr_manager import ocr_manager

# 添加專案路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "app")
sys.path.insert(0, app_dir)

# ==================== 測試參數設定 ====================

# 測試資料夾設定 (必須設定此參數，將從該資料夾載入圖片進行測試)
# 如果資料夾中有 config 檔案，將進行準確度比對；如果沒有，只顯示OCR結果
TEST_FOLDER = "_table_custom_images"  # 設定測試資料夾路徑，不能為空

# OCR引擎設定
DEFAULT_OCR_ENGINE = "tesseract"  # 使用的OCR引擎

# 顯示設定
SHOW_ENGINE_INFO = True  # 是否顯示引擎資訊
SHOW_OCR_RESULTS = True  # 是否顯示OCR結果
SHOW_BOUNDING_BOXES = True  # 是否顯示文字座標資訊
SHOW_SUMMARY = True  # 是否顯示總結資訊

# ==================== 測試參數設定結束 ====================


def demo():
    """
    演示功能
    """
    print("=== OCR測試系統演示 ===\n")

    # 檢查 TEST_FOLDER 是否設定
    if not TEST_FOLDER or TEST_FOLDER.strip() == "":
        print("❌ 錯誤: TEST_FOLDER 參數不能為空")
        print("請在 demo.py 中設定 TEST_FOLDER 參數為有效的資料夾路徑")
        print('例如: TEST_FOLDER = "your_images_folder"')
        return

    # 清空並準備結果資料夾
    import shutil

    results_dir = "_results"
    if os.path.exists(results_dir):
        print(f"清空結果資料夾: {results_dir}")
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f"結果將儲存到: {results_dir}")
    print()

    # 1. 檢查可用引擎
    if SHOW_ENGINE_INFO:
        print("1. 檢查可用OCR引擎:")
        engines = ocr_manager.list_engines()
        for engine in engines:
            info = ocr_manager.get_engine_info(engine)
            print(f"   - {engine}: {info['name']} v{info['version']}")
            print(
                f"     支持語言: {', '.join(info['supported_languages'][:5])}{'...' if len(info['supported_languages']) > 5 else ''}"
            )
        print()

    # 2. 準備測試圖片
    print(f"2. 載入測試資料夾: {TEST_FOLDER}")
    test_results = load_existing_images(TEST_FOLDER)
    if not test_results:
        print(f"❌ 未找到任何圖片檔案在 {TEST_FOLDER}")
        return

    has_config = any(result.get("has_config", False) for result in test_results)
    print(
        f"   載入 {len(test_results)} 張圖片 ({'有設定檔案' if has_config else '無設定檔案'})"
    )
    print()

    # 3. OCR測試
    if SHOW_OCR_RESULTS:
        print("3. 進行OCR識別測試...")

    total_accuracy = 0
    total_confidence = 0
    total_time = 0
    successful_tests = 0

    for i, result in enumerate(test_results, 1):
        try:
            # 進行OCR識別
            ocr_result = ocr_manager.recognize_text(
                result["path"], engine=DEFAULT_OCR_ENGINE, languages=result["ocr_lang"]
            )

            recognized_text = ocr_result.text.strip()

            if SHOW_OCR_RESULTS:
                print(f"   {i}. {os.path.basename(result['path'])} 識別結果:")
                print(f"      檔案: {result['path']}")

                # 顯示圖片尺寸（如果能獲取到）
                try:
                    from PIL import Image

                    with Image.open(result["path"]) as img:
                        print(f"      尺寸: {img.size[0]}x{img.size[1]}")
                        print(f"      格式: {img.format}")
                except:
                    pass

                print(
                    f"      語言: {', '.join(str(lang) for lang in result['ocr_lang'])}"
                )

                if result.get("has_config", False):
                    # 有設定檔案，顯示比對結果
                    original_text = result["config"]["text"]
                    accuracy = calculate_simple_accuracy(original_text, recognized_text)
                    print(f"      原始文字: {original_text}")
                    print(f"      識別結果: {recognized_text}")
                    print(f"      準確度: {accuracy:.2f}%")
                    total_accuracy += accuracy
                else:
                    # 沒有設定檔案，顯示OCR結果
                    print(f"      識別結果: {recognized_text}")

                print(f"      信心度: {ocr_result.confidence:.1f}%")
                print(f"      處理時間: {ocr_result.processing_time:.3f}秒")

                # 顯示bounding box資訊
                if SHOW_BOUNDING_BOXES and ocr_result.bounding_boxes:
                    print(f"      文字區域 ({len(ocr_result.bounding_boxes)} 個):")
                    for i, box in enumerate(
                        ocr_result.bounding_boxes[:10]
                    ):  # 只顯示前10個
                        print(
                            f"        [{i+1}] \"{box['text']}\" -> 位置:({box['x']},{box['y']}) 尺寸:{box['w']}x{box['h']} 信心:{box['confidence']:.1f}%"
                        )
                    if len(ocr_result.bounding_boxes) > 10:
                        print(
                            f"        ...還有{len(ocr_result.bounding_boxes)-10}個文字區域"
                        )

                print()  # 添加空行分隔

            # 儲存結果到檔案
            save_result_to_file(result, ocr_result, results_dir)

            # 累計統計
            total_confidence += ocr_result.confidence
            total_time += ocr_result.processing_time
            successful_tests += 1

        except Exception as e:
            if SHOW_OCR_RESULTS:
                print(f"   {i}. {os.path.basename(result['path'])} OCR測試失敗: {e}")

    if SHOW_OCR_RESULTS and successful_tests > 0:
        print()

    # 4. 總結
    if SHOW_SUMMARY:
        print("4. 測試總結:")
        print(f"   測試圖片數量: {len(test_results)}")
        if successful_tests > 0:
            # 檢查是否有任何圖片有設定檔案
            has_any_config = any(
                result.get("has_config", False) for result in test_results
            )

            if has_any_config and total_accuracy > 0:
                avg_accuracy = total_accuracy / sum(
                    1 for result in test_results if result.get("has_config", False)
                )
                print(f"   平均準確度: {avg_accuracy:.1f}%")

            avg_confidence = total_confidence / successful_tests
            avg_time = total_time / successful_tests
            print(f"   平均信心度: {avg_confidence:.1f}%")
            print(f"   平均處理時間: {avg_time:.3f}秒")

    # 儲存總結檔案
    save_summary_to_file(
        test_results,
        total_accuracy,
        total_confidence,
        total_time,
        successful_tests,
        results_dir,
    )

    print("\n=== 演示完成 ===")


def load_existing_images(test_folder=None):
    """
    載入現有的測試圖片和設定

    Args:
        test_folder: 測試資料夾路徑，如果為None則使用預設的"_testing_images"

    Returns:
        測試圖片列表，包含路徑、設定等資訊
        如果有設定檔案，包含config資訊用於準確度比對
        如果沒有設定檔案，只有圖片路徑用於純OCR測試
    """
    import json
    import os

    test_results = []

    # 確定測試資料夾路徑
    if test_folder is None:
        test_folder = "_testing_images"

    images_dir = os.path.join(test_folder, "images")
    configs_dir = os.path.join(test_folder, "configs")

    # 如果有設定檔案的資料夾結構，優先使用設定檔案
    if os.path.exists(images_dir) and os.path.exists(configs_dir):
        print(f"   發現設定檔案，載入 {configs_dir} 中的設定...")

        # 獲取所有設定文件
        for config_file in os.listdir(configs_dir):
            if config_file.endswith(".json"):
                config_path = os.path.join(configs_dir, config_file)
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    image_path = config_data.get("image_path")
                    if image_path and os.path.exists(image_path):
                        # 根據語言確定OCR語言
                        language = config_data.get("language", "english")
                        ocr_lang_map = {
                            "english": ["eng"],
                            "chinese": [
                                "chi_tra",
                                "chi_sim",
                            ],  # 優先使用繁體中文，然後簡體中文
                            "japanese": ["jpn"],
                            "korean": ["kor"],
                        }
                        ocr_lang = ocr_lang_map.get(language, ["eng"])

                        test_results.append(
                            {
                                "path": image_path,
                                "config": config_data,
                                "ocr_lang": ocr_lang,
                                "language": language,
                                "text": config_data.get("text", ""),
                                "has_config": True,  # 標記有設定檔案
                            }
                        )

                except Exception as e:
                    print(f"   警告: 載入設定文件 {config_file} 失敗: {e}")

    else:
        # 如果沒有設定檔案的資料夾結構，掃描所有圖片檔案
        print(f"   未發現設定檔案，掃描 {test_folder} 中的所有圖片...")

        # 支援的圖片副檔名
        image_extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
            ".gif",
        ]

        # 如果 test_folder 是檔案，直接處理單個檔案
        if os.path.isfile(test_folder):
            if any(test_folder.lower().endswith(ext) for ext in image_extensions):
                test_results.append(
                    {
                        "path": test_folder,
                        "config": None,
                        "ocr_lang": [
                            "eng",
                            "chi_tra",
                            "chi_sim",
                        ],  # 多語言辨識：英文+繁體+簡體中文
                        "language": "auto",
                        "text": "",
                        "has_config": False,  # 標記沒有設定檔案
                    }
                )
        elif os.path.isdir(test_folder):
            # 掃描資料夾中的所有圖片檔案
            for root, dirs, files in os.walk(test_folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_path = os.path.join(root, file)
                        test_results.append(
                            {
                                "path": image_path,
                                "config": None,
                                "ocr_lang": [
                                    "eng",
                                    "chi_tra",
                                    "chi_sim",
                                ],  # 多語言辨識：英文+繁體+簡體中文
                                "language": "auto",
                                "text": "",
                                "has_config": False,  # 標記沒有設定檔案
                            }
                        )

        if not test_results:
            print(f"   警告: 在 {test_folder} 中找不到任何圖片檔案")
            return test_results

    return test_results


def calculate_simple_accuracy(original: str, recognized: str) -> float:
    """
    計算簡單的文字準確度
    """
    if not original:
        return 100.0 if not recognized else 0.0

    # 移除空白和轉小寫進行比較
    orig_clean = "".join(original.split()).lower()
    recog_clean = "".join(recognized.split()).lower()

    if not orig_clean:
        return 100.0

    if not recog_clean:
        return 0.0

    # 計算最長公共子序列長度
    def lcs_len(s1: str, s2: str) -> int:
        if not s1 or not s2:
            return 0
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]

    lcs_length = lcs_len(orig_clean, recog_clean)
    accuracy = (lcs_length / len(orig_clean)) * 100
    return min(accuracy, 100.0)


def save_result_to_file(result, ocr_result, results_dir):
    """
    將OCR結果儲存到JSON檔案

    Args:
        result: 測試結果字典
        ocr_result: OCRResult物件
        results_dir: 結果資料夾路徑
    """
    import json
    import os

    # 建立結果資料結構
    result_data = {
        "image_info": {
            "path": result["path"],
            "filename": os.path.basename(result["path"]),
            "has_config": result.get("has_config", False),
        },
        "ocr_result": ocr_result.to_dict(),
    }

    # 如果有設定檔案，加入準確度資訊
    if result.get("has_config", False):
        accuracy = calculate_simple_accuracy(result["config"]["text"], ocr_result.text)
        result_data["accuracy_analysis"] = {
            "original_text": result["config"]["text"],
            "accuracy": accuracy,
        }

    # 加入圖片尺寸資訊（如果能獲取到）
    try:
        from PIL import Image

        with Image.open(result["path"]) as img:
            result_data["image_info"]["size"] = {
                "width": img.size[0],
                "height": img.size[1],
                "format": img.format,
            }
    except:
        pass

    # 產生檔案名稱：result_[filename].json
    filename = os.path.splitext(os.path.basename(result["path"]))[0]
    # 清理檔案名稱中的特殊字元
    safe_filename = "".join(
        c for c in filename if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    result_filename = f"result_{safe_filename}.json"
    result_path = os.path.join(results_dir, result_filename)

    # 儲存到檔案
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


def save_summary_to_file(
    test_results,
    total_accuracy,
    total_confidence,
    total_time,
    successful_tests,
    results_dir,
):
    """
    儲存測試總結到JSON檔案

    Args:
        test_results: 測試結果列表
        total_accuracy: 總準確度
        total_confidence: 總信心度
        total_time: 總處理時間
        successful_tests: 成功測試數量
        results_dir: 結果資料夾路徑
    """
    import json

    # 檢查是否有任何圖片有設定檔案
    has_any_config = any(result.get("has_config", False) for result in test_results)

    # 建立總結資料結構
    summary_data = {
        "test_summary": {
            "total_images": len(test_results),
            "successful_tests": successful_tests,
            "failed_tests": len(test_results) - successful_tests,
            "test_folder": TEST_FOLDER,
        },
        "performance_stats": {
            "average_confidence": (
                total_confidence / successful_tests if successful_tests > 0 else 0
            ),
            "average_processing_time": (
                total_time / successful_tests if successful_tests > 0 else 0
            ),
            "total_processing_time": total_time,
        },
    }

    # 如果有設定檔案，加入準確度統計
    if has_any_config and total_accuracy > 0:
        config_count = sum(
            1 for result in test_results if result.get("has_config", False)
        )
        summary_data["accuracy_stats"] = {
            "images_with_config": config_count,
            "average_accuracy": total_accuracy / config_count,
            "total_accuracy_score": total_accuracy,
        }

    # 統計檔案類型
    image_formats = {}
    for result in test_results:
        try:
            from PIL import Image

            with Image.open(result["path"]) as img:
                fmt = img.format or "Unknown"
                image_formats[fmt] = image_formats.get(fmt, 0) + 1
        except:
            image_formats["Unknown"] = image_formats.get("Unknown", 0) + 1

    summary_data["file_stats"] = {
        "formats": image_formats,
    }

    # 產生檔案名稱：summary.json
    summary_filename = "summary.json"
    summary_path = os.path.join(results_dir, summary_filename)

    # 儲存到檔案
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    demo()
