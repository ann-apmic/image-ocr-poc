# Image OCR POC - 圖片OCR測試系統

一個用於測試和比較不同OCR引擎效能的完整系統，包含圖片生成模塊和OCR測試模塊。

## 功能特點

### 圖片生成模塊
- ✅ 支持多語言文字生成（英文、中文、日文、韓文）
- ✅ 可調整字體大小、模糊程度、噪點等效果
- ✅ 自動記錄生成設定，方便後續分析
- ✅ 模擬真實拍攝環境的圖片效果

### OCR測試模塊
- ✅ 插件化架構，方便替換不同的OCR引擎
- ✅ 內建Tesseract OCR引擎支持
- ✅ 支援多語言辨識：英文、中文（繁體優先）、日文、韓文
- ✅ 繁體中文辨識準確度高達96%+
- ✅ 文字座標定位：獲取每個文字區域的 bounding box (x, y, width, height)
- ✅ 詳細的準確度分析和效能評估
- ✅ 批量測試和報告生成

## 安裝步驟

### 系統依賴
```bash
# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝Tesseract OCR
sudo apt install tesseract-ocr

# 安裝語言包（推薦安裝測試語言）
sudo apt install tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-jpn tesseract-ocr-kor

# 🔍 中文辨識說明：
# - chi_sim: 簡體中文語言包
# - chi_tra: 繁體中文語言包 (重要！系統優先使用繁體中文進行辨識)
# - 系統現在支援繁體中文辨識，提供更好的準確度 (可達96%+)

# 安裝中文字體（用於圖片生成中的中文字顯示）
sudo apt install fonts-noto-cjk fonts-noto-cjk-extra fonts-wqy-zenhei fonts-wqy-microhei fonts-arphic-ukai fonts-arphic-uming

# 安裝圖形處理依賴（用於 OpenCV x圖片處理，如噪點、旋轉等效果）
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### Python依賴
```bash
# 安裝Python套件
pip install -r requirements.txt
```

## 快速開始

### 生成測試圖片並運行OCR測試
```bash
# 生成10張測試圖片（英文和中文），使用Tesseract進行測試
python app/main.py --images 10 --languages english chinese

# 使用現有的測試圖片進行測試
python app/main.py --no-generate

# 查看所有可用的OCR引擎
python app/main.py --list-engines
```

#### 測試任意圖片（推薦用法）
```bash
# 測試預設的測試圖片（會進行準確度比對）
python demo.py

# 自定義測試資料夾
# 如果資料夾中有 config 檔案，將進行準確度比對；如果沒有，只顯示OCR結果
# 修改 demo.py 中的 TEST_FOLDER 參數來自定義測試資料夾：
# TEST_FOLDER = "your_custom_folder"  # 設定測試資料夾路徑
# TEST_FOLDER = "path/to/image.jpg"   # 或指定單個圖片檔案
# TEST_FOLDER = ""                    # 錯誤：不能為空

# 生成新測試圖片
python generate_images.py
```

**進階選項:**
- `--recursive, -r`: 遞歸掃描子資料夾
- `--max-files, -m`: 限制處理的最大檔案數量

**支援的語言:**
- `english` - 英文
- `chinese` - 中文（優先繁體中文辨識，支援簡體中文）
- `japanese` - 日文
- `korean` - 韓文
- `auto` - 自動多語言檢測（英文+繁體+簡體中文）

**中文辨識特點:**
- 優先使用繁體中文模型，提供更好的準確度
- 自動支援繁體和簡體中文混合內容
- 辨識結果保持原始文字形式（繁體輸出繁體，簡體輸出簡體）

**文字座標定位:**
- 每個文字區域的bounding box資訊：x, y, width, height
- 適用於文字定位、區域提取等進階應用
- 可在demo.py中設定`SHOW_BOUNDING_BOXES = True/False`來控制顯示

**支援的圖片格式:**
PNG, JPG, JPEG, BMP, TIFF, WebP, GIF等常見格式

### 進階用法

```bash
# 生成更多圖片測試日文和韓文
python app/main.py --images 20 --languages japanese korean

# 指定使用特定OCR引擎
python app/main.py --engine tesseract --no-generate

# 自定義測試參數
python app/main.py --images 50 --languages english chinese japanese korean
```

## 專案結構

```
image-ocr-poc/
├── app/
│   ├── main.py                    # 主程式入口
│   ├── test_runner.py            # 測試運行器
│   ├── image_generator/          # 圖片生成模塊
│   │   ├── __init__.py
│   │   └── image_generator.py    # 圖片生成器實作
│   └── ocr_engine/               # OCR引擎模塊
│       ├── __init__.py
│       ├── base_ocr.py           # OCR引擎抽象接口
│       ├── tesseract_ocr.py      # Tesseract實作
│       └── ocr_manager.py        # 引擎管理器
├── _testing_images/              # 測試檔案存放區
│   ├── images/                   # 生成的測試圖片
│   ├── configs/                  # 圖片設定檔案
│   ├── reports/                  # 測試報告
│   └── README.md                 # 存放區說明
├── requirements.txt              # Python依賴
└── README.md                     # 專案說明
```

## 測試結果分析

測試完成後會自動生成詳細的JSON報告，包含：

- **準確度統計**: 各語言、各效果下的識別準確度
- **效能分析**: OCR引擎的處理時間和信心度
- **文字座標**: 每個文字區域的bounding box (x, y, width, height)
- **圖片資訊**: 檔案尺寸、格式等元資料
- **總結報告**: 整體測試統計和效能分析

### 結果儲存結構

```
_results/
├── result_[timestamp]_[filename].json    # 每個圖片的詳細結果
├── summary_[timestamp].json              # 整體測試總結
└── README.md                             # 資料夾說明
```

**注意**: `_results` 資料夾會在每次執行 `demo.py` 前自動清空

## 擴展OCR引擎

系統採用插件化設計，要添加新的OCR引擎：

1. 創建新的引擎類，繼承 `BaseOCREngine`
2. 在 `ocr_manager.py` 中註冊新引擎
3. 或使用 `OCREngineManager.load_external_engine()` 動態載入

```python
from app.ocr_engine.base_ocr import BaseOCREngine
from app.ocr_engine.ocr_manager import ocr_manager

class MyCustomOCREngine(BaseOCREngine):
    # 實作必要的方法...

# 註冊引擎
ocr_manager.register_engine('my_engine', MyCustomOCREngine)
```

## 常見問題

### Q: 生成出來的中文字都變成框框無法顯示？
A: 這是因為系統缺少中文字體。請確保安裝了中文字體包：
```bash
sudo apt install fonts-noto-cjk fonts-noto-cjk-extra fonts-wqy-zenhei fonts-wqy-microhei fonts-arphic-ukai fonts-arphic-uming
```
安裝後重新運行程式即可。

### Q: Tesseract識別效果不佳？
A: 檢查是否安裝了對應語言包，並嘗試調整 `--oem` 和 `--psm` 參數。

### Q: 如何清理測試檔案？
A: 刪除 `_testing_images/images/` 和 `_testing_images/configs/` 中的檔案，報告檔案可保留用於分析。

## 開發與貢獻

歡迎提交Issue和Pull Request！

## 授權

本專案採用 Apache License 2.0 授權。
