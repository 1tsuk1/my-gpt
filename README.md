```
#!/usr/bin/env python3
"""
PPTXファイルの各スライドから見出しを抽出し、個別のPDFファイルに変換するメインスクリプト
"""

import sys
from pathlib import Path


from pptx_processor import PPTXProcessor

def main():
    pptx_path = "sample.pptx"
    
    try:
        processor = PPTXProcessor(pptx_path)
        output_files = processor.process()
        
        if output_files:
            print(f"\n変換完了! {len(output_files)}個のPDFファイルが作成されました:")
            for file_path in output_files:
                print(f"  - {Path(file_path).name}")
        else:
            print("変換に失敗しました")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    ```


```
#!/usr/bin/env python3
"""
PDF関連のユーティリティ関数
"""

import re
import io
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from PIL import Image


class PDFUtils:
    """PDF操作のユーティリティクラス"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """ファイル名として使用できない文字を除去・変換"""
        # 使用できない文字を除去
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # 複数の空白を1つに
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # 先頭末尾の空白を除去
        sanitized = sanitized.strip()
        # 長すぎる場合は切り詰め
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        # 空の場合はデフォルト名
        if not sanitized:
            sanitized = "untitled"
        
        return sanitized
    
    @staticmethod
    def create_pdf_from_images(images: List[Image.Image], output_path: str) -> None:
        """画像リストからPDFを作成"""
        doc = fitz.open()
        
        for img in images:
            # 画像をバイトストリームに変換
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # PDFページを作成
            page = doc.new_page(width=img.width, height=img.height)
            page.insert_image(page.rect, stream=img_bytes.getvalue())
        
        # PDFを保存
        doc.save(output_path)
        doc.close()
    
    @staticmethod
    def split_pdf_by_pages(pdf_path: str, slide_titles: List[Tuple[int, str]], output_dir: Path) -> List[str]:
        """PDFを各ページごとに分割"""
        pdf_document = fitz.open(pdf_path)
        output_files = []
        
        for slide_index, title in slide_titles:
            if slide_index < len(pdf_document):
                # 新しいPDFドキュメントを作成
                new_pdf = fitz.open()
                
                # 該当ページをコピー
                new_pdf.insert_pdf(pdf_document, from_page=slide_index, to_page=slide_index)
                
                # ファイル名を生成
                safe_title = PDFUtils.sanitize_filename(title)
                output_filename = f"{slide_index+1:02d}_{safe_title}.pdf"
                output_path = output_dir / output_filename
                
                # PDFを保存
                new_pdf.save(str(output_path))
                new_pdf.close()
                
                output_files.append(str(output_path))
                print(f"作成: {output_filename}")
        
        pdf_document.close()
        return output_files```

```
#!/usr/bin/env python3
"""
PPTXファイル処理のメインクラス
"""

import io
from pathlib import Path
from typing import List, Tuple
from pptx import Presentation
from slide_converter import SlideToImageConverter
from pdf_utils import PDFUtils


class PPTXProcessor:
    """PPTXファイルを処理するクラス"""
    
    def __init__(self, pptx_path: str, output_dir: str = "output"):
        self.pptx_path = Path(pptx_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.slide_converter = SlideToImageConverter()
        
        if not self.pptx_path.exists():
            raise FileNotFoundError(f"PPTXファイルが見つかりません: {pptx_path}")
    
    def extract_slide_titles(self) -> List[Tuple[int, str]]:
        """各スライドから見出し（タイトル）を抽出"""
        presentation = Presentation(self.pptx_path)
        slide_titles = []
        
        for i, slide in enumerate(presentation.slides):
            title = self.slide_converter.extract_title_from_slide(slide)
            if not title:
                title = f"スライド{i+1}"
            slide_titles.append((i, title))
            
        return slide_titles
    
    def convert_pptx_to_pdf(self) -> str:
        """PPTXファイルを各スライドの画像からPDFに変換"""
        presentation = Presentation(self.pptx_path)
        temp_pdf_path = self.output_dir / "temp_full.pdf"
        
        # 各スライドを画像に変換
        images = []
        for i, slide in enumerate(presentation.slides):
            img = self.slide_converter.convert_slide_to_image(slide, i)
            images.append(img)
        
        # 画像からPDFを作成
        PDFUtils.create_pdf_from_images(images, str(temp_pdf_path))
        
        return str(temp_pdf_path)
    
    def process(self) -> List[str]:
        """メイン処理"""
        print(f"PPTXファイルを処理中: {self.pptx_path}")
        
        # スライドのタイトルを抽出
        print("スライドのタイトルを抽出中...")
        slide_titles = self.extract_slide_titles()
        
        if not slide_titles:
            print("スライドが見つかりませんでした")
            return []
        
        print(f"{len(slide_titles)}枚のスライドが見つかりました")
        
        # PPTXをPDFに変換
        print("PPTXをPDFに変換中...")
        pdf_path = self.convert_pptx_to_pdf()
        
        # PDFを各スライドごとに分割
        print("PDFを各スライドごとに分割中...")
        output_files = PDFUtils.split_pdf_by_pages(pdf_path, slide_titles, self.output_dir)
        
        # 一時ファイルを削除
        temp_pdf = Path(pdf_path)
        if temp_pdf.exists() and temp_pdf.name.startswith("temp_"):
            temp_pdf.unlink()
        
        return output_files```

```
#!/usr/bin/env python3
"""
スライドを画像に変換するモジュール
"""

from typing import Optional
from PIL import Image, ImageDraw, ImageFont


class SlideToImageConverter:
    """スライドを画像に変換するクラス"""
    
    def __init__(self, width: int = 960, height: int = 540):
        self.width = width
        self.height = height
        self._setup_fonts()
    
    def _setup_fonts(self):
        """フォントを設定"""
        try:
            # システムフォントを試す（日本語対応）
            self.font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 48)
            self.small_font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", 24)
        except:
            try:
                self.font = ImageFont.truetype("arial.ttf", 48)
                self.small_font = ImageFont.truetype("arial.ttf", 24)
            except:
                self.font = ImageFont.load_default()
                self.small_font = ImageFont.load_default()
    
    def convert_slide_to_image(self, slide, slide_index: int) -> Image.Image:
        """スライドを画像に変換"""
        # 白い背景の画像を作成
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img)
        
        y_position = 100  # 上からの位置
        
        # スライドの各図形からテキストを抽出して描画
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text.strip():
                text = shape.text.strip()
                lines = text.split('\n')
                
                for line in lines:
                    if line.strip():
                        # テキストの幅を計算
                        bbox = draw.textbbox((0, 0), line, font=self.font)
                        text_width = bbox[2] - bbox[0]
                        
                        # 中央揃え
                        x_position = (self.width - text_width) // 2
                        
                        # テキストを描画
                        draw.text((x_position, y_position), line, fill='black', font=self.font)
                        y_position += 80  # 次の行の位置
                        
                        if y_position > self.height - 100:  # 画面下部に近づいたら停止
                            break
                
                if y_position > self.height - 100:
                    break
        
        return img
    
    def extract_title_from_slide(self, slide) -> Optional[str]:
        """スライドからタイトルを抽出"""
        # タイトルプレースホルダーがある場合
        if slide.shapes.title:
            return slide.shapes.title.text.strip()
        
        # 最初のテキストボックスを見出しとして使用
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                # 改行で分割して最初の行を取得
                first_line = shape.text.strip().split('\n')[0]
                if first_line:
                    return first_line
        
        return None```

        ```
        python-pptx>=0.6.21
PyMuPDF>=1.23.0
Pillow>=9.0.0```
