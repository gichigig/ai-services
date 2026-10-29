import os
import io
import re
import base64
from typing import List, Dict, Any, Optional

def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 30, max_chars: int = 30000) -> str:
    """
    Extracts structured, page-by-page text and table contents from PDF bytes using PyMuPDF (fitz).
    If a page has no digital text (e.g., scanned document), applies OCR.
    """
    text_content = []
    total_chars = 0
    
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        num_pages = len(doc)
        pages_to_read = min(num_pages, max_pages)
        
        for page_num in range(pages_to_read):
            page = doc[page_num]
            page_text_blocks = []
            
            # 1. Try table extraction
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables.tables:
                        for row in table.extract():
                            row_str = " | ".join([str(cell or '').strip() for cell in row if cell is not None])
                            if row_str.strip():
                                page_text_blocks.append(row_str)
            except Exception:
                pass
            
            # 2. Standard text extraction
            raw_text = page.get_text().strip()
            if raw_text:
                page_text_blocks.append(raw_text)
            
            # 3. Fallback: If page has no digital text, attempt OCR on page pixmap
            if not page_text_blocks:
                try:
                    import easyocr
                    import numpy as np
                    import cv2
                    
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        reader = easyocr.Reader(['en'], gpu=False)
                        results = reader.readtext(img)
                        ocr_lines = [res[1] for res in results if res[1].strip()]
                        if ocr_lines:
                            page_text_blocks.append("\n".join(ocr_lines))
                except Exception as ocr_err:
                    print(f"OCR fallback error on PDF page {page_num+1}: {ocr_err}")
            
            combined_page_text = "\n".join(page_text_blocks).strip()
            if combined_page_text:
                page_header = f"--- Page {page_num + 1} of {num_pages} ---"
                text_content.append(f"{page_header}\n{combined_page_text}")
                total_chars += len(combined_page_text)
                if total_chars >= max_chars:
                    text_content.append(f"[Note: Truncated to first {max_chars} characters for optimal processing]")
                    break
                    
        if num_pages > max_pages:
            text_content.append(f"[Note: Document has {num_pages} total pages; first {max_pages} pages processed]")
            
    except Exception as e:
        print(f"Error parsing PDF with PyMuPDF: {e}")
        return f"Error reading PDF content: {str(e)}"
        
    return "\n\n".join(text_content)


def extract_text_from_docx(file_bytes: bytes, max_chars: int = 30000) -> str:
    """
    Extracts text from a DOCX file by reading word/document.xml directly via zipfile.
    Requires no external dependencies.
    """
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if 'word/document.xml' not in zf.namelist():
                return ""
            xml_content = zf.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # XML namespace for WordML
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            
            for p in tree.iterfind('.//w:p', namespaces):
                texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
                    
            full_text = "\n\n".join(paragraphs)
            if len(full_text) > max_chars:
                return full_text[:max_chars] + f"\n[Note: Truncated to {max_chars} characters]"
            return full_text
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""


def extract_text_from_image_bytes(file_bytes: bytes, ext: str = "png") -> str:
    """
    Extracts text from image bytes using PyMuPDF image text inspection and EasyOCR.
    """
    text_content = []
    
    # 1. Plain text / CSV detection in case a text file had an image extension
    try:
        raw = file_bytes.decode('utf-8')
        if len(raw.strip()) > 10 and not any(ord(c) < 9 or (13 < ord(c) < 32) for c in raw[:200]):
            return raw
    except Exception:
        pass

    # 2. PyMuPDF OCR/image text inspection
    try:
        import fitz
        clean_ext = ext.lower().replace(".", "")
        if clean_ext in ["jpg", "jpeg"]:
            clean_ext = "jpeg"
        elif clean_ext not in ["png", "webp"]:
            clean_ext = "png"
            
        img_doc = fitz.open(stream=file_bytes, filetype=clean_ext)
        for page in img_doc:
            text = page.get_text().strip()
            if text:
                text_content.append(text)
    except Exception:
        pass

    # 3. EasyOCR for optical character and handwritten text extraction
    if not text_content or len("\n".join(text_content).strip()) < 10:
        try:
            import easyocr
            import numpy as np
            import cv2
            
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                reader = easyocr.Reader(['en'], gpu=False)
                results = reader.readtext(img)
                items = []
                heights = []
                for bbox, text, prob in results:
                    text_str = str(text).strip()
                    if not text_str:
                        continue
                    y_top = min(pt[1] for pt in bbox)
                    y_bottom = max(pt[1] for pt in bbox)
                    y_center = (y_top + y_bottom) / 2.0
                    x_left = min(pt[0] for pt in bbox)
                    h = abs(y_bottom - y_top)
                    heights.append(h)
                    items.append({
                        "y_center": y_center,
                        "x_left": x_left,
                        "height": h,
                        "text": text_str
                    })
                
                if items:
                    avg_h = (sum(heights) / len(heights)) if heights else 20.0
                    row_threshold = max(14.0, avg_h * 0.65)
                    items.sort(key=lambda it: it["y_center"])
                    rows = []
                    curr_row = []
                    curr_y = None
                    for it in items:
                        if curr_y is None or abs(it["y_center"] - curr_y) <= row_threshold:
                            curr_row.append(it)
                            if curr_y is None:
                                curr_y = it["y_center"]
                            else:
                                curr_y = (curr_y + it["y_center"]) / 2.0
                        else:
                            curr_row.sort(key=lambda it: it["x_left"])
                            rows.append(" | ".join(it["text"] for it in curr_row))
                            curr_row = [it]
                            curr_y = it["y_center"]
                    if curr_row:
                        curr_row.sort(key=lambda it: it["x_left"])
                        rows.append(" | ".join(it["text"] for it in curr_row))
                    text_content.append("\n".join(rows))
        except Exception as e:
            print(f"EasyOCR extraction error: {e}")

    return "\n".join(text_content).strip()


def inspect_image_metadata(file_bytes: bytes) -> Dict[str, Any]:
    """Inspects basic dimensions and format of image bytes."""
    meta = {"format": "Unknown", "width": 0, "height": 0}
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            h, w, _ = img.shape
            meta["width"] = w
            meta["height"] = h
            meta["format"] = "Image"
    except Exception:
        pass
    return meta


async def process_chat_attachments(attachments: List[Dict[str, Any]]) -> str:
    """
    Processes a list of chat attachments (each dict has 'name', 'mime_type', 'data' [base64]).
    Extracts text, metadata, and structural content from documents and images.
    Returns a clean, structured context string ready to be injected into the LLM prompt.
    """
    if not attachments:
        return ""
        
    context_blocks = []
    
    for idx, att in enumerate(attachments):
        name = att.get("name", f"attachment_{idx+1}")
        mime_type = (att.get("mime_type") or "").lower()
        base64_data = att.get("data", "")
        
        if not base64_data:
            continue
            
        try:
            # Handle base64 string with or without data: URL prefix
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            file_bytes = base64.b64decode(base64_data)
        except Exception as b64_err:
            print(f"Failed to decode base64 for {name}: {b64_err}")
            continue
            
        ext = os.path.splitext(name)[1].lower()
        if not ext and mime_type:
            if "pdf" in mime_type:
                ext = ".pdf"
            elif "image" in mime_type:
                ext = ".png"
            elif "docx" in mime_type:
                ext = ".docx"
            elif "text" in mime_type:
                ext = ".txt"
                
        # 1. PDF Document
        if ext == ".pdf" or "pdf" in mime_type:
            pdf_text = extract_text_from_pdf(file_bytes)
            if pdf_text.strip():
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"PDF\">\n{pdf_text}\n</ATTACHED_DOCUMENT>"
                )
            else:
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"PDF\">\n(Document was uploaded but contains no extractable text or scans)\n</ATTACHED_DOCUMENT>"
                )
                
        # 2. Word (.docx) Document
        elif ext == ".docx" or "wordprocessingml" in mime_type:
            docx_text = extract_text_from_docx(file_bytes)
            if docx_text.strip():
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"DOCX\">\n{docx_text}\n</ATTACHED_DOCUMENT>"
                )
            else:
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"DOCX\">\n(Document was uploaded but contains no extractable text)\n</ATTACHED_DOCUMENT>"
                )

        # 3. Excel Spreadsheets (.xlsx, .xls)
        elif ext in [".xlsx", ".xls"] or "spreadsheet" in mime_type or "excel" in mime_type:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
                sheet_texts = []
                for sheet_name in wb.sheetnames:
                    sheet = wb[sheet_name]
                    rows_text = []
                    for row in sheet.iter_rows(values_only=True):
                        non_empty = [str(c).strip() for c in row if c is not None and str(c).strip()]
                        if non_empty:
                            rows_text.append(" | ".join(non_empty))
                    if rows_text:
                        sheet_texts.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows_text[:400]))
                full_sheet_str = "\n\n".join(sheet_texts)
                if len(full_sheet_str) > 30000:
                    full_sheet_str = full_sheet_str[:30000] + "\n[Note: Truncated to 30,000 characters]"
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"Spreadsheet\">\n{full_sheet_str}\n</ATTACHED_DOCUMENT>"
                )
            except Exception as excel_err:
                print(f"Error reading Excel file {name}: {excel_err}")

        # 4. Plain Text / Code / CSV / Markdown
        elif ext in [".txt", ".md", ".csv", ".json", ".py", ".kt", ".java", ".html", ".css", ".xml"] or "text/" in mime_type:
            try:
                text_content = file_bytes.decode('utf-8', errors='replace').strip()
                if len(text_content) > 30000:
                    text_content = text_content[:30000] + "\n[Note: Truncated to 30,000 characters]"
                context_blocks.append(
                    f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"Text\">\n{text_content}\n</ATTACHED_DOCUMENT>"
                )
            except Exception as txt_err:
                print(f"Error reading text file {name}: {txt_err}")
                
        # 4. Images (JPG, PNG, WEBP, etc.)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"] or "image/" in mime_type:
            meta = inspect_image_metadata(file_bytes)
            ocr_text = extract_text_from_image_bytes(file_bytes, ext or "png")
            
            image_details = []
            if meta.get("width") and meta.get("height"):
                image_details.append(f"Image Dimensions: {meta['width']}x{meta['height']} pixels")
                
            if ocr_text:
                image_details.append(f"Extracted Text / Equations / Labels:\n{ocr_text}")
            else:
                image_details.append("Visual Analysis: Image contains graphics, photos, or diagrams without legible printed text.")
                
            context_blocks.append(
                f"<ATTACHED_IMAGE filename=\"{name}\">\n" + "\n\n".join(image_details) + "\n</ATTACHED_IMAGE>"
            )
            
        else:
            # Fallback: Try decoding as UTF-8 text
            try:
                fallback_text = file_bytes.decode('utf-8')
                if len(fallback_text.strip()) > 5:
                    context_blocks.append(
                        f"<ATTACHED_DOCUMENT filename=\"{name}\" type=\"Unknown\">\n{fallback_text[:20000]}\n</ATTACHED_DOCUMENT>"
                    )
            except Exception:
                context_blocks.append(
                    f"<ATTACHED_FILE filename=\"{name}\">\n(Binary attachment could not be parsed as text or supported image format)\n</ATTACHED_FILE>"
                )
                
    if not context_blocks:
        return ""
        
    return "\n\n".join(context_blocks) + "\n\n"
