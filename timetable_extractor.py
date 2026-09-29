import os
import json
import re
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text and table content from PDF bytes using PyMuPDF (fitz)."""
    text_content = []
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Try table extraction first if available
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables.tables:
                        for row in table.extract():
                            text_content.append(" | ".join([str(cell or '') for cell in row]))
            except Exception:
                pass
            # Standard text extraction
            text_content.append(page.get_text())
    except Exception as e:
        print(f"Error extracting text from PDF with fitz: {e}")
    return "\n".join(text_content)

def extract_text_from_image(file_bytes: bytes, ext: str = "png") -> str:
    """Extracts text from image bytes using PyMuPDF OCR or EasyOCR."""
    text_content = []
    
    # 1. Try decoding as plain text in case text/csv file was sent
    try:
        raw = file_bytes.decode('utf-8')
        if len(raw.strip()) > 10 and not any(ord(c) < 9 or (13 < ord(c) < 32) for c in raw[:200]):
            return raw
    except Exception:
        pass

    # 2. Try PyMuPDF image support
    try:
        import fitz
        img_doc = fitz.open(stream=file_bytes, filetype=ext if ext in ["png", "jpg", "jpeg", "webp"] else "png")
        for page in img_doc:
            text = page.get_text()
            if text.strip():
                text_content.append(text)
    except Exception:
        pass

    # 3. Try EasyOCR
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
                lines = [res[1] for res in results]
                text_content.append("\n".join(lines))
        except Exception as e:
            print(f"EasyOCR extraction error: {e}")

    return "\n".join(text_content)

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def parse_time_slot_str(text: str):
    """Extracts start and end times in HH:MM format from headers like '0800-1100' or '8:00 - 11:00'."""
    text = str(text or "").strip()
    # Matches '0800-1100' or '0800 - 1100'
    m2 = re.search(r"(\d{4})\s*[-–to]+\s*(\d{4})", text)
    if m2:
        s, e = m2.group(1), m2.group(2)
        return f"{s[:2]}:{s[2:]}", f"{e[:2]}:{e[2:]}"
    # Matches '8:00 - 11:00' or '08:00-11:00' or '8-11'
    m = re.search(r"(\d{1,2})[:.]?(\d{2})?\s*[-–to]+\s*(\d{1,2})[:.]?(\d{2})?", text, re.I)
    if m:
        s_hr, s_min, e_hr, e_min = m.group(1), m.group(2) or "00", m.group(3), m.group(4) or "00"
        return f"{int(s_hr):02d}:{s_min}", f"{int(e_hr):02d}:{e_min}"
    return None, None

KNOWN_COURSE_ACRONYMS = {
    "software engineering": ["BSE", "DSE"],
    "bachelor of software engineering": ["BSE"],
    "bachelor of science in software engineering": ["BSE"],
    "bsc software engineering": ["BSE"],
    "bse": ["BSE"],
    "diploma in software engineering": ["DSE"],
    "dip software engineering": ["DSE"],
    "dse": ["DSE"],
    "computer science": ["DCS", "BSCIT", "BSE"],
    "bachelor of science in information technology": ["BSCIT"],
    "information technology": ["BSCIT", "BBIT", "DIT", "DBIT", "CIT", "CICT"],
    "bachelor of business information technology": ["BBIT"],
    "diploma in information technology": ["DIT"],
    "diploma in business information technology": ["DBIT"],
    "business information technology": ["BBIT", "DBIT"],
    "bbit": ["BBIT"],
    "dbit": ["DBIT"],
    "dit": ["DIT"],
    "bscit": ["BSCIT"],
    "commerce": ["BCOM", "CBM", "DBMA"],
    "bachelor of commerce": ["BCOM"],
    "bcom": ["BCOM"],
    "business administration": ["BBAM", "DBMA", "CBM"],
    "bachelor of business administration": ["BBAM"],
    "bbam": ["BBAM"],
    "accounting and finance": ["BAF", "DAF", "DAC"],
    "accounting": ["BAF", "DAF", "DAC"],
    "baf": ["BAF"],
    "daf": ["DAF"],
    "dac": ["DAC"],
    "procurement": ["BPSM", "DPSM", "CPSM"],
    "purchasing and supplies": ["BPSM", "DPSM", "CPSM"],
    "supply chain": ["BPSM", "DPSM", "CPSM"],
    "bpsm": ["BPSM"],
    "dpsm": ["DPSM"],
    "cpsm": ["CPSM"],
    "journalism": ["BMDC", "BAJ", "DCMS", "CJMS"],
    "mass communication": ["BMDC", "BAJ", "DCMS", "CJMS"],
    "media": ["BMDC", "BAJ"],
    "hospitality": ["BHTM", "DHM", "CHTM"],
    "tourism": ["BHTM", "DTM", "CHTM"],
    "international relations": ["BIRD", "DIRD"],
    "diplomacy": ["BIRD", "DIRD"],
    "electrical engineering": ["DEE", "CEE"],
    "fashion": ["DFT"],
    "human resource": ["DHR"],
    "icdl": ["ICDL"]
}

def match_cohort_blocks(blocks: list, course: str = "", year: str = "", semester: str = "", cohort_hint: str = ""):
    """
    Intelligently matches user's course/year/semester to cohort sections in timetable.
    Returns (matched_blocks, all_cohorts, matched_cohort_name)
    """
    if not blocks:
        return [], [], None
        
    all_cohorts = sorted(list({b.get("cohort", "").strip() for b in blocks if b.get("cohort", "").strip()}))
    
    # 1. Exact hint match (e.g. user clicked "BSE Y1S1")
    if cohort_hint and cohort_hint.strip() in all_cohorts:
        target = cohort_hint.strip()
        matched = [b for b in blocks if b.get("cohort", "").strip() == target]
        return matched, all_cohorts, target

    combined_text = f"{course} {year} {semester} {cohort_hint}".strip().lower()
    if not combined_text:
        return blocks, all_cohorts, None
        
    y_match = re.search(r"(?:year|y|yr|level|mod|module)\s*(\d)", combined_text) or re.search(r"\b([1-4])\b", year or "")
    y_num = y_match.group(1) if y_match else ""
    
    s_match = re.search(r"(?:semester|sem|s|term|t)\s*(\d)", combined_text) or re.search(r"\b([1-4])\b", semester or "")
    s_num = s_match.group(1) if s_match else ""
    
    level = ""
    if any(k in combined_text for k in ["bachelor", "degree", "undergraduate", "bsc", "b.sc", "ba "]):
        level = "B"
    elif any(k in combined_text for k in ["diploma", "dip", "dip."]):
        level = "D"
    elif any(k in combined_text for k in ["certificate", "cert", "cert."]):
        level = "C"
        
    candidates = []
    for phrase, acronyms in KNOWN_COURSE_ACRONYMS.items():
        if phrase in combined_text or phrase.replace(" ", "") in combined_text.replace(" ", ""):
            for acr in acronyms:
                if level and acr[0] != level:
                    continue
                candidates.append(acr)
                
    if not candidates:
        words = re.findall(r"[a-zA-Z]+", str(course).upper())
        for w in words:
            if len(w) in [3, 4, 5]:
                candidates.append(w)
                
    candidates = list(dict.fromkeys(candidates))
    
    best_cohort = None
    best_score = 0
    
    for c in all_cohorts:
        c_upper = c.upper()
        score = 0
        
        # Course acronym match
        for cand in candidates:
            if cand in c_upper:
                score += 10
                if c_upper.startswith(cand):
                    score += 5
                break
                
        # Year match
        if y_num:
            if f"Y{y_num}" in c_upper or f"YEAR {y_num}" in c_upper or f"MOD {y_num}" in c_upper:
                score += 5
            elif f"SEM {y_num}" in c_upper and not s_num:
                score += 3
                
        # Semester match
        if s_num:
            if f"S{s_num}" in c_upper or f"SEM {s_num}" in c_upper or f"SEM{s_num}" in c_upper or f"TERM {s_num}" in c_upper:
                score += 5
                
        if score > best_score:
            best_score = score
            best_cohort = c
            
    if best_cohort and best_score >= 15:
        matched = [b for b in blocks if b.get("cohort", "").strip() == best_cohort]
        return matched, all_cohorts, best_cohort
        
    if best_cohort and best_score >= 10:
        matched = [b for b in blocks if b.get("cohort", "").strip().startswith(best_cohort.split()[0])]
        return matched, all_cohorts, best_cohort

    return blocks, all_cohorts, None

def parse_excel_grid(file_bytes: bytes) -> list:
    """
    Directly and deterministically parses Excel timetable spreadsheets:
    - Identifies Time Headers (e.g. 0800-1100, 1100-1400, 1400-1700)
    - Identifies Day Rows (Monday through Sunday)
    - Identifies Cohort Sections (e.g. BSE Y1S1, BSE Y2S1)
    - Maps column positions to exact lecture times and room venues without truncation.
    """
    blocks = []
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            current_cohort = sheet_name
            current_col_times = {}
            
            for row in sheet.iter_rows(values_only=True):
                non_empty = [c for c in row if c is not None and str(c).strip()]
                if not non_empty:
                    continue

                # 1. Check if row is a Day row
                day_match = None
                for cell in row[:3]:
                    if cell:
                        cell_clean = str(cell).strip().lower()
                        for d in DAYS_OF_WEEK:
                            if d.lower() == cell_clean or cell_clean.startswith(d.lower()[:3]):
                                day_match = d
                                break
                    if day_match:
                        break
                        
                if day_match and current_col_times:
                    for col_idx, (st, et) in current_col_times.items():
                        if col_idx < len(row):
                            cell_val = row[col_idx]
                            if cell_val and str(cell_val).strip():
                                val_str = str(cell_val).strip()
                                if val_str.lower() not in [d.lower() for d in DAYS_OF_WEEK] and not parse_time_slot_str(val_str)[0]:
                                    blocks.append({
                                        "title": val_str,
                                        "day_of_week": day_match,
                                        "start_time": st,
                                        "end_time": et,
                                        "type": "CLASS",
                                        "cohort": str(current_cohort).strip()
                                    })
                    continue

                # 2. Check if row is a Time Header row
                time_cols = {}
                for col_idx, cell in enumerate(row):
                    if cell:
                        st, et = parse_time_slot_str(cell)
                        if st and et:
                            time_cols[col_idx] = (st, et)
                if len(time_cols) >= 2:
                    current_col_times = time_cols
                    continue

                # 3. Otherwise, if not a day row and not a time row, it is a Cohort / Section Header!
                if len(non_empty) <= 4:
                    hdr_text = " ".join(str(c).strip() for c in non_empty)
                    if not any(d.lower() in hdr_text.lower() for d in DAYS_OF_WEEK) and not parse_time_slot_str(hdr_text)[0]:
                        current_cohort = hdr_text
    except Exception as e:
        print(f"Error in parse_excel_grid: {e}")
        
    return blocks

def clean_and_parse_json(raw_response: str) -> list:
    """Cleans markdown wrappers or explanations and parses JSON list of blocks."""
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()

    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid_blocks = []
            for item in parsed:
                if isinstance(item, dict) and "title" in item and "day_of_week" in item:
                    day = str(item.get("day_of_week", "Monday")).capitalize()
                    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    matched_day = next((d for d in valid_days if d.lower() in day.lower()), "Monday")
                    
                    b_type = str(item.get("type", "CLASS")).upper()
                    if b_type not in ["CLASS", "STUDY"]:
                        b_type = "CLASS"
                    
                    valid_blocks.append({
                        "title": str(item.get("title", "Lecture Session")).strip(),
                        "day_of_week": matched_day,
                        "start_time": str(item.get("start_time", "08:00")).strip(),
                        "end_time": str(item.get("end_time", "10:00")).strip(),
                        "type": b_type,
                        "cohort": str(item.get("cohort", "")).strip()
                    })
            return valid_blocks
    except Exception as e:
        print(f"Failed to parse JSON timetable blocks: {e}\nRaw output was: {raw_response[:300]}")
    
    return []

async def extract_timetable_from_bytes(
    file_bytes: bytes, 
    filename: str = "document.pdf", 
    course: str = "",
    year: str = "",
    semester: str = "",
    cohort_hint: str = ""
) -> dict:
    """
    Main extraction pipeline:
    1. For Excel files (.xlsx, .xls, .csv): Deterministically parses grid tables (Time columns x Day rows).
    2. Runs smart cohort matcher to filter for the student's exact cohort.
    3. Returns dict containing matched blocks, all_cohorts list, and matched_cohort name.
    """
    ext = filename.split(".")[-1].lower() if "." in filename else "pdf"
    
    # 1. Native Excel 2D Grid Parser
    if ext in ["xlsx", "xls", "csv"]:
        all_blocks = parse_excel_grid(file_bytes)
        if all_blocks:
            matched_blocks, all_cohorts, matched_cohort = match_cohort_blocks(
                all_blocks, course=course, year=year, semester=semester, cohort_hint=cohort_hint
            )
            print(f"Excel Extraction: {len(all_blocks)} total blocks, matched '{matched_cohort}' ({len(matched_blocks)} blocks), {len(all_cohorts)} cohorts found.")
            return {
                "blocks": matched_blocks,
                "all_cohorts": all_cohorts,
                "matched_cohort": matched_cohort,
                "all_blocks": all_blocks
            }

    if ext in ["pdf"]:
        raw_text = extract_text_from_pdf(file_bytes)
    elif ext in ["txt"]:
        try:
            raw_text = file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            raw_text = extract_text_from_image(file_bytes, ext)
    else:
        raw_text = extract_text_from_image(file_bytes, ext)

    if not raw_text.strip():
        print("No text could be extracted from the uploaded document.")
        return {"blocks": [], "all_cohorts": [], "matched_cohort": None, "all_blocks": []}

    system_prompt = (
        "You are an expert academic timetable extraction engine. "
        "Your task is to parse raw document/timetable text and output ONLY a valid JSON array of lecture or study blocks. "
        "Do not include conversational greetings, explanations, markdown formatting, or any text other than the JSON array."
    )

    user_prompt = f"""
Extract all distinct class, lesson, lecture, laboratory, tutorial, and study blocks from the following timetable document.

DOCUMENT CONTENT:
\"\"\"
{raw_text[:4000]}
\"\"\"

Extraction Rules:
1. Extract every distinct class/subject/unit/lesson or study block.
2. 'day_of_week' MUST be one of: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday".
3. 'start_time' and 'end_time' MUST be in 24-hour "HH:MM" format (e.g. "08:00", "09:30", "14:00", "16:30"). If end time is not stated, estimate 1 or 2 hours after start time.
4. 'title' should include the course/unit code, subject name, and room/venue if available (e.g. "CSC 312: Operating Systems (Lab 3)").
5. Return ONLY a valid JSON array of objects with keys: "title", "day_of_week", "start_time", "end_time".

Format example:
[
  {{"title": "CSC 101: Intro to Programming (Lab 2)", "day_of_week": "Monday", "start_time": "08:00", "end_time": "10:00"}},
  {{"title": "MAT 201: Calculus (Room 4B)", "day_of_week": "Wednesday", "start_time": "11:00", "end_time": "13:00"}}
]
"""

    import modal
    blocks = []
    
    # 1. Try Small Model directly
    try:
        SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
        small_model = SmallModelCls()
        response = await small_model.generate.remote.aio(system_prompt, user_prompt)
        blocks = clean_and_parse_json(response)
    except Exception as e:
        print(f"Small model extraction error: {e}")

    # 2. Fallback to Large Model if needed
    if not blocks:
        try:
            print("Retrying timetable extraction with QwenLargeModel...")
            LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
            large_model = LargeModelCls()
            response = await large_model.generate.remote.aio(system_prompt, user_prompt)
            blocks = clean_and_parse_json(response)
        except Exception as e:
            print(f"Large model extraction error: {e}")

    matched_blocks, all_cohorts, matched_cohort = match_cohort_blocks(
        blocks, course=course, year=year, semester=semester, cohort_hint=cohort_hint
    )
    return {
        "blocks": matched_blocks,
        "all_cohorts": all_cohorts,
        "matched_cohort": matched_cohort,
        "all_blocks": blocks
    }

async def generate_study_timetable_ai(class_blocks: list, course: str = "", study_mode: str = "FULL_TIME") -> list:
    """
    Uses Qwen to generate an optimized weekly study & revision timetable
    fitting around the student's scheduled class lecture blocks.
    """
    import modal
    
    classes_summary = "\n".join([
        f"- {b.get('day_of_week')}: {b.get('start_time')}-{b.get('end_time')} : {b.get('title')}"
        for b in class_blocks
    ])
    
    system_prompt = (
        "You are an expert academic tutor and study planner for university students. "
        "Your task is to generate a realistic, balanced, high-yield weekly self-study timetable. "
        "Output ONLY a valid JSON array of study session blocks. "
        "Do not include any other text, greetings, markdown codeblocks or explanations."
    )
    
    mode_text = "Part-time student (prefer evenings 18:00-21:00 and weekends)" if "PART" in str(study_mode).upper() else (
        "Online/Distance student (flexible schedule throughout the week)" if "ONLINE" in str(study_mode).upper() else
        "Full-time student (prefer weekday late afternoons/evenings 17:30-20:30 and weekend mornings)"
    )
    
    user_prompt = f"""
Student Course: {course or 'University Degree'}
Study Mode: {mode_text}

STUDENT'S EXISTING CLASS / LECTURE SCHEDULE (DO NOT SCHEDULE STUDY SESSIONS DURING THESE TIMES):
\"\"\"
{classes_summary if classes_summary else 'No fixed classes provided (create full balanced study schedule across core subjects)'}
\"\"\"

Instructions:
1. Create 6 to 10 study/revision sessions across the week (Monday through Sunday).
2. Schedule sessions during free hours (e.g. 17:30-19:30, 19:00-21:00, or weekend mornings 09:00-12:00) without overlapping any lecture above.
3. Each block 'title' must specify the subject focus and goal (e.g. "Review Structured Programming: Loops & Functions", "Math 111: Calculus Practice Problems", "Digital Literacy: Lab Assignment Prep").
4. 'day_of_week' must be: "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", or "Sunday".
5. 'start_time' and 'end_time' in 24-hour "HH:MM" format.
6. 'type' MUST be "STUDY".
7. Return ONLY a valid JSON array of objects with keys: "title", "day_of_week", "start_time", "end_time", "type".
"""

    study_blocks = []
    try:
        SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
        small_model = SmallModelCls()
        resp = await small_model.generate.remote.aio(system_prompt, user_prompt)
        study_blocks = clean_and_parse_json(resp)
    except Exception as e:
        print(f"Small model study generation error: {e}")
        
    if not study_blocks:
        try:
            LargeModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenLargeModel")
            large_model = LargeModelCls()
            resp = await large_model.generate.remote.aio(system_prompt, user_prompt)
            study_blocks = clean_and_parse_json(resp)
        except Exception as e:
            print(f"Large model study generation error: {e}")
            
    for b in study_blocks:
        b["type"] = "STUDY"
        
    return study_blocks

