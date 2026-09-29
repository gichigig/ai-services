import os
import tempfile
import requests
from datetime import datetime, timedelta

def get_next_weekday(weekday_name: str) -> datetime:
    """Returns the datetime of the upcoming specified weekday."""
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    try:
        target_day = days.index(weekday_name.lower())
    except ValueError:
        target_day = 0
        
    now = datetime.now()
    days_ahead = target_day - now.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return now + timedelta(days=days_ahead)

def generate_pdf_timetable(schedule: list) -> str:
    """Generates a PDF and returns the temporary file path."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
        
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    c = canvas.Canvas(tmp.name, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Your Study Timetable")
    
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    for item in schedule:
        title = item.get("title", "Study Session")
        day = item.get("day_of_week", "Any Day")
        start = item.get("start_time", "00:00")
        end = item.get("end_time", "00:00")
        desc = item.get("description", "")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, f"{day} ({start} - {end}): {title}")
        y_position -= 20
        
        c.setFont("Helvetica", 12)
        c.drawString(70, y_position, desc)
        y_position -= 30
        
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            
    c.save()
    return tmp.name

def generate_ics_timetable(schedule: list) -> str:
    """Generates an ICS file and returns the temporary file path."""
    try:
        from ics import Calendar, Event
    except ImportError:
        return None
        
    c = Calendar()
    
    for item in schedule:
        title = item.get("title", "Study Session")
        day = item.get("day_of_week", "Monday")
        start = item.get("start_time", "00:00")
        end = item.get("end_time", "00:00")
        desc = item.get("description", "")
        
        next_day = get_next_weekday(day)
        date_str = next_day.strftime("%Y-%m-%d")
        
        try:
            e = Event()
            e.name = title
            e.begin = f"{date_str} {start}:00"
            e.end = f"{date_str} {end}:00"
            e.description = desc
            c.events.add(e)
        except Exception as e:
            print(f"Error adding event to ICS: {e}")
            continue
        
    tmp = tempfile.NamedTemporaryFile(suffix=".ics", delete=False)
    with open(tmp.name, 'w') as f:
        f.writelines(c.serialize_iter())
        
    return tmp.name

def upload_file_to_backend(file_path: str, backend_url: str) -> str:
    """Uploads the file to the Kotlin backend and returns the URL."""
    upload_url = f"{backend_url}/api/v1/upload"
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f)}
            res = requests.post(upload_url, files=files)
            
        if res.status_code == 200:
            data = res.json()
            return f"{backend_url}{data.get('url')}"
    except Exception as e:
        print(f"Failed to upload {file_path} to backend:", e)
        
    return None

def process_and_upload_timetable(schedule: list, backend_url: str):
    """Generates both files, uploads them, and returns their URLs."""
    pdf_path = generate_pdf_timetable(schedule)
    ics_path = generate_ics_timetable(schedule)
    
    pdf_url = None
    ics_url = None
    
    if pdf_path:
        pdf_url = upload_file_to_backend(pdf_path, backend_url)
        os.remove(pdf_path)
        
    if ics_path:
        ics_url = upload_file_to_backend(ics_path, backend_url)
        os.remove(ics_path)
        
    return {
        "pdf_url": pdf_url,
        "ics_url": ics_url
    }
