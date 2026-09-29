"""
Mock University LMS Portal (e-Campus / Moodle / Canvas Simulator)
Provides a fully functional web portal for testing autonomous browser navigation, unit scanning, and assignment uploads.
"""

from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
import os
import time
import hashlib

router = APIRouter(prefix="/mock-portal", tags=["Mock University Portal"])

# In-memory mock database of units and assignments
MOCK_DATA = {
    "units": [
        {
            "code": "BBIT 204",
            "name": "Distributed Systems & Cloud Architecture",
            "lecturer": "Dr. Angela Wanjiru",
            "credits": 4,
            "assignments": [
                {
                    "id": "asg-1",
                    "title": "Assignment 2: Cloud Architectures & Distributed Consensus",
                    "due_date": "August 25, 2026, 11:59 PM",
                    "max_grade": "100 Marks",
                    "status": "Pending Submission",
                    "description": "Evaluate CAP theorem constraints and implement a resilient quorum coordinator with Raft consensus. Submit solution as a single formatted PDF.",
                    "submitted_file": None,
                    "submitted_at": None,
                },
                {
                    "id": "cat-1",
                    "title": "Continuous Assessment Test 1 (CAT 1)",
                    "due_date": "August 30, 2026, 05:00 PM",
                    "max_grade": "30 Marks",
                    "status": "Pending Submission",
                    "description": "Comprehensive timed assessment covering distributed RPC, RPC serialization protocols, and vector clocks.",
                    "submitted_file": None,
                    "submitted_at": None,
                }
            ]
        },
        {
            "code": "ICS 3101",
            "name": "Advanced Database Systems & Query Optimization",
            "lecturer": "Prof. David Mutua",
            "credits": 3,
            "assignments": [
                {
                    "id": "asg-2",
                    "title": "Assignment 1: Indexing & B-Tree Cost Estimation",
                    "due_date": "August 28, 2026, 11:59 PM",
                    "max_grade": "50 Marks",
                    "status": "Pending Submission",
                    "description": "Analyze query execution plans and calculate I/O block costs for clustered vs unclustered B+ tree indexes.",
                    "submitted_file": None,
                    "submitted_at": None,
                }
            ]
        },
        {
            "code": "CIT 102",
            "name": "Data Structures & Algorithm Design",
            "lecturer": "Dr. Kelvin Kiprono",
            "credits": 4,
            "assignments": [
                {
                    "id": "asg-3",
                    "title": "Project: Graph Algorithms & Minimum Spanning Trees",
                    "due_date": "September 05, 2026, 11:59 PM",
                    "max_grade": "100 Marks",
                    "status": "Pending Submission",
                    "description": "Implement Prim's and Kruskal's algorithms with disjoint set union-find data structure.",
                    "submitted_file": None,
                    "submitted_at": None,
                }
            ]
        }
    ]
}

PORTAL_BASE_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --bg-page: #f8fafc;
        --card-bg: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
        --success: #10b981;
        --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: var(--bg-page); color: var(--text-main); line-height: 1.5; }
    .portal-nav { background: #0f172a; color: white; padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .portal-nav .brand { font-size: 1.15rem; font-weight: 700; display: flex; align-items: center; gap: 8px; color: #38bdf8; }
    .portal-nav .user-badge { font-size: 0.875rem; background: rgba(255,255,255,0.1); padding: 6px 12px; border-radius: 9999px; }
    .container { max-width: 1080px; margin: 32px auto; padding: 0 20px; }
    .card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 24px; }
    .card-title { font-size: 1.25rem; font-weight: 700; color: var(--text-main); margin-bottom: 12px; }
    .btn { display: inline-flex; align-items: center; justify-content: center; padding: 10px 20px; font-size: 0.9rem; font-weight: 600; border-radius: 8px; border: none; cursor: pointer; text-decoration: none; transition: all 0.2s ease; }
    .btn-primary { background: var(--primary); color: white; }
    .btn-primary:hover { background: var(--primary-dark); }
    .btn-success { background: var(--success); color: white; }
    .badge { display: inline-block; padding: 4px 10px; font-size: 0.75rem; font-weight: 700; border-radius: 9999px; }
    .badge-pending { background: #fef3c7; color: #b45309; }
    .badge-submitted { background: #d1fae5; color: #065f46; }
    .dropzone { border: 2px dashed #93c5fd; background: #eff6ff; border-radius: 12px; padding: 36px 20px; text-align: center; cursor: pointer; margin: 20px 0; transition: border 0.2s ease; }
    .dropzone:hover { border-color: var(--primary); background: #dbeafe; }
    .unit-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
    .unit-card { border: 1px solid var(--border-color); border-radius: 10px; padding: 20px; background: white; transition: transform 0.2s, box-shadow 0.2s; }
    .unit-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); border-color: #93c5fd; }
</style>
"""

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>e-Campus Student Portal — Login</title>
        {PORTAL_BASE_STYLE}
    </head>
    <body style="display:flex; flex-direction:column; min-height:100vh;">
        <nav class="portal-nav">
            <div class="brand"> University e-Campus LMS</div>
            <div class="user-badge">Sandbox Environment</div>
        </nav>
        <div style="flex:1; display:flex; align-items:center; justify-content:center; padding: 20px;">
            <div class="card" style="width: 100%; max-width: 440px; padding: 36px;">
                <h2 style="font-size: 1.5rem; font-weight: 800; text-align: center; margin-bottom: 8px;">Student Portal Sign In</h2>
                <p style="color: var(--text-muted); font-size: 0.875rem; text-align: center; margin-bottom: 24px;">Enter student credentials to access course assignments</p>
                <form id="login-form" action="/mock-portal/login-action" method="POST">
                    <div style="margin-bottom: 16px;">
                        <label style="display:block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">Admission / Registration No.</label>
                        <input id="username" name="username" type="text" value="ADM/2026/0894" required style="width:100%; padding:10px 14px; border:1px solid var(--border-color); border-radius:8px; font-size:0.95rem;">
                    </div>
                    <div style="margin-bottom: 20px;">
                        <label style="display:block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px;">Portal Password</label>
                        <input id="password" name="password" type="password" value="StudentSecurePass123" required style="width:100%; padding:10px 14px; border:1px solid var(--border-color); border-radius:8px; font-size:0.95rem;">
                    </div>
                    <button id="login-btn" type="submit" class="btn btn-primary" style="width: 100%; padding: 12px; font-size: 1rem;">
                        Sign In to e-Campus
                    </button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """

@router.post("/login-action")
async def login_action(username: str = Form("ADM/2026/0894"), password: str = Form("")):
    return RedirectResponse(url="/mock-portal/dashboard", status_code=303)

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    units_html = ""
    for u in MOCK_DATA["units"]:
        pending_count = sum(1 for a in u["assignments"] if a["status"] == "Pending Submission")
        units_html += f"""
        <div class="unit-card" id="unit-card-{u['code'].replace(' ', '_')}">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                <span style="font-size:0.8rem; font-weight:700; color:var(--primary); background:#dbeafe; padding:2px 8px; border-radius:4px;">{u['code']}</span>
                <span class="badge { 'badge-pending' if pending_count > 0 else 'badge-submitted' }">{pending_count} Pending Tasks</span>
            </div>
            <h3 style="font-size:1.1rem; font-weight:700; margin-bottom:6px;">{u['name']}</h3>
            <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:16px;">Lecturer: {u['lecturer']}</p>
            <a href="/mock-portal/unit/{u['code'].replace(' ', '_')}" class="btn btn-primary" id="btn-open-{u['code'].replace(' ', '_')}" style="width:100%; font-size:0.85rem;">
                Open Unit & Assignments →
            </a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>e-Campus Student Dashboard</title>
        {PORTAL_BASE_STYLE}
    </head>
    <body>
        <nav class="portal-nav">
            <div class="brand"> University e-Campus LMS</div>
            <div class="user-badge" id="student-badge"> Alex Johnson (ADM/2026/0894)</div>
        </nav>
        <div class="container">
            <div class="card" style="background: linear-gradient(135deg, #1e3a8a, #0284c7); color: white; border:none;">
                <h1 style="font-size: 1.6rem; font-weight: 800; margin-bottom: 6px;">Active Enrolled Semester Units</h1>
                <p style="opacity: 0.9; font-size: 0.95rem;">Select a unit below to review course material, CATs, and submit coursework assignments.</p>
            </div>
            
            <h2 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;">Registered Course Units (Semester 2 - 2026)</h2>
            <div class="unit-grid">
                {units_html}
            </div>
        </div>
    </body>
    </html>
    """

@router.get("/unit/{unit_code}", response_class=HTMLResponse)
async def unit_page(unit_code: str):
    real_code = unit_code.replace('_', ' ')
    unit = next((u for u in MOCK_DATA["units"] if u["code"].lower() == real_code.lower()), None)
    if not unit:
        return HTMLResponse("Unit not found", status_code=404)

    asg_html = ""
    for a in unit["assignments"]:
        asg_html += f"""
        <div class="card" id="assignment-{a['id']}" style="margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="font-size:1.15rem; font-weight:700;">{a['title']}</h3>
                <span class="badge { 'badge-submitted' if a['status'] == 'Submitted for grading' else 'badge-pending' }" id="status-{a['id']}">
                    {a['status']}
                </span>
            </div>
            <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:12px;">{a['description']}</p>
            <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border-color); padding-top:12px; font-size:0.85rem;">
                <span style="color: #b45309; font-weight:600;">⏰ Due: {a['due_date']} ({a['max_grade']})</span>
                <a href="/mock-portal/unit/{unit_code}/assignment/{a['id']}" id="open-asg-{a['id']}" class="btn btn-primary" style="padding:8px 16px; font-size:0.85rem;">
                    { 'View Submission' if a['status'] == 'Submitted for grading' else 'Upload & Submit Solution →' }
                </a>
            </div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{unit['code']} — Course Assignments</title>
        {PORTAL_BASE_STYLE}
    </head>
    <body>
        <nav class="portal-nav">
            <div class="brand"> University e-Campus LMS</div>
            <div class="user-badge"><a href="/mock-portal/dashboard" style="color:#38bdf8; text-decoration:none;">← Back to Dashboard</a></div>
        </nav>
        <div class="container">
            <div style="margin-bottom: 20px;">
                <span style="font-size:0.85rem; font-weight:700; color:var(--primary);">{unit['code']}</span>
                <h1 style="font-size:1.6rem; font-weight:800;" id="unit-title">{unit['name']}</h1>
                <p style="color:var(--text-muted); font-size:0.9rem;">Course Instructor: {unit['lecturer']} | Credits: {unit['credits']}</p>
            </div>
            <h2 style="font-size: 1.25rem; font-weight: 700; margin-bottom: 16px;">Coursework, CATs & Assignments</h2>
            {asg_html}
        </div>
    </body>
    </html>
    """

@router.get("/unit/{unit_code}/assignment/{assignment_id}", response_class=HTMLResponse)
async def assignment_dropbox(unit_code: str, assignment_id: str):
    real_code = unit_code.replace('_', ' ')
    unit = next((u for u in MOCK_DATA["units"] if u["code"].lower() == real_code.lower()), None)
    if not unit:
        return HTMLResponse("Unit not found", status_code=404)
    asg = next((a for a in unit["assignments"] if a["id"] == assignment_id), None)
    if not asg:
        return HTMLResponse("Assignment not found", status_code=404)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{asg['title']} — Submission Dropbox</title>
        {PORTAL_BASE_STYLE}
    </head>
    <body>
        <nav class="portal-nav">
            <div class="brand"> University e-Campus LMS</div>
            <div class="user-badge"><a href="/mock-portal/unit/{unit_code}" style="color:#38bdf8; text-decoration:none;">← Back to {unit['code']}</a></div>
        </nav>
        <div class="container" style="max-width: 800px;">
            <div class="card">
                <span style="font-size:0.8rem; font-weight:700; color:var(--primary);">{unit['code']}: {unit['name']}</span>
                <h1 style="font-size:1.5rem; font-weight:800; margin: 4px 0 12px 0;" id="asg-title">{asg['title']}</h1>
                <p style="font-size:0.95rem; color:#334155; margin-bottom:16px; background:#f1f5f9; padding:12px; border-radius:8px; border-left:4px solid var(--primary);">{asg['description']}</p>
                
                <table style="width:100%; border-collapse:collapse; margin-bottom:20px; font-size:0.875rem;">
                    <tr style="border-bottom:1px solid var(--border-color);">
                        <td style="padding:8px 0; font-weight:600; color:var(--text-muted);">Submission Status</td>
                        <td style="padding:8px 0;" id="current-status"><span class="badge { 'badge-submitted' if asg['status'] == 'Submitted for grading' else 'badge-pending' }">{asg['status']}</span></td>
                    </tr>
                    <tr style="border-bottom:1px solid var(--border-color);">
                        <td style="padding:8px 0; font-weight:600; color:var(--text-muted);">Due Date</td>
                        <td style="padding:8px 0;">{asg['due_date']}</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 0; font-weight:600; color:var(--text-muted);">Accepted Formats</td>
                        <td style="padding:8px 0;">PDF Document (.pdf)</td>
                    </tr>
                </table>

                <form id="submission-form" action="/mock-portal/unit/{unit_code}/assignment/{assignment_id}/submit" method="POST" enctype="multipart/form-data">
                    <h3 style="font-size:1.1rem; font-weight:700; margin-bottom:10px;">File Upload Dropbox</h3>
                    <div class="dropzone" onclick="document.getElementById('file-input').click()">
                        <div style="font-size:2rem; margin-bottom:8px;"></div>
                        <div style="font-weight:700; color:var(--primary); margin-bottom:4px;" id="dropzone-text">Click to choose PDF or drag & drop here</div>
                        <div style="font-size:0.8rem; color:var(--text-muted);">Maximum file size: 25 MB</div>
                        <input type="file" id="file-input" name="file" accept=".pdf" style="display:block; margin: 12px auto 0 auto;" required onchange="document.getElementById('dropzone-text').innerText = 'Selected: ' + this.files[0].name">
                    </div>
                    
                    <div style="display:flex; gap:12px; justify-content:flex-end;">
                        <button type="submit" id="submit-assignment-btn" class="btn btn-success" style="padding:12px 28px; font-size:0.95rem;">
                             Confirm & Turn In Assignment
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    """

@router.post("/unit/{unit_code}/assignment/{assignment_id}/submit", response_class=HTMLResponse)
async def submit_assignment(
    unit_code: str,
    assignment_id: str,
    file: UploadFile = File(...)
):
    real_code = unit_code.replace('_', ' ')
    unit = next((u for u in MOCK_DATA["units"] if u["code"].lower() == real_code.lower()), None)
    if not unit:
        return HTMLResponse("Unit not found", status_code=404)
    asg = next((a for a in unit["assignments"] if a["id"] == assignment_id), None)
    if not asg:
        return HTMLResponse("Assignment not found", status_code=404)

    # Read bytes and compute hash
    content = await file.read()
    file_size_kb = round(len(content) / 1024, 1)
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    now_str = time.strftime("%B %d, %Y at %I:%M:%S %p")

    # Update in-memory record
    asg["status"] = "Submitted for grading"
    asg["submitted_file"] = file.filename
    asg["submitted_at"] = now_str

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Submission Confirmed — e-Campus</title>
        {PORTAL_BASE_STYLE}
    </head>
    <body>
        <nav class="portal-nav">
            <div class="brand"> University e-Campus LMS</div>
            <div class="user-badge"><a href="/mock-portal/dashboard" style="color:#38bdf8; text-decoration:none;">← Return to Dashboard</a></div>
        </nav>
        <div class="container" style="max-width: 720px;">
            <div class="card" style="text-align: center; padding: 40px 24px;" id="confirmation-receipt-card">
                <div style="font-size: 3.5rem; margin-bottom: 12px;"></div>
                <div style="display:inline-block; background:#d1fae5; color:#065f46; font-weight:800; padding:6px 16px; border-radius:9999px; font-size:0.875rem; margin-bottom:12px;" id="badge-success">
                    OFFICIAL SUBMISSION RECEIPT
                </div>
                <h1 style="font-size:1.6rem; font-weight:800; margin-bottom:8px;" id="receipt-title">Assignment Submitted Successfully!</h1>
                <p style="color:var(--text-muted); font-size:0.95rem; margin-bottom:24px;">Your document has been securely received and recorded in the academic database.</p>
                
                <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:10px; padding:20px; text-align:left; margin-bottom:24px; font-size:0.9rem;">
                    <div style="margin-bottom:8px;"><b>Course Unit:</b> {unit['code']} - {unit['name']}</div>
                    <div style="margin-bottom:8px;"><b>Assignment:</b> {asg['title']}</div>
                    <div style="margin-bottom:8px;"><b>Uploaded File:</b> <span style="color:var(--primary); font-weight:600;" id="receipt-filename">{file.filename}</span> ({file_size_kb} KB)</div>
                    <div style="margin-bottom:8px;"><b>Timestamp:</b> <span id="receipt-timestamp">{now_str}</span></div>
                    <div><b>Digital Verification Hash:</b> <code style="font-family:'JetBrains Mono',monospace; background:#e2e8f0; padding:2px 6px; border-radius:4px; font-size:0.8rem;">{file_hash}</code></div>
                </div>

                <div style="display:flex; justify-content:center; gap:12px;">
                    <a href="/mock-portal/unit/{unit_code}" class="btn btn-primary">Return to Unit</a>
                    <a href="/mock-portal/dashboard" class="btn" style="background:#e2e8f0; color:#0f172a;">Back to Dashboard</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
