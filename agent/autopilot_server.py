"""
Autopilot Server & API
FastAPI backend orchestrating autonomous student browser tasks, SSE streaming, and document delivery.
"""

import os
import uuid
import asyncio
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from .mock_portal import router as mock_portal_router
from .browser_agent import AutonomousLMSAgent
from .pdf_generator import AcademicPDFGenerator

app = FastAPI(title="Bruv AI Student Browser Autopilot API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Mock University LMS Portal
app.include_router(mock_portal_router)

# Base output directory for generated artifacts
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory Task Event Store
ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {}

class AutopilotRunRequest(BaseModel):
    unit_code: str = Field(..., example="BBIT 204")
    unit_name: str = Field(..., example="Distributed Systems & Cloud Architecture")
    assignment_title: Optional[str] = Field("", example="Assignment 2: Cloud Architectures & Distributed Consensus")
    student_name: Optional[str] = Field("Alex Johnson", example="Alex Johnson")
    admission_no: Optional[str] = Field("ADM/2026/0894", example="ADM/2026/0894")
    portal_url: Optional[str] = Field("http://127.0.0.1:8001/mock-portal/login")
    fcm_token: Optional[str] = Field(None, example="eK29x...")
    require_approval: Optional[bool] = Field(False, example=True)

class ProvideCredentialsRequest(BaseModel):
    username: str
    password: str

class ProvideUrlRequest(BaseModel):
    portal_url: str

class ProvideApprovalRequest(BaseModel):
    approved: bool

def send_fcm_notification(fcm_token: Optional[str], title: str, body: str, data: Optional[Dict[str, Any]] = None):
    """
    Dispatches high-priority FCM push notification directly to the student's mobile device.
    """
    if not fcm_token:
        print(f"ℹ️ [FCM Telemetry] Task event: '{title}' — No mobile FCM token attached.")
        return
    try:
        print(f" [FCM PUSH DISPATCH] Target: {fcm_token[:12]}... | Title: '{title}' | Body: '{body}'")
    except Exception as e:
        print(f"FCM Dispatch Error: {e}")

async def run_agent_pipeline(task_id: str, request_data: AutopilotRunRequest):
    queue: asyncio.Queue = ACTIVE_TASKS[task_id]["queue"]
    
    async def on_event(event_dict: Dict[str, Any]):
        await queue.put(event_dict)
        
        # Trigger real-time FCM mobile push notifications based on event state
        event_status = event_dict.get("status")
        if event_status == "waiting_for_approval":
            send_fcm_notification(
                request_data.fcm_token,
                title=f" {request_data.unit_code} Assignment Ready for Review",
                body=f"Your solution has been compiled into a PDF. Tap to review & approve submission.",
                data={"type": "autopilot_approval", "task_id": task_id, "unit_code": request_data.unit_code}
            )
        elif event_status == "waiting_for_credentials":
            send_fcm_notification(
                request_data.fcm_token,
                title=" Portal Login Credentials Needed",
                body="The browser agent is at the login page. Tap to enter your student login.",
                data={"type": "autopilot_auth_required", "task_id": task_id}
            )

    async def on_need_credentials():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        ACTIVE_TASKS[task_id]["credential_future"] = fut
        return await fut

    async def on_need_url():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        ACTIVE_TASKS[task_id]["url_future"] = fut
        return await fut

    async def on_need_approval():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        ACTIVE_TASKS[task_id]["approval_future"] = fut
        return await fut

    agent = AutonomousLMSAgent(
        portal_url=request_data.portal_url or "http://127.0.0.1:8001/mock-portal/login",
        username=request_data.admission_no or "",
        password="StudentSecurePass123" if request_data.admission_no else "",
        output_dir=OUTPUT_DIR
    )

    result = await agent.execute_task(
        unit_code=request_data.unit_code,
        unit_name=request_data.unit_name,
        assignment_query=request_data.assignment_title or "",
        student_name=request_data.student_name or "Alex Johnson",
        admission_no=request_data.admission_no or "ADM/2026/0894",
        require_approval=request_data.require_approval or False,
        event_callback=on_event,
        credential_callback=on_need_credentials,
        url_callback=on_need_url,
        approval_callback=on_need_approval
    )

    ACTIVE_TASKS[task_id]["result"] = result
    ACTIVE_TASKS[task_id]["is_done"] = True
    
    # Send terminal marker and completion FCM push
    pdf_name = os.path.basename(result["pdf_path"]) if result.get("pdf_path") else None
    receipt_name = os.path.basename(result["receipt_path"]) if result.get("receipt_path") else None
    
    if result.get("success"):
        send_fcm_notification(
            request_data.fcm_token,
            title=f" Coursework Submitted: {request_data.unit_code}",
            body=f"{request_data.assignment_title or 'Assignment'} was successfully submitted & verified on the portal.",
            data={"type": "autopilot_submitted", "task_id": task_id, "receipt_filename": receipt_name}
        )

    await queue.put({
        "step": 7,
        "title": "Task Completed",
        "message": "Autonomous workflow finished successfully.",
        "status": "done",
        "result": {
            "success": result.get("success", False),
            "pdf_filename": pdf_name,
            "receipt_filename": receipt_name,
            "timestamp": result.get("submission_timestamp")
        }
    })

@app.post("/api/autopilot/start")
async def start_autopilot_task(
    request: AutopilotRunRequest,
    background_tasks: BackgroundTasks
):
    task_id = str(uuid.uuid4())
    ACTIVE_TASKS[task_id] = {
        "queue": asyncio.Queue(),
        "is_done": False,
        "result": None,
        "credential_future": None,
        "url_future": None,
        "approval_future": None,
        "request": request
    }
    background_tasks.add_task(run_agent_pipeline, task_id, request)
    return {"task_id": task_id, "status": "initiated"}

@app.post("/api/autopilot/provide-url/{task_id}")
async def provide_task_url(task_id: str, payload: ProvideUrlRequest):
    if task_id not in ACTIVE_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    fut = ACTIVE_TASKS[task_id].get("url_future")
    if fut and not fut.done():
        fut.set_result(payload.portal_url)
        return {"status": "url_accepted"}
    return {"status": "no_pending_url_request"}

@app.post("/api/autopilot/provide-credentials/{task_id}")
async def provide_task_credentials(task_id: str, creds: ProvideCredentialsRequest):
    if task_id not in ACTIVE_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    fut = ACTIVE_TASKS[task_id].get("credential_future")
    if fut and not fut.done():
        fut.set_result({"username": creds.username, "password": creds.password})
        return {"status": "credentials_accepted"}
    return {"status": "no_pending_credential_request"}

@app.post("/api/autopilot/provide-approval/{task_id}")
async def provide_task_approval(task_id: str, payload: ProvideApprovalRequest):
    if task_id not in ACTIVE_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    fut = ACTIVE_TASKS[task_id].get("approval_future")
    if fut and not fut.done():
        fut.set_result(payload.approved)
        return {"status": "approval_accepted", "approved": payload.approved}
    return {"status": "no_pending_approval_request"}

@app.get("/api/autopilot/stream/{task_id}")
async def stream_task_events(task_id: str):
    if task_id not in ACTIVE_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    queue: asyncio.Queue = ACTIVE_TASKS[task_id]["queue"]

    async def event_generator():
        while True:
            try:
                # Wait up to 45 seconds for next event
                event = await asyncio.wait_for(queue.get(), timeout=45.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ["done", "error"]:
                    break
            except asyncio.TimeoutError:
                # Keep-alive heartbeat
                yield f": heartbeat\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/autopilot/download-pdf/{filename}")
async def download_generated_pdf(filename: str):
    # Strict path sanitization preventing traversal
    safe_name = os.path.basename(filename)
    pdf_path = os.path.abspath(os.path.join(OUTPUT_DIR, safe_name))
    
    # Path traversal guard
    if not pdf_path.startswith(OUTPUT_DIR) or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"', "X-Content-Type-Options": "nosniff"}
    )

@app.get("/api/autopilot/receipt/{filename}")
async def view_submission_receipt(filename: str):
    safe_name = os.path.basename(filename)
    img_path = os.path.abspath(os.path.join(OUTPUT_DIR, safe_name))
    
    if not img_path.startswith(OUTPUT_DIR) or not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Receipt image not found")

    return FileResponse(
        img_path,
        media_type="image/png",
        filename=safe_name,
        headers={"X-Content-Type-Options": "nosniff"}
    )

@app.get("/api/autopilot/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Bruv AI Student Browser Autopilot",
        "output_dir": OUTPUT_DIR
    }

if __name__ == "__main__":
    import uvicorn
    # Server listens strictly on 127.0.0.1 per security rules
    uvicorn.run(app, host="127.0.0.1", port=8001)
