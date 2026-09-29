"""
Verification Test Script for AI Student Browser Agent
Tests PDF generation and Autonomous Playwright Browser Navigation against the Mock Portal.
"""

import asyncio
import os
import uvicorn
import threading
import time

from agent.assignment_solver import AssignmentSolver
from agent.pdf_generator import AcademicPDFGenerator
from agent.autopilot_server import app
from agent.browser_agent import AutonomousLMSAgent

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

async def test_full_pipeline():
    print("\n--- 1. Testing Academic Assignment Solver & PDF Generation ---")
    data = AssignmentSolver.solve_assignment(
        unit_code="BBIT 204",
        unit_name="Distributed Systems & Cloud Architecture",
        assignment_title="Assignment 2: Cloud Architectures & Distributed Consensus",
        student_name="Alex Johnson",
        admission_no="ADM/2026/0894"
    )
    
    out_dir = os.path.abspath("test_output")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = AcademicPDFGenerator.generate(data, output_dir=out_dir)
    print(f" Generated PDF at: {pdf_path} (Size: {os.path.getsize(pdf_path)} bytes)")

    print("\n--- 2. Starting Background Mock Portal & Autopilot Server ---")
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    await asyncio.sleep(2)

    print("\n--- 3. Running Autonomous Playwright Browser Agent ---")
    agent = AutonomousLMSAgent(
        portal_url="http://127.0.0.1:8001/mock-portal/login",
        output_dir=out_dir
    )

    async def on_event(ev):
        status_sym = "" if ev.get("status") == "completed" else ""
        print(f"  {status_sym} [Step {ev.get('step')}] {ev.get('title')}: {ev.get('message')}")

    result = await agent.execute_task(
        unit_code="BBIT 204",
        unit_name="Distributed Systems & Cloud Architecture",
        assignment_query="Assignment 2",
        student_name="Alex Johnson",
        admission_no="ADM/2026/0894",
        event_callback=on_event
    )

    print("\n--- 4. Autonomous Agent Execution Results ---")
    print(f"Success: {result.get('success')}")
    print(f"Generated PDF: {result.get('pdf_path')}")
    print(f"Receipt Screenshot: {result.get('receipt_path')}")
    print(f"Timestamp: {result.get('submission_timestamp')}")

    assert result.get('success') is True, "Autonomous browser run failed!"
    assert result.get('pdf_path') is not None, "Missing PDF output!"
    assert result.get('receipt_path') is not None, "Missing Receipt output!"
    print("\n ALL AUTONOMOUS AGENT VERIFICATIONS PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
