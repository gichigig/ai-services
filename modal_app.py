import modal
import os
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form

# Define the Modal App
app = modal.App("video-analyzer-service")

# Define the image with all necessary system packages and Python dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libsndfile1", "python3-dev", "build-essential")
    .run_commands(
        "pip install --upgrade 'setuptools<70.0.0' wheel",
        "pip install --no-build-isolation openai-whisper"
    )
    .pip_install_from_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    .run_commands(
        "pip install --upgrade 'setuptools<70.0.0'",
        "playwright install --with-deps chromium"
    )
    .add_local_python_source("video_analyzer", "audio_analyzer", "llm_router", "rag_search", "timetable_extractor", "quiz_generator", "document_analyzer", "agent")
)

# Define the persistent volume for RAG storage
rag_volume = modal.Volume.from_name("bruv-schools-rag-data", create_if_missing=True)

from typing import Optional, List, Dict, Any

class VideoAnalysisRequest(BaseModel):
    media_id: str
    video_url: str
    caption: str = ""
    webhook_url: Optional[str] = "https://school.ishinadwelly.com/api/internal/media/analysis-complete"

class ModerationRequest(BaseModel):
    video_url: str

class ChatAttachment(BaseModel):
    name: str = "attachment"
    mime_type: Optional[str] = None
    data: str = ""

class QuestionRequest(BaseModel):
    student_id: Optional[str] = None
    question: str
    history: Optional[list] = None
    user_memory: Optional[dict] = None
    selected_model: Optional[str] = "small"
    use_reasoning: Optional[bool] = False
    attachments: Optional[List[ChatAttachment]] = None

class TopicRequest(BaseModel):
    first_question: str
    first_answer: Optional[str] = ""

class DocumentIndexRequest(BaseModel):
    file_url: str
    course_name: str = ""
    year: str = ""

@app.function(image=image, timeout=600, gpu="A10G")
def process_video_modal(media_id: str, video_url: str, caption: str = "", webhook_url: str = "https://school.ishinadwelly.com/api/internal/media/analysis-complete"):
    """
    Background task that downloads the video, runs inference, and sends results to Kotlin backend.
    Runs entirely on Modal.
    """
    # Import locally within the function to avoid downloading models when parsing the file
    from video_analyzer import analyze_video_frames
    from audio_analyzer import extract_and_analyze_audio
    import requests

    target_webhook = webhook_url or os.getenv("KOTLIN_BACKEND_WEBHOOK") or "https://school.ishinadwelly.com/api/internal/media/analysis-complete"
    if "localhost" in target_webhook:
        target_webhook = "https://school.ishinadwelly.com/api/internal/media/analysis-complete"

    import tempfile
    temp_video_path = None

    try:
        print(f"Starting analysis for media_id: {media_id}")
        if video_url.startswith("/"):
            video_url = f"https://school.ishinadwelly.com{video_url}"

        # Download video to local temp file for reliable frame seeking and fast audio analysis
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_path = temp_file.name
        temp_file.close()

        print(f"Downloading video {media_id} from {video_url} to {temp_video_path}")
        dl_resp = requests.get(video_url, stream=True, timeout=90)
        dl_resp.raise_for_status()
        with open(temp_video_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded video successfully ({os.path.getsize(temp_video_path)} bytes)")

        # 1. Run visual classification, face embeddings, and all new analysis on local file
        video_category, ai_tags, is_nsfw, face_embeddings, extra_analysis = analyze_video_frames(temp_video_path)
        
        # 2. Run audio classification + transcription (Whisper) on local file
        audio_hash, is_speech, language, audio_mood, transcript = extract_and_analyze_audio(temp_video_path)
        
        # 3. Final Action-Based Category Override (Combining Audio + Visual)
        final_category = video_category
        motion_intensity = extra_analysis.get("motionIntensity", "low")
        dance_confidence = extra_analysis.get("danceConfidence", 0.0)
        
        # If the pose model is highly confident it's a dance, override everything
        if dance_confidence > 0.4:
            final_category = "Entertainment"
            if "incident" in ai_tags: ai_tags.remove("incident")
            if "dancing" not in ai_tags: ai_tags.append("dancing")
        elif audio_mood == "intense" and motion_intensity == "high":
            final_category = "Action"
            if "dancing" in ai_tags:
                ai_tags.remove("dancing")
            if "incident" not in ai_tags:
                ai_tags.append("incident")
        elif audio_mood == "energetic" and "face-centric" in ai_tags:
            final_category = "Entertainment"
        elif is_speech and motion_intensity == "low" and "face-centric" in ai_tags:
            final_category = "Lifestyle"
        
        # Send results back to Kotlin backend
        payload = {
            "mediaId": media_id,
            "category": final_category,
            "aiTags": ai_tags,
            "audioHash": audio_hash,
            "hasSpeech": is_speech,
            "isNsfw": is_nsfw,
            "language": language,
            "faceEmbeddings": face_embeddings,
            "visualFingerprint": extra_analysis.get("visualFingerprint", ""),
            "thumbnailTimestampMs": extra_analysis.get("thumbnailTimestampMs", 0),
            "qualityScore": extra_analysis.get("qualityScore", 0.5),
            "colorMood": extra_analysis.get("colorMood", "neutral"),
            "dominantColors": extra_analysis.get("dominantColors", []),
            "motionIntensity": extra_analysis.get("motionIntensity", "low"),
            "audioMood": audio_mood,
            "caption": caption,
            "transcript": transcript,
            "nsfwScore": extra_analysis.get("nsfwScore", 0.0),
            "ocrText": extra_analysis.get("ocrText", ""),
            "engagementScore": extra_analysis.get("engagementScore", 0.0),
            "danceConfidence": dance_confidence
        }
        
        print(f"Analysis complete for {media_id}. Category: {video_category}, "
              f"Tags: {ai_tags}, Quality: {extra_analysis.get('qualityScore')}, "
              f"Mood: {extra_analysis.get('colorMood')}, AudioMood: {audio_mood}, NSFW: {is_nsfw} ({extra_analysis.get('nsfwScore', 0.0)})")
        
        # Send to Kotlin backend
        print(f"Sending webhook to {target_webhook}")
        response = requests.post(target_webhook, json=payload, timeout=30)
        response.raise_for_status()
        print("Webhook sent successfully.")
        
    except Exception as e:
        print(f"Error analyzing video {media_id}: {str(e)}")
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

# Define the FastAPI app
web_app = FastAPI(title="Bruv Schools AI Microservice (Modal)")

@web_app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Endpoint called by the Kotlin backend to trigger asynchronous video analysis.
    This routes the heavy work to the process_video_modal Modal function.
    """
    target_webhook = request.webhook_url or "https://school.ishinadwelly.com/api/internal/media/analysis-complete"
    v_url = request.video_url
    if v_url.startswith("/"):
        v_url = f"https://school.ishinadwelly.com{v_url}"
    process_video_modal.spawn(request.media_id, v_url, request.caption, target_webhook)
    return {"status": "Analysis queued on Modal", "media_id": request.media_id}

@app.function(image=image, timeout=60, gpu="A10G")
def quick_check_video_modal(video_url: str) -> dict:
    import tempfile, requests, os
    from video_analyzer import check_video_nsfw

    if video_url.startswith("/"):
        video_url = f"https://school.ishinadwelly.com{video_url}"

    temp_video_path = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_path = temp_file.name
        temp_file.close()

        print(f"quick_check_video_modal: Downloading {video_url} to {temp_video_path}")
        dl_resp = requests.get(video_url, stream=True, timeout=30)
        dl_resp.raise_for_status()
        with open(temp_video_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        is_nsfw, nsfw_score = check_video_nsfw(temp_video_path, max_frames=20)
        return {"isNsfw": is_nsfw, "nsfwScore": nsfw_score}
    except Exception as e:
        print(f"quick_check_video_modal error: {e}")
        return {"isNsfw": False, "nsfwScore": 0.0}
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

@web_app.post("/check-moderation")
def check_moderation_endpoint(request: ModerationRequest):
    """
    Synchronous pre-check for video upload moderation.
    Returns {"isNsfw": bool, "nsfwScore": float} within ~1-2 seconds.
    """
    return quick_check_video_modal.remote(request.video_url)

@web_app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """
    Endpoint called by the client or backend to ask the AI a question.
    """
    try:
        from rag_search import retrieve_context
        from llm_router import ask_qwen
        
        attachment_context = ""
        if request.attachments:
            from document_analyzer import process_chat_attachments
            attachment_context = await process_chat_attachments([a.model_dump() for a in request.attachments])

        context = retrieve_context(request.question, history=request.history)
        answer = await ask_qwen(
            request.question, 
            context, 
            student_id=request.student_id, 
            history=request.history,
            user_memory=request.user_memory,
            selected_model=request.selected_model,
            use_reasoning=request.use_reasoning,
            attachment_context=attachment_context
        )
        
        return {
            "status": "success",
            "student_id": request.student_id or "",
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "student_id": request.student_id,
            "question": request.question,
            "answer": f"Error: {str(e)}\n{traceback.format_exc()}"
        }

from fastapi.responses import StreamingResponse

@web_app.post("/ask-question-stream")
async def ask_question_stream_endpoint(request: QuestionRequest):
    """
    SSE endpoint for streaming real-time status and final answers.
    """
    from rag_search import retrieve_context
    from llm_router import ask_qwen_stream
    import json
    
    async def event_generator():
        try:
            attachment_context = ""
            if request.attachments:
                yield json.dumps({
                    "type": "status", 
                    "stage": "analyzing_attachments", 
                    "message": f"Analyzing {len(request.attachments)} attached document(s)/image(s)..."
                }) + "\n"
                
                from document_analyzer import process_chat_attachments
                attachment_context = await process_chat_attachments([a.model_dump() for a in request.attachments])
                
                yield json.dumps({
                    "type": "status", 
                    "stage": "synthesizing", 
                    "message": "Reading extracted content and preparing response..."
                }) + "\n"

            context = retrieve_context(request.question, history=request.history)
            async for chunk in ask_qwen_stream(
                request.question, 
                context, 
                student_id=request.student_id, 
                history=request.history,
                user_memory=request.user_memory,
                selected_model=request.selected_model,
                use_reasoning=request.use_reasoning,
                attachment_context=attachment_context
            ):
                yield chunk
        except Exception as e:
            import traceback
            yield json.dumps({"type": "answer", "content": f"Error: {str(e)}\n{traceback.format_exc()}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@web_app.post("/generate-topic")
async def generate_topic_endpoint(request: TopicRequest):
    """
    Generates a concise 3 to 6 word title for a conversation based on the initial exchange.
    """
    from llm_router import generate_conversation_topic
    topic = await generate_conversation_topic(request.first_question, request.first_answer or "")
    return {"status": "success", "topic": topic}

@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True, "environment": "modal"}

@web_app.post("/index-document")
async def index_document(request: DocumentIndexRequest):
    """
    Endpoint called by the Kotlin backend when a new document is uploaded.
    Downloads the document, extracts text, and stores it in the persistent FAISS index.
    """
    try:
        from rag_search import index_file_url
        
        # Determine file extension from URL
        ext = request.file_url.split(".")[-1].lower() if "." in request.file_url else "pdf"
        
        # Index the file
        num_chunks = await index_file_url(request.file_url, ext)
        
        return {
            "status": "success",
            "message": f"Document indexed successfully into {num_chunks} chunks.",
            "file_url": request.file_url
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Failed to index document: {str(e)}",
            "error_details": traceback.format_exc()
        }

@web_app.post("/extract-timetable")
async def extract_timetable_endpoint(
    file: UploadFile = File(...),
    cohort: Optional[str] = Form(None),
    course: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    semester: Optional[str] = Form(None)
):
    """
    Endpoint called by the client or backend when an image, PDF, or Excel (.xlsx) timetable is uploaded.
    Extracts text/grid tables, matches user cohort, and returns structured timetable blocks.
    """
    try:
        from timetable_extractor import extract_timetable_from_bytes
        file_bytes = await file.read()
        res = await extract_timetable_from_bytes(
            file_bytes,
            filename=file.filename or "timetable.pdf",
            course=course or "",
            year=year or "",
            semester=semester or "",
            cohort_hint=cohort or ""
        )
        if isinstance(res, dict):
            blocks = res.get("blocks", [])
            all_cohorts = res.get("all_cohorts", [])
            matched_cohort = res.get("matched_cohort")
            all_blocks = res.get("all_blocks", [])
        else:
            blocks = res
            all_cohorts = []
            matched_cohort = None
            all_blocks = blocks

        cohort_msg = f" for {matched_cohort}" if matched_cohort else ""
        return {
            "status": "success",
            "blocks": blocks,
            "all_cohorts": all_cohorts,
            "matched_cohort": matched_cohort,
            "all_blocks": all_blocks,
            "message": f"Successfully extracted {len(blocks)} timetable blocks{cohort_msg}."
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "blocks": [],
            "all_cohorts": [],
            "matched_cohort": None,
            "all_blocks": [],
            "message": f"Extraction failed: {str(e)}",
            "error_details": traceback.format_exc()
        }

class StudyTimetableRequest(BaseModel):
    class_blocks: list = []
    course: Optional[str] = ""
    study_mode: Optional[str] = "FULL_TIME"

@web_app.post("/generate-study-timetable")
async def generate_study_timetable_endpoint(req: StudyTimetableRequest):
    """
    Endpoint that generates an optimized weekly study timetable fitting around the student's classes.
    """
    try:
        from timetable_extractor import generate_study_timetable_ai
        study_blocks = await generate_study_timetable_ai(
            class_blocks=req.class_blocks,
            course=req.course or "",
            study_mode=req.study_mode or "FULL_TIME"
        )
        return {
            "status": "success",
            "blocks": study_blocks,
            "message": f"Successfully generated {len(study_blocks)} study blocks."
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "blocks": [],
            "message": f"Failed to generate study timetable: {str(e)}",
            "error_details": traceback.format_exc()
        }

class QuizGenerateRequest(BaseModel):
    course: str
    topic: Optional[str] = None
    num_questions: Optional[int] = 5
    difficulty: Optional[str] = "medium"
    context: Optional[str] = None
    question_type: Optional[str] = "multiple_choice"
    custom_instructions: Optional[str] = None

class GradeTheoryRequest(BaseModel):
    question: str
    student_answer: str
    subject: Optional[str] = "University Academics"
    max_marks: Optional[int] = 10
    model_answer: Optional[str] = None
    rubric_points: Optional[List[str]] = None

@web_app.post("/warmup")
@web_app.get("/warmup")
async def warmup_endpoint():
    """
    Pings QwenSmallModel on Modal to initiate container startup.
    When users enter the Quiz or Bruv AI screen, this warms the GPU so answers generate instantly.
    If no question arrives within 30 seconds, Modal's scaledown_window powers off the GPU container automatically.
    """
    try:
        SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
        small_model = SmallModelCls()
        await small_model.warmup.remote.aio()
        return {"status": "success", "message": "QwenSmallModel container initiated and warm"}
    except Exception as e:
        return {"status": "error", "message": f"Warmup ping error: {str(e)}"}

@web_app.post("/generate-quiz")
async def generate_quiz_endpoint(req: QuizGenerateRequest):
    """
    Generates tailored, course-specific multiple choice or theory quiz questions using AI.
    Learns from and retrieves unit lecture notes and syllabus materials via RAG.
    """
    try:
        from quiz_generator import generate_course_quiz_ai
        questions = await generate_course_quiz_ai(
            course=req.course,
            topic=req.topic,
            num_questions=req.num_questions or 5,
            difficulty=req.difficulty or "medium",
            context=req.context,
            question_type=req.question_type or "multiple_choice",
            custom_instructions=req.custom_instructions
        )
        return {
            "status": "success",
            "course": req.course,
            "topic": req.topic or req.course,
            "count": len(questions),
            "questions": questions
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "course": req.course,
            "topic": req.topic or req.course,
            "count": 0,
            "questions": [],
            "message": f"AI Quiz generation failed: {str(e)}",
            "error_details": traceback.format_exc()
        }

@web_app.post("/grade-theory-answer")
async def grade_theory_endpoint(req: GradeTheoryRequest):
    """
    Evaluates and grades a student's typed written answer against academic rubrics and model answers.
    """
    try:
        from quiz_generator import grade_student_theory_answer
        result = await grade_student_theory_answer(
            question=req.question,
            student_answer=req.student_answer,
            subject=req.subject or "University Academics",
            max_marks=req.max_marks or 10,
            model_answer=req.model_answer,
            rubric_points=req.rubric_points
        )
        return {"status": "success", **result}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Theory answer evaluation failed: {str(e)}",
            "error_details": traceback.format_exc()
        }

class EcitizenIntentRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, Any]]] = None
    available_services: Optional[List[Dict[str, Any]]] = None
    user_context: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = "gpt_120b"

@web_app.post("/ecitizen/parse-intent")
async def ecitizen_parse_intent_endpoint(req: EcitizenIntentRequest):
    """
    Conversational Multi-Turn AI Intent, Clarification, and Web Search engine for eCitizen.
    Uses GPT OSS 120B on Modal and live web search to answer questions or ask clarifying
    follow-up questions before initiating autonomous browser automation.
    """
    import json
    import re

    clean_prompt = req.prompt.strip()
    p_lower = clean_prompt.lower()

    # 1. Determine if live web search is needed
    search_keywords = [
        "how much", "what is the fee", "what are the fees", "how long", "requirements", 
        "what documents", "cost", "tariff", "is it possible", "can i", "check online", 
        "search web", "google", "taken", "available", "conflict", "trademark", "is the name"
    ]
    needs_web_search = any(k in p_lower for k in search_keywords) or clean_prompt.endswith("?")

    web_search_text = ""
    web_sources = []
    if needs_web_search:
        try:
            from llm_router import execute_web_search
            q = re.sub(r'^(check|search|verify|find|tell me|google)\s+', '', clean_prompt, flags=re.IGNORECASE)
            search_query = f"{q} eCitizen Kenya" if "ecitizen" not in q.lower() and "kenya" not in q.lower() else q
            print(f"[EcitizenSearch] Executing live web search for: '{search_query}'")
            raw_text, sources = await execute_web_search(search_query)
            if raw_text:
                web_search_text = raw_text[:2200]
                web_sources = sources[:4] if sources else []
        except Exception as search_err:
            print(f"[EcitizenSearch] Web search note: {search_err}")

    # 2. Format multi-turn conversation history
    history_context = ""
    if req.history:
        formatted_history = []
        for h in req.history[-6:]:
            role = h.get("sender") or h.get("role") or "user"
            content = h.get("text") or h.get("content") or ""
            if content:
                formatted_history.append(f"{role.capitalize()}: {content}")
        if formatted_history:
            history_context = "Conversation History:\n" + "\n".join(formatted_history) + "\n\n"

    system_prompt = (
        "You are an expert conversational AI Copilot for Kenyan eCitizen government portals (BRS, NTSA, DCI, Immigration).\n"
        "You can chat conversationally, answer government questions using live web search results, ask clarifying questions before execution, or trigger live browser automation when all required details are present.\n\n"
        "Decision Modes (Field 'type'):\n"
        "1. 'info': Use when the user is asking an informational question about government requirements, fees, rules, or asking to check something online. Answer the question thoroughly in 'message' using the provided web search context or your knowledge. Do NOT launch the browser (ready_to_launch=false).\n"
        "2. 'clarification': Use when the user wants to execute a government service (e.g. business registration, good conduct, driving license renewal) but has NOT provided necessary details (e.g. missing proposed business name for BRS, missing preferred Huduma station). Ask clear, friendly follow-up questions in 'message' to gather the missing details (ready_to_launch=false).\n"
        "3. 'action_ready': Use ONLY when you have sufficient details (e.g., at least the primary proposed business name for BRS, or user explicitly commands 'proceed' / 'start' / 'run') to launch the automated browser navigation on eCitizen. Summarize what you will do in 'message' and set ready_to_launch=true.\n\n"
        "Known Government Services & Portal Mapping:\n"
        "- 'business_name_search': Business Name Search & Reservation on Business Registration Service (BRS), URL: https://brs.ecitizen.go.ke\n"
        "- 'business_name_registration': Business Name Registration (Sole Proprietorship / BN2) on BRS, URL: https://brs.ecitizen.go.ke\n"
        "- 'good_conduct': Police Clearance Certificate on Directorate of Criminal Investigations (DCI), URL: https://dci.ecitizen.go.ke\n"
        "- 'driving_license_renewal': 3-Year Smart Driving License Renewal or Driving Licence / DL Application/Payment on National Transport and Safety Authority (NTSA). (Matches 'driving license', 'driving licence', 'pay for driving licence', 'smart dl', 'renew driving'). URL: https://serviceportal.ntsa.go.ke\n"
        "- 'pdl_test_booking': Provisional Driving License / PDL & Test Booking on NTSA, URL: https://serviceportal.ntsa.go.ke\n"
        "- 'passport_application': New/Renewal Passport Application on Directorate of Immigration Services (DIS), URL: https://immigration.ecitizen.go.ke\n"
        "- 'birth_certificate': Birth Certificate Replacement / Application on Civil Registration Services (CRS), URL: https://accounts.ecitizen.go.ke\n"
        "- 'helb_student_loan': Higher Education Financing / HELB Student Loan, URL: https://hef.co.ke\n"
        "- 'tax_compliance_certificate': KRA Tax Compliance Certificate (TCC), URL: https://itax.kra.go.ke\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "type": "clarification" | "info" | "action_ready",\n'
        '  "message": "<Conversational answer or follow-up question in friendly, professional tone>",\n'
        '  "service_id": "<matched service ID from list above, or empty string>",\n'
        '  "target_portal_url": "<matched URL, or empty string>",\n'
        '  "task_goal": "<clear, actionable directive for browser agent if action_ready, else empty>",\n'
        '  "detected_summary": "<short 2-5 word summary>",\n'
        '  "options": {\n'
        '    "businessName1": "<extracted primary proposed name or empty>",\n'
        '    "businessName2": "<extracted alternative proposed name or empty>",\n'
        '    "phoneNumber": "<extracted phone or empty>",\n'
        '    "selectedCentre": "<extracted Huduma Centre or empty>",\n'
        '    "licenseDuration": "<extracted duration or empty>",\n'
        '    "vehicleClass": "<extracted vehicle class or empty>",\n'
        '    "natureOfBusiness": "<extracted nature of business or empty>"\n'
        "  },\n"
        '  "missing_fields": ["<list of any required fields still missing, e.g. businessName1>"],\n'
        '  "ready_to_launch": <boolean: true only if type is action_ready>,\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reasoning": "<chain-of-thought explanation>"\n'
        "}"
    )

    search_block = f"\n<WEB_SEARCH_RESULTS>\n{web_search_text}\n</WEB_SEARCH_RESULTS>\n" if web_search_text else ""
    user_prompt = f"{history_context}Current User Message: {clean_prompt}\n{search_block}"
    if req.user_context:
        user_prompt += f"\nUser Context: {json.dumps(req.user_context)}"

    # Attempt 1: GPT OSS 120B on Modal
    try:
        import modal
        GptModelCls = modal.Cls.from_name("bruv-schools-llm", "GptOss120bModel")
        gpt_model = GptModelCls()
        raw_res = await gpt_model.generate.remote.aio(
            system_message=system_prompt,
            user_message=user_prompt,
            max_tokens=550
        )
        match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            if web_sources:
                decision["search_performed"] = True
                decision["search_sources"] = web_sources
            return {"status": "success", "source": "gpt_120b", "result": decision}
    except Exception as gpt_err:
        print(f"GPT OSS 120B intent parsing note: {gpt_err}. Trying Qwen fallback...")

    # Attempt 2: QwenSmallModel fallback
    try:
        import modal
        SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
        small_model = SmallModelCls()
        raw_res = await small_model.generate.remote.aio(
            system_message=system_prompt,
            user_message=user_prompt,
            max_tokens=450
        )
        match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            if web_sources:
                decision["search_performed"] = True
                decision["search_sources"] = web_sources
            return {"status": "success", "source": "qwen_small", "result": decision}
    except Exception as qwen_err:
        print(f"Qwen intent parsing fallback note: {qwen_err}")

    # Robust multi-service heuristic fallback resolver
    p = clean_prompt.lower().replace("licence", "license")
    is_q = clean_prompt.endswith("?") or any(w in p for w in ["what", "how", "requirements", "where", "cost", "fee"])

    service_id = ""
    target_portal_url = ""
    service_title = ""
    short_name = ""

    if any(w in p for w in ["pdl", "provisional", "driving test", "test booking"]):
        service_id = "pdl_test_booking"
        target_portal_url = "https://serviceportal.ntsa.go.ke"
        service_title = "Provisional Driving License (PDL) & Test Booking"
        short_name = "PDL & Test Booking"
    elif any(w in p for w in ["driving", "license", "dl", "ntsa", "driver"]):
        service_id = "driving_license_renewal"
        target_portal_url = "https://serviceportal.ntsa.go.ke"
        service_title = "Smart Driving License Renewal / Application"
        short_name = "Driving License"
    elif any(w in p for w in ["good conduct", "police clearance", "dci", "fingerprint", "c24"]):
        service_id = "good_conduct"
        target_portal_url = "https://dci.ecitizen.go.ke"
        service_title = "Police Clearance Certificate (Certificate of Good Conduct)"
        short_name = "Good Conduct"
    elif any(w in p for w in ["business", "company", "sole proprietorship", "bn2", "cr14"]):
        if any(w in p for w in ["search", "check", "reserve", "name"]):
            service_id = "business_name_search"
            service_title = "Business Name Search & Reservation"
            short_name = "Name Search"
        else:
            service_id = "business_name_registration"
            service_title = "Business Name Registration (Sole Proprietorship / BN2)"
            short_name = "Business Registration"
        target_portal_url = "https://brs.ecitizen.go.ke"
    elif any(w in p for w in ["passport", "immigration"]):
        service_id = "passport_application"
        target_portal_url = "https://immigration.ecitizen.go.ke"
        service_title = "Passport Application / Renewal"
        short_name = "Passport"
    elif any(w in p for w in ["helb", "student loan", "hef", "scholarship"]):
        service_id = "helb_student_loan"
        target_portal_url = "https://hef.co.ke"
        service_title = "HELB Student Loan & Scholarship"
        short_name = "HELB / HEF"
    elif any(w in p for w in ["tax", "kra", "tcc", "itax"]):
        service_id = "tax_compliance_certificate"
        target_portal_url = "https://itax.kra.go.ke"
        service_title = "Tax Compliance Certificate (TCC)"
        short_name = "KRA TCC"
    elif any(w in p for w in ["birth", "crs", "born"]):
        service_id = "birth_certificate"
        target_portal_url = "https://accounts.ecitizen.go.ke"
        service_title = "Birth Certificate Application / Replacement"
        short_name = "Birth Certificate"

    # Default options extraction
    fallback_options = {}
    if service_id == "driving_license_renewal":
        if "1 year" in p or "one year" in p or "interim" in p:
            fallback_options["licenseDuration"] = "1-Year Interim License (KES 650)"
        else:
            fallback_options["licenseDuration"] = "3-Year Smart Driving License (KES 3,050)"

    if is_q or not service_id:
        msg = f"eCitizen provides services across BRS, NTSA, DCI, and Immigration. For {service_title or 'government services'}, please let me know what you would like to process." if service_id else "I couldn't identify the specific eCitizen service. Please specify if you want Driving License (NTSA), Police Clearance (DCI), or Business Registration (BRS)."
        return {
            "status": "success",
            "source": "fallback",
            "result": {
                "type": "info" if is_q else "clarification",
                "message": msg,
                "service_id": service_id,
                "target_portal_url": target_portal_url,
                "task_goal": "",
                "detected_summary": short_name or "General Inquiry",
                "options": fallback_options,
                "missing_fields": [] if service_id else ["service"],
                "ready_to_launch": False,
                "confidence": 0.7 if service_id else 0.3,
                "reasoning": "Heuristic fallback response"
            }
        }

    return {
        "status": "success",
        "source": "fallback",
        "result": {
            "type": "action_ready",
            "message": f"I have prepared the plan for {service_title}. Click 'Start Autopilot' to launch.",
            "service_id": service_id,
            "target_portal_url": target_portal_url,
            "task_goal": f"Navigate and process {service_title} on eCitizen",
            "detected_summary": f"Target: {short_name}",
            "options": fallback_options,
            "missing_fields": [],
            "ready_to_launch": True,
            "confidence": 0.85,
            "reasoning": f"Matched intent keywords for {service_title}"
        }
    }

class EcitizenElement(BaseModel):
    id: int
    tag: str
    text: Optional[str] = ""
    aria: Optional[str] = ""
    type: Optional[str] = ""
    name: Optional[str] = ""
    placeholder: Optional[str] = ""
    role: Optional[str] = ""
    href: Optional[str] = ""
    container: Optional[str] = ""

class EcitizenActionRequest(BaseModel):
    url: str
    page_title: Optional[str] = ""
    task_goal: str
    step_title: Optional[str] = "Autonomous Navigation"
    step_description: Optional[str] = ""
    elements: List[EcitizenElement]
    form_values: Optional[Dict[str, Any]] = None
    action_history: Optional[List[Dict[str, Any]]] = None
    last_error: Optional[str] = None
    model_name: Optional[str] = "gpt_120b"

@web_app.post("/ecitizen/decide-action")
async def ecitizen_decide_action_endpoint(req: EcitizenActionRequest):
    """
    Autonomous DOM Vision & Reasoning endpoint for eCitizen government portal navigation.
    Uses GPT OSS 120B to decide which element to click, fill, or select based on live DOM perception.
    """
    import json
    import re

    # Build structured Accessibility Tree summary for the prompt with container context
    elements_summary = []
    for el in req.elements[:120]:
        role_str = el.role if el.role else el.tag
        desc = f"[{el.id}] role=\"{role_str}\" <{el.tag}>"
        if el.name:
            name_clean = " ".join(el.name.split())[:80]
            desc += f' name="{name_clean}"'
        if el.text and el.text.strip() != (el.name or '').strip():
            cleaned_text = " ".join(el.text.split())[:60]
            desc += f' text="{cleaned_text}"'
        if el.placeholder:
            desc += f' placeholder="{el.placeholder}"'
        if el.type and el.type not in ('text', ''):
            desc += f' type="{el.type}"'
        if el.href:
            desc += f' href="{el.href}"'
        if el.container:
            desc += f' | In: {el.container}'
        elements_summary.append(desc)

    elements_text = "\n".join(elements_summary) if elements_summary else "No accessible elements found."
    form_text = json.dumps(req.form_values or {}, indent=2)

    history_lines = []
    if req.action_history:
        for h in req.action_history[-6:]:
            history_lines.append(f"- Turn {h.get('turn', '?')}: {str(h.get('action', '')).upper()} [{h.get('element', '')}] -> Status: {h.get('status', 'executed')} ({h.get('reason', '')})")
    history_text = "\n".join(history_lines) if history_lines else "None yet. Starting fresh from current page."

    system_prompt = (
        "You are an autonomous AI browser agent navigating Kenyan eCitizen government portals (BRS, NTSA, DCI).\n"
        "You analyze the live DOM & Accessibility Tree (roles, accessible names, containers) and decide the exact next action to make progress toward the user's task goal.\n\n"
        "CRITICAL RULES:\n"
        "- Every element in the tree has a W3C accessible role (button, link, textbox, combobox, etc.) and an accessible name.\n"
        "- ONLY pick elements with role='button' or role='link' for click actions. NEVER pick a div, span, p, h1-h6, or any container element for clicking.\n"
        "- Each element has a container context (e.g. 'In: Modal: MANUAL SERVICES' or 'In: Card: Business Registration Service') telling you where it lives in the page hierarchy.\n"
        "- Use the container context to disambiguate: if the goal is Business Registration, click the button INSIDE the card for Business Registration, not a button inside Manual Services.\n\n"
        "Reasoning Guidelines:\n"
        "1. Prioritize elements matching the task goal (e.g. for Business Name Search, target 'Business Registration Service' or 'Name Search', NOT unrelated services like 'Manual Services' or 'Insolvency').\n"
        "2. If an unwanted popup, public notice, or modal dialog is blocking the view, dismiss it by clicking its close button (role='button' with name containing 'Close', '×', or 'Dismiss').\n"
        "3. If input fields (role='textbox') for the target form are visible, choose action 'fill' with the appropriate value from Available Form Data.\n"
        "4. If a submit/continue/proceed/next button is visible after filling fields, choose action 'click' on that button.\n"
        "5. If on a login or SMS OTP verification screen that requires user credentials, choose action 'wait'.\n"
        "6. If reached a statutory fee / payment screen, choose action 'wait' for payment confirmation.\n"
        "7. If an [ALERT / RE-INSPECTION] notice is present, the previous click FAILED. Do NOT repeat the same element_id. Choose a DIFFERENT actionable control.\n"
        "8. If the current URL is completely off-target (e.g. still on landing portal when target is BRS), you can use action 'navigate' with target_url='https://brs.ecitizen.go.ke'.\n"
        "9. Agency SSO Landing Screens (e.g. NTSA serviceportal.ntsa.go.ke): If you are on an agency landing page with a 'Sign in' button and text stating 'Use your Ecitizen login credentials', choose action 'click' on the 'Sign in' button to initiate the SSO handshake into the agency portal.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "thought": "<your chain-of-thought reasoning analyzing current page context and previous actions>",\n'
        '  "action": "click" | "fill" | "select" | "navigate" | "wait" | "done",\n'
        '  "element_id": <int: id of the chosen element, or null if wait/navigate/done>,\n'
        '  "fill_value": "<string to fill, or null>",\n'
        '  "target_url": "<url string to navigate to, or null>",\n'
        '  "milestone": "<short 2-4 word status, e.g. Dismiss Notice, Select BRS, Enter Name, Payment Checkpoint>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<short user-facing explanation of what you are doing>"\n'
        "}"
    )

    alert_text = f"\n[ALERT / RE-INSPECTION]: {req.last_error}\n" if req.last_error else ""

    user_prompt = (
        f"Task Goal: {req.task_goal}\n"
        f"Current Page URL: {req.url}\n"
        f"Page Title: {req.page_title or 'eCitizen Portal'}\n"
        f"{alert_text}\n"
        f"Recent Action History:\n{history_text}\n\n"
        f"Available Form Data to Fill:\n{form_text}\n\n"
        f"Live DOM & Accessibility Tree Nodes with Containers:\n{elements_text}\n\n"
        "Analyze the accessibility tree, review your history, and decide the exact next action to make progress toward the goal."
    )

    # First attempt: GPT OSS 120B on Modal
    try:
        import modal
        GptModelCls = modal.Cls.from_name("bruv-schools-llm", "GptOss120bModel")
        gpt_model = GptModelCls()
        raw_res = await gpt_model.generate.remote.aio(
            system_message=system_prompt,
            user_message=user_prompt,
            max_tokens=550
        )
        match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            return {"status": "success", "source": "gpt_120b", "decision": decision}
    except Exception as gpt_err:
        print(f"GPT OSS 120B call note: {gpt_err}. Trying Qwen fallback...")

    # Second attempt: QwenSmallModel fallback
    try:
        import modal
        SmallModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
        small_model = SmallModelCls()
        raw_res = await small_model.generate.remote.aio(
            system_message=system_prompt,
            user_message=user_prompt,
            max_tokens=400
        )
        match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            return {"status": "success", "source": "qwen_small", "decision": decision}
    except Exception as qwen_err:
        print(f"Qwen fallback note: {qwen_err}")

    # Heuristic fallback if LLM container is spinning up
    if "serviceportal.ntsa" in req.url or "ntsa.go.ke" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if any(w in combined for w in ["sign in", "signin", "log in", "login"]):
                return {
                    "status": "success",
                    "source": "heuristic_ntsa_sso",
                    "decision": {
                        "thought": "Detected NTSA landing page. Clicking 'Sign in' to initiate eCitizen SSO handshake.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "NTSA SSO Login",
                        "reason": "Clicking 'Sign in' to authenticate via eCitizen SSO"
                    }
                }

    for el in req.elements:
        t = (el.text or "").lower()
        if any(w in t for w in ["search", "business", "application", "continue", "next", "submit", "apply", "brs"]):
            return {
                "status": "success",
                "source": "heuristic_fallback",
                "decision": {
                    "action": "click",
                    "element_id": el.id,
                    "confidence": 0.8,
                    "reason": f"Matched action intent keyword on '{el.text}'"
                }
            }

    return {
        "status": "partial",
        "source": "fallback",
        "decision": {
            "action": "wait",
            "element_id": None,
            "confidence": 0.5,
            "reason": "Waiting for page elements to stabilize"
        }
    }

@app.function(image=image, min_containers=1, volumes={"/data": rag_volume})
@modal.asgi_app()
def fastapi_app():
    # Mount Browser Autopilot & Mock Portal Router inside container
    try:
        from agent.autopilot_server import app as autopilot_app
        web_app.include_router(autopilot_app.router)
    except Exception as e:
        print(f"Container Autopilot router inclusion note: {e}")
    return web_app
