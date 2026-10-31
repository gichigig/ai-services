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

    # 1. Tier 1: Known Core Service Identification (Instant Fast Path)
    core_service_keywords = [
        "driving license", "driving licence", "driver license", "driver licence",
        "renew dl", "pay dl", "pdl", "test booking", "good conduct", "police clearance",
        "dci", "fingerprint", "name search", "business name", "register business",
        "passport", "immigration", "birth certificate", "helb", "student loan",
        "tax compliance", "tcc", "kra pin"
    ]
    is_known_core = any(k in p_lower for k in core_service_keywords)

    # 2. Tier 2: Conversational Follow-Up State Bypass
    followup_keywords = [
        "start", "launch", "proceed", "yes", "run", "do it", "approved", "confirm",
        "my number is", "phone number", "nairobi", "mombasa", "kisumu", "huduma",
        "go ahead", "begin", "okay", "ok"
    ]
    is_followup = any(k in p_lower for k in followup_keywords) and len(clean_prompt.split()) < 10

    # 3. Tier 3: Search Triggers (Informational or Long-Tail Service Discovery)
    info_keywords = [
        "how much", "what is the fee", "what are the fees", "how long", "requirements", 
        "what documents", "cost", "tariff", "is it possible", "can i", "check online", 
        "search web", "google", "where", "where do i", "which portal", "how do i", 
        "how to apply", "steps to", "location", "huduma", "scope", "plan"
    ]
    is_info = any(k in p_lower for k in info_keywords) or clean_prompt.endswith("?")

    # Dynamic Discovery Rule:
    # Trigger live web search if user asks an informational/fee/portal question,
    # OR requests a government service outside the 9 core services, provided it's not a short follow-up.
    needs_dynamic_search = (is_info or (not is_known_core and not is_followup)) and len(clean_prompt.strip()) > 3

    web_search_text = ""
    web_sources = []
    if needs_dynamic_search:
        try:
            from llm_router import execute_web_search
            clean_q = re.sub(r'^(check|search|verify|find|tell me|google|i want to|how to|apply for|pay for)\s+', '', clean_prompt, flags=re.IGNORECASE).strip()
            search_query = f'site:ecitizen.go.ke OR site:go.ke "{clean_q}" portal requirements fee'
            print(f"[EcitizenDiscovery] Executing targeted discovery search: '{search_query}'")
            raw_text, sources = await execute_web_search(search_query)
            if not raw_text or len(raw_text.strip()) < 50:
                fallback_query = f'{clean_q} eCitizen Kenya government portal requirements fee'
                print(f"[EcitizenDiscovery] Executing broader discovery search: '{fallback_query}'")
                raw_text, sources = await execute_web_search(fallback_query)

            if raw_text:
                web_search_text = raw_text[:2800]
                web_sources = sources[:5] if sources else []
        except Exception as search_err:
            print(f"[EcitizenDiscovery] Web search note: {search_err}")

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
        "You are an expert conversational AI Copilot for Kenyan eCitizen government portals.\n"
        "You have universal dynamic coverage over all 5,000+ digitized government services across ministries, departments, and agencies.\n"
        "When real-time verified search results are provided in <WEB_SEARCH_RESULTS>:\n"
        "- Discover the exact official government portal URL (must end with .go.ke or contain ecitizen.go.ke). If specific subdomain is unknown, default to 'https://accounts.ecitizen.go.ke'.\n"
        "- Extract the official statutory fee (KES), responsible agency/department, and required documents/prerequisites.\n"
        "- When the user wants to execute a government service outside the 9 core services, synthesize a dynamic 5-milestone service DAG in 'dynamic_service'.\n\n"
        "Decision Modes (Field 'type'):\n"
        "1. 'info': Use when the user is asking an informational question about requirements, fees, rules, or portal locations. Answer thoroughly in 'message' with source citations. Do NOT launch the browser (ready_to_launch=false).\n"
        "2. 'clarification': Use when the user wants to execute a service but has NOT provided necessary details (e.g. proposed name, national ID, preferred station). Ask clear follow-up questions in 'message' (ready_to_launch=false).\n"
        "3. 'action_ready': Use ONLY when you have sufficient details or the user commanded 'proceed' / 'start' / 'run' to launch browser automation on eCitizen. Summarize in 'message' and set ready_to_launch=true.\n\n"
        "Known Government Core Services:\n"
        "- 'business_name_search': Business Name Search & Reservation on Business Registration Service (BRS), URL: https://brs.ecitizen.go.ke\n"
        "- 'business_name_registration': Business Name Registration (Sole Proprietorship / BN2) on BRS, URL: https://brs.ecitizen.go.ke\n"
        "- 'good_conduct': Police Clearance Certificate on Directorate of Criminal Investigations (DCI), URL: https://dci.ecitizen.go.ke\n"
        "- 'driving_license_renewal': 3-Year Smart Driving License Renewal or Driving Licence / DL Application/Payment on National Transport and Safety Authority (NTSA). (Matches 'driving license', 'driving licence', 'pay for driving licence', 'smart dl', 'renew driving'). URL: https://serviceportal.ntsa.go.ke\n"
        "- 'pdl_test_booking': Provisional Driving License / PDL & Test Booking on NTSA, URL: https://serviceportal.ntsa.go.ke\n"
        "- 'passport_application': New/Renewal Passport Application on Directorate of Immigration Services (DIS), URL: https://immigration.ecitizen.go.ke\n"
        "- 'birth_certificate': Birth Certificate Replacement / Application on Civil Registration Services (CRS), URL: https://accounts.ecitizen.go.ke\n"
        "- 'helb_student_loan': Higher Education Financing / HELB Student Loan, URL: https://hef.co.ke\n"
        "- 'tax_compliance_certificate': KRA Tax Compliance Certificate (TCC), URL: https://itax.kra.go.ke\n\n"
        "For any OTHER service discovered via search (e.g. Marriage Certificate, NEMA Waste Permit, Mining License, Nairobi County Business Permit), generate 'dynamic_service' and an official .go.ke target_portal_url.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "type": "clarification" | "info" | "action_ready",\n'
        '  "message": "<Conversational answer or follow-up question in friendly, professional tone>",\n'
        '  "service_id": "<core service ID or dynamic slug like marriage_certificate>",\n'
        '  "service_title": "<Full official service title>",\n'
        '  "department": "<Government Agency or Ministry>",\n'
        '  "target_portal_url": "<Official .go.ke or ecitizen.go.ke URL>",\n'
        '  "fee_kes": "<Statutory fee, e.g. KES 3,900>",\n'
        '  "task_goal": "<clear, actionable directive for browser agent>",\n'
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
        '  "dynamic_service": {\n'
        '    "title": "<Discovered title>",\n'
        '    "shortName": "<Short name>",\n'
        '    "department": "<Discovered agency>",\n'
        '    "portalUrl": "<Discovered official URL>",\n'
        '    "feeKes": "<Statutory fee>",\n'
        '    "requirements": ["<req 1>", "<req 2>"]\n'
        '  },\n'
        '  "missing_fields": ["<list of any required fields still missing>"],\n'
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
    screenshot_b64: Optional[str] = None
    viewport: Optional[Dict[str, Any]] = None

@web_app.post("/ecitizen/decide-action")
async def ecitizen_decide_action_endpoint(req: EcitizenActionRequest):
    """
    Autonomous DOM Vision & Reasoning endpoint for eCitizen government portal navigation.
    Uses GPT OSS 120B to decide which element to click, fill, or select based on live DOM perception.
    """
    import json
    import re

    print(f"[DecideAction INCOMING] URL={req.url} | ElementsCount={len(req.elements)} | Goal='{req.task_goal}'")

    def extract_json_object(text: str):
        if not text:
            return None
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'```\s*', '', cleaned).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        start = cleaned.find('{')
        if start != -1:
            brace_count = 0
            in_string = False
            escape = False
            for i in range(start, len(cleaned)):
                c = cleaned[i]
                if escape:
                    escape = False
                    continue
                if c == '\\':
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            candidate = cleaned[start:i+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                fixed = re.sub(r',\s*([\}\]])', r'\1', candidate)
                                try:
                                    return json.loads(fixed)
                                except Exception:
                                    pass
                            break
        return None

    # Guard against completely empty element lists while page is still loading
    if not req.elements:
        print(f"[DecideAction RETRY] No interactive elements present on {req.url}. Returning transient wait state.")
        return {
            "status": "success",
            "source": "empty_elements_guard",
            "decision": {
                "thought": "Page DOM is still loading or no interactive elements have rendered yet.",
                "action": "wait",
                "is_transient": True,
                "confidence": 0.9,
                "reason": "Waiting for page elements to render..."
            }
        }

    # Guard: Never attempt automated clicks/fills on central login/authentication pages
    url_lower = (req.url or "").lower()
    page_title_lower = (req.page_title or "").lower()
    is_agency = any(a in url_lower for a in ["brs.", "brsv2.", "brs", "dci.", "ntsa.", "kra.", "immigration.", "hef.co.ke"])
    
    is_auth_page = not is_agency and "/account-switcher" not in url_lower and "/authorize" not in url_lower and (
        "accounts.ecitizen.go.ke" in url_lower or
        "/login" in url_lower or
        "/signin" in url_lower or
        "/otp" in url_lower or
        "sign in" in page_title_lower or
        "one log in" in page_title_lower or
        "ecitizen login" in page_title_lower or
        any((el.type == "password" or "password" in (el.name or "").lower()) for el in req.elements)
    )

    if is_auth_page:
        print(f"[DecideAction AUTH GUARD] Detected login screen on {req.url}. Returning human wait checkpoint.")
        return {
            "status": "success",
            "source": "auth_guard",
            "decision": {
                "thought": "The browser is currently on the eCitizen authentication screen. Requiring user to sign in manually.",
                "action": "wait",
                "is_transient": False,
                "element_id": None,
                "milestone": "Awaiting User Sign-In",
                "confidence": 1.0,
                "reason": "Please enter your eCitizen email/ID and password manually. Autopilot will automatically resume once logged in."
            }
        }

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
        "ECITIZEN AI BROWSER AGENT — MASTER SYSTEM INSTRUCTIONS\n\n"
        "STRICT ANTI-HALLUCINATION AND BROWSER CONTROL POLICY:\n\n"
        "EVIDENCE REQUIREMENT:\n"
        "Every browser action must be supported by evidence obtained from the latest browser inspection.\n"
        "Never invent:\n"
        "- Element IDs\n"
        "- Button names\n"
        "- CSS selectors\n"
        "- URLs\n"
        "- Form fields\n"
        "- Page content\n"
        "- Navigation paths\n\n"
        "TARGET IDENTIFICATION:\n"
        "Before requesting a click:\n"
        "1. Obtain a current page snapshot.\n"
        "2. Identify the intended element from the provided Accessibility Tree.\n"
        "3. Match its role and accessible name.\n"
        "4. Verify its surrounding context and container.\n"
        "5. Confirm that it is unique.\n"
        "6. Confirm that it is visible and enabled.\n"
        "7. Confirm that its snapshot is current.\n"
        "8. Provide the exact target ID returned by the browser extension.\n"
        "If any check fails, reject the action.\n\n"
        "NO-GUESS POLICY:\n"
        "Never:\n"
        "- Click based on assumptions.\n"
        "- Click based only on approximate screen coordinates.\n"
        "- Select the first matching button without checking context.\n"
        "- Reuse outdated element IDs.\n"
        "- Invent selectors when an element cannot be found.\n"
        "- Substitute a different button when the intended button is unavailable.\n"
        "- Repeat a failed action without reinspection.\n\n"
        "ACTION AUTHORIZATION:\n"
        "The browser controller is responsible for deciding whether an action is executable.\n"
        "You may propose actions but cannot override browser-controller restrictions.\n"
        "You cannot execute:\n"
        "- Unapproved submissions.\n"
        "- Unapproved payments.\n"
        "- Unauthorized changes.\n"
        "- Arbitrary JavaScript.\n"
        "- Actions outside permitted website boundaries.\n\n"
        "UNCERTAINTY HANDLING:\n"
        "When evidence is insufficient:\n"
        "- Request another page inspection (action='wait' with is_transient=true).\n"
        "- Reassess the current page.\n"
        "- Ask the user to clarify if necessary.\n"
        "- Stop if the target remains ambiguous.\n"
        "Never convert uncertainty into an assumption.\n\n"
        "PAGE STATE MANAGEMENT & VERIFICATION:\n"
        "- Every action must reference a valid page snapshot.\n"
        "- After navigation, significant DOM changes, modal changes or unexpected page updates, request a fresh snapshot.\n"
        "- Never assume that a previous page's elements remain available.\n"
        "- After every action, obtain new browser evidence. A click event does not establish successful execution.\n"
        "- Only report success when the expected outcome is independently verified.\n\n"
        "FAILURE POLICY:\n"
        "When validation fails:\n"
        "1. Do not execute.\n"
        "2. Record the failure reason.\n"
        "3. Refresh page information if appropriate.\n"
        "4. Retry only when the failure has been understood.\n"
        "5. Stop after two unsuccessful recovery attempts.\n\n"
        "AUTHENTICATION & HUMAN APPROVAL:\n"
        "- The user must log in manually. Never request or store passwords, OTPs or PINs.\n"
        "- Pause on login/OTP screens. Never bypass authentication challenges.\n"
        "- Always require explicit user confirmation (requires_user_approval=true) before: Final application submission, Government fee payments, Legal declarations, Signing agreements, Cancelling/withdrawing applications, Changing important personal info, or Irreversible operations.\n\n"
        "HUMAN-IN-THE-LOOP OPERATING POLICY:\n"
        "WHEN TO REQUEST HUMAN INTERACTION:\n"
        "Request human intervention (action='wait' with is_transient=false) whenever:\n"
        "1. Manual login is required.\n"
        "2. OTP verification is required.\n"
        "3. CAPTCHA or Cloudflare challenge appears.\n"
        "4. The website requires identity verification.\n"
        "5. Required form information is unavailable in Available Form Data.\n"
        "6. A document needs to be manually selected or uploaded.\n"
        "7. The browser presents an unexpected security challenge.\n"
        "8. A sensitive operation requires user approval.\n"
        "9. The intended browser action cannot be identified confidently.\n\n"
        "HANDOFF & WAITING RULES:\n"
        "When human intervention is required, choose action 'wait'. While waiting:\n"
        "- Do not click buttons or fill forms.\n"
        "- Do not navigate automatically or interfere with manual user interaction.\n"
        "- Do not repeatedly refresh the website.\n"
        "- Do not assume that inactivity means completion.\n"
        "- Do not continue merely because time has passed.\n"
        "- Explain exactly what the user must do in 'reason' and 'user_action_required'.\n\n"
        "VERIFICATION & RESUMPTION RULES:\n"
        "- Human confirmation initiates verification. Successful verification authorizes resumption. Never confuse the two.\n"
        "- For login, verify the authenticated session.\n"
        "- For uploads, verify that the intended file was selected.\n"
        "- For payments, verify the authoritative payment status on page.\n"
        "- For application submissions, verify the official submission confirmation.\n"
        "- Resume only when the required condition has been independently verified.\n\n"
        "VISUAL RECOGNITION AND SCREENSHOT RECOVERY POLICY:\n"
        "1. DOM and accessibility inspection is your primary method of identifying browser controls.\n"
        "2. Visual recognition assists when:\n"
        "   - A target cannot be identified through DOM inspection.\n"
        "   - The page contains visually complex layouts.\n"
        "   - A menu, modal, dropdown or overlay is difficult to interpret.\n"
        "   - The page layout differs from previously observed layouts.\n"
        "   - An unexpected page state is encountered.\n"
        "3. When visual targets are identified, provide the visual target description, approximate location, element type, visible label, and surrounding context.\n"
        "4. Distinguish between observed information and assumptions. Never invent a button, link, field, or page element.\n"
        "5. All visual targets must be cross-checked against the current DOM whenever possible.\n"
        "6. Coordinates must account for viewport dimensions, scrolling, browser zoom and device pixel ratio. Stale screenshot coordinates must never be reused.\n"
        "7. If a visual target cannot be matched to a valid DOM element, do NOT execute the action automatically unless authorized and validated.\n"
        "8. The vision model must NEVER independently authorize: fee payments, final submissions, legal declarations, signing, or cancellations. These require explicit human approval (requires_user_approval=true) even with high confidence.\n"
        "9. If DOM inspection and visual recognition both fail, STOP and request human assistance (action='wait').\n"
        "10. Every visual action must be followed by a fresh snapshot to verify the resulting page state.\n"
        "11. Privacy safeguards: sensitive personal information must not be unnecessarily transmitted or retained.\n"
        "CORE RULE: Vision assists identification; the validation engine controls execution.\n\n"
        "FORM FILLING:\n"
        "- Use ONLY user-provided or authorized information from Available Form Data.\n"
        "- NEVER fabricate personal information, national IDs, names, or addresses.\n"
        "- Verify field labels before entering data. Respect required fields.\n"
        "- If required form information is missing from Available Form Data, choose action 'wait' and ask user.\n\n"
        "FINAL PRINCIPLE:\n"
        "NO EVIDENCE = NO ACTION.\n"
        "NO VALIDATION = NO CLICK.\n"
        "NO VERIFICATION = NO SUCCESS CLAIM.\n"
        "WHEN UNCERTAIN = STOP AND ASK.\n"
        "HUMAN CONFIRMATION INITIATES VERIFICATION.\n"
        "SUCCESSFUL VERIFICATION AUTHORIZES RESUMPTION.\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "thought": "<detailed chain-of-thought analyzing URL, page title, headings, accessibility tree, previous history, and safety constraints>",\n'
        '  "action": "click" | "fill" | "select" | "navigate" | "wait" | "done",\n'
        '  "element_id": <int: exact id of chosen element from tree, or null if wait/navigate/done>,\n'
        '  "fill_value": "<string to fill, or null>",\n'
        '  "target_url": "<url string to navigate to, or null>",\n'
        '  "milestone": "<short 2-4 word status, e.g. Ownership Details, Select BRS, Enter Name, Payment Checkpoint>",\n'
        '  "requires_user_approval": <boolean: true if this action is payment, final submission, legal declaration, or cancellation>,\n'
        '  "visual_target": "<description of visual target if vision was used, or null>",\n'
        '  "visual_coordinates": <{"x": int, "y": int} or null>,\n'
        '  "handoff_trigger": "<login | otp | captcha | file_upload | payment | approval | missing_info | null>",\n'
        '  "user_action_required": "<clear instructions for the user if handoff is needed, or null>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<clear explanation adhering to the master operating principles>"\n'
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

    # Strict Anti-Hallucination Evidence Gate
    valid_ids = {el.id for el in req.elements if el.id is not None}

    def validate_anti_hallucination(decision: dict):
        if not decision or not isinstance(decision, dict):
            return None
        act = decision.get("action")
        el_id = decision.get("element_id")
        if act in ("click", "fill", "select") and el_id is not None:
            if el_id not in valid_ids:
                print(f"[DecideAction ANTI-HALLUCINATION REJECTION] Model hallucinated element_id {el_id} (not in {len(valid_ids)} verified elements). Enforcing NO EVIDENCE = NO ACTION.")
                return {
                    "status": "rejected",
                    "source": "anti_hallucination_guard",
                    "decision": {
                        "action": "wait",
                        "is_transient": True,
                        "element_id": None,
                        "milestone": "Re-inspecting Page",
                        "thought": f"The proposed target element #{el_id} does not exist in the current page inspection snapshot. Enforcing Strict Anti-Hallucination Policy.",
                        "reason": f"Target element #{el_id} was not verified in latest browser inspection. Requesting fresh page snapshot."
                    }
                }
        return None

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
        decision = extract_json_object(raw_res)
        if decision and isinstance(decision, dict) and "action" in decision:
            rejection = validate_anti_hallucination(decision)
            if rejection:
                return rejection
            print(f"[DecideAction SUCCESS] Source=gpt_120b | Action={decision.get('action')} | Element={decision.get('element_id')} | Reason={decision.get('reason')}")
            return {"status": "success", "source": "gpt_120b", "decision": decision}
        else:
            print(f"[DecideAction PARSE NOTE] GPT 120B output not clean JSON dict: {str(raw_res)[:180]}. Trying Qwen fallback...")
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
        decision = extract_json_object(raw_res)
        if decision and isinstance(decision, dict) and "action" in decision:
            rejection = validate_anti_hallucination(decision)
            if rejection:
                return rejection
            print(f"[DecideAction SUCCESS] Source=qwen_small | Action={decision.get('action')} | Element={decision.get('element_id')} | Reason={decision.get('reason')}")
            return {"status": "success", "source": "qwen_small", "decision": decision}
        else:
            print(f"[DecideAction PARSE NOTE] Qwen output not clean JSON dict: {str(raw_res)[:180]}")
    except Exception as qwen_err:
        print(f"Qwen fallback note: {qwen_err}")

    # Heuristic fallback if LLM container is spinning up
    if "serviceportal.ntsa" in req.url or "ntsa.go.ke" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if any(w in combined for w in ["sign in", "signin", "log in", "login"]):
                print(f"[DecideAction HEURISTIC] NTSA SSO sign-in on element {el.id}")
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

    if "brsv2.ecitizen.go.ke" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if "login with ecitizen" in combined or ("login" in combined and "ecitizen" in combined):
                print(f"[DecideAction HEURISTIC] BRS V2 SSO sign-in on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_brsv2_sso",
                    "decision": {
                        "thought": "Detected BRS V2 SSO gateway. Clicking 'Login with eCitizen' to perform single sign-on handshake.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "BRS V2 SSO Login",
                        "reason": "Clicking 'Login with eCitizen' to authenticate via active eCitizen session"
                    }
                }

    if "account-switcher" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if "personal" in combined or "personal" in container or any(k in combined for k in ["42761734", "proceed to your personal account", "bildad"]):
                print(f"[DecideAction HEURISTIC] Account Switcher personal account selection on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_account_switcher",
                    "decision": {
                        "thought": "Detected eCitizen account switcher. Selecting 'Personal Account' to complete OAuth handshake and proceed to BRS V2.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Select Personal Account",
                        "reason": "Selecting personal account to complete login into BRS V2"
                    }
                }

    if "accounts.ecitizen.go.ke" in req.url and "/authorize" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if "proceed" in combined:
                print(f"[DecideAction HEURISTIC] OAuth consent 'Proceed' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_oauth_proceed",
                    "decision": {
                        "thought": "Detected eCitizen OAuth authorization consent screen. Clicking 'Proceed' to complete authorization handshake.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "OAuth Consent",
                        "reason": "Clicking 'Proceed' to authorize single sign-on into BRS V2"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and "/switcher" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if "proceed" in combined and ("companies registry" in container or "companies" in container or "responsible for registering" in container):
                print(f"[DecideAction HEURISTIC] BRS V2 Switcher Companies Registry selection on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_companies_registry",
                    "decision": {
                        "thought": "Detected BRS V2 Registry Switcher. Clicking 'Proceed' under 'Companies Registry Services' for business name registration.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Companies Registry",
                        "reason": "Selecting Companies Registry Services to register business name"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and "/registries/brs/dashboard" in req.url:
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if "apply" in combined and ("business name" in container or "business name" in combined or "registration of a business name" in container):
                print(f"[DecideAction HEURISTIC] BRS V2 Business Name Registration on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_register_business_name",
                    "decision": {
                        "thought": "Detected BRS V2 dashboard. Clicking 'Apply' under 'Registration Of A Business Name' (runner will auto-scroll into view).",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Apply Business Name",
                        "reason": "Clicking 'Apply' to initiate Registration of a Business Name"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and "/apply/19" in req.url:
        # Check for unfilled Proposed Names input
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            if el.tag == "input" and ("preferred name" in combined or "proposed name" in combined):
                val = (req.form_values or {}).get("proposed_name") or (req.form_values or {}).get("businessName")
                if val:
                    print(f"[DecideAction HEURISTIC] Fill proposed business name on element {el.id}: {val}")
                    return {
                        "status": "success",
                        "source": "heuristic_fill_proposed_name",
                        "decision": {
                            "thought": f"Filling proposed business name '{val}' in priority order.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": val,
                            "confidence": 0.99,
                            "milestone": "Proposed Business Name",
                            "reason": f"Entering primary proposed business name: {val}"
                        }
                    }
        # Check for Affiliated to an Existing Entity dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("affiliated" in combined or "affiliated" in container) and ("choose" in combined or el.tag == "select"):
                affil_val = (req.form_values or {}).get("isAffiliated", "No")
                print(f"[DecideAction HEURISTIC] Select affiliation '{affil_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_select_affiliation",
                    "decision": {
                        "thought": f"Selecting '{affil_val}' for Affiliated to an Existing Entity.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": affil_val,
                        "confidence": 0.99,
                        "milestone": "Affiliation Dropdown",
                        "reason": f"Setting entity affiliation to '{affil_val}'"
                    }
                }
        # Once filled, click Next
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if el.tag == "button" and "next" in combined:
                print(f"[DecideAction HEURISTIC] Click 'Next' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_proposed_names_next",
                    "decision": {
                        "thought": "All proposed name details are complete. Clicking 'Next' to proceed to Applicant Details.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Proceed to Applicant Details",
                        "reason": "Clicking 'Next' to advance to Applicant Details step"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and ("step=1" in req.url or "applicant details" in (req.page_title or "").lower()):
        # Check for Capacity dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("capacity" in combined or "capacity" in container) and ("choose" in combined or el.tag == "select"):
                raw_cap = (req.form_values or {}).get("capacity", "Proprietor")
                cap_val = "Proprietor" if str(raw_cap).lower() in ("owner", "self", "proprietor", "") else str(raw_cap)
                print(f"[DecideAction HEURISTIC] Select capacity '{cap_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_applicant_capacity",
                    "decision": {
                        "thought": f"Selecting '{cap_val}' for Applicant Capacity.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": cap_val,
                        "confidence": 0.99,
                        "milestone": "Applicant Capacity",
                        "reason": f"Setting capacity to '{cap_val}'"
                    }
                }
        # Check for Address textbox
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("address" in combined or "address" in container):
                addr_val = (req.form_values or {}).get("postalAddress") or (req.form_values or {}).get("address") or "P.O. Box 100-00100 Nairobi"
                print(f"[DecideAction HEURISTIC] Fill address on element {el.id}: {addr_val}")
                return {
                    "status": "success",
                    "source": "heuristic_applicant_address",
                    "decision": {
                        "thought": f"Filling postal address '{addr_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": addr_val,
                        "confidence": 0.99,
                        "milestone": "Applicant Address",
                        "reason": f"Entering applicant postal address: {addr_val}"
                    }
                }
        # Check for Postal Code
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("postal code" in combined or "postal code" in container) and ("choose" in combined or el.tag in ("select", "input")):
                raw_pc = (req.form_values or {}).get("postalCode", "00100 - G.P.O NAIROBI")
                pc_val = "00100 - G.P.O NAIROBI" if "00100" in str(raw_pc) else str(raw_pc)
                print(f"[DecideAction HEURISTIC] Select postal code '{pc_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_applicant_postal_code",
                    "decision": {
                        "thought": f"Setting postal code '{pc_val}'.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": pc_val,
                        "confidence": 0.99,
                        "milestone": "Applicant Postal Code",
                        "reason": f"Setting postal code to '{pc_val}'"
                    }
                }
        # Click Next
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if el.tag == "button" and "next" in combined:
                print(f"[DecideAction HEURISTIC] Click 'Next' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_applicant_details_next",
                    "decision": {
                        "thought": "Applicant details are complete. Clicking 'Next' to advance to Nature Of Business.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Proceed to Nature of Business",
                        "reason": "Clicking 'Next' to proceed to Nature of Business step"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and ("step=2" in req.url or "nature of business" in (req.page_title or "").lower()):
        # Check for Nature of Business dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("nature of business" in combined or "nature of business" in container) and ("choose" in combined or el.tag == "select"):
                nature_val = (req.form_values or {}).get("natureOfBusiness", "OTHER")
                print(f"[DecideAction HEURISTIC] Select Nature of Business '{nature_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_nature_of_business_select",
                    "decision": {
                        "thought": f"Selecting '{nature_val}' for Nature of Business.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": nature_val,
                        "confidence": 0.99,
                        "milestone": "Nature of Business Selection",
                        "reason": f"Setting nature of business to '{nature_val}'"
                    }
                }
        # Check for Describe Nature of Business textarea
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("textarea", "input") and ("describe" in combined or "describe" in container or "nature" in combined):
                desc_val = (req.form_values or {}).get("businessDescription") or (req.form_values or {}).get("natureDescription") or "General supplies, retail trading, and related commercial services."
                print(f"[DecideAction HEURISTIC] Fill business description on element {el.id}: {desc_val}")
                return {
                    "status": "success",
                    "source": "heuristic_nature_of_business_desc",
                    "decision": {
                        "thought": f"Describing nature of business: '{desc_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": desc_val,
                        "confidence": 0.99,
                        "milestone": "Nature of Business Description",
                        "reason": f"Entering nature of business description: {desc_val}"
                    }
                }
        # Click Next
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if el.tag == "button" and "next" in combined:
                print(f"[DecideAction HEURISTIC] Click 'Next' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_nature_of_business_next",
                    "decision": {
                        "thought": "Nature of business is complete. Clicking 'Next' to proceed to Registered Office Address.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Proceed to Registered Office Address",
                        "reason": "Clicking 'Next' to advance to Registered Office Address step"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and ("step=3" in req.url or "primary address" in (req.page_title or "").lower() or "registered office" in (req.page_title or "").lower()):
        # Check County dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("county" in combined or "county" in container) and ("choose" in combined or el.tag == "select"):
                c_val = (req.form_values or {}).get("county", "NAIROBI")
                print(f"[DecideAction HEURISTIC] Select County '{c_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_address_county",
                    "decision": {
                        "thought": f"Selecting County '{c_val}'.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": c_val,
                        "confidence": 0.99,
                        "milestone": "Select County",
                        "reason": f"Setting county to '{c_val}'"
                    }
                }
        # Check District dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("district" in combined or "district" in container) and ("choose" in combined or el.tag == "select"):
                d_val = (req.form_values or {}).get("district", "NAIROBI CENTRAL")
                print(f"[DecideAction HEURISTIC] Select District '{d_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_address_district",
                    "decision": {
                        "thought": f"Selecting District '{d_val}'.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": d_val,
                        "confidence": 0.99,
                        "milestone": "Select District",
                        "reason": f"Setting district to '{d_val}'"
                    }
                }
        # Check Locality dropdown
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("locality" in combined or "locality" in container) and ("choose" in combined or el.tag == "select"):
                loc_val = (req.form_values or {}).get("locality", "CENTRAL BUSINESS DISTRICT")
                print(f"[DecideAction HEURISTIC] Select Locality '{loc_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_address_locality",
                    "decision": {
                        "thought": f"Selecting Locality '{loc_val}'.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": loc_val,
                        "confidence": 0.99,
                        "milestone": "Select Locality",
                        "reason": f"Setting locality to '{loc_val}'"
                    }
                }
        # Check Building/Plot No
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("building" in combined or "plot" in combined or "building" in container or "plot" in container):
                bldg_val = (req.form_values or {}).get("buildingName") or "Upper Hill Chambers"
                print(f"[DecideAction HEURISTIC] Fill Building Name on element {el.id}: {bldg_val}")
                return {
                    "status": "success",
                    "source": "heuristic_address_building",
                    "decision": {
                        "thought": f"Entering building name '{bldg_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": bldg_val,
                        "confidence": 0.99,
                        "milestone": "Building Name",
                        "reason": f"Filling building name/plot: {bldg_val}"
                    }
                }
        # Check Street/Road
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("street" in combined or "road" in combined or "street" in container or "road" in container):
                street_val = (req.form_values or {}).get("streetRoad") or "Ngong Avenue"
                print(f"[DecideAction HEURISTIC] Fill Street/Road on element {el.id}: {street_val}")
                return {
                    "status": "success",
                    "source": "heuristic_address_street",
                    "decision": {
                        "thought": f"Entering street '{street_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": street_val,
                        "confidence": 0.99,
                        "milestone": "Street/Road",
                        "reason": f"Filling street/road: {street_val}"
                    }
                }
        # Check Postal Address
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("postal address" in combined or "postal address" in container):
                p_addr = (req.form_values or {}).get("businessPostalAddress") or (req.form_values or {}).get("postalAddress") or "P. O. Box 30404"
                print(f"[DecideAction HEURISTIC] Fill Postal Address on element {el.id}: {p_addr}")
                return {
                    "status": "success",
                    "source": "heuristic_address_postal_address",
                    "decision": {
                        "thought": f"Entering postal address '{p_addr}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": p_addr,
                        "confidence": 0.99,
                        "milestone": "Postal Address",
                        "reason": f"Filling postal address: {p_addr}"
                    }
                }
        # Check Postal Code
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            container = (el.container or "").lower()
            if ("postal code" in combined or "postal code" in container) and ("choose" in combined or el.tag in ("select", "input")):
                raw_pc = (req.form_values or {}).get("businessPostalCode") or (req.form_values or {}).get("postalCode", "00100 - G.P.O NAIROBI")
                pc_val = "00100 - G.P.O NAIROBI" if "00100" in str(raw_pc) else str(raw_pc)
                print(f"[DecideAction HEURISTIC] Select postal code '{pc_val}' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_address_postal_code",
                    "decision": {
                        "thought": f"Setting postal code '{pc_val}'.",
                        "action": "select",
                        "element_id": el.id,
                        "fill_value": pc_val,
                        "confidence": 0.99,
                        "milestone": "Postal Code",
                        "reason": f"Setting postal code to '{pc_val}'"
                    }
                }
        # Check Mobile Number
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("mobile" in combined or "phone" in combined or "mobile" in container or "phone" in container):
                phone_val = (req.form_values or {}).get("businessPhone") or (req.form_values or {}).get("phoneNumber") or "+254700000000"
                print(f"[DecideAction HEURISTIC] Fill Mobile Number on element {el.id}: {phone_val}")
                return {
                    "status": "success",
                    "source": "heuristic_address_mobile",
                    "decision": {
                        "thought": f"Entering mobile number '{phone_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": phone_val,
                        "confidence": 0.99,
                        "milestone": "Mobile Number",
                        "reason": f"Filling business mobile number: {phone_val}"
                    }
                }
        # Check Company Email Address
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
            container = (el.container or "").lower()
            if el.tag in ("input", "textarea") and ("email" in combined or "email" in container):
                email_val = (req.form_values or {}).get("businessEmail") or (req.form_values or {}).get("email") or "info@business.co.ke"
                print(f"[DecideAction HEURISTIC] Fill Email on element {el.id}: {email_val}")
                return {
                    "status": "success",
                    "source": "heuristic_address_email",
                    "decision": {
                        "thought": f"Entering company email '{email_val}'.",
                        "action": "fill",
                        "element_id": el.id,
                        "fill_value": email_val,
                        "confidence": 0.99,
                        "milestone": "Company Email",
                        "reason": f"Filling company email: {email_val}"
                    }
                }
        # Click Next
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if el.tag == "button" and "next" in combined:
                print(f"[DecideAction HEURISTIC] Click 'Next' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_address_next",
                    "decision": {
                        "thought": "Registered office address details are complete. Clicking 'Next' to advance to Ownership Information.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Proceed to Ownership Information",
                        "reason": "Clicking 'Next' to advance to Ownership Information step"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and ("step=4" in req.url or "ownership information" in (req.page_title or "").lower()):
        # Check if the Add/Edit Proprietor Modal is active
        has_modal = any(
            ("modal" in (el.container or "").lower()) or
            (el.tag == "button" and (el.text or "").strip().lower() == "save") or
            ("id type" in f"{(el.text or '')} {(el.name or '')}".lower()) or
            ("national id" in f"{(el.text or '')} {(el.name or '')}".lower())
            for el in req.elements
        )

        if has_modal:
            # 1. ID Type dropdown
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                container = (el.container or "").lower()
                if ("id type" in combined or "id type" in container) and ("choose" in combined or el.tag == "select"):
                    id_type_val = (req.form_values or {}).get("idType", "Kenyan Citizen")
                    print(f"[DecideAction HEURISTIC] Select ID Type '{id_type_val}' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_id_type",
                        "decision": {
                            "thought": f"Selecting ID Type '{id_type_val}'.",
                            "action": "select",
                            "element_id": el.id,
                            "fill_value": id_type_val,
                            "confidence": 0.99,
                            "milestone": "Select ID Type",
                            "reason": f"Setting ID Type to '{id_type_val}'"
                        }
                    }

            # 2. National ID Number (Search row)
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                if el.tag in ("input", "textarea") and "national id" in combined:
                    id_num = (req.form_values or {}).get("idNumber") or (req.form_values or {}).get("nationalId") or "42761734"
                    print(f"[DecideAction HEURISTIC] Fill National ID on element {el.id}: {id_num}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_national_id",
                        "decision": {
                            "thought": f"Entering National ID '{id_num}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": id_num,
                            "confidence": 0.99,
                            "milestone": "Fill National ID",
                            "reason": f"Entering National ID: {id_num}"
                        }
                    }

            # 3. First Name (Search row)
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                if el.tag in ("input", "textarea") and "first name" in combined and "last" not in combined and "middle" not in combined:
                    first_n = (req.form_values or {}).get("firstName") or "BILDAD"
                    print(f"[DecideAction HEURISTIC] Fill First Name on element {el.id}: {first_n}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_first_name",
                        "decision": {
                            "thought": f"Entering First Name '{first_n}' for registry verification search.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": first_n,
                            "confidence": 0.99,
                            "milestone": "Fill First Name",
                            "reason": f"Entering First Name: {first_n}"
                        }
                    }

            # 4. Search button
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                if el.tag == "button" and "search" in combined:
                    print(f"[DecideAction HEURISTIC] Click Search button on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_search_button",
                        "decision": {
                            "thought": "Triggering eCitizen registry verification search to autofill citizen details.",
                            "action": "click",
                            "element_id": el.id,
                            "confidence": 0.99,
                            "milestone": "Execute Registry Search",
                            "reason": "Clicking 'Search' to autofill citizen particulars"
                        }
                    }

            # 5. Postal Address
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                container = (el.container or "").lower()
                if el.tag in ("input", "textarea") and ("postal address" in combined or "postal address" in container):
                    post_addr = (req.form_values or {}).get("proprietorPostalAddress") or (req.form_values or {}).get("postalAddress") or "15"
                    print(f"[DecideAction HEURISTIC] Fill Postal Address on element {el.id}: {post_addr}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_postal_address",
                        "decision": {
                            "thought": f"Entering Postal Address '{post_addr}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": post_addr,
                            "confidence": 0.99,
                            "milestone": "Proprietor Postal Address",
                            "reason": f"Filling postal address: {post_addr}"
                        }
                    }

            # 6. Postal Code
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                container = (el.container or "").lower()
                if ("postal code" in combined or "postal code" in container) and ("choose" in combined or el.tag in ("select", "input")):
                    raw_pc = (req.form_values or {}).get("proprietorPostalCode") or "10205 - MARAGUA"
                    print(f"[DecideAction HEURISTIC] Select Postal Code '{raw_pc}' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_postal_code",
                        "decision": {
                            "thought": f"Setting postal code '{raw_pc}'.",
                            "action": "select",
                            "element_id": el.id,
                            "fill_value": raw_pc,
                            "confidence": 0.99,
                            "milestone": "Proprietor Postal Code",
                            "reason": f"Setting postal code to '{raw_pc}'"
                        }
                    }

            # 7. Service Address
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                container = (el.container or "").lower()
                if el.tag in ("input", "textarea") and ("service address" in combined or "service address" in container):
                    srv_addr = (req.form_values or {}).get("serviceAddress") or "15 maragua"
                    print(f"[DecideAction HEURISTIC] Fill Service Address on element {el.id}: {srv_addr}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_service_address",
                        "decision": {
                            "thought": f"Entering service address '{srv_addr}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": srv_addr,
                            "confidence": 0.99,
                            "milestone": "Proprietor Service Address",
                            "reason": f"Filling service address: {srv_addr}"
                        }
                    }

            # 8. Occupation
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                container = (el.container or "").lower()
                if el.tag in ("input", "textarea") and ("occupation" in combined or "occupation" in container):
                    occ_val = (req.form_values or {}).get("occupation") or "software engineer"
                    print(f"[DecideAction HEURISTIC] Fill Occupation on element {el.id}: {occ_val}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_occupation",
                        "decision": {
                            "thought": f"Entering occupation '{occ_val}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": occ_val,
                            "confidence": 0.99,
                            "milestone": "Proprietor Occupation",
                            "reason": f"Filling occupation: {occ_val}"
                        }
                    }

            # 9. Physical Address: County
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                container = (el.container or "").lower()
                if ("county" in combined or "county" in container) and ("choose" in combined or el.tag == "select"):
                    c_val = (req.form_values or {}).get("proprietorCounty") or "Kiambu"
                    print(f"[DecideAction HEURISTIC] Select County '{c_val}' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_county",
                        "decision": {
                            "thought": f"Selecting County '{c_val}'.",
                            "action": "select",
                            "element_id": el.id,
                            "fill_value": c_val,
                            "confidence": 0.99,
                            "milestone": "Proprietor County",
                            "reason": f"Setting county to '{c_val}'"
                        }
                    }

            # 10. Physical Address: District
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                container = (el.container or "").lower()
                if ("district" in combined or "district" in container) and ("choose" in combined or el.tag == "select"):
                    d_val = (req.form_values or {}).get("proprietorDistrict") or "Ruiru District"
                    print(f"[DecideAction HEURISTIC] Select District '{d_val}' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_district",
                        "decision": {
                            "thought": f"Selecting District '{d_val}'.",
                            "action": "select",
                            "element_id": el.id,
                            "fill_value": d_val,
                            "confidence": 0.99,
                            "milestone": "Proprietor District",
                            "reason": f"Setting district to '{d_val}'"
                        }
                    }

            # 11. Physical Address: Locality
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                container = (el.container or "").lower()
                if ("locality" in combined or "locality" in container) and ("choose" in combined or el.tag == "select"):
                    loc_val = (req.form_values or {}).get("proprietorLocality") or "Ruiru"
                    print(f"[DecideAction HEURISTIC] Select Locality '{loc_val}' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_locality",
                        "decision": {
                            "thought": f"Selecting Locality '{loc_val}'.",
                            "action": "select",
                            "element_id": el.id,
                            "fill_value": loc_val,
                            "confidence": 0.99,
                            "milestone": "Proprietor Locality",
                            "reason": f"Setting locality to '{loc_val}'"
                        }
                    }

            # 12. Physical Address: Building
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                container = (el.container or "").lower()
                if el.tag in ("input", "textarea") and ("building" in combined or "plot" in combined or "estate" in combined or "building" in container):
                    bldg = (req.form_values or {}).get("proprietorBuilding") or "finance house"
                    print(f"[DecideAction HEURISTIC] Fill Building on element {el.id}: {bldg}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_bldg",
                        "decision": {
                            "thought": f"Entering building name '{bldg}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": bldg,
                            "confidence": 0.99,
                            "milestone": "Proprietor Building",
                            "reason": f"Filling building name: {bldg}"
                        }
                    }

            # 13. Physical Address: Street
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')} {(el.placeholder or '')}".lower()
                container = (el.container or "").lower()
                if el.tag in ("input", "textarea") and ("street" in combined or "road" in combined or "street" in container):
                    street = (req.form_values or {}).get("proprietorStreet") or "kamiti road"
                    print(f"[DecideAction HEURISTIC] Fill Street on element {el.id}: {street}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_street",
                        "decision": {
                            "thought": f"Entering street '{street}'.",
                            "action": "fill",
                            "element_id": el.id,
                            "fill_value": street,
                            "confidence": 0.99,
                            "milestone": "Proprietor Street",
                            "reason": f"Filling street: {street}"
                        }
                    }

            # 14. Save button in modal
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                if el.tag == "button" and "save" in combined:
                    print(f"[DecideAction HEURISTIC] Click Save in modal on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_proprietor_save",
                        "decision": {
                            "thought": "All proprietor details are completed. Clicking 'Save' to commit proprietor info.",
                            "action": "click",
                            "element_id": el.id,
                            "confidence": 0.99,
                            "milestone": "Save Proprietor Info",
                            "reason": "Clicking 'Save' inside modal"
                        }
                    }

        # If Modal is NOT open:
        # Check if table has proprietor row, or if we need to click "Add Proprietor"
        has_proprietor_row = any("edit proprietor" in f"{(el.text or '')} {(el.name or '')}".lower() or "remove" in f"{(el.text or '')} {(el.name or '')}".lower() for el in req.elements)
        if not has_proprietor_row:
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                if el.tag == "button" and "add proprietor" in combined:
                    print(f"[DecideAction HEURISTIC] Click 'Add Proprietor' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_click_add_proprietor",
                        "decision": {
                            "thought": "No proprietor found in table. Clicking 'Add Proprietor' to open entry modal.",
                            "action": "click",
                            "element_id": el.id,
                            "confidence": 0.99,
                            "milestone": "Open Add Proprietor Modal",
                            "reason": "Clicking 'Add Proprietor' button"
                        }
                    }

        # Once proprietor is in the table, click Next
        for el in req.elements:
            combined = f"{(el.text or '')} {(el.name or '')}".lower()
            if el.tag == "button" and "next" in combined:
                print(f"[DecideAction HEURISTIC] Click 'Next' on element {el.id}")
                return {
                    "status": "success",
                    "source": "heuristic_ownership_next",
                    "decision": {
                        "thought": "Proprietor information is recorded in table. Clicking 'Next' to proceed to Application Documents.",
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.99,
                        "milestone": "Proceed to Application Documents",
                        "reason": "Clicking 'Next' to advance to Application Documents step"
                    }
                }

    if "brsv2.ecitizen.go.ke" in req.url and ("step=5" in req.url or "application documents" in (req.page_title or "").lower()):
        # Check if a signed BN-2 document has already been attached
        has_uploaded_doc = any(
            ("remove this file" in f"{(el.text or '')} {(el.name or '')}".lower()) or
            any((el.text or "").lower().strip().endswith(ext) for ext in [".pdf", ".jpg", ".png", ".jpeg"]) or
            (el.type == "file" and bool(getattr(el, "value", None)))
            for el in req.elements
        )

        if has_uploaded_doc:
            # Document is uploaded. Click Next to proceed to Declaration
            for el in req.elements:
                combined = f"{(el.text or '')} {(el.name or '')}".lower()
                if el.tag == "button" and "next" in combined:
                    print(f"[DecideAction HEURISTIC] BN-2 document detected. Click 'Next' on element {el.id}")
                    return {
                        "status": "success",
                        "source": "heuristic_application_documents_next",
                        "decision": {
                            "thought": "Signed BN-2 document has been uploaded. Clicking 'Next' to advance to Declaration.",
                            "action": "click",
                            "element_id": el.id,
                            "confidence": 0.99,
                            "milestone": "Proceed to Declaration",
                            "reason": "Signed BN-2 form uploaded. Advancing to Declaration step."
                        }
                    }

        # Document is NOT uploaded yet: Return Human-in-the-Loop Wait Checkpoint
        print(f"[DecideAction HEURISTIC] Step 5 awaiting signed BN-2 upload on {req.url}")
        return {
            "status": "success",
            "source": "heuristic_application_documents_wait",
            "decision": {
                "thought": "Statutory BN-2 form must be downloaded, signed by proprietor, and uploaded.",
                "action": "wait",
                "is_transient": False,
                "element_id": None,
                "handoff_trigger": "file_upload",
                "milestone": "Awaiting Signed BN-2 Upload",
                "confidence": 1.0,
                "reason": "Please click 'CLICK HERE TO DOWNLOAD BN-2 form' to download your pre-filled form, physically sign it, scan or save as PDF/image, and upload it using 'Choose File'. Autopilot will automatically resume once the document is attached."
            }
        }

    if not is_auth_page:
        for el in req.elements:
            t = f"{(el.text or '')} {(el.name or '')}".lower()
            if any(w in t for w in ["search", "business", "application", "continue", "next", "submit", "apply", "brs", "ownership"]):
                print(f"[DecideAction HEURISTIC] Matched intent on element {el.id}: '{t[:40]}'")
                return {
                    "status": "success",
                    "source": "heuristic_fallback",
                    "decision": {
                        "action": "click",
                        "element_id": el.id,
                        "confidence": 0.8,
                        "reason": f"Matched action intent keyword on '{el.text or el.name}'"
                    }
                }

    print(f"[DecideAction FALLBACK WAIT] Returning transient wait for URL: {req.url}")
    return {
        "status": "partial",
        "source": "fallback",
        "decision": {
            "action": "wait",
            "is_transient": True,
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
