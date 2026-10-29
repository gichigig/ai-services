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

class EcitizenActionRequest(BaseModel):
    url: str
    page_title: Optional[str] = ""
    task_goal: str
    step_title: str
    step_description: Optional[str] = ""
    elements: List[EcitizenElement]
    form_values: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    model_name: Optional[str] = "gpt_120b"

@web_app.post("/ecitizen/decide-action")
async def ecitizen_decide_action_endpoint(req: EcitizenActionRequest):
    """
    Autonomous DOM Vision/Reasoning endpoint for eCitizen government portal navigation.
    Uses GPT OSS 120B to decide which element to click, fill, or select based on live DOM perception.
    """
    import json
    import re

    # Build structured elements summary for the prompt
    elements_summary = []
    for el in req.elements[:65]:
        desc = f"[{el.id}] <{el.tag}"
        if el.type: desc += f' type="{el.type}"'
        if el.name: desc += f' name="{el.name}"'
        if el.placeholder: desc += f' placeholder="{el.placeholder}"'
        if el.aria: desc += f' aria-label="{el.aria}"'
        if el.href: desc += f' href="{el.href}"'
        desc += ">"
        if el.text:
            cleaned_text = " ".join(el.text.split())[:80]
            desc += f" {cleaned_text}"
        elements_summary.append(desc)

    elements_text = "\n".join(elements_summary) if elements_summary else "No interactive elements found."
    form_text = json.dumps(req.form_values or {}, indent=2)

    system_prompt = (
        "You are an expert autonomous browser agent navigating the Kenyan eCitizen portal (e.g. BRS, NTSA, DCI). "
        "Your task is to analyze the page state and visible interactive elements, then choose the single best element to interact with to accomplish the user's current step. "
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "action": "click" | "fill" | "select" | "wait",\n'
        '  "element_id": <int: id of the chosen element, or null if wait>,\n'
        '  "fill_value": "<string or null>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<short explanation>"\n'
        "}"
    )

    user_prompt = (
        f"Task Goal: {req.task_goal}\n"
        f"Current Step: {req.step_title}\n"
        f"Step Description: {req.step_description or 'Navigate and proceed'}\n"
        f"Current Page URL: {req.url}\n"
        f"Page Title: {req.page_title or 'eCitizen Portal'}\n"
        f"Available Form Data to Fill:\n{form_text}\n\n"
        f"Visible Interactive Elements on Page:\n{elements_text}\n\n"
        "Analyze the available elements and decide the exact next action to make progress."
    )

    # First attempt: GPT OSS 120B on Modal
    try:
        import modal
        GptModelCls = modal.Cls.from_name("bruv-schools-llm", "GptOss120bModel")
        gpt_model = GptModelCls()
        raw_res = await gpt_model.generate.remote.aio(
            system_message=system_prompt,
            user_message=user_prompt,
            max_tokens=450
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
            max_tokens=350
        )
        match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        if match:
            decision = json.loads(match.group(0))
            return {"status": "success", "source": "qwen_small", "decision": decision}
    except Exception as qwen_err:
        print(f"Qwen fallback note: {qwen_err}")

    # Heuristic fallback if LLM container is spinning up
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
