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
    .pip_install_from_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    .run_commands("playwright install --with-deps chromium")
    .add_local_python_source("video_analyzer", "audio_analyzer", "llm_router", "rag_search", "timetable_extractor")
)

# Define the persistent volume for RAG storage
rag_volume = modal.Volume.from_name("bruv-schools-rag-data", create_if_missing=True)

class VideoAnalysisRequest(BaseModel):
    media_id: str
    video_url: str
    caption: str = ""

from typing import Optional

class QuestionRequest(BaseModel):
    student_id: Optional[str] = None
    question: str
    history: Optional[list] = None
    user_memory: Optional[dict] = None
    selected_model: Optional[str] = "small"
    use_reasoning: Optional[bool] = False

class DocumentIndexRequest(BaseModel):
    file_url: str
    course_name: str = ""
    year: str = ""

@app.function(image=image, timeout=600, gpu="A10G")
def process_video_modal(media_id: str, video_url: str, caption: str = ""):
    """
    Background task that downloads the video, runs inference, and sends results to Kotlin backend.
    Runs entirely on Modal.
    """
    # Import locally within the function to avoid downloading models when parsing the file
    from video_analyzer import analyze_video_frames
    from audio_analyzer import extract_and_analyze_audio
    import requests

    # Note: If running a local Kotlin backend, KOTLIN_BACKEND_WEBHOOK needs to be a public URL (e.g. via ngrok)
    # because Modal runs in the cloud and cannot reach your localhost.
    KOTLIN_BACKEND_WEBHOOK = os.getenv("KOTLIN_BACKEND_WEBHOOK", "http://localhost:8080/api/internal/media/analysis-complete")

    try:
        print(f"Starting analysis for media_id: {media_id}")
        
        # 1. Run visual classification, face embeddings, and all new analysis
        video_category, ai_tags, is_nsfw, face_embeddings, extra_analysis = analyze_video_frames(video_url)
        
        # 2. Run audio classification + transcription (Whisper)
        audio_hash, is_speech, language, audio_mood, transcript = extract_and_analyze_audio(video_url)
        
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
              f"Mood: {extra_analysis.get('colorMood')}, AudioMood: {audio_mood}")
        
        # Send to Kotlin backend
        print(f"Sending webhook to {KOTLIN_BACKEND_WEBHOOK}")
        response = requests.post(KOTLIN_BACKEND_WEBHOOK, json=payload)
        response.raise_for_status()
        print("Webhook sent successfully.")
        
    except Exception as e:
        print(f"Error analyzing video {media_id}: {str(e)}")

# Define the FastAPI app
web_app = FastAPI(title="Bruv Schools AI Microservice (Modal)")

@web_app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Endpoint called by the Kotlin backend to trigger asynchronous video analysis.
    This routes the heavy work to the process_video_modal Modal function.
    """
    process_video_modal.spawn(request.media_id, request.video_url, request.caption)
    return {"status": "Analysis queued on Modal", "media_id": request.media_id}

@web_app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """
    Endpoint called by the client or backend to ask the AI a question.
    """
    try:
        from rag_search import retrieve_context
        from llm_router import ask_qwen
        
        context = retrieve_context(request.question, history=request.history)
        answer = await ask_qwen(
            request.question, 
            context, 
            student_id=request.student_id, 
            history=request.history,
            user_memory=request.user_memory,
            selected_model=request.selected_model,
            use_reasoning=request.use_reasoning
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
    
    async def event_generator():
        try:
            context = retrieve_context(request.question, history=request.history)
            async for chunk in ask_qwen_stream(
                request.question, 
                context, 
                student_id=request.student_id, 
                history=request.history,
                user_memory=request.user_memory,
                selected_model=request.selected_model,
                use_reasoning=request.use_reasoning
            ):
                yield chunk
        except Exception as e:
            import traceback
            import json
            yield json.dumps({"type": "answer", "content": f"Error: {str(e)}\n{traceback.format_exc()}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

@app.function(image=image, min_containers=1, volumes={"/data": rag_volume})
@modal.asgi_app()
def fastapi_app():
    return web_app
