import os
import hashlib
from typing import List
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import requests

from video_analyzer import analyze_video_frames
from audio_analyzer import extract_and_analyze_audio

app = FastAPI(title="Bruv Schools AI Microservice")

# The URL of your Kotlin Backend webhook
KOTLIN_BACKEND_WEBHOOK = os.getenv("KOTLIN_BACKEND_WEBHOOK", "http://localhost:8080/api/internal/media/analysis-complete")

class VideoAnalysisRequest(BaseModel):
    media_id: str
    video_url: str
    caption: str = ""

def process_video_background(media_id: str, video_url: str, caption: str = ""):
    """
    Background task that downloads the video, runs inference, and sends results to Kotlin backend.
    """
    try:
        print(f"Starting analysis for media_id: {media_id}")
        
        # 1. Run visual classification, face embeddings, and all new analysis
        video_category, ai_tags, is_nsfw, face_embeddings, extra_analysis = analyze_video_frames(video_url)
        
        # 2. Run audio classification + mood detection
        audio_hash, is_speech, language, audio_mood = extract_and_analyze_audio(video_url)
        
        # 3. Send results back to Kotlin backend
        payload = {
            "mediaId": media_id,
            "category": video_category,
            "aiTags": ai_tags,
            "audioHash": audio_hash,
            "hasSpeech": is_speech,
            "isNsfw": is_nsfw,
            "language": language,
            "faceEmbeddings": face_embeddings,
            # New analysis fields
            "visualFingerprint": extra_analysis.get("visualFingerprint", ""),
            "thumbnailTimestampMs": extra_analysis.get("thumbnailTimestampMs", 0),
            "qualityScore": extra_analysis.get("qualityScore", 0.5),
            "colorMood": extra_analysis.get("colorMood", "neutral"),
            "dominantColors": extra_analysis.get("dominantColors", []),
            "motionIntensity": extra_analysis.get("motionIntensity", "low"),
            "audioMood": audio_mood,
            "caption": caption
        }
        
        print(f"Analysis complete for {media_id}. Category: {video_category}, "
              f"Tags: {ai_tags}, Quality: {extra_analysis.get('qualityScore')}, "
              f"Mood: {extra_analysis.get('colorMood')}, AudioMood: {audio_mood}")
        
        # Send to Kotlin backend
        response = requests.post(KOTLIN_BACKEND_WEBHOOK, json=payload)
        response.raise_for_status()
        
    except Exception as e:
        print(f"Error analyzing video {media_id}: {str(e)}")

@app.post("/analyze-video")
async def analyze_video(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Endpoint called by the Kotlin backend to trigger asynchronous video analysis.
    """
    background_tasks.add_task(process_video_background, request.media_id, request.video_url, request.caption)
    return {"status": "Analysis queued", "media_id": request.media_id}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
