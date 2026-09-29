import hashlib
import os
import subprocess
import librosa
import numpy as np

def extract_and_analyze_audio(video_url: str):
    """
    Uses FFmpeg to extract the audio track from the video.
    Hashes the raw audio bytes (to group identical trending sounds).
    Uses Librosa to calculate BPM, speech/music detection, and audio mood.
    
    Returns:
        audio_hash (str): MD5 hash of audio for duplicate sound detection
        is_speech (bool): Whether the audio is primarily speech
        language_code (str | None): Detected language code (e.g., 'en')
        audio_mood (str): Detected mood — "energetic", "calm", "melancholic", "intense", "neutral"
        transcript (str): The transcribed text of the audio using Whisper
    """
    
    temp_audio_file = f"temp_audio_{hash(video_url)}.wav"
    
    try:
        # 1. Use FFmpeg to extract audio from the video URL
        command = [
            "ffmpeg", "-y",
            "-i", video_url,
            "-t", "10",
            "-q:a", "0",
            "-map", "a",
            temp_audio_file
        ]
        
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(temp_audio_file):
            print(f"Audio extraction failed for {video_url}")
            return "no_audio_hash", False, None, "neutral"
            
        # 2. Hash the actual extracted audio file
        hasher = hashlib.md5()
        with open(temp_audio_file, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        real_audio_hash = hasher.hexdigest()
        
        # 3. Analyze Audio using Librosa
        y, sr = librosa.load(temp_audio_file)
        
        # Estimate BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        # Spectral centroid (high = bright/harsh, low = warm/dark)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_cent = float(np.mean(cent))
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=y)
        avg_rms = float(np.mean(rms))
        max_rms = float(np.max(rms))
        
        # Spectral rolloff (frequency below which 85% of energy is concentrated)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        avg_rolloff = float(np.mean(rolloff))
        
        # Zero crossing rate (higher = noisier/percussive)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        avg_zcr = float(np.mean(zcr))
        
        # ── Speech vs Music Detection ──────────────────────────────
        is_speech = True
        if bpm > 100:
            is_speech = False
            print(f"Detected High Energy Music (BPM: {bpm:.1f})")
        elif avg_cent > 2500:
            is_speech = True
            print(f"Detected Speech/Vocal-heavy audio (Centroid: {avg_cent:.1f})")
        else:
            is_speech = False
            print("Detected Acoustic/Mellow Music")
        
        # ── Audio Mood Classification ──────────────────────────────
        # Combines BPM, energy, spectral features, and brightness to
        # classify the audio into one of 5 mood categories.
        audio_mood = classify_audio_mood(bpm, avg_rms, max_rms, avg_cent, avg_rolloff, avg_zcr)
        print(f"Audio Mood: {audio_mood} (BPM: {bpm:.1f}, RMS: {avg_rms:.4f}, Centroid: {avg_cent:.1f})")
            
        # ── Language & Transcription (Whisper) ──────────────────────
        language_code = None
        transcript = ""
        
        if is_speech:
            try:
                import whisper
                # Load the model (this will cache it on the Modal container)
                model = whisper.load_model("base")
                
                # Transcribe the audio
                result = model.transcribe(temp_audio_file)
                transcript = result.get("text", "").strip()
                language_code = result.get("language", "en")
                
                print(f"Detected Language: {language_code}")
                print(f"Transcript: {transcript}")
            except Exception as e:
                print(f"Language/Transcription detection failed: {e}")
            
        return real_audio_hash, is_speech, language_code, audio_mood, transcript
        
    except Exception as e:
        print(f"Error in audio extraction/analysis: {e}")
        return "error_hash", False, None, "neutral", ""
        
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


def classify_audio_mood(
    bpm: float,
    avg_rms: float,
    max_rms: float,
    avg_centroid: float,
    avg_rolloff: float,
    avg_zcr: float
) -> str:
    """
    Classify the audio mood based on acoustic features.
    
    Mood categories:
    - "energetic": High BPM + high energy — hype music, EDM, party
    - "calm": Low BPM + low energy — ambient, lo-fi, acoustic
    - "melancholic": Low-mid BPM + low-mid energy + warm tone — sad songs, ballads
    - "intense": High energy + high spectral brightness — rock, metal, dramatic
    - "neutral": Everything else — speech, average music
    
    Based on music information retrieval research:
    - BPM correlates with energy/arousal
    - RMS correlates with perceived loudness
    - Spectral centroid correlates with brightness/timbre
    - Zero crossing rate correlates with noisiness
    """
    
    # Normalize features to comparable scales
    energy_high = avg_rms > 0.08
    energy_very_high = avg_rms > 0.15
    energy_low = avg_rms < 0.03
    
    bright_sound = avg_centroid > 3000
    warm_sound = avg_centroid < 2000
    
    fast_tempo = bpm > 120
    slow_tempo = bpm < 80
    mid_tempo = 80 <= bpm <= 120
    
    noisy = avg_zcr > 0.1
    
    # Classification rules
    if fast_tempo and energy_high and bright_sound:
        return "energetic"
    
    if energy_very_high and bright_sound and noisy:
        return "intense"
    
    if fast_tempo and energy_high:
        return "energetic"
    
    if slow_tempo and energy_low and warm_sound:
        return "melancholic"
    
    if slow_tempo and energy_low:
        return "calm"
    
    if mid_tempo and energy_low and warm_sound:
        return "melancholic"
    
    if slow_tempo and not energy_high:
        return "calm"
    
    return "neutral"
