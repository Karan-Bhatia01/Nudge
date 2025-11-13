from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import io
import docx
from PyPDF2 import PdfReader
from datetime import datetime

# --- Internal imports ---
from AudioAnalyser.services.audio_transcript import upload_to_assemblyai, transcribe_and_poll
from AudioAnalyser.services.evaluation import analyze_technical_answer
from VideoAnalyser.video_processing import process_video
from ReportGeneration.connection import generate_interview_report
from QuestionGeneration.context_generation import generate_interview_questions
import shared_state

# --- ElevenLabs TTS ---
from elevenlabs import ElevenLabs

# Initialize FastAPI
app = FastAPI()

# Initialize ElevenLabs client
tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Resume Parsing Helpers ---
async def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
        return ""

async def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_content))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""

# --- Health Check ---
@app.get("/")
def healthy():
    return {"healthy": "API working ✅"}

# --- Start Interview ---
@app.post("/start-interview")
async def start_interview(
    candidate_name: str = Form("Anonymous"),
    job_role: str = Form(...),
    company_name: str = Form("Not specified"),
    job_description: str = Form("No description provided"),
    other_details: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None)
):
    resume_text_content = None
    if resume_file:
        file_extension = os.path.splitext(resume_file.filename)[1].lower()
        file_content_bytes = await resume_file.read()

        if file_extension == ".pdf":
            resume_text_content = await extract_text_from_pdf(file_content_bytes)
        elif file_extension == ".docx":
            resume_text_content = await extract_text_from_docx(file_content_bytes)
        elif file_extension == ".txt":
            try:
                resume_text_content = file_content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                resume_text_content = file_content_bytes.decode("latin-1", errors="ignore")

    shared_state.stored_job_info = {
        "candidate_name": candidate_name,
        "job_role": job_role,
        "company_name": company_name,
        "job_description": job_description,
        "other_details": other_details,
        "resume_text_content": resume_text_content,
    }

    return {"message": "✅ Interview setup details saved", "data": shared_state.stored_job_info}

# --- Retrieve Saved Job Info ---
@app.get("/get-job-info")
async def get_job_info():
    if shared_state.stored_job_info:
        return {"job_info": shared_state.stored_job_info}
    return {"message": "❌ No job info saved yet."}

# --- Generate Interview Questions ---
@app.get("/generate-problems")
async def generate_problems_endpoint():
    details = shared_state.stored_job_info
    if not details:
        raise HTTPException(status_code=400, detail="Job info not set. Please use /start-interview first.")

    questions = generate_interview_questions(
        job_role=details.get("job_role"),
        company_name=details.get("company_name"),
        job_description=details.get("job_description"),
        other_details=details.get("other_details"),
        resume_text=details.get("resume_text_content"),
    )

    shared_state.questions_generated = questions
    return questions

# --- Upload and Analyze Audio ---
@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        audio_url = upload_to_assemblyai(audio.file)
        transcript_text = transcribe_and_poll(audio_url)
        analysis_result = analyze_technical_answer(transcript_text)

        timestamp = datetime.utcnow().isoformat()
        shared_state.stored_audio_transcripts[timestamp] = {
            "transcription": transcript_text,
            "analysis": analysis_result,
        }

        return {
            "timestamp": timestamp,
            "transcription": transcript_text,
            "analysis": analysis_result,
            "job_info_used": shared_state.stored_job_info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Analyze Video ---
@app.post("/analyze-video")
async def analyze_video(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        analysis_result = process_video(video_bytes)
        return {
            "message": "✅ Video processed successfully",
            "total_frames": analysis_result.get("total_frames"),
            "frames_analyzed": analysis_result.get("frames_analyzed"),
            "emotions": analysis_result.get("emotion_analysis"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Generate Final Report ---
@app.post("/generate-report")
async def generate_report():
    try:
        report = generate_interview_report()
        if report:
            return {"message": "✅ Report generated successfully", "report": report}
        raise HTTPException(status_code=500, detail="❌ Failed to generate report")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ElevenLabs TTS ---
def generate_tts_audio(question_text: str, voice: str = "Rachel") -> bytes:
    """
    Generate ElevenLabs TTS audio and return it as bytes (no file saving required).
    """
    if not question_text:
        raise ValueError("Question text cannot be empty.")

    try:
        # convert() returns bytes
        audio_bytes = tts_client.text_to_speech.convert(
            text=question_text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with desired voice ID
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        # Ensure we have bytes
        if hasattr(audio_bytes, "__iter__") and not isinstance(audio_bytes, bytes):
            audio_bytes = b"".join(audio_bytes)
        return audio_bytes
    except Exception as e:
        raise RuntimeError(f"TTS generation failed: {e}")

# --- Question TTS Endpoint ---
@app.get("/question-tts/{question_id}")
async def question_tts(question_id: int, voice: str = "Rachel"):
    q_data = getattr(shared_state, "questions_generated", None)
    if not q_data:
        raise HTTPException(status_code=400, detail="No questions generated yet.")

    questions_list = q_data.get("questions", [])
    if question_id < 1 or question_id > len(questions_list):
        raise HTTPException(status_code=404, detail="Invalid question ID")

    question_text = questions_list[question_id - 1]

    try:
        audio_bytes = generate_tts_audio(question_text, voice)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
