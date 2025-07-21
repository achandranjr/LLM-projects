from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from openai import OpenAI
import logging
import os
import re
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Transcript Summarizer API",
    description="Generate AI-powered summaries from YouTube video transcripts using OpenAI's GPT models",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YouTube Transcript API
ytt_api = YouTubeTranscriptApi()

# Pydantic models for request/response validation
class SummarizeRequest(BaseModel):
    video_id: str
    api_key: str

class SummarizeResponse(BaseModel):
    success: bool
    video_id: str
    summary: str
    transcript_length: int

class ErrorResponse(BaseModel):
    error: str

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str

class DetailedHealthResponse(BaseModel):
    status: str
    youtube_api: str
    openai_api: str
    timestamp: str

class TranscriptTestResponse(BaseModel):
    video_id: str
    has_transcripts: bool
    transcript_segments: Optional[int] = None
    sample_text: Optional[str] = None
    error: Optional[str] = None

def validate_video_id(video_id: str) -> bool:
    """Validate YouTube video ID format"""
    if not video_id or len(video_id) != 11:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]{11}$', video_id))

def extract_video_id_from_url(url: str) -> str:
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url

def get_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video using simple direct fetch"""
    try:
        logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
        
        # Use the simple direct fetch method - let the library handle language selection
        transcript_data = ytt_api.get_transcript(video_id)
        logger.info(f"Fetched {len(transcript_data)} transcript segments using direct fetch")
        
        if not transcript_data:
            raise Exception("Transcript data is empty")
        
        # Combine all text parts
        full_text = ' '.join([entry.get('text', '') for entry in transcript_data if entry.get('text')])
        
        if not full_text.strip():
            raise Exception("Transcript text is empty after processing")
        
        logger.info(f"Successfully extracted transcript: {len(full_text)} characters")
        return full_text
        
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video. The video owner has turned off captions.")
    except NoTranscriptFound:
        raise HTTPException(status_code=400, detail="No transcript found for this video. The video may not have captions available.")
    except VideoUnavailable:
        raise HTTPException(status_code=400, detail="Video is unavailable, private, or does not exist. Please check the video ID.")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error fetching transcript: {error_msg}")
        
        if "no element found" in error_msg.lower():
            raise HTTPException(status_code=400, detail="Unable to parse video data. The video may be private, deleted, or have restricted access.")
        elif "http" in error_msg.lower() and "error" in error_msg.lower():
            raise HTTPException(status_code=400, detail="Network error while fetching transcript. Please try again.")
        elif "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail="YouTube API quota exceeded. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {error_msg}")

def generate_summary(transcript_text: str, api_key: str) -> str:
    """Generate summary using OpenAI"""
    try:
        # Initialize OpenAI client with provided API key
        client = OpenAI(api_key=api_key)
        
        # Truncate transcript if it's too long (GPT has token limits)
        max_chars = 12000  # Rough estimate to stay under token limits
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars] + "..."
            logger.info("Transcript truncated due to length")
        
        # Create completion
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise, well-structured summaries of video transcripts. Focus on the main points, key insights, and actionable information. Structure your summary with clear sections and bullet points where appropriate."
                },
                {
                    "role": "user",
                    "content": f"Please provide a comprehensive summary of the following video transcript. Structure it with clear sections and bullet points where appropriate:\n\n{transcript_text}"
                }
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        error_msg = str(e).lower()
        
        if "api_key" in error_msg or "unauthorized" in error_msg:
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key. Please check your API key and try again.")
        elif "quota" in error_msg or "billing" in error_msg:
            raise HTTPException(status_code=402, detail="OpenAI API quota exceeded or billing issue. Please check your OpenAI account.")
        elif "model" in error_msg:
            raise HTTPException(status_code=503, detail="OpenAI model not available. Please try again later.")
        elif "rate" in error_msg:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment and try again.")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def home():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="YouTube Transcript Summarizer API",
        version="1.0.0"
    )

@app.get("/api/health", response_model=DetailedHealthResponse)
async def health_check():
    """Detailed health check"""
    return DetailedHealthResponse(
        status="healthy",
        youtube_api="available",
        openai_api="ready",
        timestamp="2025-06-24T00:00:00Z"
    )

@app.get("/api/test-video/{video_id}", response_model=TranscriptTestResponse)
async def test_video(video_id: str):
    """Test endpoint to check if a video has transcripts available"""
    try:
        #clean_video_id = extract_video_id_from_url(video_id)
        
        if not validate_video_id(video_id):
            raise HTTPException(status_code=400, detail="Invalid video ID format")
        
        # Use direct fetch to test
        transcript_data = ytt_api.fetch(video_id)
        
        return TranscriptTestResponse(
            video_id=video_id,
            has_transcripts=True,
            transcript_segments=len(transcript_data),
            sample_text=transcript_data[0].text if transcript_data else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return TranscriptTestResponse(
            video_id=video_id,
            has_transcripts=False,
            error=str(e)
        )

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_video(request: SummarizeRequest):
    """
    Main endpoint to summarize YouTube video
    
    - **video_id**: YouTube video ID or full URL
    - **api_key**: Your OpenAI API key
    """
    try:
        # Extract and validate video ID
        video_id = extract_video_id_from_url(request.video_id.strip())
        
        if not validate_video_id(video_id):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid YouTube video ID format. Got: '{video_id}'. Expected 11-character alphanumeric string."
            )
        
        logger.info(f"Processing video ID: {video_id}")
        
        # Step 1: Fetch transcript
        transcript_text = get_transcript(video_id)
        logger.info(f"Transcript fetched successfully, length: {len(transcript_text)} characters")
        
        # Step 2: Generate summary
        summary = generate_summary(transcript_text, request.api_key)
        logger.info("Summary generated successfully")
        
        # Return successful response
        return SummarizeResponse(
            success=True,
            video_id=video_id,
            summary=summary,
            transcript_length=len(transcript_text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    import uvicorn
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )