from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.services.speech import speech_service
from app.services.translator import translator_service

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str
    target_lang: str
    source_lang: str = "en"
    use_llm: Optional[bool] = None

@router.get("/health")
async def health_check():
    """Verify service is active"""
    return {"status": "healthy", "service": "speech-translation-service"}

@router.post("/speech-to-text")
async def transcribe_voice(
    file: UploadFile = File(..., description="The audio file to transcribe (WebM, WAV, OGG supported)"),
    lang: str = Form("en", description="Language code of the speaker: 'en', 'hi', 'te'"),
    chat_id: str = Form(None, description="Optional session or chat identifier for tracking"),
    use_llm: Optional[bool] = Form(None, description="Force LLM (true), force Standard (false), or Autonomous (null)")
):
    """
    **Transcribe Speech to Native & English Text**
    
    Processes the audio and returns binary text outputs. 
    If the language is non-English, it automatically generates an English translation for the backend.
    
    *Autonomous Mode*: If `use_llm` is not provided, the service will pick the best engine based on text complexity.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    result = await speech_service.transcribe_audio(file, lang=lang, chat_id=chat_id, use_llm=use_llm)
    return result

@router.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    **Localize Text (AI Response Bridge)**
    
    Translates English AI responses back into the user's selected language.
    
    *Agricultural Context*: When the LLM is used, it applies deep domain knowledge for crop-specific terminology.
    """
    translated = await translator_service.translate_text(
        text=request.text,
        target_lang=request.target_lang,
        source_lang=request.source_lang,
        use_llm=request.use_llm
    )
    return {
        "original_text": request.text,
        "translated_text": translated,
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
        "llm_used": request.use_llm if request.use_llm is not None else "autonomous"
    }
