import os
import shutil
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
from fastapi import UploadFile, HTTPException
from typing import Optional
import logging
from app.services.translator import translator_service

logger = logging.getLogger(__name__)

class SpeechService:
    """
    Service layer for handling audio processing and speech-to-text conversion.
    Supports multi-language transcription and automatic translation.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Mapping simple language codes to Google Speech recognition codes
        self.lang_map = {
            'en': 'en-US',
            'hi': 'hi-IN',
            'te': 'te-IN'
        }

    async def transcribe_audio(self, file: UploadFile, lang: str = 'en', chat_id: str = None, use_llm: Optional[bool] = None) -> dict:
        """
        Processes an uploaded audio file and converts it to text.
        Automatically translates to English if the source is not English.
        
        Args:
            file: The uploaded audio file.
            lang: Language code ('en', 'hi', 'te'). Defaults to 'en'.
            chat_id: Optional session identifier for tracking metrics.
            use_llm: Whether to use advanced LLM-based translation.
            
        Returns:
            dict: { "native_text": "...", "english_text": "..." }
        """
        if chat_id:
            logger.info(f"Processing speech ({lang}) for chat session: {chat_id} (LLM: {use_llm})")
            
        # Get full recognition language code
        recognition_lang = self.lang_map.get(lang, 'en-US')
            
        # 1. Create a temporary file to store the upload
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)
            temp_audio_path = temp_audio.name

        wav_path = None
        try:
            # 2. Convert to standard WAV format
            # This requires FFmpeg to be installed on the system
            wav_path = temp_audio_path.rsplit(".", 1)[0] + "_converted.wav"
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                audio.export(wav_path, format="wav")
            except Exception as e:
                # If conversion fails and it's not a WAV file, we can't proceed
                if suffix.lower() not in ['.wav', '.wave']:
                    logger.error(f"Audio conversion failed: {e}. Ensure FFmpeg is installed.")
                    raise HTTPException(
                        status_code=500, 
                        detail="Could not process audio format. Ensure FFmpeg is installed on the server (brew install ffmpeg)."
                    )
                else:
                    # It's already a WAV, maybe it works directly
                    wav_path = temp_audio_path

            # 3. Perform Recognition
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                native_text = self.recognizer.recognize_google(audio_data, language=recognition_lang)
                
                # 4. Handle Translation to English if necessary
                english_text = native_text
                if lang != 'en':
                    english_text = await translator_service.translate_text(
                        text=native_text, 
                        target_lang='en', 
                        source_lang=lang,
                        use_llm=use_llm
                    )
                
                return {
                    "native_text": native_text,
                    "english_text": english_text,
                    "language": lang
                }

        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech could not be understood")
        except sr.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Speech service unavailable: {e}")
        except Exception as e:
            print(f"Speech processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # 5. Cleanup temporary files
            for path in [temp_audio_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

# Create a singleton instance
speech_service = SpeechService()
