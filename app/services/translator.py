import asyncio
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator
from openai import AsyncOpenAI
from google import genai
from typing import Optional
import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)

class TranslatorService:
    """
    Service to handle language translation for the AgriGPT platform.
    Supports English, Hindi, and Telugu.
    Optimized for Alumnx OSS endpoint with Gemini fallback.
    """
    
    def __init__(self):
        # Cache translators for common language pairs to improve performance
        self.translators = {}
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize OpenAI client with dummy API key for OSS endpoint
        self.openai_client = None
        try:
            self.openai_client = AsyncOpenAI(
                base_url=os.getenv("OSS_LLM_URL", "http://3.109.63.164:11434/v1"),
                api_key="dummy"
            )
            logger.info("OpenAI client initialized for advanced translation.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            
        # Initialize Gemini client as fallback
        self.gemini_client = None
        if settings.GOOGLE_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
                logger.info("Gemini client initialized for fallback translation.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini fallback client: {e}")

    def _get_translator(self, source: str, target: str):
        key = f"{source}_{target}"
        if key not in self.translators:
            self.translators[key] = GoogleTranslator(source=source, target=target)
        return self.translators[key]

    async def translate_text(self, text: str, target_lang: str, source_lang: str = 'auto', use_llm: Optional[bool] = None) -> str:
        """
        Translates text from source language to target language.
        
        ACTS AUTONOMOUSLY: 
        1. If use_llm is passed (True/False), it follows that.
        2. If use_llm is None, it uses the LLM automatically if:
           - GOOGLE_API_KEY is present.
           - The text is more than 3 words (contextual query).
        3. Fallback: Always falls back to standard translator on failure.
        """
        if not text or not text.strip():
            return ""
            
        if source_lang == target_lang:
            return text

        # Determine if we should use LLM autonomously
        should_use_llm = use_llm
        if should_use_llm is None:
            # Autonomous logic: Use LLM if key is present AND it's a real query (not just "hello")
            word_count = len(text.split())
            should_use_llm = (self.openai_client is not None or self.gemini_client is not None) and word_count > 2

        if should_use_llm and (self.openai_client or self.gemini_client):
            try:
                return await self.translate_text_llm(text, target_lang, source_lang)
            except Exception as e:
                logger.warning(f"Autonomous LLM failed completely, falling back to standard: {e}")

        # Standard / Fallback translation
        try:
            translator = self._get_translator(source_lang, target_lang)
            loop = asyncio.get_event_loop()
            translated = await loop.run_in_executor(
                self.executor, 
                translator.translate, 
                text
            )
            return translated
        except Exception as e:
            logger.error(f"Translation Error ({source_lang} -> {target_lang}): {e}")
            return text

    async def translate_text_llm(self, text: str, target_lang: str, source_lang: str = 'auto') -> str:
        """
        Uses OSS LLM for high-accuracy contextual translation.
        Includes a 10-second timeout to prevent service hanging.
        """
        if not self.openai_client:
            return text

        lang_names = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu'
        }
        
        src = lang_names.get(source_lang, source_lang)
        tgt = lang_names.get(target_lang, target_lang)

        prompt = f"""You are a strict technical translator specialized in Agriculture.
Your task is to translate from {src} to {tgt}.

RULES:
1. Output ONLY the translated text. 
2. Do not include phrases like "Here is the translation" or "I can't translate".
3. Maintain technical accuracy for crops, pests, and schemes.
4.The context is Agriculture, specifically serving farmers, students, and researchers. The content covers crops, diseases, government schemes, and modern farming practices.Maintain a professional yet helpful tone. Ensure technical terms, chemical names, and scheme titles are translated accurately and appropriately for the agricultural domain.Only return the translated text.
5. If the input is not {src}, just translate it to the best of your ability into {tgt}.

TEXT TO TRANSLATE:
{text}"""

        try:
            # Try primary OSS endpoint with a 5s fast timeout
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model="gpt-oss:20b",
                    messages=[
                        {"role": "system", "content": "You are an expert translator specializing in the agricultural domain."},
                        {"role": "user", "content": prompt}
                    ]
                ),
                timeout=25.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OSS LLM Translation failed or timed out: {e}. Falling back to Gemini.")
            if self.gemini_client:
                try:
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.gemini_client.models.generate_content,
                            model='gemini-2.0-flash',
                            contents=prompt
                        ),
                        timeout=10.0
                    )
                    return response.text.strip()
                except Exception as e_gemini:
                    logger.error(f"Fallback Gemini Translation Error: {e_gemini}")
                    return await self.translate_text(text, target_lang, source_lang, use_llm=False)
            
            # Fallback to normal deep-translator if Gemini fails or is not available
            return await self.translate_text(text, target_lang, source_lang, use_llm=False)

# Global singleton instance
translator_service = TranslatorService()
