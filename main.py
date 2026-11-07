from datetime import datetime, timedelta
import json
import logging
import os
import random
import re
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# ==========================================
# 0. SETUP (API key comes from the .env file, NOT from this code)
# Exception handling
# ==========================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocab")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set. AI features will not work.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="VocabBuilder API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. OOP CLASSES
# ==========================================

class Word:
    """Represents a vocabulary word entity."""

    def __init__(
        self,
        word: str,
        phonetic: str,
        definition: str,
        examples: List[str],
        synonyms: List[str],
        antonyms: List[str],
        simple_explanation: str,
        memory_trick: str,
    ):
        self.word = word
        self.phonetic = phonetic
        self.definition = definition
        self.examples = examples
        self.synonyms = synonyms
        self.antonyms = antonyms
        self.simple_explanation = simple_explanation
        self.memory_trick = memory_trick


class Flashcard:
    """Represents an individual flashcard with spaced-repetition attributes."""

    def __init__(
        self,
        word_data: dict,
        interval_days: int = 1,
        next_review_date: Optional[str] = None,
    ):
        self.id = str(datetime.now().timestamp())
        self.word_data = word_data
        self.interval_days = interval_days
        self.next_review_date = next_review_date or datetime.now().isoformat()

    def to_dict(self):
        return {
            "id": self.id,
            **self.word_data,
            "intervalDays": self.interval_days,
            "nextReviewDate": self.next_review_date,
        }


class DictionaryClient:
    """Handles external API requests, dictionary parsing, and Gemini AI generation."""

    FREE_DICT_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"

    @staticmethod
    def validate_and_clean_input(word: str) -> str:
        """Applies Regular Expressions to validate and sanitize user input."""
        if not word or not word.strip():
            raise ValueError("Input cannot be empty.")

        cleaned = re.sub(r"[^a-zA-Z]", "", word).strip().lower()
        if not cleaned:
            raise ValueError("Invalid word format. Letters only.")
        return cleaned

    @staticmethod
    def clean_sentence(text: str) -> str:
        """Regex cleanup for AI-generated text: strip markdown symbols and extra spaces."""
        text = re.sub(r"[*_`#]", "", str(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fetch_ai_content(self, word: str, fallback_definition: str) -> dict:
        """Uses Gemini to generate a plain-English explanation, memory trick, and examples."""
        prompt = f"""You are an expert vocabulary tutor.
Word: "{word}"
Dictionary definition: "{fallback_definition}"

Return JSON with exactly these keys:
- "simple_explanation": ONE sentence a 10-year-old would understand. Do NOT copy or
  rephrase the dictionary definition. Explain it a different way, for example with a
  comparison or a everyday situation.
- "memory_trick": ONE catchy mnemonic, rhyme, or word-association that helps someone
  remember the word. It must be different from the simple explanation.
- "examples": an array of 2 simple, everyday sentences that use the word.
"""
        try:
            if gemini_client is None:
                raise RuntimeError("GEMINI_API_KEY is not configured.")

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                ),
            )

            raw = response.text.strip()
           
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)

            examples = data.get("examples", [])
            if isinstance(examples, str):
                examples = [examples]

            return {
                "simple_explanation": self.clean_sentence(data["simple_explanation"]),
                "memory_trick": self.clean_sentence(data["memory_trick"]),
                "examples": [self.clean_sentence(e) for e in examples][:2],
            }

        except Exception as e:
            
            logger.error("Gemini failed for '%s': %s", word, e)
            return {
                "simple_explanation": "AI explanation is unavailable right now. Please try again.",
                "memory_trick": "AI memory trick is unavailable right now. Please try again.",
                "examples": [],
            }

    def fetch_word(self, raw_word: str) -> Word:
        try:
            word = self.validate_and_clean_input(raw_word)

            try:
                response = requests.get(f"{self.FREE_DICT_URL}{word}", timeout=10)
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Could not reach the dictionary service: {e}")

            if response.status_code == 404:
                raise LookupError(f"The word '{word}' was not found in the dictionary.")
            elif response.status_code != 200:
                raise ConnectionError("External Dictionary API service error.")

            data = response.json()[0]

            
            phonetic = data.get("phonetic")
            if not phonetic:
                for p in data.get("phonetics", []):
                    if p.get("text"):
                        phonetic = p["text"]
                        break
            phonetic = phonetic or f"/{word}/"

            meanings = data.get("meanings") or [{}]
            definitions = meanings[0].get("definitions") or [{}]
            definition = definitions[0].get("definition", "No definition available.")

           
            synonyms, antonyms = [], []
            for meaning in meanings:
                synonyms.extend(meaning.get("synonyms", []))
                antonyms.extend(meaning.get("antonyms", []))
                for d in meaning.get("definitions", []):
                    synonyms.extend(d.get("synonyms", []))
                    antonyms.extend(d.get("antonyms", []))
            synonyms = list(dict.fromkeys(synonyms))[:5]
            antonyms = list(dict.fromkeys(antonyms))[:5]

           
            ai_data = self.fetch_ai_content(word, definition)

           
            examples = ai_data["examples"]
            if not examples:
                for meaning in meanings:
                    for d in meaning.get("definitions", []):
                        if d.get("example"):
                            examples = [d["example"]]
                            break
                    if examples:
                        break

            return Word(
                word=word,
                phonetic=phonetic,
                definition=definition,
                examples=examples,
                synonyms=synonyms,
                antonyms=antonyms,
                simple_explanation=ai_data["simple_explanation"],
                memory_trick=ai_data["memory_trick"],
            )

        except (ValueError, LookupError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ConnectionError as e:
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            logger.error("Unexpected error in fetch_word: %s", e)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


class QuizGenerator:
    """Generates quiz questions based on saved words."""

    @staticmethod
    def generate_quiz(saved_words: list):
        if not saved_words:
            return {"question": "No words saved yet!", "options": [], "correct": ""}

        target = random.choice(saved_words)

       
        distractors = [
            w["definition"]
            for w in saved_words
            if w.get("word") != target.get("word") and w.get("definition")
        ]
        random.shuffle(distractors)
        distractors = distractors[:2]

        
        generic = [
            "A feeling of extreme happiness after winning something.",
            "A tool used for measuring distance in construction.",
            "The act of moving quickly from one place to another.",
        ]
        for g in generic:
            if len(distractors) >= 2:
                break
            distractors.append(g)

        options = [target["definition"]] + distractors
        random.shuffle(options)

        return {
            "word": target["word"],
            "question": f"What is the definition of '{target['word']}'?",
            "correctAnswer": target["definition"],
            "options": options,
        }


class SpacedRepetitionManager:
    """Manages review intervals based on user feedback (Hard, Good, Easy)."""

    @staticmethod
    def calculate_next_interval(current_interval: int, quality: str) -> int:
        if quality == "hard":
            return 1
        elif quality == "medium":
            return current_interval * 2
        elif quality == "easy":
            return current_interval * 3
        return current_interval

    @staticmethod
    def next_review_date(interval_days: int) -> str:
        return (datetime.now() + timedelta(days=interval_days)).isoformat()


class FileManager:
    """Handles file read/write operations for local persistence."""

    FILE_PATH = "vocab_storage.json"

    @classmethod
    def load_data(cls) -> dict:
        try:
            if not os.path.exists(cls.FILE_PATH):
                return {"flashcards": [], "quiz_history": []}
            with open(cls.FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                data.setdefault("flashcards", [])
                data.setdefault("quiz_history", [])
                return data
        except (IOError, json.JSONDecodeError) as e:
            logger.error("File read error: %s", e)
            return {"flashcards": [], "quiz_history": []}

    @classmethod
    def save_data(cls, data: dict):
        try:
            with open(cls.FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            logger.error("File write error: %s", e)
            raise HTTPException(status_code=500, detail="Failed to persist data to storage file.")


# ==========================================
# 2. FASTAPI ENDPOINTS
# ==========================================

client = DictionaryClient()


@app.get("/api/word")
def get_word(query: str):
    word_obj = client.fetch_word(query)
    data_dict = word_obj.__dict__

    
    formatted_data = {
        "word": data_dict["word"],
        "phonetic": data_dict["phonetic"],
        "definition": data_dict["definition"],
        "examples": data_dict["examples"],
        "synonyms": data_dict["synonyms"],
        "antonyms": data_dict["antonyms"],
        "simpleExplanation": data_dict["simple_explanation"],
        "memoryTrick": data_dict["memory_trick"],
    }
    return {"success": True, "data": formatted_data}


@app.get("/api/flashcards")
def get_flashcards():
    data = FileManager.load_data()
    return {"success": True, "flashcards": data.get("flashcards", [])}


@app.post("/api/flashcards")
def save_flashcard(word_payload: dict):
    if not word_payload.get("word"):
        raise HTTPException(status_code=400, detail="Missing 'word' in request body.")

    data = FileManager.load_data()

    if any(f.get("word", "").lower() == word_payload["word"].lower() for f in data["flashcards"]):
        return {"success": False, "message": "Word already exists in flashcards."}

    flashcard = Flashcard(word_payload)
    data["flashcards"].append(flashcard.to_dict())
    FileManager.save_data(data)
    return {"success": True, "message": "Flashcard saved successfully."}


@app.get("/api/quiz")
def get_quiz():
    data = FileManager.load_data()
    quiz = QuizGenerator.generate_quiz(data.get("flashcards", []))
    return {"success": True, "quiz": quiz}





