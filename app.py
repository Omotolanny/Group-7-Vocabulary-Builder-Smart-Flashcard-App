import json
import logging
import os
import random
import re
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# 0. SETUP (the API key comes from the .env file, NOT from this code)
# ==========================================

st.set_page_config(page_title="Vocab Builder", page_icon="📚", layout="centered")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vocab")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

GEMINI_FALLBACKS = ["gemini-flash-latest", "gemini-3.7-flash", "gemini-3.6-flash"]
MODELS_TO_TRY = [GEMINI_MODEL] + [model for model in GEMINI_FALLBACKS if model != GEMINI_MODEL]


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
        ai_error: Optional[str] = None,
    ):
        self.word = word
        self.phonetic = phonetic
        self.definition = definition
        self.examples = examples
        self.synonyms = synonyms
        self.antonyms = antonyms
        self.simple_explanation = simple_explanation
        self.memory_trick = memory_trick
        self.ai_error = ai_error  

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "phonetic": self.phonetic,
            "definition": self.definition,
            "examples": self.examples,
            "synonyms": self.synonyms,
            "antonyms": self.antonyms,
            "simple_explanation": self.simple_explanation,
            "memory_trick": self.memory_trick,
            "ai_error": self.ai_error,
        }


class Flashcard:
    """A flashcard with spaced-repetition attributes."""

    def __init__(
        self,
        word_data: dict,
        interval_days: int = 1,
        next_review_date: Optional[str] = None,
        card_id: Optional[str] = None,
    ):
        self.id = card_id or str(datetime.now().timestamp())
        self.word_data = word_data
        self.interval_days = interval_days
        self.next_review_date = next_review_date or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            **self.word_data,
            "interval_days": self.interval_days,
            "next_review_date": self.next_review_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Flashcard":
        extra_keys = ("id", "interval_days", "next_review_date")
        word_data = {k: v for k, v in d.items() if k not in extra_keys}
        return cls(word_data, d.get("interval_days", 1), d.get("next_review_date"), d.get("id"))

    def is_due(self) -> bool:
        return datetime.fromisoformat(self.next_review_date) <= datetime.now()



def friendly_ai_error(error) -> str:
    """Turns technical Gemini errors into short messages users can understand."""
    text = str(error)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "The AI has reached its daily free limit. Please try again later."
    if "503" in text or "UNAVAILABLE" in text:
        return "The AI service is busy right now. Please try again in a moment."
    if "404" in text or "NOT_FOUND" in text:
        return "The AI model is not available. Please check the model setting."
    if "API key" in text or "401" in text or "403" in text or "PERMISSION_DENIED" in text:
        return "The AI service could not be accessed. Please check the API key."
    if "GEMINI_API_KEY is missing" in text:
        return "The AI service is not set up yet. Please add the API key."
    return "The AI explanation could not be generated. Please try again."
class DictionaryClient:
    """Handles the dictionary API and Gemini AI generation with offline fallback."""

    FREE_DICT_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"
    GENERIC_WRONG = [
        "A feeling of extreme happiness after winning something.",
        "A tool used for measuring distance in construction.",
        "The act of moving quickly from one place to another.",
        "A small container used for storing liquids.",
    ]

    def __init__(self):
        self.ai = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=10000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) if GEMINI_API_KEY else None

    @staticmethod
    def validate_and_clean_input(word: str) -> str:
        if not word or not word.strip():
            raise ValueError("Please type a word first.")
        cleaned = re.sub(r"[^a-zA-Z]", "", word).strip().lower()
        if not cleaned:
            raise ValueError("Invalid word. Please use letters only.")
        return cleaned

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[*_`#]", "", str(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _ask_gemini_json(self, prompt: str) -> dict:
        if self.ai is None:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        
        try:
            response = self.ai.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as e:
            logger.error("Gemini connection error: %s", e)
            
            return {
                "definition": "Definition unavailable due to network connection reset.",
                "phonetic": "/word/",
                "simple_explanation": "Could not generate explanation right now.",
                "memory_trick": "Try searching again in a moment.",
                "examples": ["Example sentence unavailable."],
                "synonyms": [],
                "antonyms": []
            }

    def _ask_gemini_json(self, prompt: str) -> dict:
        if self.ai is None:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        response = self.ai.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)

    def fetch_ai_content(self, word: str, definition: str) -> dict:
        prompt = f"""You are an expert vocabulary tutor.
Word: "{word}"
Dictionary definition: "{definition}"

Return JSON with exactly these keys:
- "simple_explanation": ONE sentence a 10-year-old would understand. Do NOT copy or rephrase the dictionary definition.
- "memory_trick": ONE catchy mnemonic, rhyme, or word-association that helps someone remember the word.
- "examples": an array of 2 simple, everyday sentences that use the word.
"""
        try:
            data = self._ask_gemini_json(prompt)
            examples = data.get("examples", [])
            if isinstance(examples, str):
                examples = [examples]
            return {
                "simple_explanation": self.clean_text(data.get("simple_explanation", "")),
                "memory_trick": self.clean_text(data.get("memory_trick", "")),
                "examples": [self.clean_text(e) for e in examples][:2],
                "error": None,
            }
        except Exception as e:
            logger.error("Gemini failed for '%s': %s", word, e)
            return {"simple_explanation": "", "memory_trick": "", "examples": [], "error": str(e)}

    def generate_fallback_definition(self, word: str) -> dict:
        """Fallback method when the external dictionary API is unreachable (e.g. offline)."""
        prompt = f"""You are a comprehensive dictionary. Provide details for the word: "{word}"
Return JSON with exactly these keys:
- "phonetic": phonetic spelling (e.g. /ɛˈfɪmərəl/)
- "definition": clear formal definition
- "simple_explanation": ONE simple sentence explanation
- "memory_trick": a catchy memory hook
- "examples": array of 2 simple sentences using the word
- "synonyms": array of up to 3 synonyms
- "antonyms": array of up to 3 antonyms
"""
        return self._ask_gemini_json(prompt)

    def fetch_word(self, raw_word: str) -> Word:
        word = self.validate_and_clean_input(raw_word)

        definition = ""
        phonetic = f"/{word}/"
        synonyms, antonyms = [], []
        ai_data = {}

        try:
            
            response = requests.get(f"{self.FREE_DICT_URL}{word}", timeout=(5, 10))
            if response.status_code == 200:
                data = response.json()[0]
                phonetic = data.get("phonetic") or phonetic
                meanings = data.get("meanings") or [{}]
                definitions = meanings[0].get("definitions") or [{}]
                definition = definitions[0].get("definition", "")
                
                for meaning in meanings:
                    synonyms.extend(meaning.get("synonyms", []))
                    antonyms.extend(meaning.get("antonyms", []))
                
                ai = self.fetch_ai_content(word, definition)
                ai_data = {
                    "simple_explanation": ai["simple_explanation"],
                    "memory_trick": ai["memory_trick"],
                    "examples": ai["examples"],
                    "ai_error": ai["error"]
                }
            else:
                if response.status_code == 404:
                    raise LookupError(f"The word '{word}' was not found in the dictionary.")
                raise ConnectionError("The dictionary service returned an error")
                # Exception("Dictionary API returned non-200")
        except Exception:
            fallback = self.generate_fallback_definition(word)
            definition = fallback.get("definition", "Definition generated via AI fallback.")
            phonetic = fallback.get("phonetic", phonetic)
            synonyms = fallback.get("synonyms", [])
            antonyms = fallback.get("antonyms", [])
            ai_data = {
                "simple_explanation": self.clean_text(fallback.get("simple_explanation", "")),
                "memory_trick": self.clean_text(fallback.get("memory_trick", "")),
                "examples": [self.clean_text(e) for e in fallback.get("examples", [])][:2],
                "ai_error": None
            }

        return Word(
            word=word,
            phonetic=phonetic,
            definition=definition,
            examples=ai_data.get("examples", []),
            synonyms=synonyms[:5],
            antonyms=antonyms[:5],
            simple_explanation=ai_data.get("simple_explanation", ""),
            memory_trick=ai_data.get("memory_trick", ""),
            ai_error=ai_data.get("ai_error"),
        )

# class DictionaryClient:
    """Handles the dictionary API and Gemini AI generation."""

    FREE_DICT_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/"
    GENERIC_WRONG = [
        "A feeling of extreme happiness after winning something.",
        "A tool used for measuring distance in construction.",
        "The act of moving quickly from one place to another.",
        "A small container used for storing liquids.",
    ]

    def __init__(self):
        self.ai = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=10000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) if GEMINI_API_KEY else None

    # Regex helpers 
    @staticmethod
    def validate_and_clean_input(word: str) -> str:
        """Regex: validate and sanitize the user's word."""
        if not word or not word.strip():
            raise ValueError("Please type a word first.")
        cleaned = re.sub(r"[^a-zA-Z]", "", word).strip().lower()
        if not cleaned:
            raise ValueError("Invalid word. Please use letters only.")
        return cleaned

    @staticmethod
    def clean_text(text: str) -> str:
        """Regex: remove markdown symbols and extra spaces from AI text."""
        text = re.sub(r"[*_`#]", "", str(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # Gemini 
    def _ask_gemini_json(self, prompt: str) -> dict:
        if self.ai is None:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        retryable_errors = (
            "429",
            "RESOURCE_EXHAUSTED",
            "503",
            "UNAVAILABLE",
            "404",
            "NOT_FOUND",
        )
        for index, model in enumerate(MODELS_TO_TRY):
            try:
                response = self.ai.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.8,
                    ),
                )
                raw = response.text.strip()
                raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
                return json.loads(raw)
            except Exception as e:
                if index == len(MODELS_TO_TRY) - 1 or not any(
                    marker in str(e) for marker in retryable_errors
                ):
                    raise
                logger.warning(
                    "Gemini model %s is unavailable; trying the next configured model: %s",
                    model,
                    e,
                )

    def fetch_ai_content(self, word: str, definition: str) -> dict:
        prompt = f"""You are an expert vocabulary tutor.
Word: "{word}"
Dictionary definition: "{definition}"

Return JSON with exactly these keys:
- "simple_explanation": ONE sentence a 10-year-old would understand. Do NOT copy or
  rephrase the dictionary definition. Explain it a different way, for example with a
  comparison or an everyday situation.
- "memory_trick": ONE catchy mnemonic, rhyme, or word-association that helps someone
  remember the word. It must be different from the simple explanation.
- "examples": an array of 2 simple, everyday sentences that use the word.
"""
        try:
            data = self._ask_gemini_json(prompt)
            examples = data.get("examples", [])
            if isinstance(examples, str):
                examples = [examples]
            return {
                "simple_explanation": self.clean_text(data["simple_explanation"]),
                "memory_trick": self.clean_text(data["memory_trick"]),
                "examples": [self.clean_text(e) for e in examples][:2],
                "error": None,
            }
        except Exception as e:
            logger.error("Gemini failed for '%s': %s", word, e)
            return {"simple_explanation": "", "memory_trick": "", "examples": [], "error": str(e)}

    def generate_distractors(self, word: str, definition: str, count: int = 2) -> List[str]:
        """Wrong-but-believable definitions for quiz options (used when few words are saved)."""
        prompt = f"""Word: "{word}"
Correct definition: "{definition}"

Write {count} WRONG but believable definitions for a multiple-choice quiz.
Keep each about the same length as the correct one and clearly different in meaning.
Return JSON like: {{"wrong_definitions": ["...", "..."]}}
"""
        try:
            data = self._ask_gemini_json(prompt)
            wrong = [self.clean_text(x) for x in data["wrong_definitions"]][:count]
            if len(wrong) == count:
                return wrong
        except Exception as e:
            logger.error("Distractor generation failed: %s", e)
        return random.sample(self.GENERIC_WRONG, count)

    # Dictionary API 
    def fetch_word(self, raw_word: str) -> Word:
        word = self.validate_and_clean_input(raw_word)

        try:
            response = requests.get(f"{self.FREE_DICT_URL}{word}", timeout=(5, 10))
        except requests.exceptions.Timeout as e:
            logger.warning("Dictionary lookup timed out for '%s': %s", word, e)
            raise ConnectionError(
                "The dictionary lookup timed out. Check your internet connection and try again."
            ) from e
        except requests.exceptions.ConnectionError as e:
            logger.warning("Could not connect to the dictionary service for '%s': %s", word, e)
            raise ConnectionError(
                "Could not connect to the dictionary service. Check your internet connection and try again."
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error("Dictionary request failed for '%s': %s", word, e)
            raise ConnectionError(
                "The dictionary request failed. Please try again."
            ) from e

        if response.status_code == 404:
            raise LookupError(f"The word '{word}' was not found in the dictionary.")
        if response.status_code != 200:
            raise ConnectionError("The dictionary service returned an error. Try again later.")

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

        ai = self.fetch_ai_content(word, definition)

        
        examples = ai["examples"]
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
            simple_explanation=ai["simple_explanation"],
            memory_trick=ai["memory_trick"],
            ai_error=ai["error"],
        )


class QuizGenerator:
    """Builds multiple-choice quizzes from saved words."""

    def __init__(self, dictionary_client: DictionaryClient):
        self.dictionary_client = dictionary_client

    def generate_quiz(self, flashcards: list, num_questions: int = 5) -> list:
        if not flashcards:
            return []

        targets = random.sample(flashcards, min(num_questions, len(flashcards)))
        questions = []
        for card in targets:
            other_defs = [
                c["definition"] for c in flashcards
                if c["word"] != card["word"] and c.get("definition")
            ]
            random.shuffle(other_defs)
            wrong = other_defs[:2]
            if len(wrong) < 2:
                wrong += self.dictionary_client.generate_distractors(
                    card["word"], card["definition"], 2 - len(wrong)
                )

            options = [card["definition"]] + wrong
            random.shuffle(options)
            questions.append({
                "word": card["word"],
                "question": f"What does '{card['word']}' mean?",
                "correct": card["definition"],
                "options": options,
            })
        return questions


class SpacedRepetitionManager:
    """Decides when a flashcard should be reviewed again."""

    @staticmethod
    def calculate_next_interval(current_interval: int, rating: str) -> int:
        if rating == "hard":
            return 1
        if rating == "good":
            return min(current_interval * 2, 365)
        if rating == "easy":
            return min(current_interval * 3, 365)
        return current_interval

    @staticmethod
    def next_review_date(interval_days: int) -> str:
        return (datetime.now() + timedelta(days=interval_days)).isoformat()


class FileManager:
    """Reads and writes flashcards and quiz scores to a local JSON file."""

    FILE_PATH = "vocab_data.json"

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
            st.error("Could not read your saved data. Starting with an empty list.")
            return {"flashcards": [], "quiz_history": []}

    @classmethod
    def save_data(cls, data: dict) -> bool:
        try:
            with open(cls.FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except IOError as e:
            logger.error("File write error: %s", e)
            st.error("Could not save your data to the file.")
            return False


# ==========================================
# 2. HELPERS
# ==========================================

dictionary_client = DictionaryClient()
quiz_generator = QuizGenerator(dictionary_client)


def fmt_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d %b %Y")


def save_flashcard(word: Word) -> str:
    data = FileManager.load_data()
    if any(c["word"] == word.word for c in data["flashcards"]):
        return "exists"
    data["flashcards"].append(Flashcard(word.to_dict()).to_dict())
    return "saved" if FileManager.save_data(data) else "error"


# ==========================================
# 3. PAGES
# ==========================================

def page_search():
    st.header("🔍 Search a word")

    with st.form("search_form"):
        query = st.text_input("Enter a word", placeholder="e.g. ephemeral")
        submitted = st.form_submit_button("Search")

    if submitted:
        try:
            with st.spinner("Looking it up..."):
                st.session_state.current_word = dictionary_client.fetch_word(query)
        except (ValueError, LookupError) as e:
            st.session_state.current_word = None
            st.warning(str(e))
        except ConnectionError as e:
            st.session_state.current_word = None
            st.error(str(e))
        except Exception as e:
            logger.error("Unexpected error: %s", e)
            st.session_state.current_word = None
            st.error(f"Something went wrong: {e}")

    word = st.session_state.get("current_word")
    if not word:
        return

    st.divider()
    st.subheader(word.word.capitalize())
    st.caption(word.phonetic)

    st.markdown("**Definition**")
    st.write(word.definition)

    if word.ai_error:
        st.warning(f"AI explanation and memory trick are unavailable: {friendly_ai_error(word.ai_error)}")
    else:
        col1, col2 = st.columns(2)
        col1.info(f"💡 **Simple explanation**\n\n{word.simple_explanation}")
        col2.success(f"🧠 **AI memory trick**\n\n{word.memory_trick}")

    if word.examples:
        st.markdown("**Example sentences**")
        for ex in word.examples:
            st.markdown(f"- *{ex}*")

    if word.synonyms:
        st.markdown("**Synonyms:** " + ", ".join(word.synonyms))
    if word.antonyms:
        st.markdown("**Antonyms:** " + ", ".join(word.antonyms))

    if st.button("💾 Save to flashcards"):
        result = save_flashcard(word)
        if result == "saved":
            st.success("Saved! You can review it in the Review page.")
        elif result == "exists":
            st.info("This word is already in your flashcards.")


def page_review():
    st.header("🃏 Review flashcards")

    data = FileManager.load_data()
    cards = [Flashcard.from_dict(c) for c in data["flashcards"]]

    if not cards:
        st.info("No saved words yet. Search for a word and save it first.")
        return

    due = sorted((c for c in cards if c.is_due()), key=lambda c: c.next_review_date)
    st.caption(f"{len(due)} due now · {len(cards)} saved in total")

    if not due:
        upcoming = min(cards, key=lambda c: c.next_review_date)
        st.success(f"🎉 All caught up! Next review: {fmt_date(upcoming.next_review_date)}")
        return

    card = due[0]
    w = card.word_data

    st.subheader(w["word"].capitalize())
    st.caption(w.get("phonetic", ""))

    if st.session_state.get("show_answer") != card.id:
        st.write("Try to remember the meaning, then reveal the answer.")
        if st.button("Show answer"):
            st.session_state.show_answer = card.id
            st.rerun()
        return

    st.markdown("**Definition**")
    st.write(w["definition"])
    if w.get("simple_explanation"):
        st.info(f"💡 {w['simple_explanation']}")
    if w.get("memory_trick"):
        st.success(f"🧠 {w['memory_trick']}")
    if w.get("examples"):
        st.markdown(f"*{w['examples'][0]}*")

    st.write("How well did you remember it?")
    c1, c2, c3 = st.columns(3)
    for col, label, rating in ((c1, "😓 Hard", "hard"), (c2, "🙂 Good", "good"), (c3, "😎 Easy", "easy")):
        if col.button(label, key=f"{rating}_{card.id}"):
            card.interval_days = SpacedRepetitionManager.calculate_next_interval(card.interval_days, rating)
            card.next_review_date = SpacedRepetitionManager.next_review_date(card.interval_days)
            data["flashcards"] = [
                card.to_dict() if c["id"] == card.id else c for c in data["flashcards"]
            ]
            FileManager.save_data(data)
            st.session_state.show_answer = None
            st.rerun()


def page_quiz():
    st.header("📝 Quiz")

    data = FileManager.load_data()
    flashcards = data["flashcards"]

    if not flashcards:
        st.info("Save some words first, then come back to take a quiz.")
    else:
        if st.button("Start new quiz"):
            with st.spinner("Preparing your quiz..."):
                st.session_state.quiz = quiz_generator.generate_quiz(flashcards, 5)
            st.session_state.quiz_result = None
            st.session_state.quiz_id = str(datetime.now().timestamp())

        quiz = st.session_state.get("quiz")
        result = st.session_state.get("quiz_result")

        if quiz and not result:
            with st.form("quiz_form"):
                answers = []
                for i, q in enumerate(quiz):
                    answers.append(
                        st.radio(
                            f"{i + 1}. {q['question']}",
                            q["options"],
                            index=None,
                            key=f"q_{st.session_state.quiz_id}_{i}",
                        )
                    )
                submitted = st.form_submit_button("Submit answers")

            if submitted:
                if any(a is None for a in answers):
                    st.warning("Please answer every question.")
                else:
                    score = sum(1 for a, q in zip(answers, quiz) if a == q["correct"])
                    data["quiz_history"].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "score": score,
                        "total": len(quiz),
                    })
                    FileManager.save_data(data)
                    st.session_state.quiz_result = {
                        "score": score,
                        "total": len(quiz),
                        "review": [
                            {"word": q["word"], "your": a, "correct": q["correct"], "ok": a == q["correct"]}
                            for a, q in zip(answers, quiz)
                        ],
                    }
                    st.rerun()

        if result:
            st.success(f"Your score: {result['score']} / {result['total']}")
            for r in result["review"]:
                if r["ok"]:
                    st.markdown(f"✅ **{r['word']}**: correct")
                else:
                    st.markdown(f"❌ **{r['word']}**: the answer was *{r['correct']}*")

    history = data["quiz_history"]
    if history:
        st.divider()
        st.subheader("Past scores")
        best = max(h["score"] / h["total"] for h in history)
        st.caption(f"Quizzes taken: {len(history)} · Best score: {round(best * 100)}%")
        rows = [
            {"Date": h["date"], "Score": f"{h['score']}/{h['total']}"}
            for h in reversed(history)
        ]
        st.dataframe(rows)


def page_my_words():
    st.header("📚 My words")

    data = FileManager.load_data()
    if not data["flashcards"]:
        st.info("You haven't saved any words yet.")
        return

    for c in sorted(data["flashcards"], key=lambda x: x["word"]):
        with st.expander(c["word"].capitalize()):
            st.write(c["definition"])
            if c.get("simple_explanation"):
                st.info(f"💡 {c['simple_explanation']}")
            if c.get("memory_trick"):
                st.success(f"🧠 {c['memory_trick']}")
            st.caption(f"Next review: {fmt_date(c['next_review_date'])}")
            if st.button("🗑️ Delete", key=f"del_{c['id']}"):
                data["flashcards"] = [x for x in data["flashcards"] if x["id"] != c["id"]]
                FileManager.save_data(data)
                st.rerun()


# ==========================================
# 4. MAIN APP
# ==========================================

st.title("📚 Vocab Builder")

if not GEMINI_API_KEY:
    st.sidebar.warning("GEMINI_API_KEY not found. Add it to your .env file and restart.")

page = st.sidebar.radio("Menu", ["🔍 Search", "🃏 Review", "📝 Quiz", "📚 My Words"])

if page == "🔍 Search":
    page_search()
elif page == "🃏 Review":
    page_review()
elif page == "📝 Quiz":
    page_quiz()
else:
    page_my_words()
