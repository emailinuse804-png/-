#!/usr/bin/env python3
"""
=============================================================================
AURORA - Advanced Unified Reasoning and Operational Response Assistant
=============================================================================
A fully functional, local AI assistant with NO external API dependencies.
Features:
- Natural Language Processing with pattern matching
- Intent classification using Naive Bayes
- Entity extraction (names, dates, numbers, emails, etc.)
- Knowledge base with learning capability
- Sentiment analysis
- Multiple skills: math, conversions, weather simulation, jokes, facts, games
- Conversation context and memory
- Task management (reminders, notes, todos)
- Text analysis tools
=============================================================================
"""

import re
import os
import json
import math
import random
import hashlib
import datetime
import calendar
import pickle
import string
import functools
import operator
import statistics
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import difflib
import textwrap
import time


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class Config:
    """Global configuration settings."""
    APP_NAME = "AURORA"
    VERSION = "3.0.0"
    DATA_DIR = Path.home() / ".aurora_ai"
    KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"
    MEMORY_FILE = DATA_DIR / "memory.json"
    NOTES_FILE = DATA_DIR / "notes.json"
    TODOS_FILE = DATA_DIR / "todos.json"
    REMINDERS_FILE = DATA_DIR / "reminders.json"
    CONVERSATION_FILE = DATA_DIR / "conversations.json"
    USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
    PERSONAL_FACTS_FILE = DATA_DIR / "personal_facts.json"
    MAX_CONTEXT_LENGTH = 20
    LEARNING_ENABLED = True
    DEBUG_MODE = False


# Initialize data directory
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Intent(Enum):
    """All recognized user intents."""
    GREETING = auto()
    FAREWELL = auto()
    THANKS = auto()
    HELP = auto()
    MATH = auto()
    WEATHER = auto()
    TIME = auto()
    DATE = auto()
    JOKE = auto()
    FACT = auto()
    DEFINE = auto()
    TRANSLATE = auto()
    CONVERT = auto()
    REMINDER = auto()
    NOTE = auto()
    TODO = auto()
    SEARCH = auto()
    CALCULATE = auto()
    QUESTION = auto()
    SENTIMENT = auto()
    SUMMARIZE = auto()
    GAME = auto()
    QUOTE = auto()
    ADVICE = auto()
    STORY = auto()
    TRIVIA = auto()
    RIDDLE = auto()
    COMPLIMENT = auto()
    INSULT_RESPONSE = auto()
    IDENTITY = auto()
    CAPABILITY = auto()
    MOOD = auto()
    LEARN = auto()
    RECALL = auto()
    FORGET = auto()
    LIST_NOTES = auto()
    LIST_TODOS = auto()
    LIST_REMINDERS = auto()
    COMPLETE_TODO = auto()
    DELETE_NOTE = auto()
    DELETE_TODO = auto()
    DELETE_REMINDER = auto()
    COUNTDOWN = auto()
    TIMER = auto()
    RANDOM_NUMBER = auto()
    COIN_FLIP = auto()
    DICE_ROLL = auto()
    PICK_RANDOM = auto()
    SPELL_CHECK = auto()
    WORD_COUNT = auto()
    CHARACTER_COUNT = auto()
    REVERSE_TEXT = auto()
    UPPERCASE = auto()
    LOWERCASE = auto()
    TITLE_CASE = auto()
    PALINDROME = auto()
    ANAGRAM = auto()
    RHYME = auto()
    SYNONYM = auto()
    ANTONYM = auto()
    ACRONYM = auto()
    ENCRYPT = auto()
    DECRYPT = auto()
    HASH = auto()
    PASSWORD = auto()
    COLOR = auto()
    ASCII_ART = auto()
    MOTIVATE = auto()
    MEDITATE = auto()
    BREATHE = auto()
    AFFIRMATION = auto()
    HOROSCOPE = auto()
    FORTUNE = auto()
    MAGIC_8BALL = auto()
    MY_PROFILE = auto()  # View stored user info
    FORGET_ME = auto()   # Delete user data
    UNKNOWN = auto()


@dataclass
class Entity:
    """Extracted entity from text."""
    entity_type: str
    value: Any
    original_text: str
    start: int
    end: int


@dataclass
class ParsedInput:
    """Result of parsing user input."""
    original: str
    cleaned: str
    tokens: List[str]
    intent: Intent
    confidence: float
    entities: List[Entity]
    sentiment: float
    keywords: List[str]


@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    user_input: str
    bot_response: str
    intent: Intent
    timestamp: datetime.datetime
    entities: List[Entity] = field(default_factory=list)


@dataclass
class Memory:
    """Long-term memory item."""
    key: str
    value: Any
    created_at: datetime.datetime
    accessed_count: int = 0
    last_accessed: datetime.datetime = None


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Handles text cleaning and normalization."""
    
    CONTRACTIONS = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "he'll": "he will", "he'd": "he would",
        "she's": "she is", "she'll": "she will", "she'd": "she would",
        "it's": "it is", "it'll": "it will", "it'd": "it would",
        "we're": "we are", "we've": "we have", "we'll": "we will", "we'd": "we would",
        "they're": "they are", "they've": "they have", "they'll": "they will", "they'd": "they would",
        "that's": "that is", "that'll": "that will", "that'd": "that would",
        "what's": "what is", "what're": "what are", "what'll": "what will",
        "who's": "who is", "who're": "who are", "who'll": "who will",
        "where's": "where is", "where'll": "where will",
        "when's": "when is", "when'll": "when will",
        "why's": "why is", "why'll": "why will",
        "how's": "how is", "how'll": "how will",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "won't": "will not", "wouldn't": "would not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "can't": "cannot",
        "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
        "mustn't": "must not", "let's": "let us", "here's": "here is",
        "there's": "there is", "gonna": "going to", "wanna": "want to",
        "gotta": "got to", "kinda": "kind of", "sorta": "sort of",
        "lemme": "let me", "gimme": "give me", "dunno": "do not know",
        "ain't": "is not", "y'all": "you all", "c'mon": "come on",
    }
    
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'at', 'by',
        'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    }
    
    @classmethod
    def clean(cls, text: str) -> str:
        """Clean and normalize text."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\'\-\.\,\?\!]', ' ', text)
        for contraction, expansion in cls.CONTRACTIONS.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize text into words."""
        return [w for w in re.findall(r'\b\w+\b', text.lower()) if len(w) > 0]
    
    @classmethod
    def remove_stopwords(cls, tokens: List[str]) -> List[str]:
        """Remove common stopwords."""
        return [t for t in tokens if t not in cls.STOPWORDS]
    
    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 5) -> List[str]:
        """Extract important keywords from text."""
        tokens = cls.tokenize(text)
        tokens = cls.remove_stopwords(tokens)
        freq = Counter(tokens)
        return [word for word, _ in freq.most_common(top_n)]


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

class EntityExtractor:
    """Extracts entities from text."""
    
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
        'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)\b',
        'time': r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm|AM|PM)|(?:2[0-3]|[01]?[0-9]):[0-5][0-9](?::[0-5][0-9])?\b',
        'number': r'\b-?\d+(?:\.\d+)?\b',
        'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|euros?|eur|pounds?|gbp)',
        'percentage': r'\b\d+(?:\.\d+)?%\b',
        'ordinal': r'\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
        'duration': r'\b\d+\s*(?:second|minute|hour|day|week|month|year)s?\b',
        'temperature': r'-?\d+(?:\.\d+)?\s*°?\s*(?:celsius|fahrenheit|c|f)\b',
        'distance': r'\b\d+(?:\.\d+)?\s*(?:km|kilometer|mile|meter|foot|feet|inch|yard)s?\b',
        'weight': r'\b\d+(?:\.\d+)?\s*(?:kg|kilogram|pound|lb|ounce|oz|gram|g)s?\b',
        'name': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    }
    
    MONTHS = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    RELATIVE_DATES = {
        'today': 0, 'tomorrow': 1, 'yesterday': -1,
        'next week': 7, 'next month': 30, 'next year': 365,
    }
    
    @classmethod
    def extract_all(cls, text: str) -> List[Entity]:
        """Extract all entities from text."""
        entities = []
        for entity_type, pattern in cls.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = cls._parse_value(entity_type, match.group())
                entities.append(Entity(
                    entity_type=entity_type,
                    value=value,
                    original_text=match.group(),
                    start=match.start(),
                    end=match.end()
                ))
        entities.extend(cls._extract_relative_dates(text))
        return entities
    
    @classmethod
    def _parse_value(cls, entity_type: str, text: str) -> Any:
        """Parse entity value to appropriate type."""
        if entity_type == 'number':
            try:
                return float(text) if '.' in text else int(text)
            except ValueError:
                return text
        elif entity_type == 'percentage':
            try:
                return float(text.replace('%', ''))
            except ValueError:
                return text
        return text
    
    @classmethod
    def _extract_relative_dates(cls, text: str) -> List[Entity]:
        """Extract relative date references."""
        entities = []
        text_lower = text.lower()
        for phrase, days in cls.RELATIVE_DATES.items():
            if phrase in text_lower:
                target_date = datetime.date.today() + datetime.timedelta(days=days)
                start = text_lower.find(phrase)
                entities.append(Entity(
                    entity_type='relative_date',
                    value=target_date,
                    original_text=phrase,
                    start=start,
                    end=start + len(phrase)
                ))
        return entities
    
    @classmethod
    def extract_numbers(cls, text: str) -> List[float]:
        """Extract all numbers from text."""
        return [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)]


# =============================================================================
# INTENT CLASSIFICATION - NAIVE BAYES
# =============================================================================

class NaiveBayesClassifier:
    """Simple Naive Bayes classifier for intent detection."""
    
    def __init__(self):
        self.word_counts: Dict[Intent, Counter] = defaultdict(Counter)
        self.intent_counts: Counter = Counter()
        self.vocabulary: Set[str] = set()
        self.total_docs = 0
        
    def train(self, text: str, intent: Intent):
        """Train on a single example."""
        tokens = TextPreprocessor.tokenize(text)
        self.word_counts[intent].update(tokens)
        self.intent_counts[intent] += 1
        self.vocabulary.update(tokens)
        self.total_docs += 1
    
    def train_batch(self, examples: List[Tuple[str, Intent]]):
        """Train on multiple examples."""
        for text, intent in examples:
            self.train(text, intent)
    
    def predict(self, text: str) -> Tuple[Intent, float]:
        """Predict intent with confidence score."""
        tokens = TextPreprocessor.tokenize(text)
        scores = {}
        vocab_size = len(self.vocabulary) + 1
        
        for intent in self.intent_counts:
            log_prob = math.log(self.intent_counts[intent] / self.total_docs)
            total_words = sum(self.word_counts[intent].values())
            for token in tokens:
                word_count = self.word_counts[intent].get(token, 0)
                log_prob += math.log((word_count + 1) / (total_words + vocab_size))
            scores[intent] = log_prob
        
        if not scores:
            return Intent.UNKNOWN, 0.0
        
        best_intent = max(scores, key=scores.get)
        probs = self._softmax(list(scores.values()))
        confidence = max(probs)
        return best_intent, confidence
    
    @staticmethod
    def _softmax(x: List[float]) -> List[float]:
        """Compute softmax probabilities."""
        max_x = max(x)
        exp_x = [math.exp(i - max_x) for i in x]
        sum_exp = sum(exp_x)
        return [i / sum_exp for i in exp_x]


# =============================================================================
# PATTERN-BASED INTENT MATCHER
# =============================================================================

class PatternMatcher:
    """Pattern-based intent matching for high confidence cases."""
    
    PATTERNS = {
        Intent.GREETING: [
            r'\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|day)|sup|yo|hiya|aloha|namaste)\b',
            r'^(hi|hello|hey)[\s\!\.\,]*$',
        ],
        Intent.FAREWELL: [
            r'\b(bye|goodbye|farewell|see\s*you|later|take\s*care|good\s*night|cya|ttyl|peace\s*out|adios|cheerio)\b',
        ],
        Intent.THANKS: [
            r'\b(thanks?|thank\s*you|thx|ty|appreciate|grateful|cheers)\b',
        ],
        Intent.HELP: [
            r'\b(help|assist|support|guide|what\s*can\s*you\s*do|features|commands|options)\b',
            r'how\s*(do|can)\s*(i|you)',
        ],
        Intent.MATH: [
            r'\b(calculate|compute|solve|math|equation|formula)\b',
            r'what\s*(is|are|\'s)\s*\d+\s*[\+\-\*\/\^\%]\s*\d+',
            r'\d+\s*[\+\-\*\/\^\%]\s*\d+',
        ],
        Intent.TIME: [
            r'\b(what\s*(time|is\s*the\s*time)|current\s*time|time\s*(now|is\s*it)|clock)\b',
        ],
        Intent.DATE: [
            r'\b(what\s*(date|day|is\s*the\s*date|is\s*today)|current\s*date|today\'?s?\s*date|which\s*day)\b',
        ],
        Intent.WEATHER: [
            r'\b(weather|temperature|forecast|rain|sunny|cloudy|storm|snow|humid)\b',
        ],
        Intent.JOKE: [
            r'\b(joke|funny|laugh|humor|humour|make\s*me\s*laugh|something\s*funny|comedy)\b',
        ],
        Intent.FACT: [
            r'\b(fact|did\s*you\s*know|interesting|trivia|fun\s*fact|tell\s*me\s*something)\b',
        ],
        Intent.DEFINE: [
            r'\b(define|definition|meaning\s*of|what\s*(does|is)\s*\w+\s*mean|dictionary)\b',
        ],
        Intent.CONVERT: [
            r'\b(convert|conversion|how\s*many\s*\w+\s*in|transform|change\s*\w+\s*to)\b',
            r'\d+\s*\w+\s*(to|in)\s*\w+',
        ],
        Intent.REMINDER: [
            r'\b(remind|reminder|set\s*a?\s*reminder|alert|notify|don\'t\s*let\s*me\s*forget)\b',
        ],
        Intent.NOTE: [
            r'\b(note|write\s*down|jot|record|save\s*this|remember\s*this|take\s*a?\s*note)\b',
        ],
        Intent.TODO: [
            r'\b(todo|to-do|task|add\s*to\s*list|add\s*task)\b',
        ],
        Intent.QUOTE: [
            r'\b(quote|quotation|wisdom|inspirational|inspiring|motivational)\b',
        ],
        Intent.STORY: [
            r'\b(story|tale|narrative|once\s*upon|tell\s*me\s*a\s*story)\b',
        ],
        Intent.IDENTITY: [
            r'\b(who\s*are\s*you|your\s*name|what\s*are\s*you|introduce\s*yourself)\b',
        ],
        Intent.CAPABILITY: [
            r'\b(what\s*can\s*you\s*(do|help)|your\s*(abilities|capabilities|features)|help\s*me\s*with)\b',
        ],
        Intent.MOOD: [
            r'\b(how\s*are\s*you|how\'s\s*it\s*going|what\'s\s*up|how\s*do\s*you\s*feel)\b',
        ],
        Intent.COIN_FLIP: [
            r'\b(flip\s*a?\s*coin|coin\s*flip|heads\s*or\s*tails|toss\s*a?\s*coin)\b',
        ],
        Intent.DICE_ROLL: [
            r'\b(roll\s*(a\s*)?(dice?|d\d+)|dice?\s*roll|throw\s*(the\s*)?dice?)\b',
        ],
        Intent.RANDOM_NUMBER: [
            r'\b(random\s*number|pick\s*a\s*number|generate\s*(a\s*)?number)\b',
        ],
        Intent.PASSWORD: [
            r'\b(password|generate\s*password|random\s*password|secure\s*password|passphrase)\b',
        ],
        Intent.MAGIC_8BALL: [
            r'\b(magic\s*8\s*ball|8\s*ball|fortune|predict|will\s*i|should\s*i|is\s*it\s*true)\b',
        ],
        Intent.MOTIVATE: [
            r'\b(motivate|motivation|inspire|encourage|cheer\s*me\s*up|i\s*need\s*motivation)\b',
        ],
        Intent.AFFIRMATION: [
            r'\b(affirmation|self\s*love|positive\s*thought|i\s*am\s*enough)\b',
        ],
        Intent.MEDITATE: [
            r'\b(meditat|mindful|calm|relax|breathing\s*exercise|peace)\b',
        ],
        Intent.LEARN: [
            r'\b(learn|teach\s*you|remember|memorize|know\s*that)\b',
        ],
        Intent.RECALL: [
            r'\b(recall|what\s*do\s*you\s*know|what\s*did\s*i|do\s*you\s*remember)\b',
        ],
        Intent.MY_PROFILE: [
            r'\b(my\s*profile|about\s*me|what\s*do\s*you\s*know\s*about\s*me|my\s*info|my\s*information)\b',
            r'\b(what\s*have\s*you\s*learned|what\s*did\s*you\s*learn|show\s*my\s*data)\b',
            r'\b(who\s*am\s*i|what\s*is\s*my\s*name|my\s*details)\b',
        ],
        Intent.FORGET_ME: [
            r'\b(forget\s*(about\s*)?me|delete\s*my\s*(data|info|profile)|clear\s*my\s*data)\b',
            r'\b(erase\s*(my\s*)?(data|memory|profile)|reset\s*my\s*profile)\b',
        ],
        Intent.SENTIMENT: [
            r'\b(sentiment|analyze\s*feeling|emotion|mood\s*of|how\s*does\s*this\s*sound)\b',
        ],
        Intent.SUMMARIZE: [
            r'\b(summarize|summary|brief|tldr|shorten|condense)\b',
        ],
        Intent.WORD_COUNT: [
            r'\b(word\s*count|how\s*many\s*words|count\s*words)\b',
        ],
        Intent.ASCII_ART: [
            r'\b(ascii|text\s*art|draw|art\s*of)\b',
        ],
        Intent.ENCRYPT: [
            r'\b(encrypt|encode|cipher|secret\s*message)\b',
        ],
        Intent.DECRYPT: [
            r'\b(decrypt|decode|decipher)\b',
        ],
        Intent.REVERSE_TEXT: [
            r'\b(reverse|backwards|flip\s*text)\b',
        ],
        Intent.TRIVIA: [
            r'\b(trivia|quiz|test\s*me|question\s*me)\b',
        ],
        Intent.RIDDLE: [
            r'\b(riddle|puzzle|brain\s*teaser)\b',
        ],
    }
    
    @classmethod
    def match(cls, text: str) -> Optional[Tuple[Intent, float]]:
        """Match text against patterns, return intent and confidence."""
        text_lower = text.lower()
        for intent, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent, 0.9
        return None


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================

class SentimentAnalyzer:
    """Analyze sentiment of text."""
    
    POSITIVE_WORDS = {
        'good', 'great', 'awesome', 'amazing', 'wonderful', 'fantastic', 'excellent',
        'love', 'happy', 'joy', 'beautiful', 'perfect', 'best', 'brilliant',
        'superb', 'outstanding', 'magnificent', 'delightful', 'pleasant', 'nice',
        'positive', 'exciting', 'incredible', 'remarkable', 'fabulous', 'terrific',
        'lovely', 'grateful', 'thankful', 'blessed', 'cheerful', 'glad', 'pleased',
        'satisfied', 'content', 'thrilled', 'ecstatic', 'elated', 'overjoyed',
        'like', 'enjoy', 'appreciate', 'admire', 'adore', 'cherish', 'treasure',
    }
    
    NEGATIVE_WORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate', 'sad',
        'angry', 'ugly', 'wrong', 'fail', 'failure', 'disappointing', 'disappointed',
        'negative', 'boring', 'annoying', 'frustrating', 'irritating', 'depressing',
        'miserable', 'unhappy', 'upset', 'worried', 'anxious', 'stressed', 'tired',
        'exhausted', 'painful', 'hurt', 'broken', 'ruined', 'destroyed', 'useless',
        'stupid', 'dumb', 'idiotic', 'ridiculous', 'pathetic', 'worthless', 'lousy',
        'dislike', 'despise', 'loathe', 'detest', 'regret', 'resent',
    }
    
    INTENSIFIERS = {
        'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
        'incredibly': 2.0, 'super': 1.5, 'so': 1.3, 'too': 1.3, 'quite': 1.2,
        'pretty': 1.2, 'especially': 1.5, 'particularly': 1.5, 'truly': 1.5,
    }
    
    NEGATORS = {'not', 'no', 'never', 'none', 'neither', 'nobody', 'nothing', 'nowhere', "n't", 'nt'}
    
    @classmethod
    def analyze(cls, text: str) -> float:
        """Analyze sentiment, returns score from -1 (negative) to 1 (positive)."""
        tokens = TextPreprocessor.tokenize(text)
        if not tokens:
            return 0.0
        
        score = 0.0
        negation = False
        intensifier = 1.0
        
        for i, token in enumerate(tokens):
            if token in cls.NEGATORS:
                negation = True
                continue
            
            if token in cls.INTENSIFIERS:
                intensifier = cls.INTENSIFIERS[token]
                continue
            
            word_score = 0.0
            if token in cls.POSITIVE_WORDS:
                word_score = 1.0
            elif token in cls.NEGATIVE_WORDS:
                word_score = -1.0
            
            if negation:
                word_score *= -0.5
                negation = False
            
            word_score *= intensifier
            intensifier = 1.0
            score += word_score
        
        normalized = score / (len(tokens) ** 0.5) if tokens else 0.0
        return max(-1.0, min(1.0, normalized))
    
    @classmethod
    def get_sentiment_label(cls, score: float) -> str:
        """Convert score to human-readable label."""
        if score >= 0.5:
            return "very positive 😄"
        elif score >= 0.2:
            return "positive 🙂"
        elif score >= -0.2:
            return "neutral 😐"
        elif score >= -0.5:
            return "negative 😕"
        else:
            return "very negative 😢"


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class KnowledgeBase:
    """Stores and retrieves knowledge."""
    
    BUILT_IN_KNOWLEDGE = {
        "what is python": "Python is a high-level, interpreted programming language known for its readability and versatility. Created by Guido van Rossum in 1991.",
        "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence by machines, enabling them to learn, reason, and solve problems.",
        "what is machine learning": "Machine Learning is a subset of AI where systems learn from data to improve performance on tasks without being explicitly programmed.",
        "who created you": "I was created as Aurora, a local AI assistant built in Python. I don't require internet or APIs to function!",
        "meaning of life": "According to Douglas Adams' 'The Hitchhiker's Guide to the Galaxy', the answer is 42. Philosophically, it's about finding purpose and happiness.",
        "what is love": "Love is a complex emotion involving care, attachment, and affection. It comes in many forms: romantic, familial, platonic, and self-love.",
        "what is consciousness": "Consciousness is the state of being aware of one's surroundings, thoughts, and existence. It remains one of philosophy's and science's greatest mysteries.",
        "what is the universe": "The universe is all of space, time, matter, and energy that exists. It's approximately 13.8 billion years old and contains over 100 billion galaxies.",
        "what is gravity": "Gravity is a fundamental force that attracts objects with mass toward each other. Einstein described it as the curvature of spacetime caused by mass.",
        "what is time": "Time is the indefinite continued progress of existence. Einstein showed it's relative - it passes differently depending on speed and gravity.",
        "what is happiness": "Happiness is a state of well-being characterized by positive emotions, satisfaction, and contentment. It's often found in relationships, purpose, and gratitude.",
        "what is the internet": "The Internet is a global network of interconnected computers that communicate using standardized protocols, enabling information sharing worldwide.",
        "what is a computer": "A computer is an electronic device that processes data according to instructions (programs), performing calculations and operations at high speed.",
        "what is programming": "Programming is the process of creating instructions (code) that tell computers what to do. It involves logic, algorithms, and specific languages.",
        "what is an algorithm": "An algorithm is a step-by-step procedure for solving a problem or accomplishing a task, forming the foundation of computer programming.",
        "what is data": "Data is information in a form suitable for processing by computers. It can be numbers, text, images, sounds, or any other digitized content.",
        "what is a database": "A database is an organized collection of structured data stored electronically, designed for efficient storage, retrieval, and management.",
        "what is the sun": "The Sun is a star at the center of our solar system. It's a nearly perfect sphere of hot plasma, about 4.6 billion years old.",
        "what is the moon": "The Moon is Earth's only natural satellite, formed about 4.5 billion years ago. It influences tides and stabilizes Earth's axial tilt.",
        "what is earth": "Earth is the third planet from the Sun, the only known world with liquid water on its surface and the only confirmed home of life.",
        "what is water": "Water (H2O) is a transparent, colorless chemical substance essential for all known forms of life. It covers about 71% of Earth's surface.",
        "what is fire": "Fire is the rapid oxidation of material in combustion, releasing heat, light, and various reaction products. Humans have used it for over 1 million years.",
        "what is energy": "Energy is the capacity to do work. It exists in many forms: kinetic, potential, thermal, electrical, chemical, and nuclear.",
        "what is evolution": "Evolution is the process of change in living organisms over generations through natural selection, genetic drift, and mutation.",
        "what is dna": "DNA (Deoxyribonucleic acid) is the molecule that carries genetic instructions for development, functioning, and reproduction of all living organisms.",
        "what is a cell": "A cell is the basic structural and functional unit of all living organisms. Humans have about 37 trillion cells.",
        "what is photosynthesis": "Photosynthesis is the process by which green plants convert sunlight, water, and CO2 into glucose and oxygen.",
        "what is democracy": "Democracy is a system of government where citizens exercise power by voting to elect representatives who make decisions on their behalf.",
        "what is philosophy": "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language.",
        "what is science": "Science is the systematic study of the natural world through observation, experimentation, and the formulation of testable hypotheses.",
        "what is art": "Art is the expression of human creativity and imagination in various forms such as painting, music, literature, and dance.",
        "what is music": "Music is an art form that combines sounds and silence organized in time, often evoking emotional responses in listeners.",
        "what is mathematics": "Mathematics is the abstract science of number, quantity, and space, using logical reasoning to study patterns and relationships.",
        "what is physics": "Physics is the natural science that studies matter, energy, and their interactions through fundamental forces.",
        "what is chemistry": "Chemistry is the scientific study of matter and its properties, composition, structure, and the changes it undergoes during chemical reactions.",
        "what is biology": "Biology is the scientific study of life and living organisms, including their structure, function, growth, evolution, and distribution.",
        "what is history": "History is the study of past events, particularly in human affairs, analyzed and interpreted to understand how societies developed.",
        "what is language": "Language is a structured system of communication using sounds, symbols, or gestures, enabling humans to express thoughts and ideas.",
    }
    
    def __init__(self):
        self.custom_knowledge: Dict[str, str] = {}
        self._load()
    
    def _load(self):
        """Load custom knowledge from file."""
        if Config.KNOWLEDGE_FILE.exists():
            try:
                with open(Config.KNOWLEDGE_FILE, 'r') as f:
                    self.custom_knowledge = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.custom_knowledge = {}
    
    def _save(self):
        """Save custom knowledge to file."""
        with open(Config.KNOWLEDGE_FILE, 'w') as f:
            json.dump(self.custom_knowledge, f, indent=2)
    
    def add(self, key: str, value: str) -> bool:
        """Add new knowledge."""
        key = key.lower().strip()
        self.custom_knowledge[key] = value
        self._save()
        return True
    
    def get(self, query: str) -> Optional[str]:
        """Get knowledge matching query."""
        query = query.lower().strip()
        if query in self.custom_knowledge:
            return self.custom_knowledge[query]
        if query in self.BUILT_IN_KNOWLEDGE:
            return self.BUILT_IN_KNOWLEDGE[query]
        best_match = None
        best_ratio = 0.0
        all_knowledge = {**self.BUILT_IN_KNOWLEDGE, **self.custom_knowledge}
        for key in all_knowledge:
            ratio = difflib.SequenceMatcher(None, query, key).ratio()
            if ratio > best_ratio and ratio > 0.6:
                best_ratio = ratio
                best_match = key
        if best_match:
            return all_knowledge[best_match]
        return None
    
    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """Search knowledge base, return matches with relevance scores."""
        results = []
        query_lower = query.lower()
        all_knowledge = {**self.BUILT_IN_KNOWLEDGE, **self.custom_knowledge}
        for key, value in all_knowledge.items():
            score = 0.0
            if query_lower in key or query_lower in value.lower():
                score = 0.8
            else:
                score = difflib.SequenceMatcher(None, query_lower, key).ratio()
            if score > 0.3:
                results.append((key, value, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]
    
    def forget(self, key: str) -> bool:
        """Remove custom knowledge."""
        key = key.lower().strip()
        if key in self.custom_knowledge:
            del self.custom_knowledge[key]
            self._save()
            return True
        return False
    
    def list_custom(self) -> List[str]:
        """List all custom knowledge keys."""
        return list(self.custom_knowledge.keys())


# =============================================================================
# MEMORY MANAGER
# =============================================================================

class MemoryManager:
    """Manages user-related memories and preferences."""
    
    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.user_name: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}
        self._load()
    
    def _load(self):
        """Load memories from file."""
        if Config.MEMORY_FILE.exists():
            try:
                with open(Config.MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.user_name = data.get('user_name')
                    self.user_preferences = data.get('preferences', {})
                    for key, mem in data.get('memories', {}).items():
                        self.memories[key] = Memory(
                            key=mem['key'],
                            value=mem['value'],
                            created_at=datetime.datetime.fromisoformat(mem['created_at']),
                            accessed_count=mem.get('accessed_count', 0),
                            last_accessed=datetime.datetime.fromisoformat(mem['last_accessed']) if mem.get('last_accessed') else None
                        )
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save(self):
        """Save memories to file."""
        data = {
            'user_name': self.user_name,
            'preferences': self.user_preferences,
            'memories': {
                key: {
                    'key': mem.key,
                    'value': mem.value,
                    'created_at': mem.created_at.isoformat(),
                    'accessed_count': mem.accessed_count,
                    'last_accessed': mem.last_accessed.isoformat() if mem.last_accessed else None
                }
                for key, mem in self.memories.items()
            }
        }
        with open(Config.MEMORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember(self, key: str, value: Any) -> bool:
        """Store a memory."""
        self.memories[key.lower()] = Memory(
            key=key.lower(),
            value=value,
            created_at=datetime.datetime.now()
        )
        self._save()
        return True
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall a memory."""
        key = key.lower()
        if key in self.memories:
            mem = self.memories[key]
            mem.accessed_count += 1
            mem.last_accessed = datetime.datetime.now()
            self._save()
            return mem.value
        return None
    
    def forget(self, key: str) -> bool:
        """Forget a memory."""
        key = key.lower()
        if key in self.memories:
            del self.memories[key]
            self._save()
            return True
        return False
    
    def set_user_name(self, name: str):
        """Set the user's name."""
        self.user_name = name
        self._save()
    
    def get_user_name(self) -> Optional[str]:
        """Get the user's name."""
        return self.user_name
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.user_preferences[key] = value
        self._save()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.user_preferences.get(key, default)


# =============================================================================
# USER PROFILE - Persistent Identity & Personal Facts
# =============================================================================

class UserProfile:
    """Manages persistent user identity, personal facts, and learned information about the user."""
    
    def __init__(self):
        self.profile: Dict[str, Any] = {
            'name': None,
            'nickname': None,
            'email': None,
            'age': None,
            'birthday': None,
            'location': None,
            'occupation': None,
            'created_at': None,
            'last_seen': None,
            'interaction_count': 0,
        }
        self.personal_facts: Dict[str, str] = {}  # Things the user has told us about themselves
        self.preferences: Dict[str, Any] = {}  # User preferences
        self.relationships: Dict[str, str] = {}  # People the user has mentioned (name -> relationship)
        self.interests: List[str] = []  # User's interests/hobbies
        self.conversation_topics: List[str] = []  # Topics frequently discussed
        self._load()
    
    def _load(self):
        """Load user profile from file."""
        if Config.USER_PROFILE_FILE.exists():
            try:
                with open(Config.USER_PROFILE_FILE, 'r') as f:
                    data = json.load(f)
                    self.profile = data.get('profile', self.profile)
                    self.personal_facts = data.get('personal_facts', {})
                    self.preferences = data.get('preferences', {})
                    self.relationships = data.get('relationships', {})
                    self.interests = data.get('interests', [])
                    self.conversation_topics = data.get('conversation_topics', [])
            except (json.JSONDecodeError, IOError):
                pass
    
    def _save(self):
        """Save user profile to file."""
        data = {
            'profile': self.profile,
            'personal_facts': self.personal_facts,
            'preferences': self.preferences,
            'relationships': self.relationships,
            'interests': self.interests,
            'conversation_topics': self.conversation_topics,
        }
        with open(Config.USER_PROFILE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update_last_seen(self):
        """Update last seen timestamp and interaction count."""
        self.profile['last_seen'] = datetime.datetime.now().isoformat()
        self.profile['interaction_count'] += 1
        if not self.profile['created_at']:
            self.profile['created_at'] = datetime.datetime.now().isoformat()
        self._save()
    
    # --- Name Management ---
    def set_name(self, name: str):
        """Set the user's name."""
        self.profile['name'] = name.strip().title()
        self._save()
    
    def get_name(self) -> Optional[str]:
        """Get the user's name."""
        return self.profile.get('nickname') or self.profile.get('name')
    
    def set_nickname(self, nickname: str):
        """Set a nickname for the user."""
        self.profile['nickname'] = nickname.strip()
        self._save()
    
    # --- Email Management ---
    def set_email(self, email: str):
        """Set the user's email."""
        self.profile['email'] = email.strip().lower()
        self._save()
    
    def get_email(self) -> Optional[str]:
        """Get the user's email."""
        return self.profile.get('email')
    
    # --- Personal Info ---
    def set_age(self, age: int):
        """Set the user's age."""
        self.profile['age'] = age
        self._save()
    
    def set_birthday(self, birthday: str):
        """Set the user's birthday."""
        self.profile['birthday'] = birthday
        self._save()
    
    def set_location(self, location: str):
        """Set the user's location."""
        self.profile['location'] = location.strip().title()
        self._save()
    
    def set_occupation(self, occupation: str):
        """Set the user's occupation."""
        self.profile['occupation'] = occupation.strip()
        self._save()
    
    # --- Personal Facts (Things user tells us) ---
    def add_fact(self, category: str, fact: str):
        """Add a personal fact about the user."""
        category = category.lower().strip()
        self.personal_facts[category] = fact.strip()
        self._save()
    
    def get_fact(self, category: str) -> Optional[str]:
        """Get a personal fact about the user."""
        return self.personal_facts.get(category.lower().strip())
    
    def get_all_facts(self) -> Dict[str, str]:
        """Get all personal facts."""
        return self.personal_facts.copy()
    
    # --- Relationships ---
    def add_relationship(self, name: str, relationship: str):
        """Add a person the user knows (e.g., 'Mom' -> 'mother', 'John' -> 'friend')."""
        self.relationships[name.strip().title()] = relationship.strip().lower()
        self._save()
    
    def get_relationships(self) -> Dict[str, str]:
        """Get all known relationships."""
        return self.relationships.copy()
    
    # --- Interests ---
    def add_interest(self, interest: str):
        """Add a user interest."""
        interest = interest.strip().lower()
        if interest not in self.interests:
            self.interests.append(interest)
            self._save()
    
    def get_interests(self) -> List[str]:
        """Get user interests."""
        return self.interests.copy()
    
    # --- Preference Management ---
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key.lower()] = value
        self._save()
    
    def get_preference(self, key: str) -> Optional[Any]:
        """Get a user preference."""
        return self.preferences.get(key.lower())
    
    # --- Summary & Context ---
    def get_summary(self) -> str:
        """Get a summary of what we know about the user."""
        lines = []
        name = self.get_name()
        if name:
            lines.append(f"Name: {name}")
        if self.profile.get('email'):
            lines.append(f"Email: {self.profile['email']}")
        if self.profile.get('age'):
            lines.append(f"Age: {self.profile['age']}")
        if self.profile.get('location'):
            lines.append(f"Location: {self.profile['location']}")
        if self.profile.get('occupation'):
            lines.append(f"Occupation: {self.profile['occupation']}")
        if self.interests:
            lines.append(f"Interests: {', '.join(self.interests[:5])}")
        if self.personal_facts:
            lines.append(f"Known facts: {len(self.personal_facts)}")
        if self.relationships:
            lines.append(f"Known people: {len(self.relationships)}")
        if self.profile.get('interaction_count'):
            lines.append(f"Interactions: {self.profile['interaction_count']}")
        return '\n'.join(lines) if lines else "No information stored yet."
    
    def get_greeting_context(self) -> Dict[str, Any]:
        """Get context for personalized greetings."""
        return {
            'name': self.get_name(),
            'interaction_count': self.profile.get('interaction_count', 0),
            'last_seen': self.profile.get('last_seen'),
            'is_returning': self.profile.get('interaction_count', 0) > 1,
            'time_since_last': self._get_time_since_last_seen(),
        }
    
    def _get_time_since_last_seen(self) -> Optional[str]:
        """Calculate time since last interaction."""
        last_seen = self.profile.get('last_seen')
        if not last_seen:
            return None
        try:
            last = datetime.datetime.fromisoformat(last_seen)
            now = datetime.datetime.now()
            delta = now - last
            if delta.days > 0:
                return f"{delta.days} day{'s' if delta.days > 1 else ''}"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours > 1 else ''}"
            elif delta.seconds >= 60:
                mins = delta.seconds // 60
                return f"{mins} minute{'s' if mins > 1 else ''}"
            return "just now"
        except:
            return None


# =============================================================================
# INTELLIGENT FACT EXTRACTOR - Learns from conversations
# =============================================================================

class FactExtractor:
    """Extracts personal facts from user messages."""
    
    # Patterns for extracting information from casual conversation
    PATTERNS = {
        'name_request': [
            r"call me (\w+)",
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"everyone calls me (\w+)",
            r"you can call me (\w+)",
            r"just call me (\w+)",
        ],
        'email': [
            r"my email is ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"email me at ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) is my email",
        ],
        'age': [
            r"i am (\d{1,3}) years old",
            r"i'm (\d{1,3}) years old",
            r"my age is (\d{1,3})",
            r"i'm (\d{1,3})",
        ],
        'location': [
            r"i live in (.+?)(?:\.|$|,)",
            r"i'm from (.+?)(?:\.|$|,)",
            r"i am from (.+?)(?:\.|$|,)",
            r"my city is (.+?)(?:\.|$|,)",
            r"i'm in (.+?)(?:\.|$|,)",
        ],
        'occupation': [
            r"i work as a (.+?)(?:\.|$|,)",
            r"i am a (.+?)(?:\.|$|,|\s+and)",
            r"i'm a (.+?)(?:\.|$|,|\s+and)",
            r"my job is (.+?)(?:\.|$|,)",
            r"i work in (.+?)(?:\.|$|,)",
        ],
        'favorite': [
            r"my favorite (.+?) is (.+?)(?:\.|$|,)",
            r"i love (.+?)(?:\.|$|,)",
            r"i really like (.+?)(?:\.|$|,)",
        ],
        'relationship': [
            r"my (\w+)'s name is (\w+)",
            r"(\w+) is my (\w+)",
            r"my (\w+) (\w+)",
        ],
        'interest': [
            r"i enjoy (.+?)(?:\.|$|,)",
            r"i like (.+?)(?:\.|$|,)",
            r"my hobby is (.+?)(?:\.|$|,)",
            r"i'm interested in (.+?)(?:\.|$|,)",
        ],
        'birthday': [
            r"my birthday is (.+?)(?:\.|$)",
            r"i was born on (.+?)(?:\.|$)",
            r"born on (.+?)(?:\.|$)",
        ],
    }
    
    # Words to ignore when extracting names
    IGNORE_WORDS = {'a', 'the', 'just', 'here', 'not', 'very', 'really', 'so', 'too', 
                    'doing', 'going', 'tired', 'happy', 'sad', 'good', 'bad', 'fine',
                    'okay', 'ok', 'great', 'well', 'sure'}
    
    @classmethod
    def extract_name(cls, text: str) -> Optional[str]:
        """Extract a name from text."""
        text_lower = text.lower()
        for pattern in cls.PATTERNS['name_request']:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).strip()
                if name.lower() not in cls.IGNORE_WORDS and len(name) > 1:
                    return name.title()
        return None
    
    @classmethod
    def extract_email(cls, text: str) -> Optional[str]:
        """Extract an email from text."""
        for pattern in cls.PATTERNS['email']:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).lower()
        return None
    
    @classmethod
    def extract_age(cls, text: str) -> Optional[int]:
        """Extract age from text."""
        for pattern in cls.PATTERNS['age']:
            match = re.search(pattern, text.lower())
            if match:
                age = int(match.group(1))
                if 1 <= age <= 120:  # Reasonable age range
                    return age
        return None
    
    @classmethod
    def extract_location(cls, text: str) -> Optional[str]:
        """Extract location from text."""
        for pattern in cls.PATTERNS['location']:
            match = re.search(pattern, text.lower())
            if match:
                location = match.group(1).strip()
                if len(location) > 1:
                    return location.title()
        return None
    
    @classmethod
    def extract_occupation(cls, text: str) -> Optional[str]:
        """Extract occupation from text."""
        for pattern in cls.PATTERNS['occupation']:
            match = re.search(pattern, text.lower())
            if match:
                occupation = match.group(1).strip()
                if len(occupation) > 1:
                    return occupation
        return None
    
    @classmethod
    def extract_favorites(cls, text: str) -> List[Tuple[str, str]]:
        """Extract favorite things from text."""
        favorites = []
        for pattern in cls.PATTERNS['favorite']:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if len(match) == 2:
                    favorites.append((match[0].strip(), match[1].strip()))
        return favorites
    
    @classmethod
    def extract_interests(cls, text: str) -> List[str]:
        """Extract interests from text."""
        interests = []
        for pattern in cls.PATTERNS['interest']:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, str) and len(match) > 1:
                    interests.append(match.strip())
        return interests
    
    @classmethod
    def extract_all(cls, text: str) -> Dict[str, Any]:
        """Extract all possible facts from text."""
        return {
            'name': cls.extract_name(text),
            'email': cls.extract_email(text),
            'age': cls.extract_age(text),
            'location': cls.extract_location(text),
            'occupation': cls.extract_occupation(text),
            'favorites': cls.extract_favorites(text),
            'interests': cls.extract_interests(text),
        }


# =============================================================================
# TASK MANAGERS (Notes, Todos, Reminders)

class NoteManager:
    """Manages user notes."""
    
    def __init__(self):
        self.notes: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self):
        if Config.NOTES_FILE.exists():
            try:
                with open(Config.NOTES_FILE, 'r') as f:
                    self.notes = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.notes = []
    
    def _save(self):
        with open(Config.NOTES_FILE, 'w') as f:
            json.dump(self.notes, f, indent=2)
    
    def add(self, content: str, title: str = None) -> int:
        """Add a note, return its ID."""
        note = {
            'id': len(self.notes) + 1,
            'title': title or f"Note {len(self.notes) + 1}",
            'content': content,
            'created_at': datetime.datetime.now().isoformat(),
            'updated_at': datetime.datetime.now().isoformat()
        }
        self.notes.append(note)
        self._save()
        return note['id']
    
    def get(self, note_id: int) -> Optional[Dict]:
        """Get a note by ID."""
        for note in self.notes:
            if note['id'] == note_id:
                return note
        return None
    
    def list_all(self) -> List[Dict]:
        """List all notes."""
        return self.notes
    
    def delete(self, note_id: int) -> bool:
        """Delete a note by ID."""
        for i, note in enumerate(self.notes):
            if note['id'] == note_id:
                self.notes.pop(i)
                self._save()
                return True
        return False
    
    def search(self, query: str) -> List[Dict]:
        """Search notes by content or title."""
        query = query.lower()
        return [n for n in self.notes if query in n['content'].lower() or query in n['title'].lower()]


class TodoManager:
    """Manages user todos."""
    
    def __init__(self):
        self.todos: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self):
        if Config.TODOS_FILE.exists():
            try:
                with open(Config.TODOS_FILE, 'r') as f:
                    self.todos = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.todos = []
    
    def _save(self):
        with open(Config.TODOS_FILE, 'w') as f:
            json.dump(self.todos, f, indent=2)
    
    def add(self, task: str, priority: str = "medium", due_date: str = None) -> int:
        """Add a todo, return its ID."""
        todo = {
            'id': len(self.todos) + 1,
            'task': task,
            'priority': priority,
            'due_date': due_date,
            'completed': False,
            'created_at': datetime.datetime.now().isoformat()
        }
        self.todos.append(todo)
        self._save()
        return todo['id']
    
    def complete(self, todo_id: int) -> bool:
        """Mark a todo as complete."""
        for todo in self.todos:
            if todo['id'] == todo_id:
                todo['completed'] = True
                todo['completed_at'] = datetime.datetime.now().isoformat()
                self._save()
                return True
        return False
    
    def list_all(self, include_completed: bool = True) -> List[Dict]:
        """List all todos."""
        if include_completed:
            return self.todos
        return [t for t in self.todos if not t['completed']]
    
    def list_pending(self) -> List[Dict]:
        """List pending todos."""
        return [t for t in self.todos if not t['completed']]
    
    def delete(self, todo_id: int) -> bool:
        """Delete a todo by ID."""
        for i, todo in enumerate(self.todos):
            if todo['id'] == todo_id:
                self.todos.pop(i)
                self._save()
                return True
        return False


class ReminderManager:
    """Manages user reminders."""
    
    def __init__(self):
        self.reminders: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self):
        if Config.REMINDERS_FILE.exists():
            try:
                with open(Config.REMINDERS_FILE, 'r') as f:
                    self.reminders = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.reminders = []
    
    def _save(self):
        with open(Config.REMINDERS_FILE, 'w') as f:
            json.dump(self.reminders, f, indent=2)
    
    def add(self, message: str, remind_at: datetime.datetime = None, relative: str = None) -> int:
        """Add a reminder."""
        if relative and not remind_at:
            remind_at = self._parse_relative_time(relative)
        reminder = {
            'id': len(self.reminders) + 1,
            'message': message,
            'remind_at': remind_at.isoformat() if remind_at else None,
            'created_at': datetime.datetime.now().isoformat(),
            'triggered': False
        }
        self.reminders.append(reminder)
        self._save()
        return reminder['id']
    
    def _parse_relative_time(self, relative: str) -> datetime.datetime:
        """Parse relative time like '10 minutes', '2 hours'."""
        now = datetime.datetime.now()
        match = re.search(r'(\d+)\s*(second|minute|hour|day|week|month)s?', relative.lower())
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit == 'second':
                return now + datetime.timedelta(seconds=amount)
            elif unit == 'minute':
                return now + datetime.timedelta(minutes=amount)
            elif unit == 'hour':
                return now + datetime.timedelta(hours=amount)
            elif unit == 'day':
                return now + datetime.timedelta(days=amount)
            elif unit == 'week':
                return now + datetime.timedelta(weeks=amount)
            elif unit == 'month':
                return now + datetime.timedelta(days=amount * 30)
        return now + datetime.timedelta(hours=1)
    
    def check_due(self) -> List[Dict]:
        """Check for due reminders."""
        now = datetime.datetime.now()
        due = []
        for reminder in self.reminders:
            if not reminder['triggered'] and reminder['remind_at']:
                remind_at = datetime.datetime.fromisoformat(reminder['remind_at'])
                if remind_at <= now:
                    reminder['triggered'] = True
                    due.append(reminder)
        if due:
            self._save()
        return due
    
    def list_all(self) -> List[Dict]:
        """List all reminders."""
        return self.reminders
    
    def list_pending(self) -> List[Dict]:
        """List pending reminders."""
        return [r for r in self.reminders if not r['triggered']]
    
    def delete(self, reminder_id: int) -> bool:
        """Delete a reminder."""
        for i, reminder in enumerate(self.reminders):
            if reminder['id'] == reminder_id:
                self.reminders.pop(i)
                self._save()
                return True
        return False


# =============================================================================
# MATH ENGINE
# =============================================================================

class MathEngine:
    """Handles mathematical calculations."""
    
    OPERATORS = {
        '+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv,
        '^': operator.pow, '**': operator.pow, '%': operator.mod, '//': operator.floordiv,
    }
    
    FUNCTIONS = {
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'log': math.log, 'log10': math.log10, 'log2': math.log2,
        'exp': math.exp, 'abs': abs, 'floor': math.floor, 'ceil': math.ceil,
        'round': round, 'factorial': math.factorial, 'pow': pow,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'degrees': math.degrees, 'radians': math.radians,
    }
    
    CONSTANTS = {
        'pi': math.pi, 'e': math.e, 'tau': math.tau,
        'inf': math.inf, 'golden': (1 + math.sqrt(5)) / 2,
    }
    
    @classmethod
    def evaluate(cls, expression: str) -> Tuple[Optional[float], Optional[str]]:
        """Safely evaluate a mathematical expression."""
        try:
            expression = expression.lower().strip()
            for const, value in cls.CONSTANTS.items():
                expression = re.sub(r'\b' + const + r'\b', str(value), expression)
            expression = expression.replace('^', '**')
            expression = expression.replace('×', '*').replace('÷', '/')
            allowed_names = {**cls.FUNCTIONS, 'abs': abs}
            safe_dict = {"__builtins__": {}}
            safe_dict.update(allowed_names)
            safe_dict.update(cls.CONSTANTS)
            if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', expression):
                pass
            result = eval(expression, safe_dict)
            return float(result), None
        except ZeroDivisionError:
            return None, "Cannot divide by zero!"
        except ValueError as e:
            return None, f"Math error: {str(e)}"
        except SyntaxError:
            return None, "Invalid expression syntax"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    @classmethod
    def solve_equation(cls, equation: str) -> Optional[str]:
        """Solve simple linear equations."""
        try:
            equation = equation.lower().replace(' ', '')
            if '=' not in equation:
                return None
            left, right = equation.split('=')
            if 'x' in left and 'x' not in right:
                match = re.match(r'([+-]?\d*)x([+-]\d+)?', left)
                if match:
                    a = int(match.group(1)) if match.group(1) and match.group(1) not in ['+', '-'] else (1 if match.group(1) != '-' else -1)
                    b = int(match.group(2)) if match.group(2) else 0
                    c = float(right)
                    x = (c - b) / a
                    return f"x = {x}"
            return None
        except Exception:
            return None
    
    @classmethod
    def statistics(cls, numbers: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of numbers."""
        if not numbers:
            return {}
        return {
            'count': len(numbers),
            'sum': sum(numbers),
            'mean': statistics.mean(numbers),
            'median': statistics.median(numbers),
            'mode': statistics.mode(numbers) if len(numbers) > 1 else numbers[0],
            'min': min(numbers),
            'max': max(numbers),
            'range': max(numbers) - min(numbers),
            'std_dev': statistics.stdev(numbers) if len(numbers) > 1 else 0,
            'variance': statistics.variance(numbers) if len(numbers) > 1 else 0,
        }
    
    @classmethod
    def prime_check(cls, n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @classmethod
    def factorize(cls, n: int) -> List[int]:
        """Find prime factors of a number."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    @classmethod
    def gcd(cls, a: int, b: int) -> int:
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    @classmethod
    def lcm(cls, a: int, b: int) -> int:
        """Least common multiple."""
        return abs(a * b) // cls.gcd(a, b)
    
    @classmethod
    def fibonacci(cls, n: int) -> List[int]:
        """Generate Fibonacci sequence."""
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        seq = [0, 1]
        for _ in range(2, n):
            seq.append(seq[-1] + seq[-2])
        return seq


# =============================================================================
# UNIT CONVERTER
# =============================================================================

class UnitConverter:
    """Handles unit conversions."""
    
    CONVERSIONS = {
        # Length
        'km_to_miles': lambda x: x * 0.621371,
        'miles_to_km': lambda x: x * 1.60934,
        'meters_to_feet': lambda x: x * 3.28084,
        'feet_to_meters': lambda x: x * 0.3048,
        'inches_to_cm': lambda x: x * 2.54,
        'cm_to_inches': lambda x: x / 2.54,
        'yards_to_meters': lambda x: x * 0.9144,
        'meters_to_yards': lambda x: x / 0.9144,
        
        # Weight
        'kg_to_pounds': lambda x: x * 2.20462,
        'pounds_to_kg': lambda x: x / 2.20462,
        'kg_to_ounces': lambda x: x * 35.274,
        'ounces_to_kg': lambda x: x / 35.274,
        'grams_to_ounces': lambda x: x / 28.3495,
        'ounces_to_grams': lambda x: x * 28.3495,
        
        # Temperature
        'celsius_to_fahrenheit': lambda x: (x * 9/5) + 32,
        'fahrenheit_to_celsius': lambda x: (x - 32) * 5/9,
        'celsius_to_kelvin': lambda x: x + 273.15,
        'kelvin_to_celsius': lambda x: x - 273.15,
        'fahrenheit_to_kelvin': lambda x: (x - 32) * 5/9 + 273.15,
        'kelvin_to_fahrenheit': lambda x: (x - 273.15) * 9/5 + 32,
        
        # Volume
        'liters_to_gallons': lambda x: x * 0.264172,
        'gallons_to_liters': lambda x: x * 3.78541,
        'ml_to_ounces': lambda x: x / 29.5735,
        'ounces_to_ml': lambda x: x * 29.5735,
        'cups_to_ml': lambda x: x * 236.588,
        'ml_to_cups': lambda x: x / 236.588,
        
        # Time
        'hours_to_minutes': lambda x: x * 60,
        'minutes_to_hours': lambda x: x / 60,
        'days_to_hours': lambda x: x * 24,
        'hours_to_days': lambda x: x / 24,
        'weeks_to_days': lambda x: x * 7,
        'days_to_weeks': lambda x: x / 7,
        'years_to_days': lambda x: x * 365,
        'days_to_years': lambda x: x / 365,
        
        # Digital storage
        'kb_to_mb': lambda x: x / 1024,
        'mb_to_kb': lambda x: x * 1024,
        'mb_to_gb': lambda x: x / 1024,
        'gb_to_mb': lambda x: x * 1024,
        'gb_to_tb': lambda x: x / 1024,
        'tb_to_gb': lambda x: x * 1024,
        
        # Speed
        'mph_to_kph': lambda x: x * 1.60934,
        'kph_to_mph': lambda x: x / 1.60934,
        'mps_to_mph': lambda x: x * 2.23694,
        'mph_to_mps': lambda x: x / 2.23694,
        
        # Area
        'sqm_to_sqft': lambda x: x * 10.7639,
        'sqft_to_sqm': lambda x: x / 10.7639,
        'acres_to_hectares': lambda x: x * 0.404686,
        'hectares_to_acres': lambda x: x / 0.404686,
    }
    
    UNIT_ALIASES = {
        'kilometers': 'km', 'kilometer': 'km', 'mile': 'miles',
        'meter': 'meters', 'metre': 'meters', 'metres': 'meters',
        'foot': 'feet', 'inch': 'inches', 'centimeter': 'cm',
        'centimeters': 'cm', 'centimetre': 'cm', 'centimetres': 'cm',
        'kilogram': 'kg', 'kilograms': 'kg', 'pound': 'pounds', 'lb': 'pounds', 'lbs': 'pounds',
        'ounce': 'ounces', 'oz': 'ounces', 'gram': 'grams', 'g': 'grams',
        'c': 'celsius', 'f': 'fahrenheit', 'k': 'kelvin',
        'liter': 'liters', 'litre': 'liters', 'litres': 'liters', 'l': 'liters',
        'gallon': 'gallons', 'gal': 'gallons',
        'milliliter': 'ml', 'milliliters': 'ml', 'millilitre': 'ml',
        'cup': 'cups',
        'hour': 'hours', 'hr': 'hours', 'hrs': 'hours',
        'minute': 'minutes', 'min': 'minutes', 'mins': 'minutes',
        'second': 'seconds', 'sec': 'seconds', 'secs': 'seconds',
        'day': 'days', 'week': 'weeks', 'year': 'years',
        'kilobyte': 'kb', 'kilobytes': 'kb',
        'megabyte': 'mb', 'megabytes': 'mb',
        'gigabyte': 'gb', 'gigabytes': 'gb',
        'terabyte': 'tb', 'terabytes': 'tb',
    }
    
    @classmethod
    def convert(cls, value: float, from_unit: str, to_unit: str) -> Tuple[Optional[float], Optional[str]]:
        """Convert value between units."""
        from_unit = cls._normalize_unit(from_unit)
        to_unit = cls._normalize_unit(to_unit)
        
        conversion_key = f"{from_unit}_to_{to_unit}"
        if conversion_key in cls.CONVERSIONS:
            result = cls.CONVERSIONS[conversion_key](value)
            return result, None
        
        reverse_key = f"{to_unit}_to_{from_unit}"
        if reverse_key in cls.CONVERSIONS:
            result = 1 / cls.CONVERSIONS[reverse_key](1 / value) if value != 0 else 0
            return value / cls.CONVERSIONS[reverse_key](1), None
        
        return None, f"Cannot convert from {from_unit} to {to_unit}"
    
    @classmethod
    def _normalize_unit(cls, unit: str) -> str:
        """Normalize unit name."""
        unit = unit.lower().strip()
        return cls.UNIT_ALIASES.get(unit, unit)
    
    @classmethod
    def parse_conversion_request(cls, text: str) -> Optional[Tuple[float, str, str]]:
        """Parse conversion request from natural language."""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(\w+)\s*(?:to|in|into|as)\s*(\w+)',
            r'convert\s*(\d+(?:\.\d+)?)\s*(\w+)\s*(?:to|into)\s*(\w+)',
            r'how\s*many\s*(\w+)\s*(?:in|is)\s*(\d+(?:\.\d+)?)\s*(\w+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                groups = match.groups()
                if pattern.startswith('how'):
                    return float(groups[1]), groups[2], groups[0]
                return float(groups[0]), groups[1], groups[2]
        return None


# =============================================================================
# CONTENT DATABASE (Jokes, Facts, Quotes, etc.)
# =============================================================================

class ContentDatabase:
    """Stores various content like jokes, facts, quotes, etc."""
    
    JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "Why did the AI go to therapy? It had too many deep issues! 🤖",
        "What's a computer's least favorite food? Spam! 🥫",
        "Why do Java developers wear glasses? Because they can't C#! 👓",
        "What do you call a fish without eyes? A fsh! 🐟",
        "Why don't scientists trust atoms? Because they make up everything! ⚛️",
        "What do you call a bear with no teeth? A gummy bear! 🐻",
        "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
        "What do you call a fake noodle? An impasta! 🍝",
        "Why did the bicycle fall over? Because it was two-tired! 🚲",
        "Why don't eggs tell jokes? They'd crack each other up! 🥚",
        "Why did the coffee file a police report? It got mugged! ☕",
        "What do computers eat for a snack? Microchips! 💻",
        "Why was the math book sad? Because it had too many problems! 📚",
        "What did the ocean say to the beach? Nothing, it just waved! 🌊",
    ]
    
    FACTS = [
        "Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible! 🍯",
        "Octopuses have three hearts and blue blood. 🐙",
        "A group of flamingos is called a 'flamboyance'. 🦩",
        "Bananas are berries, but strawberries aren't! 🍌",
        "The Eiffel Tower can grow 6 inches taller in summer due to thermal expansion. 🗼",
        "Venus is the only planet that spins clockwise. 🪐",
        "Cows have best friends and get stressed when separated. 🐄",
        "Sharks are older than trees - over 400 million years! 🦈",
        "A day on Venus is longer than a year on Venus. 🌅",
        "The human brain uses about 20% of the body's total energy. 🧠",
        "Dolphins have names for each other using unique whistles. 🐬",
        "The inventor of the microwave was only paid $2 for his discovery. 📡",
    ]
    
    QUOTES = [
        ("The only way to do great work is to love what you do.", "Steve Jobs"),
        ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
        ("Be the change you wish to see in the world.", "Mahatma Gandhi"),
        ("The future belongs to those who believe in the beauty of their dreams.", "Eleanor Roosevelt"),
        ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
        ("Be yourself; everyone else is already taken.", "Oscar Wilde"),
        ("A journey of a thousand miles begins with a single step.", "Lao Tzu"),
        ("Talk is cheap. Show me the code.", "Linus Torvalds"),
        ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci"),
        ("The mind is everything. What you think you become.", "Buddha"),
    ]
    
    MOTIVATIONAL = [
        "You're doing amazing! Every step forward is progress. 🌟",
        "Remember: the experts were once beginners. Keep going! 💪",
        "Your potential is endless. Go do what you were created to do! 🚀",
        "Believe in yourself! You have everything it takes. ✨",
        "Today is a new day full of new possibilities. 🌈",
        "You've overcome challenges before, and you'll do it again. 💫",
        "Great things never came from comfort zones. 🎯",
        "You're stronger than you think. 🦁",
        "The best view comes after the hardest climb. ⛰️",
    ]
    
    AFFIRMATIONS = [
        "I am worthy of love, success, and happiness. 💖",
        "I trust in my abilities and express my true self with ease. 🌸",
        "Every day, in every way, I am getting better and better. 📈",
        "I am resilient, strong, and brave. 💫",
        "My potential to succeed is infinite. 🌌",
        "I choose to be proud of myself and all I have accomplished. 🏆",
    ]
    
    RIDDLES = [
        ("I have cities, but no houses. Mountains, but no trees. Water, but no fish. What am I?", "A map"),
        ("The more you take, the more you leave behind. What am I?", "Footsteps"),
        ("What has keys but no locks?", "A keyboard"),
        ("I have a head and a tail but no body. What am I?", "A coin"),
        ("What gets wetter the more it dries?", "A towel"),
    ]
    
    MAGIC_8BALL = [
        "It is certain. 🎱", "Without a doubt. 🎱", "Yes, definitely. 🎱",
        "Most likely. 🎱", "Outlook good. 🎱", "Signs point to yes. 🎱",
        "Reply hazy, try again. 🎱", "Ask again later. 🎱",
        "Don't count on it. 🎱", "My sources say no. 🎱", "Very doubtful. 🎱",
    ]
    
    ASCII_ART = {
        'heart': "   ♥♥   ♥♥\n ♥♥♥♥♥♥♥♥♥♥♥\n  ♥♥♥♥♥♥♥♥♥\n    ♥♥♥♥♥\n      ♥",
        'star': "    ★\n   ★ ★\n★★★★★★★\n  ★   ★\n ★     ★",
        'cat': "  /\\_/\\\n ( o.o )\n  > ^ <",
        'dog': " / \\__\n(    @\\___\n/         O\n/   (_____/\n/_____/   U",
        'robot': "╔═══════╗\n║ ◉   ◉ ║\n║  ═══  ║\n╚═══════╝",
        'smile': "  ☺☺☺☺\n ☺    ☺\n☺ ◕  ◕ ☺\n☺  __  ☺\n ☺    ☺\n  ☺☺☺☺",
    }
    
    @classmethod
    def get_joke(cls) -> str: return random.choice(cls.JOKES)
    @classmethod
    def get_fact(cls) -> str: return random.choice(cls.FACTS)
    @classmethod
    def get_quote(cls) -> tuple: return random.choice(cls.QUOTES)
    @classmethod
    def get_motivation(cls) -> str: return random.choice(cls.MOTIVATIONAL)
    @classmethod
    def get_affirmation(cls) -> str: return random.choice(cls.AFFIRMATIONS)
    @classmethod
    def get_riddle(cls) -> tuple: return random.choice(cls.RIDDLES)
    @classmethod
    def get_8ball(cls) -> str: return random.choice(cls.MAGIC_8BALL)
    @classmethod
    def get_ascii(cls, name: str) -> str: return cls.ASCII_ART.get(name.lower(), "Art not found")


# =============================================================================
# TEXT UTILITIES
# =============================================================================

class TextUtilities:
    """Various text manipulation utilities."""
    
    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())
    
    @staticmethod
    def char_count(text: str, include_spaces: bool = True) -> int:
        return len(text) if include_spaces else len(text.replace(' ', ''))
    
    @staticmethod
    def reverse_text(text: str) -> str:
        return text[::-1]
    
    @staticmethod
    def to_uppercase(text: str) -> str:
        return text.upper()
    
    @staticmethod
    def to_lowercase(text: str) -> str:
        return text.lower()
    
    @staticmethod
    def to_title_case(text: str) -> str:
        return text.title()
    
    @staticmethod
    def is_palindrome(text: str) -> bool:
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        return cleaned == cleaned[::-1]
    
    @staticmethod
    def is_anagram(text1: str, text2: str) -> bool:
        clean1 = re.sub(r'[^a-zA-Z]', '', text1.lower())
        clean2 = re.sub(r'[^a-zA-Z]', '', text2.lower())
        return sorted(clean1) == sorted(clean2)
    
    @staticmethod
    def caesar_cipher(text: str, shift: int = 3, decrypt: bool = False) -> str:
        if decrypt:
            shift = -shift
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)
    
    @staticmethod
    def generate_password(length: int = 16, use_special: bool = True) -> str:
        chars = string.ascii_letters + string.digits
        if use_special:
            chars += "!@#$%^&*"
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def hash_text(text: str, algorithm: str = 'sha256') -> str:
        if algorithm == 'md5':
            return hashlib.md5(text.encode()).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(text.encode()).hexdigest()
        else:
            return hashlib.sha256(text.encode()).hexdigest()
    
    @staticmethod
    def summarize(text: str, max_sentences: int = 3) -> str:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return '. '.join(sentences[:max_sentences]) + '.' if sentences else text


# =============================================================================
# WEATHER SIMULATOR (No API - Generates realistic fake weather)
# =============================================================================

class WeatherSimulator:
    """Simulates weather data without API calls."""
    
    CONDITIONS = ['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy', 'Stormy', 'Snowy', 'Foggy', 'Windy']
    EMOJIS = {'Sunny': '☀️', 'Partly Cloudy': '⛅', 'Cloudy': '☁️', 'Rainy': '🌧️', 
              'Stormy': '⛈️', 'Snowy': '❄️', 'Foggy': '🌫️', 'Windy': '💨'}
    
    @classmethod
    def get_weather(cls, location: str = "your area") -> Dict[str, Any]:
        """Generate simulated weather data."""
        now = datetime.datetime.now()
        month = now.month
        
        # Seasonal temperature ranges
        if month in [12, 1, 2]:  # Winter
            temp_range = (-5, 45)
        elif month in [3, 4, 5]:  # Spring
            temp_range = (40, 70)
        elif month in [6, 7, 8]:  # Summer
            temp_range = (65, 95)
        else:  # Fall
            temp_range = (35, 65)
        
        temp = random.randint(*temp_range)
        condition = random.choice(cls.CONDITIONS)
        
        # Adjust condition based on temperature
        if temp < 32 and condition == 'Rainy':
            condition = 'Snowy'
        
        return {
            'location': location.title(),
            'temperature': temp,
            'condition': condition,
            'emoji': cls.EMOJIS.get(condition, ''),
            'humidity': random.randint(30, 90),
            'wind_speed': random.randint(0, 30),
            'feels_like': temp + random.randint(-5, 5),
        }
    
    @classmethod
    def format_weather(cls, weather: Dict) -> str:
        """Format weather data as readable string."""
        return f"""
🌡️ Weather for {weather['location']}:
{weather['emoji']} Condition: {weather['condition']}
🌡️ Temperature: {weather['temperature']}°F (feels like {weather['feels_like']}°F)
💧 Humidity: {weather['humidity']}%
💨 Wind: {weather['wind_speed']} mph

Note: This is simulated weather data (no API used).
"""


# =============================================================================
# GAMES
# =============================================================================

class Games:
    """Simple text-based games."""
    
    @staticmethod
    def number_guessing_game() -> Dict[str, Any]:
        """Initialize a number guessing game."""
        return {
            'type': 'number_guess',
            'number': random.randint(1, 100),
            'attempts': 0,
            'max_attempts': 7,
            'active': True
        }
    
    @staticmethod
    def check_guess(game: Dict, guess: int) -> Tuple[str, bool]:
        """Check a guess in the number game."""
        game['attempts'] += 1
        if guess == game['number']:
            game['active'] = False
            return f"🎉 Correct! You got it in {game['attempts']} attempts!", True
        elif game['attempts'] >= game['max_attempts']:
            game['active'] = False
            return f"😢 Game over! The number was {game['number']}.", True
        elif guess < game['number']:
            return f"📈 Too low! {game['max_attempts'] - game['attempts']} attempts left.", False
        else:
            return f"📉 Too high! {game['max_attempts'] - game['attempts']} attempts left.", False
    
    @staticmethod
    def rock_paper_scissors(user_choice: str) -> str:
        """Play rock paper scissors."""
        choices = ['rock', 'paper', 'scissors']
        user_choice = user_choice.lower().strip()
        if user_choice not in choices:
            return "Please choose rock, paper, or scissors!"
        
        computer = random.choice(choices)
        emojis = {'rock': '🪨', 'paper': '📄', 'scissors': '✂️'}
        
        result = f"You: {emojis[user_choice]} vs Me: {emojis[computer]}\n"
        
        if user_choice == computer:
            result += "It's a tie! 🤝"
        elif (user_choice == 'rock' and computer == 'scissors') or \
             (user_choice == 'paper' and computer == 'rock') or \
             (user_choice == 'scissors' and computer == 'paper'):
            result += "You win! 🎉"
        else:
            result += "I win! 🤖"
        
        return result
    
    @staticmethod
    def hangman_game(word: str = None) -> Dict[str, Any]:
        """Initialize a hangman game."""
        words = ['python', 'programming', 'computer', 'algorithm', 'database', 
                 'internet', 'software', 'artificial', 'intelligence', 'machine']
        word = word or random.choice(words)
        return {
            'type': 'hangman',
            'word': word.lower(),
            'guessed': set(),
            'wrong_guesses': 0,
            'max_wrong': 6,
            'active': True
        }
    
    @staticmethod
    def hangman_guess(game: Dict, letter: str) -> Tuple[str, bool]:
        """Make a guess in hangman."""
        letter = letter.lower()
        if len(letter) != 1 or not letter.isalpha():
            return "Please guess a single letter!", False
        
        if letter in game['guessed']:
            return f"You already guessed '{letter}'!", False
        
        game['guessed'].add(letter)
        
        if letter in game['word']:
            display = ''.join(c if c in game['guessed'] else '_' for c in game['word'])
            if '_' not in display:
                game['active'] = False
                return f"🎉 You won! The word was: {game['word']}", True
            return f"Good guess! {display}", False
        else:
            game['wrong_guesses'] += 1
            remaining = game['max_wrong'] - game['wrong_guesses']
            if remaining <= 0:
                game['active'] = False
                return f"💀 Game over! The word was: {game['word']}", True
            display = ''.join(c if c in game['guessed'] else '_' for c in game['word'])
            return f"Wrong! {remaining} lives left. {display}", False


# =============================================================================
# CONVERSATION CONTEXT MANAGER
# =============================================================================

class ConversationContext:
    """Manages conversation history and context."""
    
    def __init__(self, max_turns: int = 20):
        self.turns: deque = deque(maxlen=max_turns)
        self.current_game: Optional[Dict] = None
        self.current_riddle: Optional[Tuple[str, str]] = None
        self.current_trivia: Optional[Tuple[str, List[str], str]] = None
        self.topic: Optional[str] = None
        self.last_intent: Optional[Intent] = None
    
    def add_turn(self, user_input: str, bot_response: str, intent: Intent, entities: List[Entity] = None):
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(
            user_input=user_input,
            bot_response=bot_response,
            intent=intent,
            timestamp=datetime.datetime.now(),
            entities=entities or []
        ))
        self.last_intent = intent
    
    def get_last_n_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the last n conversation turns."""
        return list(self.turns)[-n:]
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context."""
        if not self.turns:
            return "No previous conversation."
        recent = self.get_last_n_turns(3)
        summary = "Recent topics: "
        topics = [turn.intent.name for turn in recent]
        return summary + ", ".join(set(topics))
    
    def clear(self):
        """Clear conversation history."""
        self.turns.clear()
        self.current_game = None
        self.current_riddle = None
        self.current_trivia = None
        self.topic = None


# =============================================================================
# RESPONSE GENERATOR
# =============================================================================

class ResponseGenerator:
    """Generates responses based on intent and entities."""
    
    def __init__(self, knowledge: KnowledgeBase, memory: MemoryManager,
                 notes: NoteManager, todos: TodoManager, reminders: ReminderManager,
                 user_profile: UserProfile = None):
        self.knowledge = knowledge
        self.memory = memory
        self.notes = notes
        self.todos = todos
        self.reminders = reminders
        self.user_profile = user_profile or UserProfile()
        self.context = ConversationContext()
    
    def learn_from_message(self, text: str) -> Optional[str]:
        """Extract and learn personal facts from user message."""
        learned = []
        
        # Extract name
        name = FactExtractor.extract_name(text)
        if name:
            self.user_profile.set_nickname(name)
            learned.append(f"name ({name})")
        
        # Extract email
        email = FactExtractor.extract_email(text)
        if email:
            self.user_profile.set_email(email)
            learned.append(f"email ({email})")
        
        # Extract age
        age = FactExtractor.extract_age(text)
        if age:
            self.user_profile.set_age(age)
            learned.append(f"age ({age})")
        
        # Extract location
        location = FactExtractor.extract_location(text)
        if location:
            self.user_profile.set_location(location)
            learned.append(f"location ({location})")
        
        # Extract occupation
        occupation = FactExtractor.extract_occupation(text)
        if occupation:
            self.user_profile.set_occupation(occupation)
            learned.append(f"occupation ({occupation})")
        
        # Extract interests
        interests = FactExtractor.extract_interests(text)
        for interest in interests:
            self.user_profile.add_interest(interest)
            learned.append(f"interest ({interest})")
        
        # Extract favorites
        favorites = FactExtractor.extract_favorites(text)
        for category, value in favorites:
            self.user_profile.add_fact(f"favorite_{category}", value)
            learned.append(f"favorite {category} ({value})")
        
        return learned if learned else None
    
    def generate(self, parsed: ParsedInput) -> str:
        """Generate a response for parsed input."""
        intent = parsed.intent
        text = parsed.original
        
        # LEARN from every message - extract personal facts
        learned = self.learn_from_message(text)
        learned_msg = ""
        if learned:
            learned_msg = f"\n\n💾 *I'll remember that! Learned: {', '.join(learned)}*"
        
        # Check for active game
        if self.context.current_game and self.context.current_game.get('active'):
            return self._handle_game_input(text)
        
        # Handle different intents
        handlers = {
            Intent.GREETING: self._greeting,
            Intent.FAREWELL: self._farewell,
            Intent.THANKS: self._thanks,
            Intent.HELP: self._help,
            Intent.IDENTITY: self._identity,
            Intent.CAPABILITY: self._capability,
            Intent.MOOD: self._mood,
            Intent.TIME: self._time,
            Intent.DATE: self._date,
            Intent.MATH: lambda p: self._math(p.original),
            Intent.CALCULATE: lambda p: self._math(p.original),
            Intent.CONVERT: lambda p: self._convert(p.original),
            Intent.WEATHER: lambda p: self._weather(p.original),
            Intent.JOKE: self._joke,
            Intent.FACT: self._fact,
            Intent.QUOTE: self._quote,
            Intent.MOTIVATE: self._motivate,
            Intent.AFFIRMATION: self._affirmation,
            Intent.RIDDLE: self._riddle,
            Intent.MAGIC_8BALL: self._magic_8ball,
            Intent.COIN_FLIP: self._coin_flip,
            Intent.DICE_ROLL: lambda p: self._dice_roll(p.original),
            Intent.RANDOM_NUMBER: lambda p: self._random_number(p.original),
            Intent.PASSWORD: lambda p: self._password(p.original),
            Intent.SENTIMENT: lambda p: self._analyze_sentiment(p.original),
            Intent.WORD_COUNT: lambda p: self._word_count(p.original),
            Intent.REVERSE_TEXT: lambda p: self._reverse_text(p.original),
            Intent.ASCII_ART: lambda p: self._ascii_art(p.original),
            Intent.NOTE: lambda p: self._add_note(p.original),
            Intent.TODO: lambda p: self._add_todo(p.original),
            Intent.REMINDER: lambda p: self._add_reminder(p.original),
            Intent.LIST_NOTES: self._list_notes,
            Intent.LIST_TODOS: self._list_todos,
            Intent.LIST_REMINDERS: self._list_reminders,
            Intent.LEARN: lambda p: self._learn(p.original),
            Intent.RECALL: lambda p: self._recall(p.original),
            Intent.QUESTION: lambda p: self._answer_question(p.original),
            Intent.DEFINE: lambda p: self._define(p.original),
            Intent.GAME: lambda p: self._start_game(p.original),
            Intent.ENCRYPT: lambda p: self._encrypt(p.original),
            Intent.DECRYPT: lambda p: self._decrypt(p.original),
            Intent.MY_PROFILE: self._my_profile,
            Intent.FORGET_ME: self._forget_me,
        }
        
        handler = handlers.get(intent, self._unknown)
        response = handler(parsed) if callable(handler) else handler
        
        # Add learned confirmation if applicable (but not for greetings where we do it inline)
        if learned and intent != Intent.GREETING:
            response += learned_msg
        
        return response
    
    def _greeting(self, _) -> str:
        """Generate personalized greeting based on user profile."""
        name = self.user_profile.get_name()
        context = self.user_profile.get_greeting_context()
        
        if name and context['is_returning']:
            # Returning user with name
            time_away = context.get('time_since_last')
            interactions = context.get('interaction_count', 0)
            
            if interactions > 50:
                return f"Hey {name}! 🌟 Always great to chat with you! What's up?"
            elif interactions > 10:
                if time_away and 'day' in time_away:
                    return f"Welcome back, {name}! 👋 It's been {time_away}. How can I help?"
                return f"Hey {name}! 😊 Good to see you again! What can I do for you?"
            else:
                return f"Hello, {name}! 👋 How can I help you today?"
        
        elif name:
            # New user but we just learned their name
            return f"Nice to meet you, {name}! 🎉 I'm Aurora. I'll remember your name! How can I help?"
        
        elif context['is_returning']:
            # Returning user without name
            greetings = [
                "Welcome back! 👋 How can I help you today?",
                "Hey there! 😊 Good to see you again!",
                "Hello again! 🌟 What can I do for you?",
            ]
            return random.choice(greetings) + "\n\n💡 *Tip: Tell me your name and I'll remember it!*"
        
        else:
            # First time user
            return """Hello! 👋 I'm **Aurora**, your personal AI assistant!

I can remember things about you if you share them with me:
• "Call me [name]" - I'll remember your name
• "My email is [email]" - I'll save your email
• "I live in [city]" - I'll remember your location

What can I do for you today?"""
    
    def _farewell(self, _) -> str:
        """Generate personalized farewell."""
        name = self.user_profile.get_name()
        self.user_profile.update_last_seen()
        
        if name:
            farewells = [
                f"Goodbye, {name}! 👋 Have a wonderful day!",
                f"See you later, {name}! 🌟 Take care!",
                f"Bye, {name}! 😊 I'll be here when you need me!",
                f"Until next time, {name}! ✨ Stay awesome!",
            ]
            return random.choice(farewells)
        
        return random.choice([
            "Goodbye! 👋 Have a wonderful day!",
            "See you later! 🌟 Take care!",
            "Bye! 😊 It was great chatting with you!",
            "Until next time! 🎉 Stay awesome!",
        ])
    
    def _thanks(self, _) -> str:
        name = self.user_profile.get_name()
        if name:
            responses = [
                f"You're welcome, {name}! 😊 Happy to help!",
                f"My pleasure, {name}! ✨ Anytime!",
                f"No problem, {name}! 👍 That's what I'm here for!",
            ]
            return random.choice(responses)
        return random.choice([
            "You're welcome! 😊 Happy to help!",
            "My pleasure! ✨ Anytime!",
            "No problem! 👍 That's what I'm here for!",
            "Glad I could help! 🌟",
        ])
    
    def _help(self, _) -> str:
        return """
🤖 **Aurora AI Assistant - Help Menu**

Here's what I can do for you:

📊 **Information & Knowledge**
• Answer questions about various topics
• Define words and concepts
• Share interesting facts

🔢 **Math & Calculations**
• Calculate expressions (try: "what is 15 * 7 + 23")
• Solve simple equations
• Unit conversions (try: "convert 100 km to miles")

⏰ **Time & Date**
• Current time and date
• Countdown calculations

📝 **Productivity**
• Take notes, create todos, set reminders
• List and manage your tasks

🎮 **Fun & Games**
• Tell jokes, share quotes
• Play games (number guessing, rock paper scissors)
• Tell riddles, give fortunes
• Magic 8-ball predictions

🎲 **Random & Utilities**
• Flip coins, roll dice
• Generate passwords
• Text analysis (word count, sentiment)

😊 **Wellness**
• Motivational messages
• Positive affirmations

💡 **Tips:**
• Just type naturally - I understand casual language!
• Try: "tell me a joke" or "what's 2+2" or "flip a coin"
• Say "remember that X" to teach me something new!
"""

    def _identity(self, _) -> str:
        return f"""
🤖 I'm **Aurora** - Advanced Unified Reasoning and Operational Response Assistant!

Version: {Config.VERSION}
Type: Local AI Assistant (No API required!)

I'm a fully functional AI assistant built entirely in Python. I can:
• Understand natural language using pattern matching and ML
• Remember things you teach me
• Help with math, conversions, and more
• Keep notes, todos, and reminders
• Play games and tell jokes
• And much more!

All processing happens locally - your data stays on your device! 🔒
"""
    
    def _capability(self, _) -> str:
        return self._help(None)
    
    def _mood(self, _) -> str:
        moods = [
            "I'm doing great, thank you for asking! 😊 Ready to help you!",
            "I'm fantastic! 🌟 What can I do for you today?",
            "Excellent! I love being helpful. ✨ How are you?",
            "I'm operating at peak performance! 🚀 What's on your mind?",
        ]
        return random.choice(moods)
    
    def _time(self, _) -> str:
        now = datetime.datetime.now()
        return f"🕐 The current time is **{now.strftime('%I:%M:%S %p')}**"
    
    def _date(self, _) -> str:
        now = datetime.datetime.now()
        day_name = now.strftime('%A')
        date_str = now.strftime('%B %d, %Y')
        return f"📅 Today is **{day_name}, {date_str}**"
    
    def _math(self, text: str) -> str:
        # Extract mathematical expression
        text = text.lower()
        text = re.sub(r'^(what is|calculate|compute|solve|whats|what\'s)\s*', '', text)
        text = re.sub(r'\?$', '', text).strip()
        
        result, error = MathEngine.evaluate(text)
        if error:
            return f"❌ {error}"
        if result is not None:
            if result == int(result):
                return f"🔢 The answer is **{int(result)}**"
            return f"🔢 The answer is **{result:.6g}**"
        return "I couldn't calculate that. Try something like '5 + 3 * 2'"
    
    def _convert(self, text: str) -> str:
        parsed = UnitConverter.parse_conversion_request(text)
        if not parsed:
            return "I couldn't understand the conversion. Try: 'convert 100 km to miles'"
        
        value, from_unit, to_unit = parsed
        result, error = UnitConverter.convert(value, from_unit, to_unit)
        
        if error:
            return f"❌ {error}"
        return f"📐 {value} {from_unit} = **{result:.4g} {to_unit}**"
    
    def _weather(self, text: str) -> str:
        # Extract location if mentioned
        location = "your area"
        match = re.search(r'(?:weather|temperature)\s+(?:in|for|at)\s+(.+)', text.lower())
        if match:
            location = match.group(1).strip()
        
        weather = WeatherSimulator.get_weather(location)
        return WeatherSimulator.format_weather(weather)
    
    def _joke(self, _) -> str:
        return f"😄 Here's one for you:\n\n{ContentDatabase.get_joke()}"
    
    def _fact(self, _) -> str:
        return f"🧠 Did you know?\n\n{ContentDatabase.get_fact()}"
    
    def _quote(self, _) -> str:
        quote, author = ContentDatabase.get_quote()
        return f'💬 *"{quote}"*\n\n— {author}'
    
    def _motivate(self, _) -> str:
        return f"💪 {ContentDatabase.get_motivation()}"
    
    def _affirmation(self, _) -> str:
        return f"🌸 {ContentDatabase.get_affirmation()}"
    
    def _riddle(self, _) -> str:
        riddle, answer = ContentDatabase.get_riddle()
        self.context.current_riddle = (riddle, answer)
        return f"🧩 Riddle time!\n\n{riddle}\n\n(Say 'answer' to reveal the solution!)"
    
    def _magic_8ball(self, _) -> str:
        return f"🎱 *shakes the magic 8-ball*\n\n{ContentDatabase.get_8ball()}"
    
    def _coin_flip(self, _) -> str:
        result = random.choice(['Heads', 'Tails'])
        emoji = '🪙'
        return f"{emoji} *flips coin*\n\nIt's **{result}**!"
    
    def _dice_roll(self, text: str) -> str:
        # Check for specific dice (d6, d20, etc.)
        match = re.search(r'd(\d+)', text.lower())
        sides = int(match.group(1)) if match else 6
        result = random.randint(1, sides)
        return f"🎲 *rolls a d{sides}*\n\nYou rolled: **{result}**!"
    
    def _random_number(self, text: str) -> str:
        numbers = EntityExtractor.extract_numbers(text)
        if len(numbers) >= 2:
            low, high = int(min(numbers)), int(max(numbers))
        else:
            low, high = 1, 100
        result = random.randint(low, high)
        return f"🎯 Random number between {low} and {high}: **{result}**"
    
    def _password(self, text: str) -> str:
        # Extract length if specified
        numbers = EntityExtractor.extract_numbers(text)
        length = int(numbers[0]) if numbers else 16
        length = max(8, min(length, 64))  # Clamp between 8 and 64
        
        password = TextUtilities.generate_password(length)
        return f"🔐 Generated secure password ({length} characters):\n\n`{password}`"
    
    def _analyze_sentiment(self, text: str) -> str:
        # Extract the text to analyze (after "sentiment of" or similar)
        text = re.sub(r'^.*?(?:sentiment|analyze|feeling)\s*(?:of|in)?\s*', '', text, flags=re.I)
        text = text.strip('"\'')
        
        if len(text) < 5:
            return "Please provide some text to analyze. Try: 'analyze sentiment of I love this!'"
        
        score = SentimentAnalyzer.analyze(text)
        label = SentimentAnalyzer.get_sentiment_label(score)
        
        return f"""
📊 **Sentiment Analysis**

Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

Sentiment: {label}
Score: {score:.2f} (range: -1 to 1)
"""
    
    def _word_count(self, text: str) -> str:
        # Extract text to count
        text = re.sub(r'^.*?(?:word count|count words)\s*(?:of|in|for)?\s*', '', text, flags=re.I)
        text = text.strip('"\'')
        
        words = TextUtilities.word_count(text)
        chars = TextUtilities.char_count(text)
        chars_no_space = TextUtilities.char_count(text, False)
        
        return f"""
📝 **Text Statistics**

Words: {words}
Characters (with spaces): {chars}
Characters (no spaces): {chars_no_space}
"""
    
    def _reverse_text(self, text: str) -> str:
        text = re.sub(r'^.*?(?:reverse|backwards)\s*', '', text, flags=re.I)
        text = text.strip('"\'')
        return f"🔄 Reversed: **{TextUtilities.reverse_text(text)}**"
    
    def _ascii_art(self, text: str) -> str:
        available = list(ContentDatabase.ASCII_ART.keys())
        for name in available:
            if name in text.lower():
                art = ContentDatabase.get_ascii(name)
                return f"🎨 ASCII Art - {name.title()}:\n```\n{art}\n```"
        return f"🎨 Available ASCII art: {', '.join(available)}\nTry: 'show me a heart' or 'ascii cat'"
    
    def _add_note(self, text: str) -> str:
        content = re.sub(r'^.*?(?:note|write down|remember)\s*(?:that|this)?\s*:?\s*', '', text, flags=re.I)
        if len(content) < 3:
            return "What would you like me to note down?"
        note_id = self.notes.add(content)
        return f"📝 Note #{note_id} saved: '{content[:50]}{'...' if len(content) > 50 else ''}'"
    
    def _add_todo(self, text: str) -> str:
        task = re.sub(r'^.*?(?:todo|task|add)\s*:?\s*', '', text, flags=re.I)
        if len(task) < 3:
            return "What task would you like to add?"
        todo_id = self.todos.add(task)
        return f"✅ Todo #{todo_id} added: '{task}'"
    
    def _add_reminder(self, text: str) -> str:
        # Extract time and message
        message = re.sub(r'^.*?(?:remind|reminder)\s*(?:me)?\s*(?:to|that)?\s*', '', text, flags=re.I)
        reminder_id = self.reminders.add(message)
        return f"⏰ Reminder #{reminder_id} set: '{message}'"
    
    def _list_notes(self, _) -> str:
        notes = self.notes.list_all()
        if not notes:
            return "📝 You have no notes yet. Say 'note: your text' to create one!"
        
        result = "📝 **Your Notes:**\n\n"
        for note in notes[-10:]:  # Show last 10
            result += f"• #{note['id']}: {note['content'][:50]}{'...' if len(note['content']) > 50 else ''}\n"
        return result
    
    def _list_todos(self, _) -> str:
        todos = self.todos.list_pending()
        if not todos:
            return "✅ All caught up! No pending todos."
        
        result = "📋 **Your Todos:**\n\n"
        for todo in todos:
            result += f"{'☑️' if todo['completed'] else '⬜'} #{todo['id']}: {todo['task']}\n"
        return result
    
    def _list_reminders(self, _) -> str:
        reminders = self.reminders.list_pending()
        if not reminders:
            return "⏰ No pending reminders."
        
        result = "⏰ **Your Reminders:**\n\n"
        for r in reminders:
            result += f"• #{r['id']}: {r['message']}\n"
        return result
    
    def _learn(self, text: str) -> str:
        # Pattern: "learn that X is Y" or "remember X means Y"
        match = re.search(r'(?:learn|remember|know)\s+(?:that\s+)?(.+?)\s+(?:is|means|=)\s+(.+)', text, flags=re.I)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            self.knowledge.add(key, value)
            return f"🧠 Got it! I'll remember that **{key}** → {value}"
        return "I couldn't understand. Try: 'learn that python is a programming language'"
    
    def _recall(self, text: str) -> str:
        query = re.sub(r'^.*?(?:recall|remember|know about|what is)\s*', '', text, flags=re.I)
        query = query.strip('?').strip()
        
        result = self.knowledge.get(query)
        if result:
            return f"🧠 Here's what I know about **{query}**:\n\n{result}"
        return f"🤔 I don't have information about '{query}'. You can teach me with 'learn that {query} is ...'"
    
    def _answer_question(self, text: str) -> str:
        return self._recall(text)
    
    def _define(self, text: str) -> str:
        word = re.sub(r'^.*?(?:define|definition|meaning)\s*(?:of)?\s*', '', text, flags=re.I)
        word = word.strip('?').strip()
        
        result = self.knowledge.get(f"what is {word}")
        if result:
            return f"📖 **{word.title()}**: {result}"
        return f"📖 I don't have a definition for '{word}'. You can teach me!"
    
    def _start_game(self, text: str) -> str:
        text_lower = text.lower()
        if 'rock' in text_lower or 'paper' in text_lower or 'scissors' in text_lower:
            choice = 'rock' if 'rock' in text_lower else 'paper' if 'paper' in text_lower else 'scissors'
            return Games.rock_paper_scissors(choice)
        
        if 'number' in text_lower or 'guess' in text_lower:
            self.context.current_game = Games.number_guessing_game()
            return "🎮 Let's play! I'm thinking of a number between 1 and 100.\nYou have 7 attempts. Make your guess!"
        
        if 'hangman' in text_lower:
            self.context.current_game = Games.hangman_game()
            display = '_' * len(self.context.current_game['word'])
            return f"🎮 Let's play Hangman!\nWord: {display}\nGuess a letter!"
        
        return """
🎮 **Available Games:**
• Number guessing - "play number guessing game"
• Rock Paper Scissors - "rock paper scissors"
• Hangman - "play hangman"

Which would you like to play?
"""
    
    def _handle_game_input(self, text: str) -> str:
        game = self.context.current_game
        if not game or not game.get('active'):
            return None
        
        if game['type'] == 'number_guess':
            numbers = EntityExtractor.extract_numbers(text)
            if numbers:
                guess = int(numbers[0])
                response, game_over = Games.check_guess(game, guess)
                return response
            return "Please guess a number! (1-100)"
        
        elif game['type'] == 'hangman':
            letters = re.findall(r'[a-zA-Z]', text)
            if letters:
                response, game_over = Games.hangman_guess(game, letters[0])
                return response
            return "Please guess a letter!"
        
        return "I'm not sure what you mean. Type a guess or say 'quit' to end the game."
    
    def _encrypt(self, text: str) -> str:
        text = re.sub(r'^.*?(?:encrypt|encode)\s*', '', text, flags=re.I)
        text = text.strip('"\'')
        encrypted = TextUtilities.caesar_cipher(text)
        return f"🔐 Encrypted (Caesar cipher, shift 3):\n\n`{encrypted}`"
    
    def _decrypt(self, text: str) -> str:
        text = re.sub(r'^.*?(?:decrypt|decode)\s*', '', text, flags=re.I)
        text = text.strip('"\'')
        decrypted = TextUtilities.caesar_cipher(text, decrypt=True)
        return f"🔓 Decrypted:\n\n`{decrypted}`"
    
    def _my_profile(self, _) -> str:
        """Show user's stored profile information."""
        name = self.user_profile.get_name()
        if not name and not self.user_profile.personal_facts and self.user_profile.profile.get('interaction_count', 0) < 2:
            return """📋 **Your Profile**

I don't know much about you yet! Tell me things like:
• "Call me [your name]"
• "My email is [email@example.com]"
• "I live in [city]"
• "I'm a [occupation]"
• "I'm [age] years old"
• "I enjoy [hobby]"

I'll remember everything you share! 💾"""
        
        lines = ["📋 **Your Profile**\n"]
        
        # Basic info
        if name:
            lines.append(f"👤 **Name**: {name}")
        if self.user_profile.profile.get('email'):
            lines.append(f"📧 **Email**: {self.user_profile.profile['email']}")
        if self.user_profile.profile.get('age'):
            lines.append(f"🎂 **Age**: {self.user_profile.profile['age']}")
        if self.user_profile.profile.get('location'):
            lines.append(f"📍 **Location**: {self.user_profile.profile['location']}")
        if self.user_profile.profile.get('occupation'):
            lines.append(f"💼 **Occupation**: {self.user_profile.profile['occupation']}")
        if self.user_profile.profile.get('birthday'):
            lines.append(f"🎈 **Birthday**: {self.user_profile.profile['birthday']}")
        
        # Interests
        interests = self.user_profile.get_interests()
        if interests:
            lines.append(f"\n🎯 **Interests**: {', '.join(interests)}")
        
        # Personal facts (favorites, etc.)
        facts = self.user_profile.get_all_facts()
        if facts:
            lines.append("\n📝 **Things I Know About You**:")
            for key, value in list(facts.items())[:10]:
                key_display = key.replace('_', ' ').title()
                lines.append(f"  • {key_display}: {value}")
        
        # Relationships
        relationships = self.user_profile.get_relationships()
        if relationships:
            lines.append("\n👥 **People You've Mentioned**:")
            for person, relation in list(relationships.items())[:5]:
                lines.append(f"  • {person} ({relation})")
        
        # Stats
        interactions = self.user_profile.profile.get('interaction_count', 0)
        created = self.user_profile.profile.get('created_at')
        if interactions > 0:
            lines.append(f"\n📊 **Stats**: {interactions} conversations")
            if created:
                try:
                    created_date = datetime.datetime.fromisoformat(created).strftime('%B %d, %Y')
                    lines.append(f"📅 **First met**: {created_date}")
                except:
                    pass
        
        return '\n'.join(lines)
    
    def _forget_me(self, _) -> str:
        """Delete all user profile data."""
        name = self.user_profile.get_name()
        
        # Reset the profile
        self.user_profile.profile = {
            'name': None,
            'nickname': None,
            'email': None,
            'age': None,
            'birthday': None,
            'location': None,
            'occupation': None,
            'created_at': None,
            'last_seen': None,
            'interaction_count': 0,
        }
        self.user_profile.personal_facts = {}
        self.user_profile.preferences = {}
        self.user_profile.relationships = {}
        self.user_profile.interests = []
        self.user_profile.conversation_topics = []
        self.user_profile._save()
        
        if name:
            return f"💔 Done, {name}. I've forgotten everything about you.\n\nYour profile has been completely reset. We can start fresh whenever you're ready."
        return "💔 Done. I've deleted all your stored information.\n\nYour profile has been completely reset. We're now strangers again."
    
    def _unknown(self, parsed: ParsedInput) -> str:
        # Check if it's a riddle answer
        if self.context.current_riddle:
            if 'answer' in parsed.original.lower() or 'give up' in parsed.original.lower():
                answer = self.context.current_riddle[1]
                self.context.current_riddle = None
                return f"🧩 The answer is: **{answer}**"
        
        # Try to find in knowledge base
        result = self.knowledge.get(parsed.original)
        if result:
            return f"💡 {result}"
        
        # Fallback responses
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that? 🤔",
            "Hmm, I don't quite get that. Try asking differently! 💭",
            "I'm still learning! Try saying 'help' to see what I can do. 📚",
            "That's beyond my current abilities. Type 'help' for options! 🌟",
        ]
        return random.choice(fallbacks)


# =============================================================================
# MAIN AURORA AI CLASS
# =============================================================================

class Aurora:
    """The main AI assistant class that ties everything together."""
    
    def __init__(self):
        print(f"\n{'='*60}")
        print(f"  🌟 AURORA AI v{Config.VERSION} - Initializing...")
        print(f"{'='*60}\n")
        
        # Initialize components
        self.knowledge = KnowledgeBase()
        self.memory = MemoryManager()
        self.notes = NoteManager()
        self.todos = TodoManager()
        self.reminders = ReminderManager()
        self.user_profile = UserProfile()  # NEW: Persistent user identity
        
        # Initialize classifier
        self.classifier = NaiveBayesClassifier()
        self._train_classifier()
        
        # Initialize response generator with user profile
        self.generator = ResponseGenerator(
            self.knowledge, self.memory, 
            self.notes, self.todos, self.reminders,
            self.user_profile  # Pass user profile to generator
        )
        
        print("  ✅ Knowledge base loaded")
        print("  ✅ Memory system initialized")
        print("  ✅ User profile loaded")
        print("  ✅ Intent classifier trained")
        print("  ✅ All systems operational!")
        print(f"\n{'='*60}\n")
    
    def _train_classifier(self):
        """Train the intent classifier with examples."""
        training_data = [
            # Greetings
            ("hello", Intent.GREETING), ("hi", Intent.GREETING), ("hey", Intent.GREETING),
            ("good morning", Intent.GREETING), ("hi there", Intent.GREETING),
            ("howdy", Intent.GREETING), ("whats up", Intent.GREETING),
            
            # Farewells
            ("bye", Intent.FAREWELL), ("goodbye", Intent.FAREWELL), ("see you", Intent.FAREWELL),
            ("talk later", Intent.FAREWELL), ("i have to go", Intent.FAREWELL),
            
            # Thanks
            ("thanks", Intent.THANKS), ("thank you", Intent.THANKS), ("appreciate it", Intent.THANKS),
            ("that helped", Intent.THANKS), ("you're helpful", Intent.THANKS),
            
            # Help
            ("help", Intent.HELP), ("what can you do", Intent.HELP), ("features", Intent.HELP),
            ("commands", Intent.HELP), ("how do i use you", Intent.HELP),
            
            # Math
            ("calculate", Intent.MATH), ("what is 2 plus 2", Intent.MATH),
            ("solve this equation", Intent.MATH), ("do math", Intent.MATH),
            ("5 times 3", Intent.MATH), ("15 divided by 3", Intent.MATH),
            
            # Time
            ("what time is it", Intent.TIME), ("current time", Intent.TIME),
            ("time please", Intent.TIME), ("whats the time", Intent.TIME),
            
            # Date
            ("what is the date", Intent.DATE), ("what day is it", Intent.DATE),
            ("today's date", Intent.DATE), ("current date", Intent.DATE),
            
            # Weather
            ("how is the weather", Intent.WEATHER), ("weather forecast", Intent.WEATHER),
            ("is it going to rain", Intent.WEATHER), ("temperature outside", Intent.WEATHER),
            
            # Jokes
            ("tell me a joke", Intent.JOKE), ("make me laugh", Intent.JOKE),
            ("say something funny", Intent.JOKE), ("got any jokes", Intent.JOKE),
            
            # Facts
            ("tell me a fact", Intent.FACT), ("interesting fact", Intent.FACT),
            ("did you know", Intent.FACT), ("random fact", Intent.FACT),
            
            # Quotes
            ("give me a quote", Intent.QUOTE), ("inspirational quote", Intent.QUOTE),
            ("famous quote", Intent.QUOTE), ("wisdom", Intent.QUOTE),
            
            # Motivation
            ("motivate me", Intent.MOTIVATE), ("i need motivation", Intent.MOTIVATE),
            ("cheer me up", Intent.MOTIVATE), ("encourage me", Intent.MOTIVATE),
            
            # Affirmation
            ("give me an affirmation", Intent.AFFIRMATION),
            ("positive affirmation", Intent.AFFIRMATION),
            ("self love", Intent.AFFIRMATION),
            
            # Riddles
            ("tell me a riddle", Intent.RIDDLE), ("got a puzzle", Intent.RIDDLE),
            ("brain teaser", Intent.RIDDLE),
            
            # Magic 8 ball
            ("magic 8 ball", Intent.MAGIC_8BALL), ("will i", Intent.MAGIC_8BALL),
            ("should i", Intent.MAGIC_8BALL), ("is it true that", Intent.MAGIC_8BALL),
            
            # Coin flip
            ("flip a coin", Intent.COIN_FLIP), ("heads or tails", Intent.COIN_FLIP),
            ("coin toss", Intent.COIN_FLIP),
            
            # Dice
            ("roll a dice", Intent.DICE_ROLL), ("roll d20", Intent.DICE_ROLL),
            ("throw dice", Intent.DICE_ROLL),
            
            # Random number
            ("random number", Intent.RANDOM_NUMBER), ("pick a number", Intent.RANDOM_NUMBER),
            ("generate number", Intent.RANDOM_NUMBER),
            
            # Password
            ("generate password", Intent.PASSWORD), ("create password", Intent.PASSWORD),
            ("random password", Intent.PASSWORD), ("secure password", Intent.PASSWORD),
            
            # Sentiment
            ("analyze sentiment", Intent.SENTIMENT), ("how does this sound", Intent.SENTIMENT),
            ("feeling of", Intent.SENTIMENT),
            
            # Word count
            ("word count", Intent.WORD_COUNT), ("count words", Intent.WORD_COUNT),
            ("how many words", Intent.WORD_COUNT),
            
            # Reverse
            ("reverse text", Intent.REVERSE_TEXT), ("backwards", Intent.REVERSE_TEXT),
            ("flip text", Intent.REVERSE_TEXT),
            
            # ASCII
            ("ascii art", Intent.ASCII_ART), ("draw a", Intent.ASCII_ART),
            ("text art", Intent.ASCII_ART),
            
            # Notes
            ("take a note", Intent.NOTE), ("write down", Intent.NOTE),
            ("note this", Intent.NOTE), ("remember this", Intent.NOTE),
            
            # Todos
            ("add todo", Intent.TODO), ("add task", Intent.TODO),
            ("to do list", Intent.TODO), ("add to list", Intent.TODO),
            
            # List notes/todos
            ("show notes", Intent.LIST_NOTES), ("my notes", Intent.LIST_NOTES),
            ("show todos", Intent.LIST_TODOS), ("my tasks", Intent.LIST_TODOS),
            ("pending tasks", Intent.LIST_TODOS),
            
            # Reminders
            ("remind me", Intent.REMINDER), ("set reminder", Intent.REMINDER),
            ("remind me to", Intent.REMINDER),
            ("show reminders", Intent.LIST_REMINDERS),
            
            # Learning
            ("learn that", Intent.LEARN), ("remember that", Intent.LEARN),
            ("know that", Intent.LEARN),
            
            # Recall
            ("what do you know about", Intent.RECALL),
            ("tell me about", Intent.RECALL),
            ("do you remember", Intent.RECALL),
            
            # Questions
            ("what is", Intent.QUESTION), ("who is", Intent.QUESTION),
            ("why is", Intent.QUESTION), ("how does", Intent.QUESTION),
            ("where is", Intent.QUESTION), ("when was", Intent.QUESTION),
            
            # Define
            ("define", Intent.DEFINE), ("meaning of", Intent.DEFINE),
            ("what does mean", Intent.DEFINE), ("definition of", Intent.DEFINE),
            
            # Games
            ("play a game", Intent.GAME), ("lets play", Intent.GAME),
            ("number guessing", Intent.GAME), ("hangman", Intent.GAME),
            ("rock paper scissors", Intent.GAME),
            
            # Convert
            ("convert", Intent.CONVERT), ("how many miles", Intent.CONVERT),
            ("to celsius", Intent.CONVERT), ("in kilometers", Intent.CONVERT),
            
            # Encrypt/Decrypt
            ("encrypt", Intent.ENCRYPT), ("encode", Intent.ENCRYPT),
            ("decrypt", Intent.DECRYPT), ("decode", Intent.DECRYPT),
            
            # Identity
            ("who are you", Intent.IDENTITY), ("what are you", Intent.IDENTITY),
            ("your name", Intent.IDENTITY),
            
            # Capability
            ("what can you do", Intent.CAPABILITY), ("help me with", Intent.CAPABILITY),
            ("your abilities", Intent.CAPABILITY),
            
            # Mood
            ("how are you", Intent.MOOD), ("how are you doing", Intent.MOOD),
            ("how do you feel", Intent.MOOD),
        ]
        self.classifier.train_batch(training_data)
    
    def parse_input(self, text: str) -> ParsedInput:
        """Parse user input into structured format."""
        cleaned = TextPreprocessor.clean(text)
        tokens = TextPreprocessor.tokenize(cleaned)
        
        # Try pattern matching first
        pattern_result = PatternMatcher.match(text)
        if pattern_result:
            intent, confidence = pattern_result
        else:
            # Fall back to classifier
            intent, confidence = self.classifier.predict(cleaned)
        
        # Extract entities
        entities = EntityExtractor.extract_all(text)
        
        # Analyze sentiment
        sentiment = SentimentAnalyzer.analyze(text)
        
        # Extract keywords
        keywords = TextPreprocessor.extract_keywords(cleaned)
        
        return ParsedInput(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            intent=intent,
            confidence=confidence,
            entities=entities,
            sentiment=sentiment,
            keywords=keywords
        )
    
    def process(self, user_input: str) -> str:
        """Process user input and generate response."""
        if not user_input or not user_input.strip():
            return "I didn't catch that. Could you say something? 💭"
        
        # Parse input
        parsed = self.parse_input(user_input)
        
        # Check for due reminders
        due_reminders = self.reminders.check_due()
        reminder_alert = ""
        if due_reminders:
            reminder_alert = "⏰ **Reminder Alert!**\n"
            for r in due_reminders:
                reminder_alert += f"• {r['message']}\n"
            reminder_alert += "\n"
        
        # Generate response
        response = self.generator.generate(parsed)
        
        # Add conversation turn
        self.generator.context.add_turn(
            user_input=user_input,
            bot_response=response,
            intent=parsed.intent,
            entities=parsed.entities
        )
        
        # Check if user mentioned their name
        name_match = re.search(r"(?:i am|i'm|my name is|call me)\s+(\w+)", user_input, re.I)
        if name_match:
            name = name_match.group(1).title()
            if name.lower() not in ['a', 'the', 'just', 'here', 'not']:
                self.memory.set_user_name(name)
                response += f"\n\nNice to meet you, {name}! I'll remember your name. 😊"
        
        return reminder_alert + response
    
    def run(self):
        """Run the interactive REPL."""
        print("🤖 Aurora AI Assistant is ready!")
        print("💡 Type 'help' for commands, 'quit' to exit.\n")
        
        name = self.memory.get_user_name()
        if name:
            print(f"Welcome back, {name}! 👋\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🌟 Aurora: Goodbye! Have a wonderful day! See you soon! 👋✨\n")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                response = self.process(user_input)
                print(f"\n🌟 Aurora: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n🌟 Aurora: Caught you trying to escape! 😄 Type 'quit' to exit properly.\n")
            except Exception as e:
                print(f"\n❌ Oops! Something went wrong: {str(e)}\n")
                if Config.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the Aurora AI Assistant."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗        ║
    ║  ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗       ║
    ║  ███████║██║   ██║██████╔╝██║   ██║██████╔╝███████║       ║
    ║  ██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══██║       ║
    ║  ██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║  ██║       ║
    ║  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝       ║
    ║                                                           ║
    ║       Advanced Unified Reasoning & Operational            ║
    ║              Response Assistant                           ║
    ║                                                           ║
    ║       🤖 Local AI • No API Required • 100% Python         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    aurora = Aurora()
    aurora.run()


if __name__ == "__main__":
    main()
