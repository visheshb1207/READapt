import os
import uuid
import pyttsx3
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


TTS_OUTPUT_DIR = "data/tts/"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


BASE_TTS_RATE = 180       
BASE_TTS_VOLUME = 0.9


DYSLEXIC_PATTERNS = {
    "b_d": ["b", "d"],          
    "p_q": ["p", "q"],          
    "m_w": ["m", "w"],           
    "n_u": ["n", "u"],           
    "vowels": ["a", "e", "i", "o", "u"]
}


_tts_engine = None


def get_tts_engine():
    """Get or initialize the TTS engine singleton"""
    global _tts_engine
    if _tts_engine is None:
        engine = pyttsx3.init()
        engine.setProperty("volume", BASE_TTS_VOLUME)
        
        
        voices = engine.getProperty("voices")
        
        
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
        
        _tts_engine = engine
    return _tts_engine


DYSLEXIA_FRIENDLY_COLORS = {
    
    "light": {
        "background":"#F4F1DE",           
        "text": "#2D3142",                
        "highlight_primary": "#F2CC8F",    
        "highlight_secondary": "#81B29A",  
        "syllable_separator": "#E07A5F",   
        "error_underline": "#E63946",     
        "success": "#06D6A0",             
        "focus_word": "#FFA69E"           
    },
    
    "dark": {
        "background": "#1A1A2E",
        "text": "#E8E8E8",
        "highlight_primary": "#0F3460",
        "highlight_secondary": "#16213E",
        "syllable_separator": "#E94560",
        "error_underline": "#FF6B6B",
        "success": "#06D6A0",
        "focus_word": "#FFD93D"
    },
   
    "high_contrast": {
        "background": "#000000",
        "text": "#FFFF00",  
        "highlight_primary": "#00FF00",
        "highlight_secondary": "#00FFFF",
        "syllable_separator": "#FF00FF",
        "error_underline": "#FF0000",
        "success": "#00FF00",
        "focus_word": "#FFFFFF"
    }
}


DYSLEXIA_FRIENDLY_FONTS = {
    "primary": "Atkinson Hyperlegible",  
    "alternative": "OpenDyslexic",      
    "fallback": "Arial, sans-serif",     
    "letter_spacing": 0.12,              
    "word_spacing": 0.16,               
    "line_height": 1.8                   
}



@dataclass
class TextToken:
    """Structured representation of a word with visual cues"""
    word: str
    syllables: List[str]
    highlight: bool = False
    highlight_color: Optional[str] = None
    underline_error: bool = False
    underline_color: Optional[str] = None
    is_difficult: bool = False
    phonetic_hint: Optional[str] = None
    word_type: str = "normal" 



VOWELS = set("aeiouyAEIOUY")
CONSONANTS = set("bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ")


def syllabify_word_enhanced(word: str) -> List[str]:
    """
    Enhanced syllable segmentation with better heuristics
    Handles common patterns like:
    - Silent 'e' (make → ma-ke)
    - Double consonants (little → lit-tle)
    - Consonant clusters (str, spr, thr)
    """
    # Remove punctuation and get clean word
    clean_word = re.sub(r'[^\w]', '', word)
    
    if len(clean_word) <= 3:
        return [word]
    
    w = clean_word.lower()
    syllables = []
    current = ""
    
    i = 0
    while i < len(w):
        current += w[i]
        
        
        if i < len(w) - 1:
            curr_char = w[i]
            next_char = w[i + 1]
            
            
            if (i < len(w) - 2 and 
                curr_char in VOWELS and 
                next_char in CONSONANTS and 
                w[i + 2] in VOWELS):
                
                current += next_char
                syllables.append(current)
                current = ""
                i += 1
            
            
            elif (i < len(w) - 2 and 
                  curr_char in CONSONANTS and 
                  next_char == curr_char):
                syllables.append(current)
                current = ""
        
        i += 1
    
    if current:
        syllables.append(current)
    
    
    if syllables and word[0].isupper():
        syllables[0] = syllables[0].capitalize()
    
    return syllables if syllables else [word]


def get_phonetic_hint(word: str) -> Optional[str]:
    """
    Generate phonetic hints for commonly confused letter patterns
    """
    word_lower = word.lower()
    
    # Check for common dyslexic confusion patterns
    if any(char in word_lower for char in DYSLEXIC_PATTERNS["b_d"]):
        if 'b' in word_lower:
            return "Remember: 'b' has belly on the right →"
        elif 'd' in word_lower:
            return "Remember: 'd' has belly on the left ←"
    
    # Vowel confusion
    if sum(1 for c in word_lower if c in "aeiou") >= 2:
        return "Focus on the vowel sounds"
    
    # Silent 'e'
    if word_lower.endswith('e') and len(word_lower) > 3:
        return "Silent 'e' at the end makes vowel say its name"
    
    return None




def generate_tts_audio(text: str, 
                      tts_speed: float = 1.0,
                      emphasis_words: Optional[List[str]] = None) -> str:
    """
    Generate TTS audio with optional word emphasis
    
    Args:
        text: Text to convert to speech
        tts_speed: Speed multiplier (0.7-1.2)
        emphasis_words: Words to emphasize with SSML-like tags
    
    Returns:
        Path to generated audio file
    """
    engine = get_tts_engine()

    # Clamp speed to safe range
    tts_speed = float(np.clip(tts_speed, 0.7, 1.2))
    rate = int(BASE_TTS_RATE * tts_speed)
    engine.setProperty("rate", rate)

    # Add emphasis to specific words if provided
    processed_text = text
    if emphasis_words:
        for word in emphasis_words:
            # Add pauses and slight emphasis (pyttsx3 limited, but helps)
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            processed_text = pattern.sub(f"... {word} ...", processed_text)

    # Generate unique filename
    out_filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(TTS_OUTPUT_DIR, out_filename)

    try:
        engine.save_to_file(processed_text, out_path)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS generation failed: {e}")
        # Create a dummy file to prevent errors
        with open(out_path, 'w') as f:
            f.write("")

    return out_path


def generate_word_audio(word: str, speed: float = 0.8) -> str:
    """Generate slower audio for individual word practice"""
    return generate_tts_audio(word, tts_speed=speed)



# 6. ENHANCED WORD-LEVEL VISUAL CUES


def build_word_cues(text: str,
                   adapt_result: Dict,
                   listen_result: Optional[Dict] = None,
                   color_scheme: str = "light") -> List[Dict]:
    """
    Build enhanced token-level visual cues with:
    - Error highlighting
    - Difficulty-based hints
    - Phonetic hints
    - Syllable breakdown
    """
    words = text.split()
    tokens = []

    difficulty_level = adapt_result.get("difficulty", "level_2")
    hint_level = adapt_result.get("hint_level", "medium")
    
    # Get color scheme
    colors = DYSLEXIA_FRIENDLY_COLORS.get(color_scheme, DYSLEXIA_FRIENDLY_COLORS["light"])

    # Map difficulty to highlighting aggressiveness
    difficulty_factor = {
        "level_1": 1.0,   
        "level_2": 0.7,
        "level_3": 0.4,
        "level_4": 0.2    
    }.get(difficulty_level, 0.7)

    # Build error map from listen results
    word_error_map = {}
    error_details = {}
    
    if listen_result and "word_results" in listen_result:
        for wr in listen_result["word_results"]:
            target = wr.get("target_word", "").lower()
            has_error = (
                len(wr.get("phonetic_errors", [])) > 0 or 
                wr.get("error_type") in ["dyslexic_error", "substitution", "deletion"]
            )
            
            if has_error:
                word_error_map[target] = True
                error_details[target] = {
                    "spoken": wr.get("spoken_word", ""),
                    "error_type": wr.get("error_type", "unknown"),
                    "phonetic_errors": wr.get("phonetic_errors", [])
                }

    # Process each word
    for idx, w in enumerate(words):
        
        clean_word = re.sub(r'[^\w]', '', w).lower()
        syllables = syllabify_word_enhanced(w)
        
        has_error = word_error_map.get(clean_word, False)
        
        
        highlight = False
        highlight_color = None
        underline_error = False
        underline_color = None
        is_difficult = False
        phonetic_hint = None
        word_type = "normal"

        
        if has_error:
            underline_error = True
            underline_color = colors["error_underline"]
            highlight = True
            highlight_color = colors["highlight_secondary"]
            is_difficult = True
            word_type = "error"
            
            
            if clean_word in error_details:
                error_info = error_details[clean_word]
                phonetic_hint = f"You said '{error_info['spoken']}' - try '{clean_word}'"
        
        
        else:
            
            phonetic_hint = get_phonetic_hint(clean_word)
            
            if phonetic_hint:
                is_difficult = True
                word_type = "challenging"
            
           
            highlight_prob = {
                "high": 0.7 * difficulty_factor,
                "medium": 0.4 * difficulty_factor,
                "low": 0.2 * difficulty_factor
            }.get(hint_level, 0.4) * difficulty_factor
            
            if np.random.rand() < highlight_prob or is_difficult:
                highlight = True
                highlight_color = colors["highlight_primary"]

        
        token = {
            "word": w,
            "index": idx,
            "syllables": syllables,
            "highlight": highlight,
            "highlight_color": highlight_color,
            "underline_error": underline_error,
            "underline_color": underline_color,
            "is_difficult": is_difficult,
            "word_type": word_type,
            "phonetic_hint": phonetic_hint,
            "audio_path": None  
        }
        
        tokens.append(token)

    return tokens




def build_layout_spec(adapt_result: Dict, 
                     color_scheme: str = "light",
                     user_preferences: Optional[Dict] = None) -> Dict:
    """
    Build comprehensive layout specification for UI rendering
    
    Considers:
    - Adaptive difficulty settings
    - User preferences (font size overrides, etc.)
    - Accessibility requirements
    """
    font_size = adapt_result.get("font_size", 24)
    sentence_length = adapt_result.get("sentence_length", 10)
    word_spacing = adapt_result.get("word_spacing", 1.2)
    line_spacing = adapt_result.get("line_spacing", 1.5)
    
    # Apply user preference overrides
    if user_preferences:
        font_size = user_preferences.get("font_size_override", font_size)
        color_scheme = user_preferences.get("color_scheme", color_scheme)
    
    colors = DYSLEXIA_FRIENDLY_COLORS.get(color_scheme, DYSLEXIA_FRIENDLY_COLORS["light"])
    
    layout = {
        
        "font_family": DYSLEXIA_FRIENDLY_FONTS["primary"],
        "alternative_font": DYSLEXIA_FRIENDLY_FONTS["alternative"],
        "fallback_font": DYSLEXIA_FRIENDLY_FONTS["fallback"],
        "font_size": int(font_size),
        "font_weight": 400,  
        
        
        "line_height": float(line_spacing),
        "letter_spacing_em": DYSLEXIA_FRIENDLY_FONTS["letter_spacing"],
        "word_spacing_em": word_spacing,
        "paragraph_spacing": 2.0,
        
        
        "background_color": colors["background"],
        "text_color": colors["text"],
        "highlight_color": colors["highlight_primary"],
        "error_color": colors["error_underline"],
        "success_color": colors["success"],
        "focus_color": colors["focus_word"],
        
        
        "max_words_per_line": int(sentence_length),
        "text_align": "left",  
        "max_line_width": 600,  
        
        
        "enable_animations": True,
        "transition_speed": "0.3s",
        
        
        "color_scheme": color_scheme,
        "high_contrast_mode": color_scheme == "high_contrast",
        "reduced_motion": False  
    }

    return layout




def generate_reading_hints(tokens: List[Dict], difficulty: str) -> List[str]:
    """
    Generate contextual hints for challenging words
    """
    hints = []
    
    for token in tokens:
        if token.get("is_difficult") and token.get("phonetic_hint"):
            hints.append({
                "word": token["word"],
                "hint": token["phonetic_hint"],
                "syllables": token["syllables"]
            })
    
    
    if difficulty == "level_1":
        hints.append({
            "general": True,
            "hint": "Take your time. Sound out each word slowly."
        })
    
    return hints




def run_assist_module(text: str,
                     adapt_result: Dict,
                     listen_result: Optional[Dict] = None,
                     color_scheme: str = "light",
                     user_preferences: Optional[Dict] = None,
                     generate_word_audio: bool = False) -> Dict:
    """
    Main entry point for ASSIST module - Multisensory learning layer
    
    Args:
        text: Target passage/sentence to display and voice
        adapt_result: Output from Adapt module (difficulty settings)
        listen_result: Optional output from Listen module (for error highlighting)
        color_scheme: Visual theme ("light", "dark", "high_contrast")
        user_preferences: User-specific overrides
        generate_word_audio: Whether to generate individual word audio files
    
    Returns:
        {
            "audio_path": str,           # TTS audio file path
            "tokens": List[Dict],        # Word-level visual cues
            "layout": Dict,              # UI layout specification
            "hints": List[Dict],         # Reading hints
            "syllable_count": int,       # Total syllables (complexity metric)
            "assist_type": str           # Type of assistance provided
        }
    """
    
    # 1. Generate main TTS audio
    tts_speed = adapt_result.get("tts_speed", 0.9)
    
    # Identify words to emphasize (errors or difficult words)
    emphasis_words = []
    if listen_result and "word_results" in listen_result:
        for wr in listen_result["word_results"]:
            if wr.get("error_type"):
                emphasis_words.append(wr.get("target_word", ""))
    
    audio_path = generate_tts_audio(text, tts_speed=tts_speed, emphasis_words=emphasis_words)

    
    tokens = build_word_cues(
        text=text,
        adapt_result=adapt_result,
        listen_result=listen_result,
        color_scheme=color_scheme
    )
    
    
    if generate_word_audio:
        for token in tokens:
            if token.get("is_difficult"):
                word_audio = generate_word_audio(token["word"], speed=0.75)
                token["audio_path"] = word_audio

    
    layout = build_layout_spec(
        adapt_result=adapt_result,
        color_scheme=color_scheme,
        user_preferences=user_preferences
    )

    
    difficulty = adapt_result.get("difficulty", "level_2")
    hints = generate_reading_hints(tokens, difficulty)
    
    
    syllable_count = sum(len(token["syllables"]) for token in tokens)
    
    
    assist_type = "basic"
    if listen_result:
        assist_type = "error_aware"
    if generate_word_audio:
        assist_type = "comprehensive"

    return {
        "audio_path": audio_path,
        "tokens": tokens,
        "layout": layout,
        "hints": hints,
        "syllable_count": syllable_count,
        "word_count": len(tokens),
        "assist_type": assist_type,
        "color_scheme": color_scheme
    }



def cleanup_old_audio_files(max_age_seconds: int = 3600):
    """Clean up old TTS files to save disk space"""
    import time
    
    current_time = time.time()
    deleted_count = 0
    
    for filename in os.listdir(TTS_OUTPUT_DIR):
        filepath = os.path.join(TTS_OUTPUT_DIR, filename)
        
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"Could not delete {filename}: {e}")
    
    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} old TTS files")




if __name__ == "__main__":
    print("=" * 60)
    print("ASSIST MODULE - Multisensory Learning Layer Test")
    print("=" * 60)
    
    sample_text = "The friendly dragon played gently in the beautiful garden."

    fake_adapt = {
        "difficulty": "level_2",
        "font_size": 24,
        "sentence_length": 8,
        "tts_speed": 0.9,
        "hint_level": "medium",
        "word_spacing": 1.5,
        "line_spacing": 1.8
    }

    fake_listen = {
        "word_results": [
            {
                "target_word": "friendly",
                "spoken_word": "frendly",
                "phonetic_errors": [
                    {"target_phoneme": "i", "spoken_phoneme": "e", "type": "vowel_change"}
                ],
                "error_type": "substitution"
            },
            {
                "target_word": "beautiful",
                "spoken_word": "beutiful",
                "phonetic_errors": [],
                "error_type": "deletion"
            }
        ]
    }

    print("\nTest 1: Basic Assist (No Errors)")
    result1 = run_assist_module(sample_text, fake_adapt)
    print(f"  Audio: {result1['audio_path']}")
    print(f"  Word Count: {result1['word_count']}")
    print(f"  Syllable Count: {result1['syllable_count']}")
    print(f"  Assist Type: {result1['assist_type']}")
    
    print("\nTest 2: Error-Aware Assist")
    result2 = run_assist_module(sample_text, fake_adapt, fake_listen)
    print(f"  Assist Type: {result2['assist_type']}")
    print(f"  Hints Generated: {len(result2['hints'])}")
    
    error_tokens = [t for t in result2['tokens'] if t['underline_error']]
    print(f"  Error Words Highlighted: {len(error_tokens)}")
    for token in error_tokens:
        print(f"    - {token['word']}: {token['phonetic_hint']}")
    
    print("\nLayout Specification:")
    for key, value in result2['layout'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("\n Assist module test complete!")
    
    print("\n Testing cleanup...")
    cleanup_old_audio_files(max_age_seconds=0)  
