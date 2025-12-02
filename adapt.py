import numpy as np
import joblib
import os
import json
from typing import Dict, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# CONFIG & CONSTANTS

MODEL_DIR = "data/adapt"
MODEL_PATH = os.path.join(MODEL_DIR, "adapt_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "adapt_scaler.joblib")
HISTORY_PATH = os.path.join(MODEL_DIR, "adaptation_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)

DIFFICULTY_LEVELS = ["level_1", "level_2", "level_3", "level_4"]
DIFFICULTY_LABELS = {
    "level_1": "Beginner - Extra Support",
    "level_2": "Elementary - Guided Practice",
    "level_3": "Intermediate - Growing Confidence",
    "level_4": "Advanced - Independent Reading"
}

FONT_SIZE_RANGE = (18, 32)
SENTENCE_LENGTH_RANGE = (4, 16)
TTS_SPEED_RANGE = (0.7, 1.1)
WORD_SPACING_RANGE = (1.0, 1.8)

HINT_LEVELS = ["minimal", "low", "medium", "high", "maximum"]

# Weight for blending previous difficulty with new estimate
ADAPTATION_SMOOTHING = 0.7   # 0.7 new, 0.3 previous by default


# FEATURE ENGINEERING

def build_features(listen_result: Dict[str, Any],
                   observe_result: Dict[str, Any],
                   session_duration_sec: float,
                   previous_difficulty: Optional[str] = None) -> np.ndarray:
    """
    Converts outputs from Listen and Observe modules into feature vector for ML/rules.
    
    Features (8+ dimensions):
      0. WER (Word Error Rate) - pronunciation difficulty
      1. Reading instability (1 - confidence)
      2. Deletions (skipped words)
      3. Insertions (extra words)
      4. Normalized focus score
      5. Normalized blink rate
      6. Fatigue index (0-1)
      7. Session duration (minutes) - fatigue accumulation
      8. Total error density
      9. (optional) previous difficulty (normalized)
    """
    wer = listen_result.get("wer", 0.0)
    sentence_conf = listen_result.get("sentence_confidence", 1.0)
    deletions = listen_result.get("deletions", 0)
    insertions = listen_result.get("insertions", 0)
    substitutions = listen_result.get("substitutions", 0)

    focus = observe_result.get("focus_score", 100.0)
    blink_rate = observe_result.get("blink_rate_per_min", 18.0)
    fatigue = observe_result.get("fatigue_level", "low")

    # Convert fatigue to numeric
    fatigue_num = {
        "low": 0.0,
        "moderate": 0.5,
        "high": 1.0
    }.get(fatigue, 0.5)

    # Base features
    features = [
        wer,                            # 0: Pronunciation difficulty
        1.0 - sentence_conf,            # 1: Reading instability
        deletions / 10.0,               # 2: Normalized deletions
        insertions / 10.0,              # 3: Normalized insertions
        focus / 100.0,                  # 4: Normalized focus
        blink_rate / 30.0,              # 5: Normalized blink rate
        fatigue_num,                    # 6: Fatigue index
        session_duration_sec / 60.0     # 7: Session minutes
    ]

    # Total error density
    total_errors = deletions + insertions + substitutions
    features.append(total_errors / 15.0)  # 8: Normalized total errors

    # Previous difficulty for temporal smoothing (if available)
    if previous_difficulty and previous_difficulty in DIFFICULTY_LEVELS:
        prev_level_idx = DIFFICULTY_LEVELS.index(previous_difficulty)
        features.append(prev_level_idx / 3.0)  # 9: Normalized previous level

    return np.array(features).reshape(1, -1)

 
# RULE-BASED ADAPTATION ENGINE

def rule_based_adaptation(listen_result: Dict[str, Any],
                          observe_result: Dict[str, Any],
                          previous_difficulty: Optional[str] = None,
                          smooth_transition: bool = True) -> Dict[str, Any]:
    """
    Safe, explainable baseline adaptation engine.
    Uses multi-factor decision tree for difficulty adjustment.
    """
    wer = listen_result.get("wer", 0.0)
    focus = observe_result.get("focus_score", 100.0)
    fatigue = observe_result.get("fatigue_level", "low")
    attention_span = observe_result.get("attention_span_seconds", 30.0)
    needs_break = observe_result.get("needs_break", False)

    # ----- Difficulty Level Selection -----
    difficulty_score = 0.0
    reasons = []

    # Factor 1: Word Error Rate (most important)
    if wer > 0.35:
        difficulty_score += 0
        reasons.append("High error rate")
    elif wer > 0.22:
        difficulty_score += 1
        reasons.append("Moderate error rate")
    elif wer > 0.12:
        difficulty_score += 2
        reasons.append("Good accuracy")
    else:
        difficulty_score += 3
        reasons.append("Excellent accuracy")

    # Factor 2: Focus Score
    if focus < 50:
        difficulty_score -= 1
        reasons.append("Low focus")
    elif focus > 80:
        difficulty_score += 0.5
        reasons.append("High focus")

    # Factor 3: Fatigue Level
    if fatigue == "high":
        difficulty_score -= 1
        reasons.append("High fatigue")
    elif fatigue == "low":
        difficulty_score += 0.3

    # Factor 4: Attention Span
    if attention_span < 15:
        difficulty_score -= 0.5
        reasons.append("Short attention span")
    elif attention_span > 60:
        difficulty_score += 0.5
        reasons.append("Sustained attention")

    # Smooth transition if enabled and we have previous difficulty
    if smooth_transition and previous_difficulty and previous_difficulty in DIFFICULTY_LEVELS:
        prev_idx = DIFFICULTY_LEVELS.index(previous_difficulty)
        # Blend new score with previous level index
        difficulty_score = (
            difficulty_score * ADAPTATION_SMOOTHING +
            prev_idx * (1 - ADAPTATION_SMOOTHING)
        )

    # Convert score to level (clamp to 0-3)
    level_idx = int(np.clip(round(difficulty_score), 0, 3))
    difficulty = DIFFICULTY_LEVELS[level_idx]

    # ----- Font Size Adjustment -----
    if fatigue == "high" or focus < 55 or needs_break:
        font_size = 28  # Larger for tired readers
        reasons.append("Using larger font for comfort")
    elif focus > 80 and wer < 0.15:
        font_size = 20  # Smaller for confident readers
        reasons.append("Using smaller font for confident reader")
    else:
        font_size = 24  # Standard

    # ----- Sentence Length (words per sentence) -----
    if fatigue == "high" or attention_span < 20:
        sentence_length = 5  # Very short for fatigue
        reasons.append("Short sentences due to fatigue/attention")
    elif wer > 0.25:
        sentence_length = 7  # Short for struggling readers
    elif wer > 0.15:
        sentence_length = 10  # Medium
    else:
        sentence_length = 13  # Longer for advanced readers

    # ----- Text-to-Speech Speed -----
    if focus < 60 or wer > 0.3:
        tts_speed = 0.8  # Slower for struggling readers
        reasons.append("Slow TTS to support decoding")
    elif fatigue == "high":
        tts_speed = 0.85  # Slightly slower when tired
    elif wer < 0.1 and focus > 75:
        tts_speed = 1.0  # Normal speed for advanced readers
    else:
        tts_speed = 0.9  # Slightly slower (default for dyslexia)

    # ----- Hint Level -----
    if wer > 0.35 or focus < 50:
        hint_level = "maximum"
    elif wer > 0.25:
        hint_level = "high"
    elif wer > 0.15:
        hint_level = "medium"
    elif wer > 0.08:
        hint_level = "low"
    else:
        hint_level = "minimal"

    # ----- Word Spacing (for dyslexic readers) -----
    if fatigue == "high" or focus < 60:
        word_spacing = 1.6  # Extra spacing when tired
    elif wer > 0.2:
        word_spacing = 1.5  # More spacing for struggling readers
    else:
        word_spacing = 1.3  # Standard dyslexia-friendly spacing

    return {
        "difficulty": difficulty,
        "difficulty_label": DIFFICULTY_LABELS[difficulty],
        "difficulty_level_numeric": level_idx,
        "font_size": int(font_size),
        "sentence_length": int(sentence_length),
        "tts_speed": round(float(tts_speed), 2),
        "hint_level": hint_level,
        "word_spacing": round(float(word_spacing), 2),
        "engine": "rule_based",
        "reasons": reasons,
        "needs_break": needs_break,
        "timestamp": datetime.now().isoformat()
    }


 
# ML-BASED ADAPTATION ENGINE

def train_adapt_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, StandardScaler]:
    """
    Train ML adaptation model.
    
    Args:
        X: Feature matrix (N samples × 8-10 features)
        y: Target difficulty levels (N samples, values 0-3)
    
    Returns:
        Tuple of (trained_model, fitted_scaler)
    """
    if len(X) < 10:
        raise ValueError("Need at least 10 samples to train model")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"   Model trained and saved to {MODEL_PATH}")
    print(f"   Training samples: {len(X)}")
    print(f"   R² score: {model.score(X_scaled, y):.3f}")

    return model, scaler


def load_adapt_model() -> Tuple[Optional[LinearRegression], Optional[StandardScaler]]:
    """Load saved ML model and scaler"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def ml_based_adaptation(features: np.ndarray,
                        previous_difficulty: Optional[str] = None,
                        smooth_transition: bool = True) -> Optional[Dict[str, Any]]:
    """
    ML-based adaptation using trained regression model.
    Provides smoother, data-driven difficulty adjustments.
    """
    model, scaler = load_adapt_model()

    if model is None or scaler is None:
        return None  # Fallback to rule-based

    try:
        X_scaled = scaler.transform(features)
        pred = model.predict(X_scaled)[0]

        # Smooth transition with previous difficulty (if enabled)
        if smooth_transition and previous_difficulty and previous_difficulty in DIFFICULTY_LEVELS:
            prev_idx = DIFFICULTY_LEVELS.index(previous_difficulty)
            pred = pred * ADAPTATION_SMOOTHING + prev_idx * (1 - ADAPTATION_SMOOTHING)

        # Convert to difficulty level
        level_idx = int(np.clip(round(pred), 0, 3))
        difficulty = DIFFICULTY_LEVELS[level_idx]

        # Generate continuous parameters based on predicted level
        font_size = int(np.interp(level_idx, [0, 3], [28, 20]))
        tts_speed = np.interp(level_idx, [0, 3], [0.8, 1.0])
        sentence_length = int(np.interp(level_idx, [0, 3], [5, 13]))
        word_spacing = np.interp(level_idx, [0, 3], [1.6, 1.2])

        # Hint level from prediction
        if pred < 0.8:
            hint_level = "maximum"
        elif pred < 1.5:
            hint_level = "high"
        elif pred < 2.2:
            hint_level = "medium"
        elif pred < 2.8:
            hint_level = "low"
        else:
            hint_level = "minimal"

        return {
            "difficulty": difficulty,
            "difficulty_label": DIFFICULTY_LABELS[difficulty],
            "difficulty_level_numeric": level_idx,
            "predicted_score": round(float(pred), 2),
            "font_size": int(font_size),
            "sentence_length": int(sentence_length),
            "tts_speed": round(float(tts_speed), 2),
            "hint_level": hint_level,
            "word_spacing": round(float(word_spacing), 2),
            "engine": "ml_based",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"ML adaptation error: {e}")
        return None


 
# RL POLICY PLACEHOLDER
 

class AdaptRLPolicy:
    """
    Placeholder for Reinforcement Learning agent.
    
    State: [wer, focus, fatigue, previous_difficulty, session_time]
    Action: {decrease_difficulty, maintain, increase_difficulty}
    Reward: improvement_in_wer + improvement_in_focus - fatigue_penalty
    """

    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

    def get_state_key(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete key"""
        discrete = (state * 10).astype(int)
        return str(discrete.tolist())

    def select_action(self, state: np.ndarray) -> str:
        """
        Select action using epsilon-greedy policy
        Actions: "decrease", "maintain", "increase"
        """
        state_key = self.get_state_key(state)

        # Simple rule-based policy (placeholder)
        wer, focus = state[0], state[4] * 100

        if wer > 0.3 or focus < 50:
            return "decrease"
        elif wer < 0.1 and focus > 80:
            return "increase"
        return "maintain"

    def update(self, state, action, reward, next_state):
        """Update Q-values (to be implemented for RL version)"""
        pass  # Placeholder for future RL implementation


 
# LOGGING & HISTORY
 

def save_adaptation_history(adaptation_result: Dict[str, Any],
                            session_id: Optional[int] = None):
    """Save adaptation decisions for analysis and model improvement"""
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
        else:
            history = []

        entry = {
            "session_id": session_id,
            "timestamp": adaptation_result.get("timestamp"),
            "difficulty": adaptation_result.get("difficulty"),
            "engine": adaptation_result.get("engine"),
            "params": {
                "font_size": adaptation_result.get("font_size"),
                "sentence_length": adaptation_result.get("sentence_length"),
                "tts_speed": adaptation_result.get("tts_speed"),
                "hint_level": adaptation_result.get("hint_level")
            }
        }

        history.append(entry)
        history = history[-1000:]  # Keep last 1000 entries

        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        print(f"Could not save adaptation history: {e}")


 
# MAIN ADAPT PIPELINE (USED BY main.py)
 

def run_adapt_module(listen_result: Dict[str, Any],
                     observe_result: Dict[str, Any],
                     session_duration_sec: float,
                     use_ml: bool = False,
                     previous_difficulty: Optional[str] = None,
                     session_id: Optional[int] = None,
                     smooth_transition: bool = True) -> Dict[str, Any]:
    """
    Main adaptation pipeline - called by main.py
    
    Args:
        listen_result: Output from Listen module
        observe_result: Output from Observe module
        session_duration_sec: Current session duration
        use_ml: Whether to use ML model (if available)
        previous_difficulty: Last difficulty level for smooth transitions
        session_id: For logging purposes
        smooth_transition: Whether to blend with previous difficulty or not
    
    Returns:
        Dict with adaptation parameters for UI and other modules
    """
    # Step 1: Build feature vector
    features = build_features(
        listen_result,
        observe_result,
        session_duration_sec,
        previous_difficulty
    )

    # Step 2: Try ML engine if requested
    adaptation_result: Optional[Dict[str, Any]] = None
    if use_ml:
        adaptation_result = ml_based_adaptation(
            features,
            previous_difficulty,
            smooth_transition
        )

    # Step 3: Fallback to rule-based engine
    if adaptation_result is None:
        adaptation_result = rule_based_adaptation(
            listen_result,
            observe_result,
            previous_difficulty,
            smooth_transition
        )

    # Step 4: Save history for analysis
    save_adaptation_history(adaptation_result, session_id)

    return adaptation_result


 
# UTILITY HELPERS
 

def get_difficulty_description(difficulty: str) -> str:
    """Get human-readable difficulty description"""
    return DIFFICULTY_LABELS.get(difficulty, "Unknown Level")


def suggest_next_difficulty(current_difficulty: str,
                            performance_improving: bool) -> str:
    """Suggest next difficulty based on performance trend"""
    if current_difficulty not in DIFFICULTY_LEVELS:
        return "level_2"

    current_idx = DIFFICULTY_LEVELS.index(current_difficulty)

    if performance_improving and current_idx < 3:
        return DIFFICULTY_LEVELS[current_idx + 1]
    elif not performance_improving and current_idx > 0:
        return DIFFICULTY_LEVELS[current_idx - 1]

    return current_difficulty


 
# SELF-TEST
 

if __name__ == "__main__":
    print("=" * 60)
    print("ADAPT MODULE - Dynamic Difficulty Engine Test")
    print("=" * 60)

    # Test case 1: Struggling reader
    print("\n Test Case 1: Struggling Reader")
    fake_listen_1 = {
        "wer": 0.35,
        "sentence_confidence": 0.55,
        "deletions": 3,
        "insertions": 1,
        "substitutions": 2
    }
    fake_observe_1 = {
        "focus_score": 52,
        "blink_rate_per_min": 26,
        "fatigue_level": "high",
        "attention_span_seconds": 18,
        "needs_break": True
    }

    result_1 = run_adapt_module(
        fake_listen_1,
        fake_observe_1,
        session_duration_sec=120,
        use_ml=False
    )

    print(f"  Difficulty: {result_1['difficulty']} - {result_1['difficulty_label']}")
    print(f"  Font Size: {result_1['font_size']}px")
    print(f"  Sentence Length: {result_1['sentence_length']} words")
    print(f"  TTS Speed: {result_1['tts_speed']}x")
    print(f"  Hint Level: {result_1['hint_level']}")
    print(f"  Needs Break: {result_1['needs_break']}")
    if 'reasons' in result_1:
        print(f"  Reasons: {', '.join(result_1['reasons'])}")

    # Test case 2: Confident reader
    print("\nTest Case 2: Confident Reader")
    fake_listen_2 = {
        "wer": 0.08,
        "sentence_confidence": 0.92,
        "deletions": 0,
        "insertions": 0,
        "substitutions": 1
    }
    fake_observe_2 = {
        "focus_score": 88,
        "blink_rate_per_min": 17,
        "fatigue_level": "low",
        "attention_span_seconds": 95,
        "needs_break": False
    }

    result_2 = run_adapt_module(
        fake_listen_2,
        fake_observe_2,
        session_duration_sec=45,
        use_ml=False
    )

    print(f"  Difficulty: {result_2['difficulty']} - {result_2['difficulty_label']}")
    print(f"  Font Size: {result_2['font_size']}px")
    print(f"  Sentence Length: {result_2['sentence_length']} words")
    print(f"  TTS Speed: {result_2['tts_speed']}x")
    print(f"  Hint Level: {result_2['hint_level']}")
    if 'reasons' in result_2:
        print(f"  Reasons: {', '.join(result_2['reasons'])}")

    # Test case 3: Moderate performance with smooth transition
    print("\n Test Case 3: Moderate Reader (with previous difficulty)")
    fake_listen_3 = {
        "wer": 0.18,
        "sentence_confidence": 0.75,
        "deletions": 1,
        "insertions": 0,
        "substitutions": 2
    }
    fake_observe_3 = {
        "focus_score": 68,
        "blink_rate_per_min": 20,
        "fatigue_level": "moderate",
        "attention_span_seconds": 45,
        "needs_break": False
    }

    result_3 = run_adapt_module(
        fake_listen_3,
        fake_observe_3,
        session_duration_sec=90,
        use_ml=False,
        previous_difficulty="level_2"
    )

    print(f"  Previous: level_2")
    print(f"  Current: {result_3['difficulty']} - {result_3['difficulty_label']}")
    print(f"  Font Size: {result_3['font_size']}px")
    print(f"  TTS Speed: {result_3['tts_speed']}x")

    print("\n" + "=" * 60)
    print("All adaptation tests completed successfully!")
    print("=" * 60)

