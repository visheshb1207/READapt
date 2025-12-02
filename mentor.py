import random
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


@dataclass
class CoachPersona:
    """Defines a mentor personality"""
    name: str
    tone: str
    style: str
    emoji_style: str
    age_group: str  

PERSONAS = {
    "aarav": CoachPersona(
        name="Aarav",
        tone="calm, friendly, patient teacher",
        style="short sentences, positive framing, gentle corrections",
        emoji_style="",
        age_group="child"
    ),
    "maya": CoachPersona(
        name="Maya",
        tone="enthusiastic, energetic, cheerful",
        style="encouraging, celebrates small wins, uses analogies",
        emoji_style="",
        age_group="teen"
    ),
    "alex": CoachPersona(
        name="Alex",
        tone="professional, supportive, understanding",
        style="clear feedback, data-driven insights, respectful",
        emoji_style="",
        age_group="adult"
    )
}


DEFAULT_PERSONA = "Yogesh"


MENTOR_MODEL_NAME = "gpt-4o-mini"
LLM_TIMEOUT = 10  
MAX_TOKENS = 100

_openai_client = None


def _get_openai_client():
    """Lazy-init OpenAI client with error handling"""
    global _openai_client
    if _openai_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Run 'pip install openai'.")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


EXCELLENT_PERFORMANCE = [
    "Wow! That was perfect! You're really getting the hang of this!",
    "Amazing reading! Your hard work is really paying off!",
    "Fantastic! You read that like a pro! Keep it up! ",
    "You're on fire today! That was excellent reading! ",
    "Incredible! You didn't miss a single word! "
]

GOOD_PERFORMANCE = [
    "Great job! You're reading more smoothly now. ",
    "Wonderful progress! Keep going, you're doing amazing. ",
    "That was nicely read! I'm proud of you. ",
    "Your focus was excellent just now! Keep it up! ",
    "Nice work! You're getting better each time! "
]

MODERATE_PERFORMANCE = [
    "You're doing fine. Let's slow down a little and try again. ",
    "Good effort! We'll improve one step at a time. ",
    "Nice try! Take a deep breath and read the next one. ",
    "You're learning! Every attempt makes you stronger. ",
    "That's okay! Practice makes progress, not perfect. "
]

STRUGGLING_PERFORMANCE = [
    "It's okay to make mistakes. Let's read this slowly together. ",
    "No worries at all. You're learning, and that's what matters. ",
    "Take your time. I'm right here with you. ",
    "Every reader struggles sometimes. You've got this! ",
    "Let's break this down into smaller pieces. You can do it! "
]


FATIGUE_MESSAGES = [
    "You look a little tired. Want to take a short break? ",
    "Your eyes might need some rest. Let's pause for a moment. ",
    "Great effort so far! A quick break will help you feel better. ",
    "You've been working hard! Time for a 2-minute break? ⏸",
    "Let's rest for a moment. You've earned it! "
]

FOCUS_DROP_MESSAGES = [
    "Let's try to focus together for just one more sentence. ",
    "Look at the screen gently and try again when you're ready. ",
    "Take a deep breath and let's refocus together. ",
    "I notice you're a bit distracted. Let's center ourselves. ",
    "Let's get back on track. You can do this! "
]

REFOCUS_MESSAGES = [
    "Welcome back! Let's pick up where we left off. ",
    "Good to see you focused again! Let's continue. ",
    "That's better! Your focus is strong now. "
]


IMPROVEMENT_MESSAGES = [
    "You're improving! Your {metric} is {percentage}% better than before! ",
    "Look at that progress! You've gotten so much better at {skill}! ",
    "I'm seeing real growth in your {skill}. Keep it up! ",
    "Wow! You just beat your personal best! ",
    "Your effort is paying off! {metric} improved by {percentage}%! "
]

MILESTONE_MESSAGES = [
    " Milestone! You've read {count} sentences today!",
    " Achievement unlocked: {achievement}!",
    " You just completed your longest reading session!",
    " You maintained focus for {duration} minutes straight!",
    " That's {count} days in a row! You're on a streak!"
]


ERROR_SPECIFIC_FEEDBACK = {
    "mirror_confusion": [
        "The letters 'b' and 'd' can be tricky! Remember: 'b' has a belly on the right. ",
        "Those mirror letters are hard for many people. You're doing great! ",
        "Let's practice: 'bed' - the word shows you both letters in order! "
    ],
    "voicing_confusion": [
        "Good try! Let's make the sound a little softer this time. ",
        "Try feeling the vibration in your throat for that sound. ",
        "Think about whispering vs. speaking for voiced sounds. "
    ],
    "fricative_error": [
        "Push a little air between your teeth for that sound. You can do it! ",
        "Think of it like a gentle hissing sound. ",
        "Put your tongue close to your teeth and blow softly. "
    ],
    "substitution": [
        "That word sounds similar, but let's try the correct one together. ",
        "You're close! Listen to how I say it and try again. ",
        "Similar sounds! Let's focus on the middle sound. "
    ],
    "deletion": [
        "You skipped a word. Let's read it together slowly. ",
        "Almost! There's one more word in that sentence. Try again! ",
        "Take your time - there's a small word hiding in there! "
    ],
    "insertion": [
        "You added an extra word. Let's read exactly what's on the screen. ",
        "Almost perfect! Just stick to the words you see. "
    ]
}


PHONEME_ENCOURAGEMENT = {
    "th": "The 'th' sound is tricky! Put your tongue between your teeth. ",
    "r": "The 'r' sound takes practice. Curl your tongue slightly. ",
    "ch": "For 'ch', start with a 't' and end with a 'sh' sound. ",
    "sh": "The 'sh' sound - like telling someone to be quiet! ",
    "wh": "Start with a gentle 'h' before the 'w' sound. "
}



def select_performance_message(wer: float, confidence: float, 
                               previous_wer: Optional[float] = None) -> str:
    """
    Select appropriate message based on performance metrics with trend analysis
    """
    # Check for improvement
    if previous_wer and wer < previous_wer * 0.8:  # 20% improvement
        improvement_pct = int((1 - wer/previous_wer) * 100)
        return random.choice(IMPROVEMENT_MESSAGES).format(
            metric="accuracy",
            percentage=improvement_pct,
            skill="reading"
        )
    
    # Performance-based selection
    if confidence > 0.85 and wer < 0.08:
        return random.choice(EXCELLENT_PERFORMANCE)
    elif confidence > 0.70 and wer < 0.15:
        return random.choice(GOOD_PERFORMANCE)
    elif confidence > 0.50 and wer < 0.30:
        return random.choice(MODERATE_PERFORMANCE)
    else:
        return random.choice(STRUGGLING_PERFORMANCE)


def select_attention_message(focus: float, fatigue: str, 
                            previous_focus: Optional[float] = None) -> Optional[str]:
    """
    Select attention-related message if needed
    Returns None if no attention intervention needed
    """
    # Critical: Fatigue
    if fatigue == "high":
        return random.choice(FATIGUE_MESSAGES)
    
    # Check if focus is recovering
    if previous_focus and focus > previous_focus + 15:
        return random.choice(REFOCUS_MESSAGES)
    
    # Low focus
    if focus < 50:
        return random.choice(FOCUS_DROP_MESSAGES)
    
    return None



def generate_realtime_feedback_rule_based(
    listen_result: Dict,
    observe_result: Dict,
    session_context: Optional[Dict] = None
) -> str:
    """
    Enhanced rule-based feedback with context awareness
    
    session_context can include:
    - previous_wer
    - previous_focus
    - turn_number
    - total_errors
    """
    wer = listen_result.get("wer", 0.0)
    confidence = listen_result.get("sentence_confidence", 0.5)
    focus = observe_result.get("focus_score", 50.0)
    fatigue = observe_result.get("fatigue_level", "moderate")
    
    # Get previous metrics if available
    prev_wer = session_context.get("previous_wer") if session_context else None
    prev_focus = session_context.get("previous_focus") if session_context else None
    turn_num = session_context.get("turn_number", 0) if session_context else 0

    # Priority 1: Attention issues (fatigue/focus)
    attention_msg = select_attention_message(focus, fatigue, prev_focus)
    if attention_msg:
        return attention_msg

    # Priority 2: Performance feedback
    performance_msg = select_performance_message(wer, confidence, prev_wer)
    
    # Add milestone celebrations
    if turn_num > 0 and turn_num % 10 == 0:
        milestone = random.choice(MILESTONE_MESSAGES).format(
            count=turn_num,
            achievement=f"{turn_num} sentences read",
            duration=turn_num * 0.5  
        )
        return milestone
    
    return performance_msg



def generate_error_specific_feedback(listen_result: Dict, max_hints: int = 2) -> List[str]:
    """
    Generate targeted phonetic hints based on actual errors
    Limits to max_hints to avoid overwhelming the learner
    """
    messages = []
    error_types_seen = set()
    
    word_results = listen_result.get("word_results", [])
    
    for word_result in word_results:
        error_type = word_result.get("error_type")
        
        # Avoid duplicate error type messages
        if error_type and error_type not in error_types_seen:
            if error_type in ERROR_SPECIFIC_FEEDBACK:
                messages.append(random.choice(ERROR_SPECIFIC_FEEDBACK[error_type]))
                error_types_seen.add(error_type)
        
        # Check for specific phoneme issues
        phonetic_errors = word_result.get("phonetic_errors", [])
        for phon_err in phonetic_errors:
            target_phoneme = phon_err.get("target_phoneme", "")
            if target_phoneme in PHONEME_ENCOURAGEMENT:
                messages.append(PHONEME_ENCOURAGEMENT[target_phoneme])
                break  # One phoneme hint per word max
        
        if len(messages) >= max_hints:
            break
    
    return messages



def generate_session_summary(session_log: List[Dict], 
                            session_duration_min: float) -> Dict[str, str]:
    """
    Generate comprehensive session summary with insights
    """
    if not session_log:
        return {
            "summary": "No session data available.",
            "highlights": [],
            "areas_to_practice": []
        }
    
    total_turns = len(session_log)
    
    # Calculate averages
    avg_conf = sum(x.get("sentence_confidence", 0) for x in session_log) / total_turns
    avg_wer = sum(x.get("wer", 0) for x in session_log) / total_turns
    avg_focus = sum(x.get("focus_score", 0) for x in session_log) / total_turns
    
    # Track improvement
    first_half_wer = sum(x.get("wer", 0) for x in session_log[:len(session_log)//2]) / max(len(session_log)//2, 1)
    second_half_wer = sum(x.get("wer", 0) for x in session_log[len(session_log)//2:]) / max(len(session_log)//2, 1)
    
    improved = second_half_wer < first_half_wer
    
    # Find most common errors
    error_counter = {}
    for entry in session_log:
        error = entry.get("common_error")
        if error:
            error_counter[error] = error_counter.get(error, 0) + 1
    
    most_common_error = max(error_counter, key=error_counter.get) if error_counter else None
    
    # Build summary
    accuracy = (1 - avg_wer) * 100
    
    summary_parts = [
        f"📚 Session Complete! You read {total_turns} sentences in {session_duration_min:.1f} minutes.",
        f"📊 Your accuracy was {accuracy:.1f}% with {avg_conf*100:.1f}% confidence.",
        f"🎯 Your focus averaged {avg_focus:.1f}/100."
    ]
    
    highlights = []
    
    if improved:
        improvement_pct = ((first_half_wer - second_half_wer) / first_half_wer * 100)
        highlights.append(f"You improved {improvement_pct:.1f}% during this session!")
    
    if avg_focus > 75:
        highlights.append("Excellent focus throughout the session!")
    
    if accuracy > 90:
        highlights.append("Outstanding accuracy! You're reading like a champion!")
    
    areas_to_practice = []
    
    if most_common_error:
        areas_to_practice.append(f"Practice '{most_common_error}' sounds for next time")
    
    if avg_focus < 60:
        areas_to_practice.append("Try taking short breaks to maintain focus")
    
    summary = " ".join(summary_parts)
    
    return {
        "summary": summary,
        "highlights": highlights,
        "areas_to_practice": areas_to_practice,
        "metrics": {
            "accuracy": accuracy,
            "confidence": avg_conf * 100,
            "focus": avg_focus,
            "turns": total_turns,
            "duration_min": session_duration_min
        }
    }



def generate_realtime_feedback_llm(
    listen_result: Dict,
    observe_result: Dict,
    adapt_result: Dict,
    session_context: Optional[Dict] = None,
    persona_name: str = DEFAULT_PERSONA
) -> str:
    """
    Generate personalized feedback using LLM with rich context
    """
    try:
        client = _get_openai_client()
    except Exception as e:
        print(f" OpenAI client initialization failed: {e}")
        raise
    
    persona = PERSONAS.get(persona_name, PERSONAS[DEFAULT_PERSONA])
    
    wer = listen_result.get("wer", 0.0)
    confidence = listen_result.get("sentence_confidence", 0.5)
    focus = observe_result.get("focus_score", 50.0)
    fatigue = observe_result.get("fatigue_level", "moderate")
    difficulty = adapt_result.get("difficulty", "level_2")
    
    # Build context
    context_parts = [
        f"Reading accuracy: {(1-wer)*100:.1f}%",
        f"Confidence: {confidence*100:.1f}%",
        f"Focus: {focus:.1f}/100",
        f"Fatigue: {fatigue}",
        f"Difficulty: {difficulty}"
    ]
    
    if session_context:
        if session_context.get("previous_wer"):
            if wer < session_context["previous_wer"]:
                context_parts.append(" Improved from last attempt")
            else:
                context_parts.append("Similar to last attempt")
        
        turn_num = session_context.get("turn_number", 0)
        if turn_num:
            context_parts.append(f"Turn {turn_num} of session")
    
    learner_state = "\n".join(context_parts)
    
    
    system_prompt = (
        f"You are {persona.name}, an AI reading coach for dyslexic learners.\n"
        f"Your tone is {persona.tone}.\n"
        f"Your style: {persona.style}\n"
        f"Target age group: {persona.age_group}\n\n"
        "RULES:\n"
        "- Respond in 20-40 words maximum\n"
        "- Use simple, encouraging language\n"
        "- NEVER blame or criticize the learner\n"
        "- Celebrate small wins\n"
        f"- You may use these emojis sparingly: {persona.emoji_style}\n"
        "- If the learner is tired, suggest a break\n"
        "- If they're doing well, praise specifically\n"
        "- If struggling, offer gentle, actionable support\n"
        "- Output ONLY the message text, nothing else"
    )
    
    user_prompt = (
        "Current learner status:\n"
        f"{learner_state}\n\n"
        "Give one encouraging coaching message that responds to these metrics."
    )
    
    try:
        completion = client.chat.completions.create(
            model=MENTOR_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_TOKENS,
            timeout=LLM_TIMEOUT
        )
        
        message = completion.choices[0].message.content.strip()
        return message
    
    except Exception as e:
        print(f" LLM generation failed: {e}")
        raise



def generate_difficulty_feedback(adapt_result: Dict, previous_difficulty: Optional[str] = None) -> str:
    """Generate feedback about difficulty changes"""
    current_diff = adapt_result.get("difficulty", "level_2")
    reason = adapt_result.get("reason", "")
    
    if previous_difficulty:
        curr_idx = ["level_1", "level_2", "level_3", "level_4"].index(current_diff)
        prev_idx = ["level_1", "level_2", "level_3", "level_4"].index(previous_difficulty)
        
        if curr_idx > prev_idx:
            return f"🎉 You're ready for harder content! Moving to {adapt_result.get('difficulty_name', current_diff)}!"
        elif curr_idx < prev_idx:
            return f"Let's slow down a bit with {adapt_result.get('difficulty_name', current_diff)} to build confidence. "
        else:
            return f"Continuing with {adapt_result.get('difficulty_name', current_diff)}. You're doing great! "
    
    return f"We'll work with {adapt_result.get('difficulty_name', current_diff)} difficulty. {reason}"


def run_mentor_module(
    listen_result: Dict,
    observe_result: Dict,
    adapt_result: Dict,
    session_history: List[Dict],
    use_llm: bool = False,
    persona_name: str = DEFAULT_PERSONA,
    previous_difficulty: Optional[str] = None
) -> Dict:
    """
    Main entry point for the Mentor module
    
    Args:
        listen_result: Output from Listen module
        observe_result: Output from Observe module
        adapt_result: Output from Adapt module
        session_history: List of previous turns for trend analysis
        use_llm: Whether to use LLM-based coaching
        persona_name: Which persona to use (aarav, maya, alex)
        previous_difficulty: Previous difficulty level for transition feedback
    
    Returns:
        {
            "realtime_message": str,
            "error_messages": List[str],
            "difficulty_feedback": str,
            "encouragement_level": str,
            "session_summary": Dict (if session complete),
            "timestamp": str,
            "engine": str
        }
    """
    
    # Build session context for better feedback
    session_context = None
    if session_history:
        last_turn = session_history[-1] if session_history else {}
        session_context = {
            "previous_wer": last_turn.get("wer"),
            "previous_focus": last_turn.get("focus_score"),
            "turn_number": len(session_history) + 1,
            "total_errors": sum(1 for t in session_history if t.get("wer", 0) > 0.2)
        }
    
    # 1. Difficulty transition feedback
    difficulty_msg = generate_difficulty_feedback(adapt_result, previous_difficulty)
    
    # 2. Error-specific coaching
    error_msgs = generate_error_specific_feedback(listen_result, max_hints=2)
    
    # 3. Real-time mentor message
    engine_used = "rule_based"
    
    if use_llm:
        try:
            realtime_msg = generate_realtime_feedback_llm(
                listen_result,
                observe_result,
                adapt_result,
                session_context,
                persona_name
            )
            engine_used = "llm"
        except Exception as e:
            # Fallback to rule-based
            print(f" Mentor LLM failed, using rule-based fallback: {e}")
            realtime_msg = generate_realtime_feedback_rule_based(
                listen_result,
                observe_result,
                session_context
            )
    else:
        realtime_msg = generate_realtime_feedback_rule_based(
            listen_result,
            observe_result,
            session_context
        )
    
    # 4. Determine encouragement level for UI
    wer = listen_result.get("wer", 0.0)
    focus = observe_result.get("focus_score", 50)
    
    if wer < 0.1 and focus > 75:
        encouragement_level = "high"
    elif wer < 0.25 and focus > 55:
        encouragement_level = "medium"
    else:
        encouragement_level = "supportive"
    
    return {
        "realtime_message": realtime_msg,
        "error_messages": error_msgs,
        "difficulty_feedback": difficulty_msg,
        "encouragement_level": encouragement_level,
        "timestamp": datetime.now().isoformat(),
        "engine": engine_used,
        "persona": persona_name
    }


if __name__ == "__main__":
    print("=" * 70)
    print("MENTOR MODULE - Personalized AI Coach Test")
    print("=" * 70)
    
    # Test data
    test_listen = {
        "wer": 0.18,
        "sentence_confidence": 0.72,
        "word_results": [
            {
                "target_word": "beautiful",
                "error_type": "substitution",
                "phonetic_errors": [{"target_phoneme": "th", "type": "fricative_error"}]
            }
        ]
    }
    
    test_observe = {
        "focus_score": 68,
        "fatigue_level": "moderate"
    }
    
    test_adapt = {
        "difficulty": "level_2",
        "difficulty_name": "Elementary",
        "reason": "Good progress"
    }
    
    test_session = [
        {"wer": 0.25, "sentence_confidence": 0.65, "focus_score": 60},
        {"wer": 0.22, "sentence_confidence": 0.70, "focus_score": 65},
        {"wer": 0.18, "sentence_confidence": 0.72, "focus_score": 68}
    ]
    
    print("\n Test 1: Rule-Based Mentor (Aarav)")
    result1 = run_mentor_module(
        test_listen, test_observe, test_adapt,
        test_session, use_llm=False, persona_name="aarav"
    )
    
    print(f"  Engine: {result1['engine']}")
    print(f"  Persona: {result1['persona']}")
    print(f"  Message: {result1['realtime_message']}")
    print(f"  Difficulty: {result1['difficulty_feedback']}")
    print(f"  Encouragement: {result1['encouragement_level']}")
    if result1['error_messages']:
        print(f"  Error Hints: {result1['error_messages']}")
    
    print("\n Test 2: LLM-Based Mentor (if configured)")
    try:
        result2 = run_mentor_module(
            test_listen, test_observe, test_adapt,
            test_session, use_llm=True, persona_name="maya"
        )
        print(f"  Engine: {result2['engine']}")
        print(f"  Persona: {result2['persona']}")
        print(f"  Message: {result2['realtime_message']}")
    except Exception as e:
        print(f" LLM test skipped: {e}")
    
    print("\n Test 3: Session Summary")
    summary = generate_session_summary(test_session, session_duration_min=8.5)
    print(f"  {summary['summary']}")
    if summary['highlights']:
        print("\n  Highlights:")
        for h in summary['highlights']:
            print(f"    {h}")
    if summary['areas_to_practice']:
        print("\n  Areas to practice:")
        for a in summary['areas_to_practice']:
            print(f"    • {a}")
    
    print("\n" + "=" * 70)
    print(" Mentor module test complete!")
