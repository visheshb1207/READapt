import os
import time
import json
import threading
import streamlit as st
from huggingface_hub import InferenceClient

#  Your modules (function names unchanged) 
from Listen import run_listen_module
from Observe import run_observe_module
from Adapt import run_adapt_module
from Assist import run_assist_module, cleanup_old_audio_files
from mentor import run_mentor_module
import Database as db


# 1. STREAMLIT CONFIG


st.set_page_config(
    page_title="Dyslexia AI Reading Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .status-ready {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #155724;
        margin: 20px 0;
    }
    .status-recording {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #856404;
        margin: 20px 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# 2. HUGGING FACE LLM CONFIG


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

hf_client = None
if HF_TOKEN:
    try:
        hf_client = InferenceClient(
            model="openai/gpt-4o-mini",
            token=HF_TOKEN,
        )
    except Exception as e:
        st.warning(f"HuggingFace client init failed, using fallback sentences. Error: {e}")
        hf_client = None
else:
    hf_client = None


def generate_llm_sentence(difficulty: str = "level_1", theme: str = "general") -> str:
    """Generate one dyslexia-friendly sentence using HF LLM (function name unchanged)"""
    level_map = {
        "level_1": "very short phonics sentence with simple words",
        "level_2": "short and simple child-friendly sentence",
        "level_3": "medium length sentence with common vocabulary",
        "level_4": "slightly longer sentence with richer vocabulary",
    }
    diff_desc = level_map.get(difficulty, "simple sentence")

    # Fallback if HF not configured
    if hf_client is None:
        fallback_sentences = {
            "level_1": [
                "The cat sat on the red mat.",
                "The dog ran fast.",
                "I see a big tree.",
                "The sun is bright.",
            ],
            "level_2": [
                "The small dog ran happily across the green field.",
                "My friend likes to play with colorful toys.",
                "The bird sang a beautiful song in the morning.",
            ],
            "level_3": [
                "The curious kitten explored the garden carefully.",
                "Children laughed while playing together at the park.",
                "The rainbow appeared after the gentle rain stopped.",
            ],
            "level_4": [
                "The determined student practiced reading every single day.",
                "Colorful butterflies danced gracefully among the blooming flowers.",
                "The young explorer discovered an ancient treasure in the cave.",
            ]
        }
        import random
        level = difficulty if difficulty in fallback_sentences else "level_1"
        return random.choice(fallback_sentences[level])

    prompt = f"""
You are generating a reading sentence for a dyslexic learner.

Generate ONE {diff_desc}.
Theme: {theme}

Rules:
- Only ONE sentence
- 5–12 words
- Very simple, concrete words
- Avoid names and complex punctuation
- Return ONLY the sentence, nothing else.
"""

    try:
        resp = hf_client.text_generation(
            prompt,
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
        )
        return " ".join(resp.strip().split())
    except Exception as e:
        print(" HuggingFace generation failed:", e)
        return "The cat sat on the red mat."



# 3. DATABASE INIT


db.init_database()


# 4. SESSION STATE INIT


# User & session management
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "session_history" not in st.session_state:
    st.session_state.session_history = []

# Module results
if "previous_difficulty" not in st.session_state:
    st.session_state.previous_difficulty = None

if "last_observe_result" not in st.session_state:
    st.session_state.last_observe_result = None

if "last_listen_result" not in st.session_state:
    st.session_state.last_listen_result = None

if "last_adapt_result" not in st.session_state:
    st.session_state.last_adapt_result = None

# Reading text
if "reading_text" not in st.session_state:
    st.session_state.reading_text = generate_llm_sentence("level_1")

# Recording state - CRITICAL: This tracks if we SHOULD start recording
if "should_record" not in st.session_state:
    st.session_state.should_record = False

if "recording_complete" not in st.session_state:
    st.session_state.recording_complete = False

if "system_ready" not in st.session_state:
    st.session_state.system_ready = True


# 5. SIDEBAR – LEARNER & SESSION CONTROL


st.sidebar.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'> Learner Panel</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    with st.expander(" Create New Learner", expanded=not st.session_state.user_id):
        name = st.text_input("Name", placeholder="Enter learner's name")
        age = st.number_input("Age", min_value=5, max_value=25, value=10)
        grade = st.text_input("Grade", placeholder="e.g., Grade 5")

        if st.button(" Create Learner", use_container_width=True):
            if not name.strip():
                st.warning(" Please enter a name.")
            else:
                user_id = db.create_user(
                    name=name.strip(),
                    age=int(age),
                    grade_level=grade.strip(),
                )
                st.session_state.user_id = user_id
                st.success(f" Learner created! ID: {user_id}")
                st.rerun()

    st.markdown("---")

    # Display active learner
    if st.session_state.user_id:
        user_info = db.get_user(st.session_state.user_id)
        if user_info:
            st.markdown(f"""
            <div style='background: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3;'>
                <strong> Active Learner</strong><br>
                <strong>Name:</strong> {user_info['name']}<br>
                <strong>Age:</strong> {user_info['age']}<br>
                <strong>Grade:</strong> {user_info['grade_level']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Session Control")
        
        col_start, col_end = st.columns(2)
        
        with col_start:
            if st.button("Start Session", use_container_width=True, disabled=st.session_state.session_id is not None):
                sid = db.start_new_session(st.session_state.user_id)
                st.session_state.session_id = sid
                st.session_state.session_history = []
                st.session_state.previous_difficulty = None
                st.session_state.last_observe_result = None
                st.session_state.last_listen_result = None
                st.session_state.last_adapt_result = None
                st.session_state.reading_text = generate_llm_sentence("level_1")
                st.session_state.recording_complete = False
                st.success(f" Session {sid} started!")
                st.rerun()

        with col_end:
            if st.button(" End Session", use_container_width=True, disabled=st.session_state.session_id is None):
                if st.session_state.session_id:
                    db.end_session(
                        st.session_state.session_id,
                        session_notes="Session ended from UI",
                    )
                    st.success(" Session ended!")
                    st.session_state.session_id = None
                    st.rerun()

        if st.session_state.session_id:
            st.info(f" Active Session: **{st.session_state.session_id}**")

    st.markdown("---")
    
    # Settings
    with st.expander(" Settings"):
        use_llm_mentor = st.checkbox(" Use AI Mentor (OpenAI)", value=False, key="use_llm_mentor")
        persona = st.selectbox(" Mentor Persona", ["aarav", "maya", "alex"], key="mentor_persona")
        color_scheme = st.selectbox(" Color Scheme", ["light", "dark", "high_contrast"], key="color_scheme")
        record_duration = st.slider("⏱ Recording Duration (sec)", 3, 15, 6, key="record_duration")
    
    st.markdown("---")
    show_history = st.checkbox("Show Progress History", value=False)


# 6. MAIN HEADER


st.markdown("""
<div class='main-header'>
    <h1 style='margin: 0;'> Dyslexia AI Reading Assistant</h1>
    <p style='margin: 5px 0 0 0; font-size: 16px;'>AI-Powered Multisensory Learning Platform</p>
</div>
""", unsafe_allow_html=True)

# Guards
if not st.session_state.user_id:
    st.warning(" Please create a learner in the sidebar to begin.")
    st.stop()

if not st.session_state.session_id:
    st.info(" Start a reading session from the sidebar to continue.")
    st.stop()


# 7. SYSTEM STATUS INDICATOR


if st.session_state.system_ready and not st.session_state.should_record and not st.session_state.recording_complete:
    st.markdown("""
    <div class='status-ready'>
         System Ready - Click "Start Reading" to begin!
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.should_record:
    st.markdown("""
    <div class='status-recording'>
        RECORDING IN PROGRESS - Webcam & Microphone Active - Please Read the Text Aloud
    </div>
    """, unsafe_allow_html=True)


# 8. READING TEXT DISPLAY


st.markdown("---")
st.subheader(" Reading Text")

# Display current reading text in a styled box
st.markdown(f"""
<div style='background: #f8f9fa; padding: 30px; border-radius: 10px; border: 2px solid #667eea; text-align: center;'>
    <p style='font-size: 28px; font-weight: 500; color: #2c3e50; margin: 0; line-height: 1.6;'>
        {st.session_state.reading_text}
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(" This sentence will be used for speech analysis. Read it clearly when recording starts.")


# 9. CONTROL BUTTONS


st.markdown("---")

col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    if st.button(
        " Start Reading",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.should_record
    ):
        # Set flag to start recording
        st.session_state.should_record = True
        st.session_state.recording_complete = False
        st.session_state.system_ready = False
        st.rerun()

with col2:
    if st.session_state.last_adapt_result is not None and not st.session_state.should_record:
        if st.button(" Generate Next Sentence", use_container_width=True):
            difficulty = st.session_state.last_adapt_result.get("difficulty", "level_1")
            st.session_state.reading_text = generate_llm_sentence(difficulty=difficulty)
            st.session_state.recording_complete = False
            st.rerun()

with col3:
    st.markdown("")  # Spacer


if st.session_state.should_record and not st.session_state.recording_complete:
    
    current_text = st.session_state.reading_text
    record_duration = st.session_state.get("record_duration", 6)
    
    st.markdown("---")
    st.markdown("##  Recording & Analysis Pipeline")
    
    # Show countdown
    countdown_placeholder = st.empty()
    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f"""
        <div style='text-align: center; font-size: 48px; padding: 20px; color: #667eea;'>
            Recording starts in {i}...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
    
    countdown_placeholder.markdown(f"""
    <div style='text-align: center; font-size: 36px; padding: 20px; color: #28a745; font-weight: bold;'>
         START READING NOW! 
    </div>
    """, unsafe_allow_html=True)
    
    # Create progress container
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    #  SIMULTANEOUS: ATTENTION TRACKING (WEBCAM) & SPEECH (MICROPHONE) 
    status_text.text(f" Recording for {record_duration} seconds - Webcam & Microphone active...")
    
    # Use threading to run both simultaneously
    observe_result = None
    listen_result = None
    observe_error = None
    listen_error = None
    
    def run_observe_thread():
        global observe_result, observe_error
        try:
            observe_result = run_observe_module(
                duration_sec=record_duration,
                show_debug_window=False,
                camera_index=0,
            )
        except Exception as e:
            observe_error = e
    
    def run_listen_thread():
        global listen_result, listen_error
        try:
            focus_drop = False  # We don't have observe_result yet
            listen_result = run_listen_module(
                target_text=current_text,
                focus_drop=focus_drop,
                record_duration=record_duration,
            )
        except Exception as e:
            listen_error = e

    
    # Start both threads
    observe_thread = threading.Thread(target=run_observe_thread)
    listen_thread = threading.Thread(target=run_listen_thread)
    
    observe_thread.start()
    listen_thread.start()
    
    # Wait for both to complete
    observe_thread.join()
    listen_thread.join()
    
    # Mark recording as complete
    st.session_state.should_record = False
    st.session_state.recording_complete = True
    
    progress_bar.progress(40)
    status_text.text(" Recording complete! Processing results...")
    
    # Handle errors
    if observe_error:
        st.error(f" Attention tracking failed: {observe_error}")
        observe_result = {
            "focus_score": 50, "fatigue_level": "moderate",
            "blink_rate_per_min": 18, "gaze_stability": 0.5,
            "head_movement": 0.1, "attention_span_seconds": 0
        }
    
    if listen_error:
        st.error(f" Speech analysis failed: {listen_error}")
        listen_result = {
            "spoken_text": "", "expected_text": current_text,
            "wer": 0.5, "sentence_confidence": 0.5,
            "insertions": 0, "deletions": 0, "substitutions": 0,
            "word_results": []
        }
    
    # Store results
    st.session_state.last_observe_result = observe_result
    st.session_state.last_listen_result = listen_result
    
    # Log to database
    if observe_result:
        db.log_observe(st.session_state.session_id, observe_result)
    if listen_result:
        db.log_listen(st.session_state.session_id, listen_result)
    
    #  DISPLAY RESULTS 
    st.markdown("### Attention Analysis Results")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Focus Score", f"{observe_result.get('focus_score', 0):.1f}/100")
    with col_b:
        st.metric("Fatigue Level", observe_result.get('fatigue_level', 'unknown').title())
    with col_c:
        st.metric("Blink Rate", f"{observe_result.get('blink_rate_per_min', 0):.1f}/min")
    
    st.markdown("### Speech Analysis Results")
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("** Target Text:**")
        st.info(listen_result["expected_text"])
    with col_y:
        st.markdown("** You Said:**")
        st.info(listen_result["spoken_text"])
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        accuracy = (1 - listen_result["wer"]) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    with col_m2:
        st.metric("Confidence", f"{listen_result['sentence_confidence']*100:.1f}%")
    with col_m3:
        total_errors = listen_result.get("insertions", 0) + listen_result.get("deletions", 0) + listen_result.get("substitutions", 0)
        st.metric("Errors", total_errors)
    with col_m4:
        st.metric("WER", f"{listen_result['wer']:.3f}")
    
    progress_bar.progress(50)
    
    #  ADAPT DIFFICULTY 
    status_text.text(" Adapting difficulty level...")
    
    approx_session_duration_sec = 30 * (len(st.session_state.session_history) + 1)
    
    adapt_result = run_adapt_module(
        listen_result=listen_result,
        observe_result=observe_result,
        session_duration_sec=approx_session_duration_sec,
        use_ml=False,
        smooth_transition=True
    )
    
    st.session_state.previous_difficulty = adapt_result["difficulty"]
    st.session_state.last_adapt_result = adapt_result
    db.log_adapt(st.session_state.session_id, adapt_result)
    
    st.markdown("### Difficulty Adaptation")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Difficulty Level", adapt_result.get("difficulty_name", adapt_result["difficulty"]))
    with col_d2:
        st.metric("Font Size", f"{adapt_result['font_size']}px")
    with col_d3:
        st.metric("TTS Speed", f"{adapt_result['tts_speed']}x")
    
    if adapt_result.get("reason"):
        st.info(f" {adapt_result['reason']}")
    
    progress_bar.progress(70)
    
    #  MULTISENSORY ASSISTANCE 
    status_text.text(" Generating multisensory support...")
    
    assist_result = run_assist_module(
        text=current_text,
        adapt_result=adapt_result,
        listen_result=listen_result,
        color_scheme=st.session_state.get("color_scheme", "light"),
        user_preferences=None,
        generate_word_audio=False,
    )
    
    db.log_assist(st.session_state.session_id, assist_result)
    
    st.markdown("### Multisensory Support")
    
    # Audio playback
    st.markdown("#### Text-to-Speech")
    st.audio(assist_result["audio_path"])
    
    # Visual rendering
    st.markdown("#### Visual Text Display")
    layout = assist_result["layout"]
    tokens = assist_result["tokens"]
    
    styled_html = f"""
    <div style='
        background:{layout["background_color"]};
        color:{layout["text_color"]};
        font-size:{layout["font_size"]}px;
        line-height:{layout["line_height"]};
        font-family:{layout["font_family"]}, {layout["fallback_font"]};
        padding:30px;
        border-radius:10px;
        text-align:center;
        letter-spacing:{layout["letter_spacing_em"]}em;
        word-spacing:{layout.get("word_spacing_em", 0.16)}em;
    '>
    """
    
    for t in tokens:
        word = t["word"]
        open_tags = ""
        close_tags = ""

        if t.get("highlight"):
            color = t.get("highlight_color") or layout["highlight_color"]
            open_tags += f"<span style='background:{color};padding:4px 6px;border-radius:5px;'>"
            close_tags = "</span>" + close_tags

        if t.get("underline_error"):
            color = t.get("underline_color") or layout["error_color"]
            open_tags += f"<u style='text-decoration-color:{color};text-decoration-thickness:3px;'>"
            close_tags = "</u>" + close_tags

        styled_html += open_tags + word + close_tags + " "

    styled_html += "</div>"
    st.markdown(styled_html, unsafe_allow_html=True)
    
    # Hints
    if assist_result["hints"]:
        st.markdown("####  Reading Hints")
        for h in assist_result["hints"]:
            if h.get("word"):
                st.info(f"**{h['word']}** → {h['hint']}")
            elif h.get("general"):
                st.success(h["hint"])
    
    progress_bar.progress(90)
    
    #  MENTOR FEEDBACK 
    status_text.text("Generating personalized feedback...")
    
    mentor_result = run_mentor_module(
        listen_result=listen_result,
        observe_result=observe_result,
        adapt_result=adapt_result,
        session_history=st.session_state.session_history,
        use_llm=st.session_state.get("use_llm_mentor", False),
        persona_name=st.session_state.get("mentor_persona", "aarav"),
        previous_difficulty=st.session_state.previous_difficulty,
    )
    
    db.log_mentor(st.session_state.session_id, mentor_result)
    
    # Add to history
    st.session_state.session_history.append({
        "wer": listen_result["wer"],
        "sentence_confidence": listen_result["sentence_confidence"],
        "focus_score": observe_result.get("focus_score", 0),
    })
    
    progress_bar.progress(100)
    status_text.text(" Analysis complete!")
    
    
    st.markdown("### Mentor Feedback")
    
    # Main message
    encouragement_level = mentor_result.get("encouragement_level", "medium")
    if encouragement_level == "high":
        st.success(f" {mentor_result['realtime_message']}")
    elif encouragement_level == "supportive":
        st.info(f" {mentor_result['realtime_message']}")
    else:
        st.info(f" {mentor_result['realtime_message']}")
    
    # Difficulty feedback
    st.info(f" {mentor_result['difficulty_feedback']}")
    
    # Error-specific coaching
    if mentor_result["error_messages"]:
        with st.expander(" Personalized Tips", expanded=True):
            for msg in mentor_result["error_messages"]:
                st.warning(msg)
    
    # System ready for next turn
    st.session_state.system_ready = True



if show_history and st.session_state.user_id:
    st.markdown("---")
    st.markdown("##  Learning Progress")
    
    progress = db.get_user_progress(st.session_state.user_id, limit=10)
    
    if not progress:
        st.info(" No past sessions yet. Complete at least one session to see your progress!")
    else:
        for idx, p in enumerate(progress):
            with st.expander(f" Session {idx + 1} - {p['date']}", expanded=idx == 0):
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    st.metric("Accuracy", f"{p['accuracy_percentage']:.1f}%")
                with col_p2:
                    st.metric("Avg Focus", f"{p['avg_focus_score']:.1f}/100")
                with col_p3:
                    st.metric("Reading Time", f"{p['total_reading_time_min']:.1f} min")
                with col_p4:
                    st.metric("Words Read", p['words_read'])


st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #7f8c8d;'>
    <p> Built with for dyslexic learners | Powered by AI & Multisensory Learning</p>
    <p style='font-size: 12px;'>Session data is stored locally and used to improve your reading experience</p>
</div>
""", unsafe_allow_html=True)

# Cleanup old audio files
cleanup_old_audio_files(max_age_seconds=3600)
