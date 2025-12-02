
import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager


DB_DIR = "data/db"
DB_PATH = os.path.join(DB_DIR, "dyslexia_ai.db")

os.makedirs(DB_DIR, exist_ok=True)



@contextmanager
def get_db_connection():
    """Thread-safe, lock-resistant SQLite connection."""
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,              
        check_same_thread=False  
    )
    conn.row_factory = sqlite3.Row

    
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



def init_database():
    """Initialize all database tables with enhanced schema."""
    with get_db_connection() as conn:
        cur = conn.cursor()

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            grade_level TEXT,
            preferences_json TEXT,
            created_at TEXT NOT NULL,
            last_active TEXT
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds INTEGER,
            total_words_attempted INTEGER DEFAULT 0,
            total_errors INTEGER DEFAULT 0,
            avg_confidence REAL,
            session_notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS listen_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            spoken_text TEXT,
            expected_text TEXT,
            wer REAL,
            sentence_confidence REAL,
            insertions INTEGER DEFAULT 0,
            deletions INTEGER DEFAULT 0,
            substitutions INTEGER DEFAULT 0,
            word_results_json TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS observe_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            focus_score REAL,
            fatigue_level TEXT,
            blink_rate REAL,
            gaze_stability REAL,
            head_movement REAL,
            attention_span_seconds REAL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS adapt_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            difficulty TEXT,
            font_size INTEGER,
            sentence_length INTEGER,
            tts_speed REAL,
            hint_level TEXT,
            engine TEXT DEFAULT 'rule_based',
            reason TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS assist_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            audio_path TEXT,
            tokens_json TEXT,
            layout_json TEXT,
            assist_type TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mentor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            realtime_message TEXT,
            error_messages_json TEXT,
            difficulty_feedback TEXT,
            encouragement_level TEXT,
            engine TEXT DEFAULT 'rule_based',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """)

        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS performance_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER,
            date TEXT NOT NULL,
            avg_wer REAL,
            avg_confidence REAL,
            avg_focus_score REAL,
            total_reading_time_min REAL,
            words_read INTEGER,
            accuracy_percentage REAL,
            improvement_notes TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)

        
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_listen_session ON listen_logs(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_observe_session ON observe_logs(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_performance_user ON performance_summary(user_id)")



def create_user(
    name: str,
    age: Optional[int] = None,
    grade_level: Optional[str] = None,
    preferences: Optional[Dict] = None
) -> int:
    """Create a new user and return user_id."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO users (name, age, grade_level, preferences_json, created_at, last_active)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name,
            age,
            grade_level,
            json.dumps(preferences) if preferences else "{}",
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        return cur.lastrowid


def get_user(user_id: int) -> Optional[Dict]:
    """Retrieve user information as a dict."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def update_user_last_active(user_id: int) -> None:
    """Update last active timestamp (standalone use)."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        UPDATE users SET last_active = ? WHERE id = ?
        """, (datetime.now().isoformat(), user_id))



def start_new_session(user_id: Optional[int] = None) -> int:
    """
    Start a new reading session.
    IMPORTANT: Updates user.last_active using the SAME connection
    to avoid nested write locks.
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        start_time = datetime.now().isoformat()

        cur.execute("""
        INSERT INTO sessions (user_id, start_time)
        VALUES (?, ?)
        """, (user_id, start_time))

        session_id = cur.lastrowid

        if user_id is not None:
            cur.execute("""
            UPDATE users SET last_active = ?
            WHERE id = ?
            """, (start_time, user_id))

        return session_id


def end_session(session_id: int, session_notes: Optional[str] = None) -> None:
    """End a session, compute statistics & log performance summary."""
    with get_db_connection() as conn:
        cur = conn.cursor()

        # Get session start & user
        cur.execute("SELECT start_time, user_id FROM sessions WHERE id = ?", (session_id,))
        result = cur.fetchone()
        if not result:
            return

        start_time = datetime.fromisoformat(result["start_time"])
        user_id = result["user_id"]
        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds())

        # Compute stats from listen_logs
        cur.execute("""
        SELECT
            AVG(wer)                AS avg_wer,
            AVG(sentence_confidence) AS avg_conf,
            COUNT(*)                AS word_count,
            SUM(insertions + deletions + COALESCE(substitutions, 0)) AS total_errors
        FROM listen_logs
        WHERE session_id = ?
        """, (session_id,))
        stats = cur.fetchone()

        avg_wer = stats["avg_wer"] if stats and stats["avg_wer"] is not None else 0.0
        avg_conf = stats["avg_conf"] if stats and stats["avg_conf"] is not None else 0.0
        word_count = stats["word_count"] if stats and stats["word_count"] is not None else 0
        total_errors = stats["total_errors"] if stats and stats["total_errors"] is not None else 0

        # Update session row
        cur.execute("""
        UPDATE sessions
        SET end_time = ?, duration_seconds = ?, total_words_attempted = ?,
            total_errors = ?, avg_confidence = ?, session_notes = ?
        WHERE id = ?
        """, (
            end_time.isoformat(),
            duration,
            word_count,
            total_errors,
            avg_conf,
            session_notes,
            session_id
        ))

        # Compute avg focus from observe_logs
        cur.execute("""
        SELECT AVG(focus_score) AS avg_focus
        FROM observe_logs
        WHERE session_id = ?
        """, (session_id,))
        focus_row = cur.fetchone()
        avg_focus = focus_row["avg_focus"] if focus_row and focus_row["avg_focus"] is not None else 0.0

        # Compute accuracy
        accuracy = 100.0 - (avg_wer * 100.0) if avg_wer is not None else 0.0

        # Insert into performance_summary if there was any reading
        if word_count > 0:
            cur.execute("""
            INSERT INTO performance_summary (
                session_id, user_id, date,
                avg_wer, avg_confidence, avg_focus_score,
                total_reading_time_min, words_read,
                accuracy_percentage, improvement_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                end_time.date().isoformat(),
                avg_wer,
                avg_conf,
                avg_focus,
                duration / 60.0,
                word_count,
                accuracy,
                None  # improvement_notes can be filled later
            ))



def log_listen(session_id: int, listen_result: Dict[str, Any]) -> None:
    """Log speech recognition results."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO listen_logs (
            session_id, spoken_text, expected_text, wer,
            sentence_confidence, insertions, deletions, substitutions,
            word_results_json, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            listen_result.get("spoken_text", ""),
            listen_result.get("expected_text", ""),
            listen_result.get("wer", 0.0),
            listen_result.get("sentence_confidence", 0.0),
            listen_result.get("insertions", 0),
            listen_result.get("deletions", 0),
            listen_result.get("substitutions", 0),
            json.dumps(listen_result.get("word_results", [])),
            datetime.now().isoformat()
        ))


def log_observe(session_id: int, observe_result: Dict[str, Any]) -> None:
    """Log attention and engagement metrics."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO observe_logs (
            session_id, focus_score, fatigue_level,
            blink_rate, gaze_stability, head_movement,
            attention_span_seconds, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            observe_result.get("focus_score", 0.0),
            observe_result.get("fatigue_level", "unknown"),
            observe_result.get("blink_rate_per_min", 0.0),
            observe_result.get("gaze_stability", 0.0),
            observe_result.get("head_movement", 0.0),
            observe_result.get("attention_span_seconds", 0.0),
            datetime.now().isoformat()
        ))


def log_adapt(session_id: int, adapt_result: Dict[str, Any]) -> None:
    """Log adaptive difficulty changes."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO adapt_logs (
            session_id, difficulty, font_size,
            sentence_length, tts_speed, hint_level,
            engine, reason, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            adapt_result.get("difficulty", "medium"),
            adapt_result.get("font_size", 18),
            adapt_result.get("sentence_length", 10),
            adapt_result.get("tts_speed", 1.0),
            adapt_result.get("hint_level", "medium"),
            adapt_result.get("engine", "rule_based"),
            adapt_result.get("reason", ""),
            datetime.now().isoformat()
        ))


def log_assist(session_id: int, assist_result: Dict[str, Any]) -> None:
    """Log multisensory assistance provided."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO assist_logs (
            session_id, audio_path, tokens_json, layout_json,
            assist_type, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            assist_result.get("audio_path", ""),
            json.dumps(assist_result.get("tokens", [])),
            json.dumps(assist_result.get("layout", {})),
            assist_result.get("assist_type", "general"),
            datetime.now().isoformat()
        ))


def log_mentor(session_id: int, mentor_result: Dict[str, Any]) -> None:
    """Log AI mentor feedback."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO mentor_logs (
            session_id, realtime_message, error_messages_json,
            difficulty_feedback, encouragement_level, engine, timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            mentor_result.get("realtime_message", ""),
            json.dumps(mentor_result.get("error_messages", [])),
            mentor_result.get("difficulty_feedback", ""),
            mentor_result.get("encouragement_level", "medium"),
            mentor_result.get("engine", "rule_based"),
            datetime.now().isoformat()
        ))



def get_session_overview(session_id: int) -> Dict[str, Any]:
    """Get comprehensive session overview."""
    with get_db_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cur.fetchone()
        if not session:
            return {}

        cur.execute("""
        SELECT COUNT(*) AS count
        FROM listen_logs
        WHERE session_id = ?
        """, (session_id,))
        listen_count = cur.fetchone()["count"]

        cur.execute("""
        SELECT AVG(focus_score) AS avg_focus
        FROM observe_logs
        WHERE session_id = ?
        """, (session_id,))
        avg_focus_row = cur.fetchone()
        avg_focus = avg_focus_row["avg_focus"] if avg_focus_row and avg_focus_row["avg_focus"] is not None else 0.0

        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
            "duration_seconds": session["duration_seconds"],
            "total_turns": listen_count,
            "total_words_attempted": session["total_words_attempted"],
            "total_errors": session["total_errors"],
            "avg_confidence": session["avg_confidence"],
            "avg_focus_score": avg_focus,
            "session_notes": session["session_notes"],
        }


def get_user_progress(user_id: int, limit: int = 10) -> List[Dict]:
    """Get user's recent performance summary."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT *
        FROM performance_summary
        WHERE user_id = ?
        ORDER BY date DESC
        LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cur.fetchall()]


def get_all_listen_logs(session_id: Optional[int] = None) -> List[sqlite3.Row]:
    """Get listen logs, optionally filtered by session."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        if session_id is not None:
            cur.execute("""
            SELECT wer, sentence_confidence, spoken_text, timestamp
            FROM listen_logs
            WHERE session_id = ?
            """, (session_id,))
        else:
            cur.execute("""
            SELECT wer, sentence_confidence
            FROM listen_logs
            """)
        return cur.fetchall()


def get_recent_sessions(user_id: Optional[int] = None, limit: int = 5) -> List[Dict]:
    """Get recent sessions for a user or all users."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        if user_id is not None:
            cur.execute("""
            SELECT *
            FROM sessions
            WHERE user_id = ?
            ORDER BY start_time DESC
            LIMIT ?
            """, (user_id, limit))
        else:
            cur.execute("""
            SELECT *
            FROM sessions
            ORDER BY start_time DESC
            LIMIT ?
            """, (limit,))
        return [dict(row) for row in cur.fetchall()]



def get_error_patterns(session_id: int) -> Dict[str, Any]:
    """Analyze common error patterns in a session."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT word_results_json
        FROM listen_logs
        WHERE session_id = ?
        """, (session_id,))

        all_errors: List[Dict[str, Any]] = []
        for row in cur.fetchall():
            word_results = json.loads(row["word_results_json"] or "[]")
            for word in word_results:
                if word.get("error_type"):
                    all_errors.append(word)

        return {
            "total_errors": len(all_errors),
            "error_details": all_errors[:20],  
        }



if __name__ == "__main__":
    print("Initializing database...")
    init_database()

    print("\nCreating test user...")
    user_id = create_user(name="Test Student", age=10, grade_level="Grade 5")
    print(f"Created user ID: {user_id}")

    print("\nStarting new session...")
    sid = start_new_session(user_id=user_id)
    print(f"Session ID: {sid}")

    # Fake test data
    fake_listen = {
        "spoken_text": "the kat sat on the mat",
        "expected_text": "the cat sat on the mat",
        "wer": 0.20,
        "sentence_confidence": 0.75,
        "insertions": 0,
        "deletions": 0,
        "substitutions": 1,
        "word_results": [{"word": "kat", "error_type": "substitution"}],
    }

    fake_observe = {
        "focus_score": 72.5,
        "fatigue_level": "low",
        "blink_rate_per_min": 18.5,
        "gaze_stability": 0.92,
        "head_movement": 0.04,
        "attention_span_seconds": 120,
    }

    fake_adapt = {
        "difficulty": "level_2",
        "font_size": 22,
        "sentence_length": 8,
        "tts_speed": 0.95,
        "hint_level": "medium",
        "engine": "rule_based",
        "reason": "Good performance, maintaining level",
    }

    fake_assist = {
        "audio_path": "data/tts/sample.mp3",
        "tokens": [{"word": "cat", "syllables": ["cat"]}],
        "layout": {"spacing": 1.5},
        "assist_type": "tts",
    }

    fake_mentor = {
        "realtime_message": "Great job! You're doing well with 'cat' sounds.",
        "error_messages": ["Remember: 'cat' has a hard 'c' sound"],
        "difficulty_feedback": "You're ready for level 2!",
        "encouragement_level": "high",
        "engine": "rule_based",
    }

    print("\nLogging module data...")
    log_listen(sid, fake_listen)
    log_observe(sid, fake_observe)
    log_adapt(sid, fake_adapt)
    log_assist(sid, fake_assist)
    log_mentor(sid, fake_mentor)

    print("\nEnding session...")
    end_session(sid, session_notes="First test session completed successfully")

    print("\n📊 Session Overview:")
    overview = get_session_overview(sid)
    for key, value in overview.items():
        print(f"  {key}: {value}")

    print("\n📈 User Progress:")
    progress = get_user_progress(user_id)
    for session in progress:
        print(f"  Date: {session['date']}, Accuracy: {session['accuracy_percentage']:.1f}%")

    print("\nDatabase test completed successfully!")
