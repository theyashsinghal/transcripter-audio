# app.py — COMPLETE BATCH TRANSCRIBER (ALL ROWS VERSION)
# -----------------------------------------------------------------------------
# FEATURES INCLUDED:
# 1. Multi-file Excel Upload (Merges multiple files).
# 2. Robust Gemini API Integration (Resumable Uploads, Polling, Retries).
# 3. PROCESSES EVERY ROW (No unique mobile number filtering).
# 4. Empty Response Retry (Retries Gemini API if it returns empty text).
# 5. High Concurrency (Slider up to 128 workers).
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import mimetypes
import tempfile
import random
import math 
import html
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
# NOTE: Using "gemini-2.5-flash" for speed and stability.
MODEL_NAME = "gemini-2.5-flash" 

# Streaming download chunk size (8KB)
DOWNLOAD_CHUNK_SIZE = 8192

# Configure logging to console
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (CSS) ---
BASE_CSS = """
<style>
/* Card look for transcript entries */
.call-card {
    border: 1px solid var(--border-color, #e6e6e6);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--card-bg, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* Transcript scroll area */
.transcript-box {
    max-height: 320px;
    overflow: auto;
    padding: 8px;
    border-radius: 6px;
    background: var(--transcript-bg, #fafafa);
    border: 1px solid var(--border-color, #eee);
    font-family: monospace; /* Monospace helps alignment */
    white-space: pre-wrap;  /* Preserves newlines */
}

/* Speaker colors for Diarization */
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }

/* Compact metadata row */
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

/* Theming variables */
.dark-theme {
    --card-bg: #0f1115;
    --transcript-bg: #0b0c0f;
    --border-color: #222428;
    --meta-color: #9aa0a6;
    color: #e6eef3;
}
.light-theme {
    --card-bg: #ffffff;
    --transcript-bg: #fafafa;
    --border-color: #e6e6e6;
    --meta-color: #666666;
    color: #111;
}

/* Search box styling */
.search-box { margin-bottom: 10px; padding: 6px; border-radius: 6px; border: 1px solid var(--border-color, #eee); width:100%; }
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- NETWORK UTILITIES ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    """
    Sleeps for a random amount of time to prevent thundering herd problems.
    Formula: base * (2^attempt) * jitter
    """
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    """
    Robust wrapper for requests with exponential backoff + jitter.
    Handles 429 (Rate Limit) and 5xx (Server Errors).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            # Treat 429 and 5xx as transient errors
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s from %s (attempt %d). Retrying...", resp.status_code, url, attempt + 1)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException on %s %s: %s (attempt %d)", method, url, str(e), attempt + 1)
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
     
    # All retries exhausted
    if last_exc:
        raise last_exc
    raise Exception("make_request_with_retry: retries exhausted without a response")


# --- MIME TYPE & FILE EXTENSION HANDLING ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wave",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    """
    Determine extension and mime type from URL path and header.
    Prioritize path extension, else header, else default to MP3.
    """
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    
    # 1. Trust extension if it is a known audio type
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    
    # 2. Try header Content-Type
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        # Reverse lookup map
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype:
                return k, ctype
        
        # Attempt standard python guess
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext:
            return guessed_ext.lower(), ctype
            
    # 3. Last fallback: Default to MP3
    # Google File API is strict; MP3 container usually handles varied bitstreams well.
    return ".mp3", "audio/mpeg"


# --- GOOGLE UPLOAD PIPELINE ---

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
    """
    Starts a resumable upload session with the Google File API.
    Returns the unique upload URL.
    """
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
    }
    payload = json.dumps({"file": {"display_name": display_name}})
    
    resp = make_request_with_retry("POST", url, headers=headers, data=payload)
    
    if resp.status_code not in (200, 201):
        logger.error("initiate_upload failed: %s %s", resp.status_code, resp.text)
        raise Exception(f"Init failed ({resp.status_code}): {resp.text}")
    
    upload_url = resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise Exception("Failed to get upload URL from Google.")
    return upload_url

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Streams file bytes to the upload URL and finalizes the file.
    """
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream",
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }

    # Try POST first (standard streaming)
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)

    # Fallback: some endpoints expect PUT for finalize
    if resp.status_code == 400:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)

    if resp.status_code not in (200, 201):
        logger.error("Upload failed: %s %s", resp.status_code, resp.text)
        raise Exception(f"UPLOAD FAILED {resp.status_code}: {resp.text}")

    try:
        j = resp.json()
    except ValueError:
        raise Exception("Upload finished but server returned non-JSON response.")
    
    # Return the file metadata
    return j.get("file", j)


# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 300) -> bool:
    """
    Polls the file status endpoint until state is ACTIVE or FAILED.
    """
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code != 200:
            logger.warning("Status poll: got %s. Retrying...", resp.status_code)
            time.sleep(2)
        else:
            j = resp.json()
            state = j.get("state")
            logger.debug("Polled file %s state=%s", file_name, state)
            
            if state == "ACTIVE":
                return True
            
            if state == "FAILED":
                raise Exception(f"File processing failed: {j.get('processingError', j)}")
            
            # If PROCESSING, wait and loop
            time.sleep(2)

        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")

def delete_file(api_key: str, file_name: str):
    """
    Deletes the file from Google servers to clean up storage.
    """
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except Exception as e:
        logger.warning("delete_file failed for %s: %s", file_name, str(e))


# --- TRANSCRIPTION API CALL ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    """
    Calls Gemini v1beta generateContent to transcribe the audio.
    INCLUDES RETRY LOGIC for empty responses.
    """
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }],
        "safetySettings": safety_settings,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }

    # RETRY LOOP FOR CONTENT
    # Sometimes API returns 200 but empty content. We retry up to 3 times.
    max_content_retries = 3
    
    for attempt in range(max_content_retries):
        resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
        
        if resp.status_code != 200:
            logger.error("Transcription API returned %s: %s", resp.status_code, resp.text)
            return f"API ERROR {resp.status_code}: {resp.text}"

        try:
            body = resp.json()
        except ValueError:
            return "PARSE ERROR: Non-JSON response from transcription API."

        # Check for block reasons first
        prompt_feedback = body.get("promptFeedback", {})
        if prompt_feedback and prompt_feedback.get("blockReason"):
            return f"BLOCKED: {prompt_feedback.get('blockReason')}"

        # Check for valid candidates
        candidates = body.get("candidates") or []
        if candidates:
            first = candidates[0]
            content = first.get("content", {})
            parts = content.get("parts", [])
            if parts:
                text = parts[0].get("text") or parts[0].get("content") or ""
                if text.strip():
                    return text # Return valid text immediately
        
        # If we reached here, response was 200 OK but had no content.
        logger.warning(f"GenerateContent attempt {attempt+1}/{max_content_retries} empty. Retrying...")
        time.sleep(2 * (attempt + 1)) # Backoff

    return "NO TRANSCRIPT (Empty Response after retries)"

def build_prompt(language_label: str) -> str:
    """
    Constructs the system prompt for strict diarization and formatting.
    """
    return f"""
Transcribe this call in {language_label} exactly as spoken.

CRITICAL REQUIREMENTS — FOLLOW STRICTLY:
1. EVERY line MUST start with exactly one of these labels:
   - Speaker 1:
   - Speaker 2:
2. NEVER merge dialogue from two speakers in one line.
3. If you are unsure who is speaking, GUESS — but DO NOT leave the speaker label blank.
4. If the call sounds like a single-person monologue, STILL label every line as:
   Speaker 1: <text>
5. Do NOT summarize or improve the language. Write EXACTLY what was said.
6. Maintain natural turn-taking and break lines whenever the speaker changes.

TIMESTAMP RULES:
- Add timestamps at the start of EVERY line.
- Format MUST be: [0ms-2500ms]
- Use raw milliseconds only.
- No mm:ss format allowed.

LANGUAGE RULES:
- ALL Hindi words must be written in Hinglish (Latin script).
- NO Devanagari characters anywhere.
- English words should remain English.

STRICT FORMAT (DO NOT IGNORE):
- [timestamp] Speaker X: line of dialogue
- Only one speaker per line.
- Only one utterance per line.
- If two people speak at the same time, split into two separate lines with separate timestamps.

AUTO-CORRECTION:
- If any line is missing the speaker label, FIX IT and assign Speaker 1 or Speaker 2 based on your best guess.
- Do NOT output any unlabeled lines.

Return ONLY the transcript. No explanation.
"""


# --- DATA PREPARATION LOGIC (MODIFIED: ALL ROWS) ---

def prepare_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the DataFrame for processing.
    
    LOGIC CHANGE:
    - Iterates through EVERY row.
    - If recording_url exists -> Mark for TRANSCRIBE.
    - If recording_url missing -> Mark for SKIP.
    """
    
    # Optional: Ensure date column is datetime just for good practice, 
    # though we aren't sorting by it anymore for filtering purposes.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    final_rows = []

    for index, row in df.iterrows():
        # Create a copy to avoid SettingWithCopy warnings
        row_data = row.copy()
        
        # Check URL validity
        url_val = row_data.get('recording_url')
        
        if pd.notna(url_val) and str(url_val).strip() != "":
            # Case A: Valid URL found -> Transcribe
            row_data['processing_action'] = 'TRANSCRIBE'
            row_data['status'] = 'Pending'
        else:
            # Case B: No URL -> Skip this specific row
            row_data['processing_action'] = 'SKIP'
            row_data['transcript'] = "⚠️ Skipped: No recording URL provided in this row."
            row_data['status'] = '⚠️ Skipped'
            row_data['error'] = 'Missing recording_url'
        
        final_rows.append(row_data)
    
    return pd.DataFrame(final_rows).reset_index(drop=True)


# --- WORKER / PROCESSING FUNCTION ---

def process_single_row(index: int, row: pd.Series, api_key: str, prompt_template: str, keep_remote: bool = False) -> Dict[str, Any]:
    """
    The worker function executed by ThreadPoolExecutor.
    Handles logic based on 'processing_action' flag.
    """
    mobile = str(row.get("mobile_number", "Unknown"))
    
    result = {
        "index": index,
        "mobile_number": mobile,
        "recording_url": row.get("recording_url"),
        "transcript": row.get("transcript", ""), # Might be pre-filled with skip message
        "status": row.get("status", "Pending"),
        "error": row.get("error", None),
    }

    # Check the flag set by the preparation logic
    action = row.get("processing_action", "TRANSCRIBE")
    
    # If set to SKIP, we return the pre-calculated result immediately
    if action == "SKIP":
        return result

    # Validate URL (Double check)
    audio_url = row.get("recording_url")
    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "error": "Invalid URL"})
        return result

    tmp_path = None
    file_info = None

    try:
        parsed = urlparse(audio_url)

        # 1. Download file (streamed)
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Failed to download audio URL ({r.status_code})")

        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)

        # 2. Upload to Google (Resumable)
        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}_{random.randint(100,999)}{ext}"

        upload_url = initiate_upload(api_key, unique_name, mime_type, file_size)
        file_info = upload_bytes(upload_url, tmp_path, mime_type)

        # 3. Wait for Processing
        wait_for_active(api_key, file_info["name"])

        # 4. Transcribe
        transcript = generate_transcript(api_key, file_info["uri"], mime_type, prompt_template)
        result["transcript"] = transcript
        
        # Check for API-level errors in the text response
        if "API ERROR" in transcript or "PARSE ERROR" in transcript or "BLOCKED" in transcript:
            result["status"] = "❌ Error"
        elif "NO TRANSCRIPT" in transcript:
            result["status"] = "❌ Empty" # Distinct status for persistent empty responses
        else:
            result["status"] = "✅ Success"

    except Exception as e:
        logger.exception("Processing failed for row %s: %s", index, str(e))
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        result["error"] = str(e)

    finally:
        # Cleanup local temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning("Failed to remove tmp file %s: %s", tmp_path, str(e))
        
        # Cleanup remote file on Google (unless debugging)
        if file_info and isinstance(file_info, dict) and file_info.get("name") and not keep_remote:
            try:
                delete_file(api_key, file_info["name"])
            except Exception:
                pass

    return result


# --- RESULT MERGING & DISPLAY UTILS ---

def merge_results_with_original(df_consolidated: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    """
    Merges the worker results back into the consolidated DataFrame.
    """
    # Sort results by index to match original DF order
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    
    # Define columns to update
    cols_to_update = ["transcript", "status", "error"]
    
    # Drop these columns from the base DF if they exist (to avoid _x / _y suffixes)
    df_base = df_consolidated.drop(columns=[c for c in cols_to_update if c in df_consolidated.columns])
    
    # Merge based on the preserved index
    merged = df_base.merge(results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
    
    # Clean up the index column
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    
    return merged

def colorize_transcript_html(text: str) -> str:
    """
    Format transcript text into color-coded HTML.
    """
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"

    lines = text.splitlines()
    html_output = ""
    
    for line in lines:
        clean = line.strip()
        if not clean: continue
        
        escaped_line = html.escape(clean)
        lc = clean.lower()
        
        if "speaker 1:" in lc:
            html_output += f"<div class='speaker1'>{escaped_line}</div>"
        elif "speaker 2:" in lc:
            html_output += f"<div class='speaker2'>{escaped_line}</div>"
        else:
            html_output += f"<div class='other-speech'>{escaped_line}</div>"
            
    return f"<div>{html_output}</div>"


# --- MAIN APPLICATION ENTRY POINT ---

def main():
    # Initialize Session State
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = []
    if "final_df" not in st.session_state:
        st.session_state.final_df = pd.DataFrame()

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        
        # Adjusted max_value to 128 as requested
        max_workers = st.slider("Concurrency (Threads)", min_value=1, max_value=128, value=4,
                                help="Higher = faster but may hit API rate limits.")
        
        keep_remote = st.checkbox("Keep audio on Google", value=False,
                                help="For debugging: prevents auto-deletion of files from Gemini.")
        
        st.divider()
        
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        lang_map = {
            "English (India)": "English (Indian accent)",
            "Hindi": "Hindi (Devanagari)",
            "Mixed (Hinglish)": "Mixed English and Hindi"
        }
        
        theme_choice = st.radio("Theme", options=["Light", "Dark"], index=0, horizontal=True)
        st.caption("Use Dark theme if you prefer low-light UI.")

    # Apply Theme Class
    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)


    # --- FILE UPLOADER (MULTI-FILE) ---
    st.write("### 📂 Upload Call Data")
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

    # Status Containers
    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()

    start_button = st.button("🚀 Start Batch Processing (All Rows)", type="primary")


    # --- PROCESSING LOGIC ---
    if start_button:
        # Validation
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
            st.stop()
        
        if not uploaded_files:
            st.error("Please upload at least one Excel file.")
            st.stop()

        # 1. READ AND CONCATENATE FILES
        all_dfs = []
        for file in uploaded_files:
            try:
                df_single = pd.read_excel(file)
                all_dfs.append(df_single)
            except Exception as e:
                st.warning(f"Skipping file {file.name}: Error reading file: {e}")
                continue

        if not all_dfs:
            st.error("No valid Excel files could be read.")
            st.stop()

        # Combine into one Master DataFrame
        # ignore_index=True ensures a clean 0..N index
        raw_df = pd.concat(all_dfs, ignore_index=True)

        # Check for required columns
        required_cols = ["recording_url", "mobile_number"] 
        missing_cols = [c for c in required_cols if c not in raw_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in dataset: {', '.join(missing_cols)}")
            st.stop()

        # 2. PREPARE DATA (ALL ROWS)
        status_text.info("Preparing data for processing (Checking every row)...")
        
        # This function marks every row for transcription or skipping based on URL presence
        df_processed_ready = prepare_all_rows(raw_df)
        
        # Prepare for Processing
        prompt_template = build_prompt(lang_map[language_mode])
        total_rows = len(df_processed_ready)
        
        status_text.info(f"Processing {total_rows} items with {max_workers} threads...")
        progress_bar.progress(0.0)

        # Reset session state for new run
        processed_results = []
        st.session_state.processed_results = []

        # 3. THREAD POOL EXECUTION
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(process_single_row, idx, row, api_key, prompt_template, keep_remote): idx
                for idx, row in df_processed_ready.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                processed_results.append(res)
                completed += 1
                
                # Update UI
                progress_bar.progress(completed / total_rows)
                status_text.markdown(f"Processed **{completed}/{total_rows}** items.")
                
                # Live Preview (Last 5 items)
                recent_results = sorted(processed_results, key=lambda r: r["index"])[-5:]
                preview_df = pd.DataFrame(recent_results)
                if not preview_df.empty:
                    result_placeholder.dataframe(
                        preview_df[["mobile_number", "status", "transcript"]], 
                        width=800,
                        hide_index=True
                    )

        # 4. FINAL MERGE
        final_df = merge_results_with_original(df_processed_ready, processed_results)
        st.session_state.final_df = final_df
        
        status_text.success("Batch Processing Complete!")


    # --- RESULTS VIEWER ---
    final_df = st.session_state.final_df

    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## 🎛️ Transcript Browser")

        # Filters Layout
        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a:
            search_q = st.text_input("Search", placeholder="Search transcript, phone, or URL...")
        with col_b:
            status_sel = st.selectbox("Status", ["All", "Success", "Failed", "Skipped"])
        with col_c:
            speaker_sel = st.selectbox("Speaker", ["All", "Speaker 1", "Speaker 2"])
        with col_d:
            per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1)

        # Apply Filters
        view_df = final_df.copy()
        
        if status_sel != "All":
            if status_sel == "Success":
                view_df = view_df[view_df["status"].str.contains("Success", case=False, na=False)]
            elif status_sel == "Failed":
                view_df = view_df[view_df["status"].str.contains("Failed|Error|Empty", case=False, na=False)]
            elif status_sel == "Skipped":
                view_df = view_df[view_df["status"].str.contains("Skipped", case=False, na=False)]

        if search_q.strip():
            q = search_q.lower()
            mask = (
                view_df["transcript"].fillna("").str.lower().str.contains(q) |
                view_df["mobile_number"].astype(str).str.lower().str.contains(q) |
                view_df["recording_url"].astype(str).str.lower().str.contains(q)
            )
            view_df = view_df[mask]

        if speaker_sel != "All":
            key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
            mask = view_df["transcript"].fillna("").str.lower().str.contains(key)
            view_df = view_df[mask]

        total_items = len(view_df)
        st.markdown(f"**Showing {total_items} result(s)**")

        # Pagination Logic
        pages = max(1, math.ceil(total_items / per_page))
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page_idx - 1) * per_page
        end = start + per_page
        page_df = view_df.iloc[start:end]

        # Excel Download Button
        out_buf = BytesIO()
        view_df.to_excel(out_buf, index=False)
        st.download_button(
            "📥 Download Filtered Results",
            data=out_buf.getvalue(),
            file_name=f"transcripts_export_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Render Individual Cards
        for idx, row in page_df.iterrows():
            mobile_display = row.get('mobile_number', 'Unknown')
            status_display = row.get('status', '')
            header = f"{mobile_display} — {status_display}"
            
            with st.expander(header, expanded=False):
                # Metadata Row
                url_val = html.escape(str(row.get('recording_url', 'None')))
                meta_html = f"<div class='meta-row'><b>URL:</b> {url_val}</div>"
                st.markdown(meta_html, unsafe_allow_html=True)
                
                # Transcript Box
                transcript_text = row.get("transcript", "")
                transcript_html = colorize_transcript_html(transcript_text)
                st.markdown(f"<div class='transcript-box'>{transcript_html}</div>", unsafe_allow_html=True)
                
                # Error display if present
                if row.get("error"):
                    st.error(f"Error: {row.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
