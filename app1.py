# app.py — COMPLETE BATCH TRANSCRIBER
# Features: Multi-file support, Unique Mobile Logic, Latest Audio Priority, Outcome Fallback.

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

# --- CONFIG ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.5-flash" 

# streaming download chunk size
DOWNLOAD_CHUNK_SIZE = 8192

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("transcriber")

# --- UI STYLING (Full CSS) ---
BASE_CSS = """
<style>
/* Card look */
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
    font-family: monospace; 
    white-space: pre-wrap;  
}

/* Speaker colors */
.speaker1 { color: #1f77b4; font-weight: 600; display: block; margin-bottom: 4px; }
.speaker2 { color: #d62728; font-weight: 600; display: block; margin-bottom: 4px; }
.other-speech { color: #333; display: block; margin-bottom: 4px; }

/* compact meta row */
.meta-row { font-size: 13px; color: var(--meta-color, #666); margin-bottom: 8px; }

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

/* small search box */
.search-box { margin-bottom: 10px; padding: 6px; border-radius: 6px; border: 1px solid var(--border-color, #eee); width:100%; }
</style>
"""

st.set_page_config(page_title="Batch Transcriber", layout="wide")
st.markdown(BASE_CSS, unsafe_allow_html=True)


# --- UTILITIES & NETWORK HELPERS ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 30)
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=60, **kwargs)
            # Treat 429 and 5xx as transient
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning("Transient HTTP %s from %s (attempt %d). Retrying...", resp.status_code, url, attempt + 1)
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning("RequestException on %s %s: %s (attempt %d)", method, url, str(e), attempt + 1)
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    if last_exc:
        raise last_exc
    raise Exception("make_request_with_retry: retries exhausted without a response")

# --- MIME & EXTENSION HANDLING ---

COMMON_AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wave", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac"
}

def detect_extension_and_mime(url_path: str, header_content_type: Optional[str]) -> (str, str):
    _, ext = os.path.splitext(url_path or "")
    ext = ext.lower()
    
    # 1. Trust extension if it is a known audio type
    if ext and ext in COMMON_AUDIO_MIME:
        return ext, COMMON_AUDIO_MIME[ext]
    
    # 2. Try header
    if header_content_type:
        ctype = header_content_type.split(";")[0].strip()
        for k, v in COMMON_AUDIO_MIME.items():
            if v == ctype:
                return k, ctype
        guessed_ext = mimetypes.guess_extension(ctype)
        if guessed_ext:
            return guessed_ext.lower(), ctype
            
    # 3. Last fallback: Default to MP3
    return ".mp3", "audio/mpeg"

# --- UPLOAD PIPELINE ---

def initiate_upload(api_key: str, display_name: str, mime_type: str, file_size: int) -> str:
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
        raise Exception(f"Init failed ({resp.status_code}): {resp.text}")
    
    upload_url = resp.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise Exception("Failed to get upload URL from Google.")
    return upload_url

def upload_bytes(upload_url: str, file_path: str, mime_type: str) -> Dict[str, Any]:
    file_size = os.path.getsize(file_path)
    headers = {
        "Content-Type": mime_type or "application/octet-stream",
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    # Try POST first (streaming)
    with open(file_path, "rb") as f:
        resp = requests.post(upload_url, headers=headers, data=f, timeout=300)

    # Fallback: some endpoints expect PUT finalize
    if resp.status_code == 400:
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, headers=headers, data=f, timeout=300)

    if resp.status_code not in (200, 201):
        raise Exception(f"UPLOAD FAILED {resp.status_code}: {resp.text}")

    try:
        j = resp.json()
    except ValueError:
        raise Exception("Upload finished but server returned non-JSON response.")
    return j.get("file", j)

# --- GOOGLE FILE STATUS POLLING ---

def wait_for_active(api_key: str, file_name: str, timeout_seconds: int = 300) -> bool:
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    start = time.time()
    while True:
        resp = make_request_with_retry("GET", url)
        if resp.status_code != 200:
            time.sleep(2)
        else:
            j = resp.json()
            state = j.get("state")
            if state == "ACTIVE":
                return True
            if state == "FAILED":
                raise Exception(f"File processing failed: {j.get('processingError', j)}")
            time.sleep(2)

        if time.time() - start > timeout_seconds:
            raise Exception("Timed out waiting for file to become ACTIVE.")

def delete_file(api_key: str, file_name: str):
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}", timeout=20)
    except Exception as e:
        logger.warning("delete_file failed for %s: %s", file_name, str(e))

# --- TRANSCRIPTION CALLS ---

def generate_transcript(api_key: str, file_uri: str, mime_type: str, prompt: str) -> str:
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
    payload = {
        "contents": [{
            "parts": [{"text": prompt}, {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}]
        }],
        "safetySettings": safety_settings,
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
    }
    resp = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    if resp.status_code != 200:
        return f"API ERROR {resp.status_code}: {resp.text}"

    try:
        body = resp.json()
    except ValueError:
        return "PARSE ERROR: Non-JSON response."

    candidates = body.get("candidates") or []
    if not candidates:
        return "NO TRANSCRIPT (Empty Response)"
    
    first_part = candidates[0].get("content", {}).get("parts", [])
    if not first_part:
        return "NO TRANSCRIPT (No parts)"
    
    return first_part[0].get("text", "")

def build_prompt(language_label: str) -> str:
    return f"""
Transcribe this call in {language_label} exactly as spoken.

CRITICAL REQUIREMENTS:
1. EVERY line MUST start with 'Speaker 1:' or 'Speaker 2:'.
2. NEVER merge dialogue from two speakers.
3. If unsure, GUESS the speaker.
4. If monologue, use 'Speaker 1:'.
5. Write EXACTLY what was said.

TIMESTAMP RULES:
- Format: [0ms-2500ms] at the start of EVERY line.
- Use raw milliseconds only.

LANGUAGE RULES:
- Hindi words must be in Hinglish (Latin script).
- NO Devanagari.

STRICT FORMAT:
- [timestamp] Speaker X: line of dialogue
"""

# --- NEW: DATA CONSOLIDATION LOGIC ---

def consolidate_data_by_mobile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by mobile_number to ensure only ONE transcript per contact.
    Prioritizes the LATEST date with a VALID recording_url.
    If no URL exists for the contact, keeps the latest row and marks for Outcome Fallback.
    """
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Sort by mobile, then date descending (latest first)
        df = df.sort_values(by=['mobile_number', 'date'], ascending=[True, False])
    
    unique_mobiles = df['mobile_number'].unique()
    final_rows = []

    for mobile in unique_mobiles:
        # Get all rows for this specific mobile number
        group = df[df['mobile_number'] == mobile]
        
        # Filter for rows that actually have a URL
        valid_recording_rows = group[group['recording_url'].notna() & (group['recording_url'] != "")]
        
        if not valid_recording_rows.empty:
            # Case A: We have at least one recording.
            # Since we sorted by date DESC, the first one is the LATEST valid recording.
            selected_row = valid_recording_rows.iloc[0].copy()
            selected_row['processing_action'] = 'TRANSCRIBE'
        else:
            # Case B: No recording URL found in any attempt for this number.
            # Pick the absolute latest attempt row.
            selected_row = group.iloc[0].copy()
            selected_row['processing_action'] = 'SKIP'
            outcome_val = selected_row.get('outcome', 'N/A')
            # Pre-fill transcript with outcome
            selected_row['transcript'] = f"Outcome: {outcome_val}"
            selected_row['status'] = '⚠️ Skipped (No Audio)'
        
        final_rows.append(selected_row)
    
    return pd.DataFrame(final_rows).reset_index(drop=True)

# --- WORKER / PROCESSING FUNCTION ---

def process_single_row(index: int, row: pd.Series, api_key: str, prompt_template: str, keep_remote: bool = False) -> Dict[str, Any]:
    mobile = str(row.get("mobile_number", "Unknown"))
    
    result = {
        "index": index,
        "mobile_number": mobile,
        "recording_url": row.get("recording_url"),
        "transcript": row.get("transcript", ""), # Use pre-filled if available
        "status": row.get("status", "Pending"),
        "error": None,
    }

    # CHECK ACTION FLAG
    action = row.get("processing_action", "TRANSCRIBE")
    
    # If logic decided to skip (missing URL fallback), return immediately
    if action == "SKIP":
        return result

    audio_url = row.get("recording_url")
    if not audio_url or not isinstance(audio_url, str):
        result.update({"status": "❌ Failed", "error": "Invalid URL"})
        return result

    tmp_path = None
    file_info = None

    try:
        parsed = urlparse(audio_url)

        # Download
        r = make_request_with_retry("GET", audio_url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Failed to download audio URL ({r.status_code})")

        header_ct = r.headers.get("content-type", "")
        ext, mime_type = detect_extension_and_mime(parsed.path, header_ct)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk: tmp.write(chunk)
            tmp_path = tmp.name

        cleaned_mobile = "".join(ch for ch in mobile if ch.isalnum())
        unique_name = f"rec_{cleaned_mobile}_{int(time.time())}{ext}"

        # Upload
        upload_url = initiate_upload(api_key, unique_name, mime_type, os.path.getsize(tmp_path))
        file_info = upload_bytes(upload_url, tmp_path, mime_type)
        wait_for_active(api_key, file_info["name"])

        # Transcribe
        transcript = generate_transcript(api_key, file_info["uri"], mime_type, prompt_template)
        result["transcript"] = transcript
        
        if transcript.startswith("API ERROR") or transcript.startswith("PARSE ERROR") or transcript.startswith("BLOCKED"):
            result["status"] = "❌ Error"
        else:
            result["status"] = "✅ Success"

    except Exception as e:
        logger.exception("Processing failed for row %s: %s", index, str(e))
        result["transcript"] = f"SYSTEM ERROR: {str(e)}"
        result["status"] = "❌ Failed"
        result["error"] = str(e)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        if file_info and not keep_remote:
            try: delete_file(api_key, file_info["name"])
            except: pass

    return result

# --- RESULT MERGE UTILITIES ---

def merge_results_with_original(df_consolidated: pd.DataFrame, processed_results: list) -> pd.DataFrame:
    # We are merging back into the CONSOLIDATED dataframe, not the raw one
    results_df = pd.DataFrame(sorted(processed_results, key=lambda r: r["index"]))
    
    # We only need to update the transcript, status, error columns
    cols_to_update = ["transcript", "status", "error"]
    
    # Drop these from original if they exist to avoid collision
    df_base = df_consolidated.drop(columns=[c for c in cols_to_update if c in df_consolidated.columns])
    
    # Merge using index (which was preserved in processing)
    merged = df_base.merge(results_df[["index"] + cols_to_update], left_index=True, right_on="index", how="left")
    
    if "index" in merged.columns:
        merged = merged.drop(columns=["index"])
    return merged

# --- HTML FORMATTING ---
def colorize_transcript_html(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "<div class='other-speech'>No transcript</div>"
    lines = text.splitlines()
    html_output = ""
    for line in lines:
        clean = line.strip()
        if not clean: continue
        escaped_line = html.escape(clean)
        lc = clean.lower()
        if "speaker 1:" in lc: html_output += f"<div class='speaker1'>{escaped_line}</div>"
        elif "speaker 2:" in lc: html_output += f"<div class='speaker2'>{escaped_line}</div>"
        else: html_output += f"<div class='other-speech'>{escaped_line}</div>"
    return f"<div>{html_output}</div>"

# --- MAIN UI LOGIC ---

def main():
    if "final_df" not in st.session_state: st.session_state.final_df = pd.DataFrame()
    if "processed_results" not in st.session_state: st.session_state.processed_results = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        max_workers = st.slider("Concurrency", 1, 8, 4)
        keep_remote = st.checkbox("Keep remote files", value=False)
        st.divider()
        language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
        lang_map = {
            "English (India)": "English (Indian accent)",
            "Hindi": "Hindi (Devanagari)",
            "Mixed (Hinglish)": "Mixed English and Hindi"
        }
        theme_choice = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)

    theme_class = "dark-theme" if theme_choice == "Dark" else "light-theme"
    st.markdown(f"<div class='{theme_class}'>", unsafe_allow_html=True)

    # --- UPDATED FILE UPLOADER ---
    uploaded_files = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=True)

    progress_bar = st.empty()
    status_text = st.empty()
    result_placeholder = st.empty()

    start_button = st.button("🚀 Start Batch Processing", type="primary")

    if start_button:
        if not api_key or not uploaded_files:
            st.error("Please enter API Key and upload one or more files.")
            st.stop()

        # 1. READ & MERGE FILES
        all_dfs = []
        for file in uploaded_files:
            try:
                df_single = pd.read_excel(file)
                all_dfs.append(df_single)
            except Exception as e:
                st.warning(f"Skipping file {file.name}: Error reading file: {e}")
                continue

        if not all_dfs:
            st.error("No valid data found.")
            st.stop()

        raw_df = pd.concat(all_dfs, ignore_index=True)

        if "recording_url" not in raw_df.columns or "mobile_number" not in raw_df.columns:
            st.error("Columns 'recording_url' and 'mobile_number' are required.")
            st.stop()

        # 2. CONSOLIDATE DATA (Unique Mobile Logic)
        status_text.info("Consolidating data by unique mobile number...")
        df_consolidated = consolidate_data_by_mobile(raw_df)
        
        # We work on the consolidated DF now
        prompt_template = build_prompt(lang_map[language_mode])
        total_rows = len(df_consolidated)
        
        status_text.info(f"Processing {total_rows} unique contacts with {max_workers} threads...")
        progress_bar.progress(0.0)

        processed_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_row, idx, row, api_key, prompt_template, keep_remote): idx
                for idx, row in df_consolidated.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                processed_results.append(res)
                completed += 1
                
                progress_bar.progress(completed / total_rows)
                status_text.markdown(f"Processed **{completed}/{total_rows}** unique contacts.")
                
                # Live Preview
                recent_results = sorted(processed_results, key=lambda r: r["index"])[-5:]
                preview_df = pd.DataFrame(recent_results)
                if not preview_df.empty:
                    result_placeholder.dataframe(preview_df[["mobile_number", "status", "transcript"]], width=800)

        final_df = merge_results_with_original(df_consolidated, processed_results)
        st.session_state.final_df = final_df
        status_text.success("Batch Processing Complete!")

    # --- VIEWER ---
    final_df = st.session_state.final_df
    if not final_df.empty:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("## 🎛️ Transcript Browser")

        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a:
            search_q = st.text_input("Search", placeholder="Search text or phone...")
        with col_b:
            status_sel = st.selectbox("Status", ["All", "Success", "Failed", "Skipped"])
        with col_c:
            speaker_sel = st.selectbox("Speaker", ["All", "Speaker 1", "Speaker 2"])
        with col_d:
            per_page = st.selectbox("Per page", [5, 10, 20, 50], index=1)

        view_df = final_df.copy()
        
        # Filters
        if status_sel != "All":
            if status_sel == "Success":
                view_df = view_df[view_df["status"].str.contains("Success", case=False, na=False)]
            elif status_sel == "Failed":
                view_df = view_df[view_df["status"].str.contains("Failed|Error", case=False, na=False)]
            elif status_sel == "Skipped":
                view_df = view_df[view_df["status"].str.contains("Skipped", case=False, na=False)]

        if search_q.strip():
            q = search_q.lower()
            mask = (
                view_df["transcript"].fillna("").str.lower().str.contains(q) |
                view_df["mobile_number"].astype(str).str.lower().str.contains(q)
            )
            view_df = view_df[mask]

        if speaker_sel != "All":
            key = "speaker 1" if speaker_sel == "Speaker 1" else "speaker 2"
            mask = view_df["transcript"].fillna("").str.lower().str.contains(key)
            view_df = view_df[mask]

        total_items = len(view_df)
        st.markdown(f"**Showing {total_items} result(s)**")

        # Pagination
        pages = max(1, math.ceil(total_items / per_page))
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page_idx - 1) * per_page
        end = start + per_page
        page_df = view_df.iloc[start:end]

        # Download
        out_buf = BytesIO()
        view_df.to_excel(out_buf, index=False)
        st.download_button("📥 Download Filtered Results", out_buf.getvalue(), "transcripts.xlsx")

        # Cards
        for idx, row in page_df.iterrows():
            header = f"{row.get('mobile_number','Unknown')} — {row.get('status','')}"
            with st.expander(header, expanded=False):
                meta_html = f"<div class='meta-row'>URL: {html.escape(str(row.get('recording_url', 'None')))}</div>"
                st.markdown(meta_html, unsafe_allow_html=True)
                
                transcript_html = colorize_transcript_html(row.get("transcript", ""))
                st.markdown(f"<div class='transcript-box'>{transcript_html}</div>", unsafe_allow_html=True)
                
                if row.get("error"):
                    st.error(f"Error: {row.get('error')}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
