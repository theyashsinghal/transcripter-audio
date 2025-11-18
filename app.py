import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO
import tempfile
import os
import time
import logging
import mimetypes

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
MODEL_NAME = "gemini-2.5-flash"  # Hardcoded Model Name

# Configure logging to show progress in console
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

st.set_page_config(page_title="Gemini Call Transcriber (2.5 Flash)", layout="wide")

# --- NETWORK HELPERS ---

def make_request_with_retry(method, url, **kwargs):
    """
    Executes HTTP requests with exponential backoff for rate limits (429)
    and server errors (5xx).
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            
            # Success range
            if 200 <= response.status_code < 300:
                return response
            
            # Retry on Rate Limit (429) or Server Error (500+)
            if response.status_code == 429 or response.status_code >= 500:
                wait_time = (2 ** attempt) + 1
                logging.warning(f"Status {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            # Return other errors (400, 404) immediately for handling
            return response
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)
            
    return response

# --- CORE API FUNCTIONS ---

def initiate_upload(api_key, filename, mime_type, file_size):
    """Step 1: Tell Google we are starting an upload (Resumable Protocol)."""
    url = f"{UPLOAD_URL}?uploadType=resumable&key={api_key}"
    
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type # Strict Promise
    }
    
    data = json.dumps({"file": {"display_name": filename}})
    
    response = make_request_with_retry("POST", url, headers=headers, data=data)
    
    if response.status_code != 200:
        raise Exception(f"Init failed ({response.status_code}): {response.text}")
        
    upload_url = response.headers.get('X-Goog-Upload-URL')
    if not upload_url:
        raise Exception("No upload URL returned from Google.")
        
    return upload_url

def upload_bytes(upload_url, file_path, mime_type):
    """Step 2: Send the actual file bytes."""
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f:
        file_bytes = f.read()

    headers = {
        "Content-Length": str(file_size),
        "Content-Type": mime_type, # Must match the promise in Step 1
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }
    
    # Try POST first (Standard for Gemini Files API)
    response = requests.post(upload_url, headers=headers, data=file_bytes)
    
    # Robustness: If POST fails with 400, try PUT (Common fix for protocol quirks)
    if response.status_code == 400:
        logging.warning("POST failed with 400. Retrying with PUT...")
        response = requests.put(upload_url, headers=headers, data=file_bytes)

    if response.status_code != 200:
        raise Exception(f"Upload failed ({response.status_code}): {response.text}")
    
    # CRITICAL FIX: Return the entire file object so we have both 'name' and 'uri'
    return response.json().get("file", {})

def wait_for_active(api_key, file_name):
    """Step 3: Poll using the file NAME (e.g. files/abc) until active."""
    url = f"{BASE_URL}/v1beta/{file_name}?key={api_key}"
    
    for _ in range(40): # Wait up to ~3 minutes
        response = make_request_with_retry("GET", url)
        
        if response.status_code != 200:
            logging.warning(f"Polling error {response.status_code}. Retrying...")
            time.sleep(5)
            continue
            
        state = response.json().get("state")
        logging.info(f"File state: {state}")
        
        if state == "ACTIVE":
            return True
        elif state == "FAILED":
            raise Exception(f"File processing failed on Google side.")
            
        time.sleep(5)
        
    raise Exception("Timed out waiting for file to become ACTIVE.")

def generate_transcript(api_key, file_uri, mime_type, prompt):
    """Step 4: Ask the model to transcribe using the file URI."""
    # Uses the hardcoded MODEL_NAME
    api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                # CRITICAL FIX: Must use the full https:// URI here, NOT the name.
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 8192
        }
    }
    
    response = make_request_with_retry("POST", api_url, json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        return f"API ERROR {response.status_code}: {response.text}"
        
    try:
        candidates = response.json().get("candidates", [])
        if not candidates:
            return "NO TRANSCRIPT (Safety block or empty audio)"
        return candidates[0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"PARSE ERROR: {str(e)}"

def delete_file(api_key, file_name):
    """Step 5: Cleanup using the file NAME."""
    try:
        requests.delete(f"{BASE_URL}/v1beta/{file_name}?key={api_key}")
        logging.info(f"Deleted: {file_name}")
    except:
        pass

# --- UI LOGIC ---

st.title(f"🎙️ Gemini Call Transcriber ({MODEL_NAME})")
st.markdown("Batch transcription with correct URI handling and milliseconds.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.info(f"🔒 Model locked to: **{MODEL_NAME}**")
    
    # Default index set to 2 (Mixed/Hinglish)
    language_mode = st.selectbox("Language", ["English (India)", "Hindi", "Mixed (Hinglish)"], index=2)
    
    lang_map = {
        "English (India)": "English (Indian accent)",
        "Hindi": "Hindi (Devanagari)",
        "Mixed (Hinglish)": "Mixed English and Hindi"
    }

uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
st.caption("Required column: `recording_url`")

if st.button("🚀 Start Processing", type="primary", width='stretch'):
    if not api_key or not uploaded_file:
        st.error("Please enter API Key and Upload File.")
        st.stop()
        
    try:
        df = pd.read_excel(uploaded_file)
        if "recording_url" not in df.columns:
            st.error("Column 'recording_url' is missing.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    if "transcript" not in df.columns:
        df["transcript"] = ""

    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    total_rows = len(df)

    for i, row in df.iterrows():
        mobile = str(row.get('mobile_number', f'Row {i+1}'))
        audio_url = row['recording_url']
        
        tmp_path = None
        file_info = None # Will hold {name: 'files/x', uri: 'https://...'}
        transcript_text = ""
        status_label = ""
        
        try:
            # 1. Download
            status_text.markdown(f"### 📥 Downloading {i+1}/{total_rows}: `{mobile}`")
            logging.info(f"Downloading {audio_url}")
            r = requests.get(audio_url, stream=True, timeout=60)
            r.raise_for_status()
            
            content_type = r.headers.get('content-type', 'audio/mp3')
            ext = mimetypes.guess_extension(content_type) or ".mp3"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            mime_type = mimetypes.guess_type(tmp_path)[0] or "audio/mpeg"
            f_size = os.path.getsize(tmp_path)
            
            # 2. Upload
            status_text.markdown(f"### 📤 Sending to AI {i+1}/{total_rows}: `{mobile}`")
            up_url = initiate_upload(api_key, f"rec_{i}_{int(time.time())}{ext}", mime_type, f_size)
            
            # RETURNS DICT: {'name': 'files/abc', 'uri': 'https://...'}
            file_info = upload_bytes(up_url, tmp_path, mime_type)
            
            # 3. Wait (Uses 'name')
            status_text.markdown(f"### 👂 AI is listening... (`{mobile}`)")
            wait_for_active(api_key, file_info['name'])
            
            # 4. Transcribe (Uses 'uri')
            status_text.markdown(f"### ✍️ Writing transcript... (`{mobile}`)")
            prompt = f"""
            Transcribe this audio in {lang_map[language_mode]}.
            Requirements:
            - Identify speakers (Speaker 1, Speaker 2).
            - Add timestamps exactly in milliseconds (e.g. [0ms-2500ms]) at the start of every line.
            - Do NOT use Minutes:Seconds format. Use raw milliseconds.
            - Write exactly what is said.
            - CRITICAL: Write ALL Hindi words in Hinglish (Latin script). Do NOT use Devanagari script.
            - Keep English words in standard English.
            """
            transcript_text = generate_transcript(api_key, file_info['uri'], mime_type, prompt)
            
            status_label = "✅ Success" if "ERROR" not in transcript_text else "❌ Error"
            
        except Exception as e:
            logging.error(f"Row {i} failed: {e}")
            transcript_text = f"SYSTEM ERROR: {str(e)}"
            status_label = "❌ Failed"
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
            
            if file_info and 'name' in file_info:
                delete_file(api_key, file_info['name'])

        df.at[i, "transcript"] = transcript_text
        results.append({
            "Mobile": mobile,
            "Status": status_label,
            "Details": transcript_text[:100]
        })
        
        progress_bar.progress((i + 1) / total_rows)

    status_text.success("Processing Complete!")
    
    st.subheader("Results")
    st.dataframe(pd.DataFrame(results), width='stretch')
    
    output = BytesIO()
    df.to_excel(output, index=False)
    st.download_button(
        label="📥 Download Transcripts",
        data=output.getvalue(),
        file_name="transcripts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width='stretch'
    )
