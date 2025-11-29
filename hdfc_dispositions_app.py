# classify_calls_app.py — STEP 3: AUTOMATED DISPOSITIONING (Updated to 64 Threads)

import streamlit as st
import pandas as pd
import requests
import json
import os
import time
import logging
import random
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

# --- CONFIGURATION ---
BASE_URL = "https://generativelanguage.googleapis.com"
MODEL_NAME = "gemini-2.5-flash" 

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("classifier")

# --- DISPOSITION STRUCTURE (STRICTLY CONSTRAINED) ---
# This dictionary will be converted to a string and injected into the prompt.
DISPOSITION_STRUCTURE = {
    "Already Paid": ["To FC", "To FLS", "At Bank", "At Branch", "Online", "At OTC"],
    "Call back": ["Follow Up", "Need time", "Loan - Need time to think", "Yet to decide for loan"],
    "Call Drop": ["Agent Not Available"],
    "Change in Frequency": ["To Quarterly", "To Half yearly", "To Monthly", "To Yearly"],
    "Cheque Pick-up": ["Service Location", "Non Service Location"],
    "Do Not Call": ["Do Not Call"],
    "Escalation": ["Wrong Number", "Policy Doesn't Belongs to me"],
    "Hung up": ["Key Info Not Passed", "Key Info Passed"],
    "Interested to Pay": ["Will Pay Later", "Temporary Funds Issue", "Part Payment", "Late Fees Waiver"],
    "LA Deceased": ["Claim Not Registered", "Claim Registered"],
    "Language barrier": ["Prefer Tamil", "Prefer Telugu", "Prefer Malayalam", "Prefer Kannada", "Prefer Marathi", "Prefer Bengali", "Prefer Oria", "Prefer Hindi", "Prefer Enlgish", "Prefer Gujrathi", "Prefer Assamese", "Others"],
    "Non Connect": ["Others Non Connect", "Number Not Reachable", "Number Switched Off", "Number Busy", "Network Announcement", "Number Temporarily Disconnected", "Number Doesn't Exist"],
    "PH Deceased": ["PH Deceased"],
    "PH Out of Station": ["Call back", "Key Info Not Passed", "Key Info Passed"],
    "Promise to Pay": ["To FC", "To FLS", "At Bank", "At Branch", "Online", "At OTC", "Agreed for loan", "Through UPI"],
    "Re-Debit Request": ["Mandate Active", "Mandate In Active"],
    "Refuse to Pay": ["Financial Constrains", "Wants to Surrender", "Miss Sold", "Service Issue", "Low Returns", "Purchased new plan in HDFC Life", "Purchased Competition Plan", "Others", "Disagreed for loan"],
    "Service Request": ["Need Policy Document", "Need Last Premium Receipt", "Need Annual Premium Receipt", "Need Fund Statement", "Need Bonus statement", "Others"],
    "Third Party Pick-up": ["Left Message", "Call back"],
    "Retained": ["Willing to continue", "Satisfied and continue in PU", "Surrender retained", "Surrender retained with partial wthdrawal", "Surrender retained with loan", "Updation of Email ID and/or Contact No."],
    "Not Retained": ["Satisfied with his policy but wish to Surrender", "Dis-satisfied with policy returns.", "Need of Money - Wants to invest in other financial instruments", "Need of Money - For personal expenses or to buy asset", "Short term sale-3/5 years", "Mis sale- Complaint", "Policy servicing complaint to be resolved (delay/not processed/ error)", "Unhappy with FC service", "Unhappy with Branch Service", "Reinvestment with others", "Branch advising to buy new policy by surrendering existing policy", "Unhappy with Policy T&C", "Market Volatility"]
}

# --- NETWORK UTILITIES (WITH RETRY LOGIC) ---

def _sleep_with_jitter(base_seconds: float, attempt: int):
    """Adds exponential backoff and jitter to sleep time."""
    jitter = random.uniform(0.5, 1.5)
    to_sleep = min(base_seconds * (2 ** attempt) * jitter, 15) 
    time.sleep(to_sleep)

def make_request_with_retry(method: str, url: str, max_retries: int = 5, backoff_base: float = 0.5, **kwargs) -> requests.Response:
    """
    Robust wrapper for requests with exponential backoff + jitter.
    Handles 429 (Rate Limit) and 5xx (Server Errors).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=45, **kwargs) 
            # Check for transient errors
            if resp.status_code == 429 or (500 <= resp.status_code < 600):
                logger.warning(f"Transient HTTP {resp.status_code}. Retrying classification...")
                _sleep_with_jitter(backoff_base, attempt)
                continue
            return resp
        except requests.exceptions.RequestException as e:
            logger.warning(f"RequestException: {str(e)} (attempt {attempt + 1})")
            last_exc = e
            _sleep_with_jitter(backoff_base, attempt)
    if last_exc: raise last_exc
    raise Exception("make_request_with_retry: retries exhausted")

# --- PROMPT CONSTRUCTION ---

def build_classification_prompt(mobile_number: str, transcript: str, dispositions: Dict[str, List[str]]) -> str:
    """
    Constructs the highly constrained system prompt for dispositioning.
    """
    
    # Format the disposition structure for the AI to read easily
    disposition_list_str = ""
    for main, subs in dispositions.items():
        disposition_list_str += f"- MAIN: {main}\n"
        for sub in subs:
            disposition_list_str += f"  - SUB: {sub}\n"

    return f"""
You are a highly analytical Call Analyst specialized in contact center call classification.
Your task is to review the provided call transcript and accurately assign the single MOST appropriate Main Disposition and Sub Disposition.

CRITICAL INSTRUCTIONS:
1. You MUST choose the Main Disposition and Sub Disposition ONLY from the provided DISPOSITION LIST.
2. The output MUST be a valid JSON object.
3. If the transcript clearly indicates that NO conversation took place (e.g., 'busy busy no_answer') or contains system errors, assign 'Non Connect' as the Main Disposition and choose the closest Sub Disposition from that category.

DISPOSITION LIST:
{disposition_list_str}

METADATA:
Mobile Number: {mobile_number}

TRANSCRIPT:
---
{transcript}
---

Return your classification strictly as a JSON object with the following three keys:
{{"main_disposition": "[Your choice from the MAIN list]", "sub_disposition": "[Your choice from the corresponding SUB list]", "classification_summary": "[A brief, 10-word summary of why you chose this disposition]"}}
"""

# --- WORKER FUNCTION ---

def generate_disposition_for_call(index: int, row: pd.Series, api_key: str, dispositions: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Worker function to call the Gemini API for classification.
    """
    mobile = str(row.get("mobile_number", "Unknown"))
    transcript = row.get("transcript", "")
    
    # Initialize result structure
    result = {
        "index": index,
        "mobile_number": mobile,
        "transcript": transcript,
        "status": "Pending",
        "main_disposition": "N/A",
        "sub_disposition": "N/A",
        "classification_error": None
    }
    
    # Skip processing rows that failed transcription or were non-connects based on the previous app's logic
    if not transcript or "SYSTEM ERROR" in transcript or "SKIP" in row.get("status", "") or "NO TRANSCRIPT" in transcript or "Outcomes:" in transcript:
        # Default non-connect classification for non-conversational rows
        result.update({
            "status": "🚫 Skipped (Non-Conversational)",
            "main_disposition": "Non Connect",
            "sub_disposition": "Others Non Connect"
        })
        return result

    try:
        prompt = build_classification_prompt(mobile, transcript, dispositions)
        api_url = f"{BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
        
        # Call the API using the robust retry function
        resp = make_request_with_retry("POST", api_url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"})
        
        if resp.status_code != 200:
            raise Exception(f"API Error {resp.status_code}: {resp.text}")
            
        # Parse response
        body = resp.json()
        text = body['candidates'][0]['content']['parts'][0]['text']
        
        # Robustly extract JSON from the raw text response (in case of markdown fences)
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        json_str = text[json_start:json_end]
        
        dispo_data = json.loads(json_str)
        
        result.update({
            "status": "✅ Classified",
            "main_disposition": dispo_data.get("main_disposition", "N/A"),
            "sub_disposition": dispo_data.get("sub_disposition", "N/A"),
            "classification_summary": dispo_data.get("classification_summary", "")
        })
        
    except Exception as e:
        logger.error(f"Classification failed for {mobile}: {e}")
        result.update({
            "status": "❌ Classification Failed",
            "classification_error": str(e)
        })

    return result

# --- MAIN UI LOGIC ---

def main():
    st.set_page_config(page_title="Gemini Call Classifier (Step 3)", layout="wide")
    st.title("🧠 Automated Dispositioning (Step 3)")
    st.markdown("---")

    # --- SIDEBAR CONFIG ---
    with st.sidebar:
        st.header("Configuration")
        
        # NEW API KEY INPUT FIELD
        classification_api_key = st.text_input("Gemini Classification API Key", type="password", key="classification_key")
        st.caption("Use a separate key for this high-volume classification step.")
        
        # CONCURRENCY SLIDER UPDATED TO 64
        max_workers_cls = st.slider("Classification Concurrency", min_value=1, max_value=64, value=64,
                                    help="Threads used for parallel API calls. Be mindful of API rate limits.")

    # --- INPUT ---
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload Transcribed Excel File (.xlsx)", type=["xlsx"], 
                                     help="Use the output file generated by the Batch Transcriber app.")
    
    start_button = st.button("🚀 Start Automated Classification", type="primary")

    if 'classified_df' not in st.session_state:
        st.session_state.classified_df = pd.DataFrame()

    # --- PROCESSING ---
    if start_button:
        if not classification_api_key or not uploaded_file:
            st.error("Please enter the Classification API Key and upload the Excel file.")
            st.stop()

        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        required_cols = ["mobile_number", "transcript", "status"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Input file must contain columns: {', '.join(required_cols)}.")
            st.stop()
        
        # Prepare data for processing
        df = df.reset_index()

        st.info(f"Starting classification for {len(df)} records with {max_workers_cls} threads...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_results = []
        total_rows = len(df)
        
        # --- THREAD POOL EXECUTION ---
        with ThreadPoolExecutor(max_workers=max_workers_cls) as executor:
            futures = {
                executor.submit(generate_disposition_for_call, idx, row, classification_api_key, DISPOSITION_STRUCTURE): idx
                for idx, row in df.iterrows()
            }
            
            completed = 0
            for future in as_completed(futures):
                res = future.result()
                processed_results.append(res)
                completed += 1
                
                progress_bar.progress(completed / total_rows)
                status_text.text(f"Classified {completed}/{total_rows} calls.")
        
        # Merge Results back into the original DataFrame
        results_df = pd.DataFrame(processed_results).set_index('index')
        
        final_df = df.set_index('index').join(results_df[[
            'status', 
            'main_disposition', 
            'sub_disposition', 
            'classification_summary', 
            'classification_error'
        ]], rsuffix='_cls')
        
        final_df = final_df.reset_index()
        st.session_state.classified_df = final_df
        st.success("Classification Complete!")
    
    # --- OUTPUT DISPLAY ---
    if not st.session_state.classified_df.empty:
        st.header("2. Classified Results")
        final_df = st.session_state.classified_df

        # Clean up unnecessary columns for download view
        download_df = final_df.drop(columns=[c for c in final_df.columns if c in ['index', 'classification_error']], errors='ignore')

        st.dataframe(download_df[[
            'mobile_number', 'status', 'main_disposition', 'sub_disposition', 'classification_summary', 'transcript'
        ]])

        # Download button
        out_buf = BytesIO()
        download_df.to_excel(out_buf, index=False)
        st.download_button(
            "📥 Download Final Classified Data",
            data=out_buf.getvalue(),
            file_name=f"classified_calls_{int(time.time())}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
