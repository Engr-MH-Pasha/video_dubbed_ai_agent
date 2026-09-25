import os
import subprocess
import asyncio
import tempfile
import streamlit as st
from groq import Groq
import edge_tts
from gradio_client import Client, handle_file

st.set_page_config(
    page_title="AI Video to Urdu Dubber & Lip-Sync",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video to Urdu Dubber & Lip-Sync Studio")
st.caption("Convert English, Arabic, Turkish, and Persian videos into synchronized Urdu speech with optional Lip-Sync")

# 1. API Key Configuration
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.info("Please enter your Groq API Key in the sidebar or configure GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# 2. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Language Selection for Input Video
    language_mapping = {
        "Auto-Detect (خودکار شناخت)": None,
        "English (انگریزی)": "en",
        "Arabic (عربی)": "ar",
        "Turkish (ترکی)": "tr",
        "Persian / Farsi (فارسی)": "fa"
    }
    selected_lang_label = st.selectbox(
        "Select Video Language (ویڈیو کی زبان)",
        options=list(language_mapping.keys()),
        index=0
    )
    source_language_code = language_mapping[selected_lang_label]

    # Voice Selection
    voice_selection = st.selectbox(
        "Select Urdu Voice (اردو آواز)",
        options=["Asad (Male - Pakistani)", "Uzma (Female - Pakistani)"],
        index=0
    )
    voice_id = "ur-PK-AsadNeural" if "Asad" in voice_selection else "ur-PK-UzmaNeural"

    # Lip-Sync Option
    enable_lipsync = st.checkbox(
        "Enable AI Lip-Sync (Beta)",
        value=False,
        help="Synchronizes actor's lip movements using a free external GPU cloud space."
    )
    
    st.markdown("---")
    st.markdown("**Supported Input Languages:**")
    st.markdown("- English (انگریزی)\n- Arabic (عربی)\n- Turkish (ترکی)\n- Persian (فارسی)")

# 3. Helper Functions
def get_media_duration(file_path: str) -> float:
    """Calculates media duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

async def generate_urdu_tts(text: str, voice: str, output_path: str):
    """Synthesizes Urdu speech using Edge-TTS."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)

def adjust_audio_tempo(input_audio: str, target_duration: float, output_audio: str):
    """Adjusts audio speed using FFmpeg atempo to match video duration without pitch shift."""
    curr_duration = get_media_duration(input_audio)
    if curr_duration <= 0 or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, output_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    tempo = curr_duration / target_duration
    tempo = max(0.75, min(1.35, tempo))
    
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-filter:a", f"atempo={tempo:.3f}",
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_wav2lip_cloud(video_path: str, audio_path: str) -> str:
    """Invokes public Hugging Face Space for Wav2Lip processing."""
    hf_client = Client("fffiloni/Wav2Lip")
    result = hf_client.predict(
        face=handle_file(video_path),
        audio=handle_file(audio_path),
        api_name="/predict"
    )
    return result

# 4. Main Application Interface
uploaded_file = st.file_uploader("Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("🚀 Start Dubbing Pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, "input.mp4")
            extracted_audio_path = os.path.join(temp_dir, "extracted.mp3")
            raw_urdu_audio_path = os.path.join(temp_dir, "urdu_raw.mp3")
            synced_urdu_audio_path = os.path.join(temp_dir, "urdu_synced.mp3")
            final_output_path = os.path.join(temp_dir, "dubbed_final.mp4")

            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            video_duration = get_media_duration(input_video_path)
            
            progress = st.progress(0)
            status = st.empty()

            try:
                # Step 1: Extract Audio
                status.info("Step 1/5: Extracting audio from original video...")
                progress.progress(15)
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_video_path,
                    "-vn", "-acodec", "libmp3lame", "-ar", "16000", extracted_audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Step 2: Speech-to-Text via Groq Whisper with selected language
                status.info("Step 2/5: Transcribing original speech...")
                progress.progress(35)
                
                whisper_kwargs = {
                    "file": (os.path.basename(extracted_audio_path), open(extracted_audio_path, "rb").read()),
                    "model": "whisper-large-v3-turbo",
                    "response_format": "text"
                }
                if source_language_code:
                    whisper_kwargs["language"] = source_language_code

                transcription = client.audio.transcriptions.create(**whisper_kwargs)
                original_text = str(transcription).strip()
                
                if not original_text:
                    st.error("No clear voice or speech detected in the uploaded video.")
                    st.stop()

                # Step 3: Translation into Natural Spoken Urdu via Llama 3.3
                status.info("Step 3/5: Translating dialogue into natural Urdu...")
                progress.progress(55)
                
                lang_name = selected_lang_label.split(" (")[0]
                system_prompt = (
                    f"You are an expert dubbing translator. Translate the provided {lang_name} speech "
                    "into natural, conversational spoken Pakistani Urdu suitable for audio voiceover. "
                    "Keep the length concise so the Urdu dubbing duration naturally matches the original cadence. "
                    "Output ONLY the Urdu translation script in Urdu alphabet without any explanations, Latin script, or additional notes."
                )
                translation = client.chat.completions.create(
                    model="gemma2-9b-it",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": original_text}
                    ],
                    temperature=0.3
                )
                urdu_text = translation.choices[0].message.content.strip()

                # Step 4: Text-to-Speech & Duration Synchronization
                status.info("Step 4/5: Synthesizing Urdu speech and synchronizing duration...")
                progress.progress(75)
                asyncio.run(generate_urdu_tts(urdu_text, voice_id, raw_urdu_audio_path))
                
                adjust_audio_tempo(raw_urdu_audio_path, video_duration, synced_urdu_audio_path)

                # Step 5: Multiplexing / Lip-Sync
                status.info("Step 5/5: Rendering final video...")
                progress.progress(90)

                lip_sync_success = False
                if enable_lipsync:
                    status.info("Processing cloud-based Lip-Sync (this may take 1-2 minutes)...")
                    try:
                        cloud_result = run_wav2lip_cloud(input_video_path, synced_urdu_audio_path)
                        if os.path.exists(cloud_result):
                            final_output_path = cloud_result
                            lip_sync_success = True
                    except Exception as err:
                        st.warning(f"Cloud Lip-Sync server was busy. Falling back to standard audio dubbing. ({err})")

                if not lip_sync_success:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", input_video_path, "-i", synced_urdu_audio_path,
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                        "-shortest", final_output_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                progress.progress(100)
                status.success("Processing complete!")

                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Transcript:**")
                    st.info(original_text)
                with col2:
                    st.markdown("**Urdu Dubbing Script:**")
                    st.success(urdu_text)

                st.subheader("Dubbed Video Output")
                st.video(final_output_path)

                with open(final_output_path, "rb") as out_file:
                    st.download_button(
                        label="⬇️ Download Dubbed Video",
                        data=out_file.read(),
                        file_name="dubbed_urdu_video.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
