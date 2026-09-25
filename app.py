import os
import subprocess
import asyncio
import tempfile
import streamlit as st
from groq import Groq
import edge_tts
from gtts import gTTS
from gradio_client import Client, handle_file

st.set_page_config(
    page_title="AI Video Urdu Dubber & Lip-Sync",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video Urdu Dubber & Lip-Sync Studio")
st.caption("Dub videos into Urdu with custom script input or LLM translation, perfectly timed with AI Lip-Sync")

# 1. API Configuration
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Groq API Key (Required only if auto-translating)", type="password")

client = Groq(api_key=api_key) if api_key else None

# 2. Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    voice_selection = st.selectbox(
        "Select Urdu Voice",
        options=["Asad (Male - Pakistani)", "Uzma (Female - Pakistani)"],
        index=0
    )
    voice_id = "ur-PK-AsadNeural" if "Asad" in voice_selection else "ur-PK-UzmaNeural"

    source_language = st.selectbox(
        "Original Video Language (if using auto-translate)",
        options=["Auto-Detect", "Arabic", "Turkish", "Persian", "English"],
        index=0
    )
    lang_code_map = {"Auto-Detect": None, "Arabic": "ar", "Turkish": "tr", "Persian": "fa", "English": "en"}
    selected_lang_code = lang_code_map[source_language]

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Upload a video file.")
    st.markdown("2. Enter your own Urdu script (or leave empty for AI auto-translation).")
    st.markdown("3. Click Start to generate the Lip-Synced Urdu video.")

# 3. Helper Functions
def get_media_duration(file_path: str) -> float:
    """Calculates exact duration of media in seconds."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

async def generate_urdu_tts_robust(text: str, voice: str, output_path: str):
    """Generates natural Urdu voiceover with bulletproof fallback."""
    clean_text = text.strip()
    if not clean_text:
        clean_text = "No script provided."

    # Attempt 1: Microsoft Neural Voice
    try:
        communicate = edge_tts.Communicate(text=clean_text, voice=voice)
        await communicate.save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return
    except Exception:
        pass

    # Attempt 2: Google Urdu TTS Fallback
    try:
        tts = gTTS(text=clean_text, lang="ur")
        tts.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Urdu voice synthesis error: {str(e)}")

def adjust_audio_tempo_and_pad(input_audio: str, target_duration: float, output_audio: str):
    """Adjusts tempo and pads/trims audio to exactly match the video duration."""
    curr_duration = get_media_duration(input_audio)
    if curr_duration <= 0 or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, output_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    tempo = curr_duration / target_duration
    clamped_tempo = max(0.75, min(1.35, tempo))

    cmd = [
        "ffmpeg", "-y",
        "-i", input_audio,
        "-filter_complex", f"[0:a]atempo={clamped_tempo:.3f},apad=whole_dur={target_duration}[a]",
        "-map", "[a]",
        "-t", str(target_duration),
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_wav2lip_robust(video_path: str, audio_path: str) -> str:
    """Executes AI Lip-Sync on active public GPU spaces."""
    public_spaces = [
        "camenduru/Wav2Lip",
        "prasannas/Wav2Lip",
        "Radames/Wav2Lip"
    ]
    for space_id in public_spaces:
        try:
            hf_client = Client(space_id)
            try:
                res = hf_client.predict(
                    face=handle_file(video_path),
                    audio=handle_file(audio_path),
                    api_name="/predict"
                )
            except Exception:
                res = hf_client.predict(
                    handle_file(video_path),
                    handle_file(audio_path),
                    api_name="/predict"
                )

            if isinstance(res, (tuple, list)):
                res = res[0]
            if isinstance(res, dict) and "video" in res:
                res = res["video"]
            if isinstance(res, str) and os.path.exists(res):
                return res
        except Exception:
            continue
    return None

def translate_with_llm(groq_client: Groq, text: str) -> str:
    """Translates spoken speech into natural spoken Urdu using Groq LLM."""
    system_prompt = (
        "You are an expert dubbing translator. Translate the given spoken speech "
        "into clear, natural, spoken Pakistani Urdu dialogue for video voiceover dubbing. "
        "Output ONLY the Urdu translation script in Urdu alphabet without any English or notes."
    )
    # Line 127: LLM MODEL INVOCATION
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        max_tokens=4096,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# 4. Main Application Interface
uploaded_file = st.file_uploader("1. Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

user_script = st.text_area(
    "2. Enter Your Urdu Script (Optional - If provided, AI will directly dub this script):",
    placeholder="Enter the exact Urdu dialogue here. If left empty, AI will transcribe and translate automatically...",
    height=150
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🚀 Start Urdu Dubbing & Lip-Sync", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, "input.mp4")
            extracted_audio_path = os.path.join(temp_dir, "extracted.mp3")
            raw_urdu_audio_path = os.path.join(temp_dir, "urdu_raw.mp3")
            synced_urdu_audio_path = os.path.join(temp_dir, "urdu_synced.mp3")
            final_output_path = os.path.join(temp_dir, "final_dubbed.mp4")

            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            video_duration = get_media_duration(input_video_path)
            progress = st.progress(0)
            status = st.empty()

            try:
                # Determine Urdu Script: Custom input OR LLM translation
                if user_script.strip():
                    status.info("Using your custom Urdu script...")
                    progress.progress(30)
                    urdu_text = user_script.strip()
                else:
                    if not client:
                        st.error("Please enter a Groq API Key in the sidebar or provide a custom Urdu script above.")
                        st.stop()

                    status.info("Extracting original audio from video...")
                    progress.progress(15)
                    subprocess.run([
                        "ffmpeg", "-y", "-i", input_video_path,
                        "-vn", "-acodec", "libmp3lame", "-ar", "16000", extracted_audio_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    status.info("Transcribing speech with Whisper...")
                    progress.progress(25)
                    whisper_args = {
                        "file": (os.path.basename(extracted_audio_path), open(extracted_audio_path, "rb").read()),
                        "model": "whisper-large-v3",
                        "response_format": "text"
                    }
                    if selected_lang_code:
                        whisper_args["language"] = selected_lang_code

                    transcription = client.audio.transcriptions.create(**whisper_args)
                    original_speech = str(transcription).strip()

                    status.info("Translating dialogue into Urdu using LLM...")
                    progress.progress(35)
                    urdu_text = translate_with_llm(client, original_speech)

                # Step 1: Synthesize Urdu Voice
                status.info("Synthesizing Urdu voiceover...")
                progress.progress(50)
                asyncio.run(generate_urdu_tts_robust(urdu_text, voice_id, raw_urdu_audio_path))

                # Step 2: Time Alignment (Match original video duration)
                status.info(f"Aligning audio duration to match video ({int(video_duration)} seconds)...")
                progress.progress(70)
                adjust_audio_tempo_and_pad(raw_urdu_audio_path, video_duration, synced_urdu_audio_path)

                # Step 3: AI Lip-Sync Execution
                status.info("Processing AI Lip-Sync (Synchronizing actor lips with Urdu speech)...")
                progress.progress(85)
                lip_sync_video = run_wav2lip_robust(input_video_path, synced_urdu_audio_path)

                if lip_sync_video and os.path.exists(lip_sync_video):
                    final_output_path = lip_sync_video
                    progress.progress(100)
                    status.success("Success! AI Lip-Sync completed successfully.")
                else:
                    status.warning("Notice: Cloud Lip-Sync was busy. Generated standard high-precision audio dubbing.")
                    cmd_merge = [
                        "ffmpeg", "-y", "-i", input_video_path, "-i", synced_urdu_audio_path,
                        "-c:v", "copy", "-c:a", "aac",
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-t", str(video_duration), final_output_path
                    ]
                    subprocess.run(cmd_merge, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    progress.progress(100)

                # Final Results
                st.subheader(f"🎬 Dubbed Video Output ({int(video_duration)} seconds)")
                st.video(final_output_path)

                with open(final_output_path, "rb") as out_file:
                    st.download_button(
                        label="⬇️ Download Dubbed Video",
                        data=out_file.read(),
                        file_name="urdu_dubbed_video.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
