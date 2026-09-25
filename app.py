import os
import subprocess
import asyncio
import tempfile
import streamlit as st
from groq import Groq
import edge_tts
from gtts import gTTS

st.set_page_config(
    page_title="AI Video to Urdu Dubber Studio",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video to Urdu Dubber Studio (Auto-Language & Full Audio)")
st.caption("Convert Arabic, Turkish, Persian, or English videos into complete synchronized Urdu speech")

# 1. API Key Setup
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
    
    language_mapping = {
        "Auto-Detect (خودکار شناخت - بہترین)": None,
        "Arabic (عربی)": "ar",
        "Turkish (ترکی)": "tr",
        "Persian / Farsi (فارسی)": "fa",
        "English (انگریزی)": "en"
    }
    selected_lang_label = st.selectbox(
        "Video Language (ویڈیو کی زبان)",
        options=list(language_mapping.keys()),
        index=0  # Defaults to Auto-Detect
    )
    source_language_code = language_mapping[selected_lang_label]

    voice_selection = st.selectbox(
        "Select Urdu Voice (اردو آواز)",
        options=["Asad (Male - پاکستانی مرد)", "Uzma (Female - پاکستانی خاتون)"],
        index=0
    )
    voice_id = "ur-PK-AsadNeural" if "Asad" in voice_selection else "ur-PK-UzmaNeural"

    burn_subtitles = st.checkbox(
        "Burn Urdu Subtitles (اردو سب ٹائٹلز دکھائیں)",
        value=True,
        help="Embeds subtitles at the bottom of the video."
    )

# 3. Helper Functions
def get_media_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def get_best_groq_chat_model(groq_client: Groq) -> str:
    """Dynamically detects active chat models on user's Groq account."""
    try:
        models = groq_client.models.list().data
        valid_models = [
            m.id for m in models 
            if not any(x in m.id for x in ["whisper", "guard", "vision", "embed"])
        ]
        # Prefer powerful models first
        for preferred in ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]:
            if preferred in valid_models:
                return preferred
        if valid_models:
            return valid_models[0]
    except Exception:
        pass
    return "llama3-8b-8192"

async def generate_urdu_tts_robust(text: str, voice: str, output_path: str):
    clean_text = text.strip()
    if not clean_text:
        clean_text = "آواز ریکارڈ نہیں ہو سکی"

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
        raise RuntimeError(f"Urdu speech synthesis failed: {str(e)}")

def adjust_audio_tempo(input_audio: str, target_duration: float, output_audio: str):
    curr_duration = get_media_duration(input_audio)
    if curr_duration <= 0 or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, output_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    tempo = curr_duration / target_duration
    tempo = max(0.70, min(1.45, tempo))
    
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-filter:a", f"atempo={tempo:.3f}",
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def translate_to_urdu(groq_client: Groq, text: str) -> str:
    active_model = get_best_groq_chat_model(groq_client)
    system_prompt = (
        "You are an expert dubbing translator. Translate the given spoken speech (which could be Arabic, Turkish, or Persian) "
        "into clear, natural, spoken Pakistani Urdu dialogue for voiceover dubbing.\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Translate the entire meaning accurately into Urdu.\n"
        "2. Do NOT summarize or omit content.\n"
        "3. Output ONLY the Urdu translation script in Urdu alphabet. Do not write any English, notes, or explanations."
    )
    
    res = groq_client.chat.completions.create(
        model=active_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        max_tokens=4096,
        temperature=0.2
    )
    return res.choices[0].message.content.strip()

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 4. Main Application Interface
uploaded_file = st.file_uploader("Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("🚀 Start Full Urdu Dubbing Pipeline", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, "input.mp4")
            extracted_audio_path = os.path.join(temp_dir, "extracted.mp3")
            raw_urdu_audio_path = os.path.join(temp_dir, "urdu_raw.mp3")
            synced_urdu_audio_path = os.path.join(temp_dir, "urdu_synced.mp3")
            dubbed_video_path = os.path.join(temp_dir, "dubbed_video.mp4")
            final_output_path = os.path.join(temp_dir, "final_output.mp4")
            srt_path = os.path.join(temp_dir, "subtitles.srt")

            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            video_duration = get_media_duration(input_video_path)
            
            progress = st.progress(0)
            status = st.empty()

            try:
                # Step 1: Extract Audio
                status.info("Step 1/5: Extracting audio from original video...")
                progress.progress(20)
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_video_path,
                    "-vn", "-acodec", "libmp3lame", "-ar", "16000", extracted_audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Step 2: Speech-to-Text via Whisper
                status.info("Step 2/5: Transcribing full speech (Auto-Detecting Language)...")
                progress.progress(40)
                
                with open(extracted_audio_path, "rb") as audio_file:
                    whisper_args = {
                        "file": (os.path.basename(extracted_audio_path), audio_file.read()),
                        "model": "whisper-large-v3",
                        "response_format": "text"
                    }
                    if source_language_code:
                        whisper_args["language"] = source_language_code

                    transcription = client.audio.transcriptions.create(**whisper_args)

                original_speech_text = str(transcription).strip()
                if not original_speech_text:
                    st.error("No audible voice speech was found in the video.")
                    st.stop()

                # Step 3: Complete Urdu Translation
                status.info("Step 3/5: Translating complete speech into Urdu...")
                progress.progress(60)
                urdu_text = translate_to_urdu(client, original_speech_text)

                # Prepare Subtitle File
                with open(srt_path, "w", encoding="utf-8") as srt_file:
                    srt_file.write(f"1\n00:00:00,500 --> {format_srt_time(video_duration)}\n{urdu_text}\n\n")

                # Step 4: Synthesize Urdu Voice and Match Duration
                status.info("Step 4/5: Synthesizing full Urdu voice and matching timing...")
                progress.progress(80)
                asyncio.run(generate_urdu_tts_robust(urdu_text, voice_id, raw_urdu_audio_path))
                adjust_audio_tempo(raw_urdu_audio_path, video_duration, synced_urdu_audio_path)

                # Step 5: Merge Audio with Video (Exact Duration Lock)
                status.info("Step 5/5: Multiplexing audio and rendering final video...")
                progress.progress(90)

                cmd_merge = [
                    "ffmpeg", "-y",
                    "-i", input_video_path,
                    "-i", synced_urdu_audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-filter_complex", f"[1:a]apad=whole_dur={video_duration}[a]",
                    "-map", "0:v:0",
                    "-map", "[a]",
                    "-t", str(video_duration),
                    dubbed_video_path
                ]
                subprocess.run(cmd_merge, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Subtitle Burn-in
                if burn_subtitles and os.path.exists(srt_path):
                    escaped_srt = srt_path.replace('\\', '/').replace(':', '\\:')
                    sub_filter = f"subtitles='{escaped_srt}':force_style='FontSize=20,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Alignment=2'"
                    cmd_burn = [
                        "ffmpeg", "-y", "-i", dubbed_video_path,
                        "-vf", sub_filter,
                        "-c:a", "copy", final_output_path
                    ]
                    burn_res = subprocess.run(cmd_burn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if burn_res.returncode != 0:
                        final_output_path = dubbed_video_path
                else:
                    final_output_path = dubbed_video_path

                progress.progress(100)
                status.success("🎉 Processing complete! Full video dubbed into Urdu.")

                # Display Results
                st.subheader("📋 Transcripts Verification")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Detected Original Speech:**")
                    st.text_area("Original Transcript", value=original_speech_text, height=200)
                with col2:
                    st.markdown("**Complete Urdu Translation:**")
                    st.text_area("Urdu Dubbing Script", value=urdu_text, height=200)

                # Final Video
                st.subheader("🎬 Final Urdu Dubbed Video")
                st.video(final_output_path)

                with open(final_output_path, "rb") as out_file:
                    st.download_button(
                        label="⬇️ Download Full Dubbed Video",
                        data=out_file.read(),
                        file_name="urdu_dubbed_video.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
