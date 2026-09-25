import os
import subprocess
import asyncio
import tempfile
import streamlit as st
from groq import Groq
import edge_tts
from gradio_client import Client, handle_file

st.set_page_config(
    page_title="AI Video to Urdu Dubber, Lip-Sync & Subtitles",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Urdu Dubbing Studio with Lip-Sync & Subtitles")
st.caption("Convert multilingual videos into synchronized Urdu speech with AI Lip-Sync and burned-in Urdu Subtitles")

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

    voice_selection = st.selectbox(
        "Select Urdu Voice (اردو آواز)",
        options=["Asad (Male - Pakistani)", "Uzma (Female - Pakistani)"],
        index=0
    )
    voice_id = "ur-PK-AsadNeural" if "Asad" in voice_selection else "ur-PK-UzmaNeural"

    enable_lipsync = st.checkbox(
        "Enable AI Lip-Sync (مطلوبہ Lip-Sync)",
        value=True,
        help="Synchronizes actor's lip movements using a free GPU cloud space."
    )
    
    burn_subtitles = st.checkbox(
        "Burn Urdu Subtitles (اردو سب ٹائٹلز دکھائیں)",
        value=True,
        help="Embeds time-synced Urdu subtitles at the bottom of the video."
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

async def generate_urdu_tts(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)

def adjust_audio_tempo(input_audio: str, target_duration: float, output_audio: str):
    curr_duration = get_media_duration(input_audio)
    if curr_duration <= 0 or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, output_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    tempo = curr_duration / target_duration
    tempo = max(0.70, min(1.40, tempo))
    
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-filter:a", f"atempo={tempo:.3f}",
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_wav2lip_cloud(video_path: str, audio_path: str) -> str:
    """Invokes active public Hugging Face Space for Wav2Lip."""
    hf_client = Client("camenduru/Wav2Lip")
    result = hf_client.predict(
        face=handle_file(video_path),
        audio=handle_file(audio_path),
        api_name="/predict"
    )
    return result

def translate_text(groq_client: Groq, text: str, source_lang: str) -> str:
    system_prompt = (
        f"You are an expert dubbing translator. Translate the provided {source_lang} speech "
        "into natural, conversational spoken Pakistani Urdu suitable for audio voiceover and subtitles. "
        "Keep it concise and clear. Output ONLY the Urdu translation script in Urdu alphabet without explanations or notes."
    )
    for model_id in ["mixtral-8x7b-32768", "gemma2-9b-it"]:
        try:
            res = groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            return res.choices[0].message.content.strip()
        except Exception:
            continue
    return text

def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(int((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# 4. Main Application Interface
uploaded_file = st.file_uploader("Upload Video File (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("🚀 Start Full Dubbing, Lip-Sync & Subtitle Pipeline", type="primary"):
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
                status.info("Step 1/6: Extracting audio from video...")
                progress.progress(10)
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_video_path,
                    "-vn", "-acodec", "libmp3lame", "-ar", "16000", extracted_audio_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Step 2: Timestamped Transcription via Whisper (verbose_json)
                status.info("Step 2/6: Transcribing speech with timestamps...")
                progress.progress(25)
                
                with open(extracted_audio_path, "rb") as audio_file:
                    whisper_args = {
                        "file": (os.path.basename(extracted_audio_path), audio_file.read()),
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"]
                    }
                    if source_language_code:
                        whisper_args["language"] = source_language_code
                    
                    transcript_res = client.audio.transcriptions.create(**whisper_args)

                segments = getattr(transcript_res, "segments", [])
                full_original_text = getattr(transcript_res, "text", "")

                if not full_original_text or not segments:
                    st.error("No clear voice speech detected in the video.")
                    st.stop()

                # Step 3: Translate segments for Subtitles & Full Script
                status.info("Step 3/6: Generating Urdu translation & subtitle segments...")
                progress.progress(45)
                
                lang_name = selected_lang_label.split(" (")[0]
                urdu_subtitle_segments = []
                full_urdu_script_parts = []

                for seg in segments:
                    seg_text = seg.get("text", "").strip()
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", seg_start + 1.0)
                    
                    if seg_text:
                        translated_seg = translate_text(client, seg_text, lang_name)
                        full_urdu_script_parts.append(translated_seg)
                        urdu_subtitle_segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": translated_seg
                        })

                urdu_text = " ".join(full_urdu_script_parts)

                # Create SRT file
                with open(srt_path, "w", encoding="utf-8") as srt_file:
                    for idx, s_item in enumerate(urdu_subtitle_segments, 1):
                        start_str = format_srt_time(s_item["start"])
                        end_str = format_srt_time(s_item["end"])
                        srt_file.write(f"{idx}\n{start_str} --> {end_str}\n{s_item['text']}\n\n")

                # Step 4: Text-to-Speech & Duration Synchronization
                status.info("Step 4/6: Synthesizing Urdu voice and synchronizing timing...")
                progress.progress(65)
                asyncio.run(generate_urdu_tts(urdu_text, voice_id, raw_urdu_audio_path))
                adjust_audio_tempo(raw_urdu_audio_path, video_duration, synced_urdu_audio_path)

                # Step 5: AI Lip-Sync (Mandatory execution)
                status.info("Step 5/6: Processing AI Lip-Sync on cloud GPU...")
                progress.progress(80)

                lip_sync_success = False
                if enable_lipsync:
                    try:
                        cloud_result = run_wav2lip_cloud(input_video_path, synced_urdu_audio_path)
                        if cloud_result and os.path.exists(cloud_result):
                            dubbed_video_path = cloud_result
                            lip_sync_success = True
                    except Exception as err:
                        st.warning(f"Cloud Lip-Sync notice: {err}. Using standard precise audio multiplexing.")

                if not lip_sync_success:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", input_video_path, "-i", synced_urdu_audio_path,
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                        "-shortest", dubbed_video_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Step 6: Burn Urdu Subtitles onto Video
                status.info("Step 6/6: Burning Urdu subtitles into final video...")
                progress.progress(95)

                if burn_subtitles and os.path.exists(srt_path):
                    # FFmpeg subtitle filter escaping path for Windows/Linux compatibility
                    escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
                    sub_filter = f"subtitles='{escaped_srt_path}':force_style='FontName=Noto Sans,FontSize=22,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Alignment=2'"
                    
                    cmd_burn = [
                        "ffmpeg", "-y", "-i", dubbed_video_path,
                        "-vf", sub_filter,
                        "-c:a", "copy", final_output_path
                    ]
                    burn_res = subprocess.run(cmd_burn, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if burn_res.returncode != 0:
                        # Fallback if subtitle filter fails due to font naming
                        final_output_path = dubbed_video_path
                else:
                    final_output_path = dubbed_video_path

                progress.progress(100)
                status.success("🎉 Complete process finished successfully!")

                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Transcript:**")
                    st.info(full_original_text)
                with col2:
                    st.markdown("**Urdu Dubbing Script:**")
                    st.success(urdu_text)

                st.subheader("Final Urdu Dubbed Video (with Lip-Sync & Subtitles)")
                st.video(final_output_path)

                with open(final_output_path, "rb") as out_file:
                    st.download_button(
                        label="⬇️ Download Final Dubbed Video",
                        data=out_file.read(),
                        file_name="urdu_dubbed_final.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
