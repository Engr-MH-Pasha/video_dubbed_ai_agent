import os
import subprocess
import asyncio
import tempfile
import streamlit as st
import edge_tts
from gtts import gTTS
from gradio_client import Client, handle_file

st.set_page_config(
    page_title="Urdu Lip-Sync Dubber",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video Urdu Dubber & Lip-Sync Studio")
st.caption("ویڈیو کے چہرے کے ہونٹوں کو اپنے اردو اسکرپٹ کے مطابق سنک (Lip-Sync) کریں")

# 1. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ سیٹنگز (Settings)")
    voice_selection = st.selectbox(
        "اردو آواز کا انتخاب کریں (Urdu Voice)",
        options=["Asad (مردانہ آواز - پاکستانی)", "Uzma (زنانی آواز - پاکستانی)"],
        index=0
    )
    voice_id = "ur-PK-AsadNeural" if "Asad" in voice_selection else "ur-PK-UzmaNeural"

    st.markdown("---")
    st.markdown("**طریقہ کار:**")
    st.markdown("1. ویڈیو اپ لوڈ کریں۔")
    st.markdown("2. اپنا اردو ڈائیلاگ / اسکرپٹ درج کریں۔")
    st.markdown("3. بٹن دبائیں اور Lip-Synced ویڈیو حاصل کریں۔")

# 2. Helper Functions
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
        clean_text = "کوئی اسکرپٹ درج نہیں کیا گیا"

    # Attempt 1: Edge-TTS Neural Voice
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
        raise RuntimeError(f"آواز بنانے میں خرابی: {str(e)}")

def adjust_audio_tempo_and_pad(input_audio: str, target_duration: float, output_audio: str):
    """Adjusts tempo and pads/trims audio to exactly match the video duration."""
    curr_duration = get_media_duration(input_audio)
    if curr_duration <= 0 or target_duration <= 0:
        subprocess.run(["ffmpeg", "-y", "-i", input_audio, output_audio], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    tempo = curr_duration / target_duration
    # Clamp tempo between 0.75x and 1.35x for realistic speech cadence
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
    """Executes AI Lip-Sync on cloud GPU spaces."""
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

# 3. Main Interface
uploaded_file = st.file_uploader("1️⃣ ویڈیو اپ لوڈ کریں (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

user_script = st.text_area(
    "2️⃣ اپنا اردو اسکرپٹ یہاں لکھیں (جو ویڈیو میں بولنا ہے):",
    placeholder="مثال: السلام علیکم! حدیث شریف کے اس مقابلے میں آپ کا خیر مقدم ہے۔۔۔",
    height=150
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🚀 اردو Lip-Sync ڈبنگ تیار کریں", type="primary"):
        if not user_script.strip():
            st.warning("⚠️ برائے مہربانی پہلے اوپر والے باکس میں اپنا اردو اسکرپٹ درج کریں۔")
            st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_video_path = os.path.join(temp_dir, "input.mp4")
            raw_urdu_audio_path = os.path.join(temp_dir, "urdu_raw.mp3")
            synced_urdu_audio_path = os.path.join(temp_dir, "urdu_synced.mp3")
            final_output_path = os.path.join(temp_dir, "final_dubbed.mp4")

            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            video_duration = get_media_duration(input_video_path)
            
            progress = st.progress(0)
            status = st.empty()

            try:
                # Step 1: Synthesize Urdu Voice from User Script
                status.info("1/3: آپ کے لکھے ہوئے اسکرپٹ کی اردو آواز تیار کی جا رہی ہے...")
                progress.progress(30)
                asyncio.run(generate_urdu_tts_robust(user_script.strip(), voice_id, raw_urdu_audio_path))

                # Step 2: Time Alignment & Synchronization
                status.info(f"2/3: آواز کی ٹائمنگ کو ویڈیو کی لمبائی ({int(video_duration)} سیکنڈ) کے مطابق ڈھالا جا رہا ہے...")
                progress.progress(60)
                adjust_audio_tempo_and_pad(raw_urdu_audio_path, video_duration, synced_urdu_audio_path)

                # Step 3: AI Lip-Sync Execution
                status.info("3/3: چہرے کے ہونٹوں کی حرکت کو اردو بول کے مطابق سنک (Lip-Sync) کیا جا رہا ہے...")
                progress.progress(80)

                lip_sync_video = run_wav2lip_robust(input_video_path, synced_urdu_audio_path)
                
                if lip_sync_video and os.path.exists(lip_sync_video):
                    final_output_path = lip_sync_video
                    progress.progress(100)
                    status.success("🎉 مبارک ہو! ویڈیو کا Lip-Sync کامیابی سے مکمل ہو گیا۔")
                else:
                    st.warning("⚠️ کلاؤڈ GPU سرور پر رش کی وجہ سے ڈائریکٹ آڈیو سنک ویڈیو رینڈر کی جا رہی ہے۔")
                    cmd_merge = [
                        "ffmpeg", "-y", "-i", input_video_path, "-i", synced_urdu_audio_path,
                        "-c:v", "copy", "-c:a", "aac",
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-t", str(video_duration), final_output_path
                    ]
                    subprocess.run(cmd_merge, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    progress.progress(100)
                    status.success("🎉 ویڈیو اردو آواز کے ساتھ مکمل تیار ہے!")

                # Final Video Output Display
                st.subheader(f"🎬 حتمی ویڈیو (طوالت: {int(video_duration)} سیکنڈ)")
                st.video(final_output_path)

                with open(final_output_path, "rb") as out_file:
                    st.download_button(
                        label="⬇️ Lip-Synced ویڈیو ڈاؤنلوڈ کریں",
                        data=out_file.read(),
                        file_name="urdu_lipsynced_video.mp4",
                        mime="video/mp4",
                        type="primary"
                    )

            except Exception as e:
                st.error(f"پروسیسنگ کے دوران خرابی پیش آئی: {str(e)}")
