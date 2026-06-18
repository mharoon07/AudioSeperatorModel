# app.py

import gradio as gr
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import os
import tempfile
import subprocess
import numpy as np
import warnings
import soundfile as sf
import librosa
import time
warnings.filterwarnings("ignore")

# --- Setup the models ---
print("Setting up models...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load HT-Demucs model
print("Loading HT-Demucs model...")
htdemucs_model = get_model(name="htdemucs")
htdemucs_model = htdemucs_model.to(device)
htdemucs_model.eval()
print("HT-Demucs model loaded successfully.")

# Setup Spleeter with Python API approach
print("Setting up Spleeter...")
spleeter_separator = None
spleeter_audio_adapter = None
spleeter_available = False

def patch_spleeter_redirects():
    """Patch Spleeter to handle GitHub redirects properly"""
    try:
        import httpx
        from spleeter.model.provider.github import GithubModelProvider
        
        # Store the original download method
        original_download = GithubModelProvider.download
        
        def patched_download(self, name, model_directory):
            """Patched download method that handles redirects"""
            import os
            import tarfile
            import tempfile
            from urllib.parse import urlparse
            
            print(f"Downloading {name} model with redirect handling...")
            
            # Model URLs - only 5stems
            model_urls = {
                '5stems': 'https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz'
            }
            
            if name not in model_urls:
                return original_download(self, name, model_directory)
            
            url = model_urls[name]
            
            try:
                # Create a session that follows redirects
                with httpx.Client(follow_redirects=True, timeout=300) as client:
                    print(f"Downloading from: {url}")
                    response = client.get(url)
                    response.raise_for_status()
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                    
                    print(f"Downloaded {len(response.content)} bytes")
                    
                    # Extract the model
                    os.makedirs(model_directory, exist_ok=True)
                    with tarfile.open(tmp_file_path, 'r:gz') as tar:
                        tar.extractall(model_directory)
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    print(f"✅ Successfully downloaded and extracted {name} model")
                    
            except Exception as e:
                print(f"❌ Failed to download {name} model: {e}")
                # Fallback to original method
                return original_download(self, name, model_directory)
        
        # Apply the patch
        GithubModelProvider.download = patched_download
        print("✅ Patched Spleeter to handle GitHub redirects")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not patch Spleeter redirects: {e}")
        return False

def setup_spleeter_with_retry():
    """Setup Spleeter 5stems model only"""
    global spleeter_separator, spleeter_audio_adapter, spleeter_available
    
    try:
        from spleeter.separator import Separator
        from spleeter.audio.adapter import AudioAdapter
        import os
        
        # Patch Spleeter to handle redirects
        patch_spleeter_redirects()
        
        # Set environment variables to help with model download
        os.environ['SPLEETER_MODEL_PATH'] = '/tmp/spleeter_models'
        
        # Create the 5stems separator
        print("Creating Spleeter 5stems separator...")
        spleeter_separator = Separator('spleeter:5stems')
        spleeter_audio_adapter = AudioAdapter.default()
        spleeter_available = True
        print("✅ Spleeter 5stems model loaded successfully!")
        return True
                
    except Exception as e:
        print(f"❌ Failed to load Spleeter 5stems: {e}")
        spleeter_separator = None
        spleeter_audio_adapter = None
        spleeter_available = False
        return False

# Try to setup Spleeter
setup_spleeter_with_retry()

# --- HT-Demucs separation function ---
def separate_with_htdemucs(audio_path, drums=True, bass=True, other=True, vocals=True):
    """
    Separates an audio file using HT-Demucs into drums, bass, other, and vocals.
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, "Please upload an audio file."

    try:
        print(f"HT-Demucs: Loading audio from: {audio_path}")
        
        # Load audio with torchaudio
        wav, sr = torchaudio.load(audio_path)

        if wav.shape[0] == 1:
            print("Audio is mono, converting to stereo.")
            wav = wav.repeat(2, 1)

        wav = wav.to(device)

        print("HT-Demucs: Applying the separation model...")
        with torch.no_grad():
            sources = apply_model(htdemucs_model, wav[None], device=device, progress=True)[0]
        print("HT-Demucs: Separation complete.")

        # Save stems with timestamp to ensure uniqueness
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        output_dir = os.path.join("outputs", f"htdemucs_stems_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        stem_names = ["drums", "bass", "other", "vocals"]
        requested_stems = {
            "drums": drums,
            "bass": bass,
            "other": other,
            "vocals": vocals
        }

        output_paths = []
        for i, name in enumerate(stem_names):
            if requested_stems.get(name, True):
                out_path = os.path.join(output_dir, f"{name}_{timestamp}.wav")
                torchaudio.save(out_path, sources[i].cpu(), sr)
                output_paths.append(out_path)
                print(f"✅ HT-Demucs saved {name} to {out_path}")
            else:
                output_paths.append(None)
                print(f"ℹ️ HT-Demucs skipped saving {name} (not requested)")

        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], "✅ HT-Demucs separation successful!"

    except Exception as e:
        print(f"HT-Demucs Error: {e}")
        return None, None, None, None, f"❌ HT-Demucs Error: {str(e)}"

# --- Spleeter separation function ---
def separate_with_spleeter(audio_path, vocals=True, drums=True, bass=True, other=True, piano=True):
    """
    Separates an audio file using Spleeter into vocals, drums, bass, other, and piano.
    Uses Python API approach from stem_separation_spleeter.py
    Returns FILE PATHS.
    """
    if audio_path is None:
        return None, None, None, None, None, "Please upload an audio file."

    if not spleeter_available or spleeter_separator is None or spleeter_audio_adapter is None:
        return None, None, None, None, None, "❌ Spleeter not available. Please install Spleeter."

    try:
        print(f"Spleeter: Processing audio from: {audio_path}")
        
        # Create output directory with timestamp
        timestamp = int(time.time() * 1000)
        output_dir = os.path.join("outputs", f"spleeter_stems_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load audio using Spleeter's audio adapter (from stem_separation_spleeter.py)
        print("Spleeter: Loading audio...")
        waveform, sample_rate = spleeter_audio_adapter.load(audio_path, sample_rate=44100)
        print(f"Spleeter: Loaded audio - shape: {waveform.shape}, sr: {sample_rate}")
        
        # Perform the separation (from stem_separation_spleeter.py)
        print("Spleeter: Separating audio sources...")
        prediction = spleeter_separator.separate(waveform)
        print("Spleeter: Separation complete.")
        print(f"Spleeter: Prediction keys: {list(prediction.keys())}")
        
        # Save stems with timestamp
        output_paths = []
        stem_names = ["vocals", "drums", "bass", "other", "piano"]
        requested_stems = {
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "other": other,
            "piano": piano
        }
        
        for stem_name in stem_names:
            if stem_name in prediction:
                if requested_stems.get(stem_name, True):
                    out_path = os.path.join(output_dir, f"{stem_name}_{timestamp}.wav")
                    stem_audio = prediction[stem_name]
                    
                    print(f"Spleeter: {stem_name} audio shape: {stem_audio.shape}, dtype: {stem_audio.dtype}")
                    
                    # Save using soundfile for better compatibility
                    sf.write(out_path, stem_audio, sample_rate)
                    output_paths.append(out_path)
                    print(f"✅ Spleeter saved {stem_name} to {out_path}")
                else:
                    output_paths.append(None)
                    print(f"ℹ️ Spleeter skipped saving {stem_name} (not requested)")
            else:
                print(f"⚠️ Warning: {stem_name} not found in prediction")
                output_paths.append(None)
        
        # Ensure we have 5 outputs
        while len(output_paths) < 5:
            output_paths.append(None)

        return output_paths[0], output_paths[1], output_paths[2], output_paths[3], output_paths[4], "✅ Spleeter separation successful!"

    except Exception as e:
        print(f"Spleeter Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, f"❌ Spleeter Error: {str(e)}"

# --- Video to Audio Extraction helper ---
def extract_audio_from_video(video_path):
    """
    Extracts the audio track from a video file using ffmpeg.
    Returns the path to the extracted WAV audio file.
    """
    timestamp = int(time.time() * 1000)
    extracted_audio_path = os.path.join("outputs", f"extracted_audio_{timestamp}.wav")
    
    # Run ffmpeg to extract audio
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                 # Skip video track
        "-acodec", "pcm_s16le", # Convert to PCM WAV
        "-ar", "44100",        # Sample rate
        "-ac", "2",            # 2 channels (stereo)
        extracted_audio_path
    ]
    
    print(f"Extracting audio track from video file: {video_path}")
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        raise Exception(f"Failed to extract audio from video using ffmpeg: {result.stderr}")
        
    print(f"Successfully extracted audio track: {extracted_audio_path}")
    return extracted_audio_path

# --- Combined separation function ---
def separate_selected_models(audio_input_file, run_htdemucs, run_spleeter, vocals=True, drums=True, bass=True, other=True, piano=True):
    """
    Separates an audio or video file using selected models (HT-Demucs, Spleeter, or both).
    Returns stems from selected models.
    """
    if audio_input_file is None:
        return [None] * 9 + ["Please upload an audio or video file."]

    # Handle if audio_input_file is a file object (from Gradio gr.File) or string path
    if hasattr(audio_input_file, "name"):
        audio_path = audio_input_file.name
    else:
        audio_path = audio_input_file

    if not run_htdemucs and not run_spleeter:
        return [None] * 9 + ["❌ Please select at least one model to run."]

    extracted_path = None
    try:
        # Check if the file is a video
        video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.3gp', '.ogg', '.ogv', '.m4v')
        if audio_path.lower().endswith(video_extensions):
            try:
                extracted_path = extract_audio_from_video(audio_path)
                process_path = extracted_path
            except Exception as e:
                return [None] * 9 + [f"❌ Video Audio Extraction Error: {str(e)}"]
        else:
            process_path = audio_path

        htdemucs_results = [None] * 5  # 4 stems + 1 status
        spleeter_results = [None] * 6  # 5 stems + 1 status
        status_messages = []
        
        # Run HT-Demucs if selected
        if run_htdemucs:
            print("Running HT-Demucs...")
            htdemucs_results = separate_with_htdemucs(
                process_path,
                drums=drums,
                bass=bass,
                other=other,
                vocals=vocals
            )
            status_messages.append(htdemucs_results[-1])
        
        # Run Spleeter if selected
        if run_spleeter:
            print("Running Spleeter...")
            spleeter_results = separate_with_spleeter(
                process_path,
                vocals=vocals,
                drums=drums,
                bass=bass,
                other=other,
                piano=piano
            )
            status_messages.append(spleeter_results[-1])
        
        # Combine results: HT-Demucs (4 stems) + Spleeter (5 stems)
        all_results = list(htdemucs_results[:-1]) + list(spleeter_results[:-1])
        
        # Create combined status message
        models_used = []
        if run_htdemucs:
            models_used.append("HT-Demucs")
        if run_spleeter:
            models_used.append("Spleeter")
        
        combined_status = f"🎵 {' + '.join(models_used)} completed!\n\n" + "\n".join(status_messages)
        
        return all_results + [combined_status]

    except Exception as e:
        print(f"Combined Error: {e}")
        import traceback
        traceback.print_exc()
        return [None] * 9 + [f"❌ Error: {str(e)}"]
    finally:
        # Clean up temporary extracted WAV audio file if it was created
        if extracted_path and os.path.exists(extracted_path):
            try:
                os.remove(extracted_path)
                print(f"Cleaned up temporary extracted audio file: {extracted_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {extracted_path}: {e}")

# --- Gradio UI ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Spleeter & Demucs - Now Both Work!
    
    **Follow me on:** [ Hugging Face @ahk-d](https://huggingface.co/ahk-d) | [ GitHub @ahk-d](https://github.com/ahk-d)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.File(
                label="🎵 Upload Audio or Video File",
                file_count="single",
                type="file",
                file_types=["audio", "video"]
            )
            
            # Model selection toggles
            gr.Markdown("### 🎛️ Select Models to Run")
            with gr.Row():
                htdemucs_toggle = gr.Checkbox(label="🎯 HT-Demucs", value=True, info="Drums, Bass, Other, Vocals")
                spleeter_label = "🎵 Spleeter 2025 (5stems)" if spleeter_available else "🎵 Spleeter 2025"
                spleeter_info = "Vocals, Drums, Bass, Other, Piano" if spleeter_available else "5stems model not available"
                spleeter_toggle = gr.Checkbox(
                    label=spleeter_label, 
                    value=spleeter_available, 
                    info=spleeter_info,
                    interactive=spleeter_available
                )
            
            separate_button = gr.Button("🚀 Separate Music", variant="primary", size="lg")
            status_output = gr.Textbox(label="📊 Status", interactive=False, lines=4)

    gr.Markdown("---")

    with gr.Row():
        # HT-Demucs Results
        with gr.Column():
            gr.Markdown("### 🎯 HT-Demucs Results")
            with gr.Row():
                htdemucs_drums = gr.Audio(label="🥁 Drums", type="filepath")
                htdemucs_bass = gr.Audio(label="🎸 Bass", type="filepath")
            with gr.Row():
                htdemucs_other = gr.Audio(label="🎼 Other", type="filepath")
                htdemucs_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
        
        # Spleeter Results
        with gr.Column():
            gr.Markdown("### 🎵 Spleeter 2025 Results")
            with gr.Row():
                spleeter_vocals = gr.Audio(label="🎤 Vocals", type="filepath")
                spleeter_drums = gr.Audio(label="🥁 Drums", type="filepath")
            with gr.Row():
                spleeter_bass = gr.Audio(label="🎸 Bass", type="filepath")
                spleeter_other = gr.Audio(label="🎼 Other", type="filepath")
            with gr.Row():
                spleeter_piano = gr.Audio(label="🎹 Piano", type="filepath")
            
            if spleeter_available:
                gr.Markdown("*5stems model: Vocals, Drums, Bass, Other, Piano*")
            else:
                gr.Markdown("*Note: Spleeter 5stems model not available*")

    gr.Markdown("---")
    
    with gr.Row():
        comparison_text = f"""
        ### 📋 Model Comparison
        
        | Feature | HT-Demucs | Spleeter 2025 (5stems) |
        |---------|-----------|----------|
        | **Vocals** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Drums** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Bass** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Other** | ✅ High Quality | {'✅ Available' if spleeter_available else '❌ N/A'} |
        | **Piano** | ❌ Not Available | {'✅ **Available**' if spleeter_available else '❌ N/A'} |
        | **Speed** | ⚡ Fast | {'⚡ Fast' if spleeter_available else '❌ N/A'} |
        | **Quality** | 🏆 Excellent | {'🏆 Good' if spleeter_available else '❌ N/A'} |
        
        **💡 Tip:** Use Spleeter 2025 for piano separation, HT-Demucs for other instruments!
        """
        gr.Markdown(comparison_text)

    # Connect the button to the combined function
    separate_button.click(
        fn=separate_selected_models,
        inputs=[audio_input, htdemucs_toggle, spleeter_toggle],
        outputs=[
            htdemucs_drums, htdemucs_bass, htdemucs_other, htdemucs_vocals,  # HT-Demucs outputs
            spleeter_vocals, spleeter_drums, spleeter_bass, spleeter_other, spleeter_piano,  # Spleeter outputs
            status_output  # Status output
        ]
    )

    gr.Markdown("""
    ---
    <p style='text-align: center; font-size: small;'>
    🚀 Powered by <strong>HT-Demucs</strong> & <strong>Spleeter 2025</strong> | 
    🎵 Compare and choose your best stems!
    </p>
    """)

import fastapi
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uvicorn

# Create outputs directory
os.makedirs("outputs", exist_ok=True)

app = fastapi.FastAPI(title="Stem Separation API")

# Mount outputs folder to serve files statically
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/api/separate")
async def api_separate(
    request: Request,
    audio: UploadFile = File(...),
    run_htdemucs: bool = Form(True),
    run_spleeter: bool = Form(True),
    vocals: bool = Form(True),
    drums: bool = Form(True),
    bass: bool = Form(True),
    piano: bool = Form(True),
    other: bool = Form(True)
):
    try:
        # Save uploaded file temporarily
        temp_audio_path = os.path.join("outputs", f"temp_upload_{int(time.time() * 1000)}_{audio.filename}")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Run separation
        results = separate_selected_models(
            temp_audio_path,
            run_htdemucs,
            run_spleeter,
            vocals=vocals,
            drums=drums,
            bass=bass,
            other=other,
            piano=piano
        )
        # Check if an error occurred in separation
        status_message = results[-1]
        if status_message.startswith("❌") or "error" in status_message.lower() or "please upload" in status_message.lower():
            # Clean up temp upload
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
            return JSONResponse(status_code=400, content={"status": "error", "message": status_message})

        # Build base URL for constructing full URLs to output files
        base_url = str(request.base_url).rstrip("/")
        
        # Helper to convert local path to URL
        def path_to_url(path):
            if path is None:
                return None
            # Convert Windows backslashes to forward slashes for URLs
            normalized_path = path.replace("\\", "/")
            return f"{base_url}/{normalized_path}"

        stems_dict = {
            "htdemucs": {
                "drums": path_to_url(results[0]),
                "bass": path_to_url(results[1]),
                "other": path_to_url(results[2]),
                "vocals": path_to_url(results[3]),
            } if run_htdemucs else None,
            "spleeter": {
                "vocals": path_to_url(results[4]),
                "drums": path_to_url(results[5]),
                "bass": path_to_url(results[6]),
                "other": path_to_url(results[7]),
                "piano": path_to_url(results[8]),
            } if run_spleeter else None
        }

        # Helper function to recursively remove None/null values
        def clean_nones(val):
            if isinstance(val, dict):
                cleaned = {k: clean_nones(v) for k, v in val.items() if v is not None}
                return cleaned if cleaned else None
            return val

        cleaned_stems = clean_nones(stems_dict)

        response_data = {
            "status": "success",
            "message": status_message,
            "stems": cleaned_stems if cleaned_stems is not None else {}
        }
        
        # Clean up temp upload
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# Mount the Gradio UI onto the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)