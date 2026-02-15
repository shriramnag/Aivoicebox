import os
import torch  # फिक्स: NameError दूर करने के लिए
import re
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import hf_hub_download

# टर्बो सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# मॉडल डाउनलोड (v2 - 1000 Epochs)
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

# TTS लोड करें
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_text(text):
    # सिर्फ हिंदी अक्षरों को रहने दें (दूसरी भाषा रोकने के लिए)
    return re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)

def generate_voice(text, voice_sample, remove_silence):
    pure_text = clean_text(text)
    output_path = "shriram_final.wav"
    
    tts.tts_to_file(
        text=pure_text, 
        speaker_wav=voice_sample, 
        language="hi",              # हिंदी लॉक
        file_path=output_path,
        split_sentences=True        # हकलाना बंद
    )
    
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks: combined += chunk
        output_path = "clean_turbo.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# इंटरफ़ेस
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - Final Fix")
    input_text = gr.Textbox(label="हिंदी लिखें", value="नमस्ते, अब आवाज़ साफ़ आएगी।")
    audio_ref = gr.Audio(label="वॉइस सैंपल", type="filepath")
    silence_chk = gr.Checkbox(label="सन्नाटा हटाएँ", value=True)
    btn = gr.Button("🚀 आवाज़ बनाएँ", variant="primary")
    audio_out = gr.Audio(label="आउटपुट")
    btn.click(generate_voice, [input_text, audio_ref, silence_chk], audio_out)

demo.launch(share=True)
