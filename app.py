import os
import torch
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import hf_hub_download
import re

# 1. हगिंग फेस से मॉडल डाउनलोड (ऑटोमेटिक)
# यह हिस्सा पुराने कोड के ऊपर रहेगा
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"  # आपका 1000 Epochs वाला नया मॉडल
INDEX_FILE = "added_IVF759_Flat_nprobe_Ramai_Shri_Ram_Voice_Training.index" # पूरा नाम यहाँ लिखें

print("⏳ हगिंग फेस से नया मॉडल और इंडेक्स फाइल डाउनलोड हो रही है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
index_path = hf_hub_download(repo_id=REPO_ID, filename=INDEX_FILE)
print(f"✅ मॉडल डाउनलोड सफल: {model_path}")

# 2. टर्बो स्टार्टअप और एग्रीमेंट
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_hindi_text(text):
    # यह दूसरी भाषा (चीनी/जैपनीज) बोलने से रोकता है
    return re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)

def generate_voice(text, voice_sample, remove_silence):
    pure_text = clean_hindi_text(text)
    output_path = "final_output.wav"
    
    # शुद्ध हिंदी और हकलाना रोकने के लिए फिक्स
    tts.tts_to_file(
        text=pure_text, 
        speaker_wav=voice_sample, 
        language="hi",
        file_path=output_path,
        split_sentences=True # हकलाने का इलाज
    )
    
    # सन्नाटा हटाना (Silence Remover)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_final.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# 3. इंटरफ़ेस (Dark Mode)
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    demo.load(None, None, None, _js="() => { document.body.classList.add('dark'); }")
    gr.Markdown("# 🎙️ **श्रीराम वाणी - शुद्ध हिंदी टर्बो (v2)**")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", value="नमस्ते, मैं अब शुद्ध हिंदी बोलूँगा।")
            audio_input = gr.Audio(label="अपनी आवाज़ का सैंपल (.wav)", type="filepath")
            silence_btn = gr.Checkbox(label="सन्नाटा हटाएँ (Silence Remover)", value=True)
            btn = gr.Button("🚀 आवाज़ उत्पन्न करें", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="आउटपुट")

    btn.click(generate_voice, [input_text, audio_input, silence_btn], audio_out)

if __name__ == "__main__":
    demo.launch(share=True)
