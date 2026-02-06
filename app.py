import os
import torch
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- कॉन्फ़िगरेशन (Turbo Settings) ---
MODEL_LINK = "https://huggingface.co/Shriramnag/%E0%A4%AE%E0%A4%BE%E0%A4%80%E0%A4%88-%E0%A4%B6%E0%A5%8D%E0%A4%B0%E0%A5%80%E0%A4%B0%E0%A4%BE%E0%A4%AE-%E0%A4%B5%E0%A5%89%E0%A4%87%E0%A4%B8/resolve/main/Shriramoriginalvoice.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- मॉडल डाउनलोड और सेटअप ---
def setup_model():
    model_path = "/content/models/shriram.pth"
    if not os.path.exists(model_path):
        os.makedirs("/content/models", exist_ok=True)
        print("⚡ हगिंग फेस से मॉडल लोड हो रहा है...")
        os.system(f"wget -c {MODEL_LINK} -O {model_path}")
    return model_path

# --- वॉयस क्लोनिंग फंक्शन ---
def generate_voice(text, voice_sample, remove_silence):
    model_path = setup_model()
    # टर्बो लोड XTTS-v2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    output_path = "output.wav"
    tts.tts_to_file(text=text, speaker_wav=voice_sample, language="hi", file_path=output_path)
    
    # साइलेंस रिमूवर लॉजिक
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_output.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# --- Gradio UI (Interface) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ AI Voice Box - Turbo Hindi Clone")
    gr.Markdown("हगिंग फेस मॉडल और टर्बो स्पीड के साथ अपनी आवाज़ क्लोन करें।")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", placeholder="नमस्ते, आप कैसे हैं?")
            audio_input = gr.Audio(label="अपनी आवाज़ का सैंपल अपलोड करें", type="filepath")
            silence_btn = gr.Checkbox(label="Silence Remover (सन्नाटा हटाएँ)", value=True)
            submit_btn = gr.Button("🚀 Generate Voice (Turbo Mode)", variant="primary")
        
        with gr.Column():
            audio_output = gr.Audio(label="आपका क्लोन किया हुआ ऑडियो")

    submit_btn.click(
        fn=generate_voice, 
        inputs=[input_text, audio_input, silence_btn], 
        outputs=audio_output
    )

# कोलाब के लिए शेयर लिंक चालू करें
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
  
