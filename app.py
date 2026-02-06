import gradio as gr
from TTS.api import TTS
import torch
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 1. मॉडल और एग्रीमेंट सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 टर्बो मॉडल लोड हो रहा है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, remove_silence):
    output_path = "output.wav"
    
    # वॉयस जनरेशन
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",
        file_path=output_path,
        speed=speed,
        pitch=pitch
    )
    
    # सन्नाटा हटाना (Silence Remover)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "final_clean.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# 2. डार्क मोड के लिए कस्टम CSS
custom_css = """
body { background-color: #121212 !important; color: white !important; }
.gradio-container { background-color: #121212 !important; }
"""

# 3. इंटरफ़ेस डिज़ाइन
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - टर्बो हाई स्पीड**")
    
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", placeholder="नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?")
            audio_ref = gr.Audio(label="अपना वॉयस सैंपल दें", type="filepath")
            
            with gr.Row():
                speed_s = gr.Slider(0.5, 2.0, value=1.0, label="गति (Speed)")
                pitch_s = gr.Slider(-10, 10, value=0, label="पिच (Pitch)")
            
            silence_btn = gr.Checkbox(label="Silence Remover बटन", value=True)
            submit = gr.Button("🚀 Generate Voice", variant="primary")
        
        with gr.Column():
            out = gr.Audio(label="आउटपुट ऑडियो")

    submit.click(generate_voice, [txt, audio_ref, speed_s, pitch_s, silence_btn], out)

# बिना किसी 'dark_mode' आर्गुमेंट के लॉन्च करें
if __name__ == "__main__":
    demo.launch(share=True)
