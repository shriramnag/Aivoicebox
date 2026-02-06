import gradio as gr
from TTS.api import TTS
import torch
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 1. मॉडल सेटअप और ऑटो-एग्रीमेंट
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 श्रीराम वॉयस टर्बो इंजन लोड हो रहा है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, remove_silence):
    output_path = "output.wav"
    
    # पिच एरर को ठीक करने के लिए 'pitch' को हटा दिया गया है
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language="hi",
        file_path=output_path,
        speed=speed
    )
    
    # सन्नाटा हटाना (Silence Remover - Working Smooth)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "final_clean_voice.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# 2. डार्क मोड और यूआई डिज़ाइन
custom_css = "body { background-color: #121212 !important; color: white !important; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), css=custom_css) as demo:
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - श्रीराम वाणी (Turbo)**")
    
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", placeholder="जय श्री गणेश...")
            audio_ref = gr.Audio(label="वॉइस सैंपल अपलोड करें (.wav)", type="filepath")
            speed_slider = gr.Slider(0.5, 2.0, value=1.0, label="गति (Speed)")
            silence_btn = gr.Checkbox(label="Silence Remover (सन्नाटा हटाएँ)", value=True)
            submit = gr.Button("🚀 Generate Perfect Voice", variant="primary")
        
        with gr.Column():
            out = gr.Audio(label="आपका फाइनल ऑडियो")

    submit.click(generate_voice, [txt, audio_ref, speed_slider, silence_btn], out)

if __name__ == "__main__":
    demo.launch(share=True)
