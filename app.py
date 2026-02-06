import gradio as gr
from TTS.api import TTS
import torch

# मॉडल लोड (GPU न होने पर CPU पर चलेगा)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, language_strict):
    output_path = "final_output.wav"
    
    # भाषा की गड़बड़ी रोकने के लिए 'Strict' मोड
    lang = "hi" if language_strict else "en"
    
    tts.tts_to_file(
        text=text,
        speaker_wav=voice_sample,
        language=lang,
        file_path=output_path,
        speed=speed,         # स्पीड कंट्रोल
        pitch=pitch          # पिच कंट्रोल (आवाज़ मोटी या पतली करने के लिए)
    )
    return output_path

# --- UI Layout (Dark Mode Enabled) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", dark_mode=True)) as demo:
    gr.Markdown("# 🎙️ AI Voice Box - Turbo v2 (Pitch & Speed Control)")
    
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="हिंदी टेक्स्ट लिखें", placeholder="यहाँ अपना संदेश लिखें...")
            audio_ref = gr.Audio(label="वॉइस सैंपल (.wav)", type="filepath")
            
            # नए कंट्रोल्स
            speed_slider = gr.Slider(0.5, 2.0, value=1.0, label="Speed (गति)")
            pitch_slider = gr.Slider(-10, 10, value=0, label="Pitch (आवाज़ का भारीपन)")
            lang_fix = gr.Checkbox(label="Strict Hindi Mode (दूसरी भाषा रोकने के लिए)", value=True)
            
            btn = gr.Button("🚀 Generate Voice", variant="primary")
            
        with gr.Column():
            audio_out = gr.Audio(label="आपका आउटपुट")

    btn.click(generate_voice, [txt, audio_ref, speed_slider, pitch_slider, lang_fix], audio_out)

demo.launch(share=True)
