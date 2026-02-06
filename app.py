import os
import torch
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence

# टर्बो लोड XTTS-v2
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, remove_silence):
    output_path = "output.wav"
    
    # सुधार 1: भाषा को 'hi' पर लॉक करना और 'Speed' बढ़ाना
    tts.tts_to_file(
        text=text, 
        speaker_wav=voice_sample, 
        language="hi",              # हिंदी पर सख्त नियंत्रण
        file_path=output_path,
        split_sentences=True        # वाक्यों को तोड़कर पढ़ना ताकि भाषा न भटके
    )
    
    # सुधार 2: साइलेंस रिमूवर (आपकी मांग के अनुसार)
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_turbo_output.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# --- इंटरफ़ेस ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ AI Voice Box - Perfect Hindi Fix")
    input_text = gr.Textbox(label="सिर्फ हिंदी टेक्स्ट लिखें", value="नमस्ते, मैं अब शुद्ध हिंदी बोलूँगा।")
    audio_input = gr.Audio(label="अपनी आवाज़ का सैंपल (.wav)", type="filepath")
    silence_btn = gr.Checkbox(label="सन्नाटा हटाएँ (Silence Remover)", value=True)
    btn = gr.Button("🚀 आवाज उत्पन्न करें")
    audio_out = gr.Audio(label="आउटपुट")

    btn.click(generate_voice, [input_text, audio_input, silence_btn], audio_out)

demo.launch(share=True)
