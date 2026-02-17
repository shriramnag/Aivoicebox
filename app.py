import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from pydub import AudioSegment

# 🚩 पुराने प्रोजेक्ट की फाइलें इम्पोर्ट
try:
    from text_engine import split_into_chunks
    from parallel_processor import combine_chunks
except ImportError:
    pass

# ⚡ टर्बो इंजन सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def apply_shriram_magic(file_path, bass_gain, amp_gain):
    """आवाज़ में भारीपन और पावर जोड़ना (बिना एरर के)"""
    sound = AudioSegment.from_wav(file_path)
    
    # एमप्लीफायर (Power)
    sound = sound + amp_gain
    
    # भारीपन (Bass) के लिए पिच को थोड़ा सा कम करना (Deep Voice Logic)
    if bass_gain > 0:
        new_sample_rate = int(sound.frame_rate * (1.0 - (bass_gain / 100)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        sound = sound.set_frame_rate(44100)
    
    final_path = "shriram_final_pro_v2.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, pitch, emotion, bass, amp, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("स्क्रिप्ट और सैंपल दें।") 

    # 🚀 आपका पुराना चंक प्रोसेसिंग (Unchanged) [cite: 2026-02-16]
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    chunks = split_into_chunks(text) 
    chunk_files = []

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")

        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=12.0, temperature=emotion,
            top_p=0.85, gpt_cond_len=3
        )
        chunk_files.append(name)

    combined_temp = "combined_temp.wav"
    combine_chunks(chunk_files, output_file=combined_temp)
    
    # ✨ भारीपन और एमप्लीफायर जोड़ें
    return apply_shriram_magic(combined_temp, bass, amp)

# 🎨 रॉयल UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - भारी आवाज़ स्टूडियो")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="सैंपल", type="filepath")
            with gr.Accordion("⚙️ पुराने स्लाइडर्स", open=True):
                speed_s = gr.Slider(label="Speed", minimum=0.5, maximum=1.5, value=1.0)
                pitch_s = gr.Slider(label="Deep Match", minimum=0.5, maximum=1.0, value=0.9)
            with gr.Accordion("🎭 भारीपन और पावर (Bass/Amp)", open=True):
                emo_s = gr.Slider(label="ह्यूमन टच (Emotion)", minimum=0.1, maximum=1.0, value=0.8)
                bass_s = gr.Slider(label="भारी वजन (Deep Voice)", minimum=0, maximum=10, value=2)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=15, value=0)
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    out = gr.Audio(label="आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, emo_s, bass_s, amp_s], out)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
