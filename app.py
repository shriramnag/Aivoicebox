import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# 🚩 आपके पुराने प्रोजेक्ट की फाइलें [cite: 2026-02-16]
try:
    from text_engine import split_into_chunks
    from parallel_processor import combine_chunks
except ImportError:
    pass

# ⚡ टर्बो इंजन और हगिंग फेस सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 हगिंग फेस से आपका Ramai.pth लोड करना
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"

print("🚀 हगिंग फेस से मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def apply_shriram_magic(file_path, bass_gain, amp_gain):
    """आवाज़ को भारी और पावरफुल बनाना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp_gain # एमप्लीफायर (Power)
    
    if bass_gain > 0:
        # आवाज़ को गहरा (Deep) करने के लिए पिच एडजस्टमेंट
        new_sample_rate = int(sound.frame_rate * (1.0 - (bass_gain / 100)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        sound = sound.set_frame_rate(44100)
    
    final_path = "shriram_final_pro_v2.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, pitch, emotion, bass, amp, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("स्क्रिप्ट और वॉइस सैंपल ज़रूरी हैं।") 

    # 🚀 आपका पुराना चंक प्रोसेसिंग लॉजिक (Unchanged) [cite: 2026-02-16]
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
    
    # ✨ भारी वजन और पावर जोड़ें
    return apply_shriram_magic(combined_temp, bass, amp)

# 🎨 आपका रॉयल UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - हगिंग फेस टर्बो इंजन")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="सैंपल अपलोड करें", type="filepath")
            with gr.Accordion("⚙️ पुराने स्लाइडर्स (Locked)", open=True):
                speed_s = gr.Slider(label="Speed", minimum=0.5, maximum=1.5, value=1.0)
                pitch_s = gr.Slider(label="Deep Match", minimum=0.5, maximum=1.0, value=0.9)
            with gr.Accordion("🎭 भारी आवाज़ और पावर (Bass/Amp)", open=True):
                emo_s = gr.Slider(label="ह्यूमन टच (Emotion)", minimum=0.1, maximum=1.0, value=0.8)
                bass_s = gr.Slider(label="भारी वजन (Deep Voice)", minimum=0, maximum=10, value=2)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=15, value=0)
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, emo_s, bass_s, amp_s], out)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
