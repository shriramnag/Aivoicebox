import os
import torch
import gradio as gr
import shutil
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain  # आपके brain.py से कनेक्शन

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (Ramai.pth) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन लोड करना
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 
    'hindi_grammar.json', 
    'english_knowledge.json', 
    'prosody_config.json'
)

def split_into_chunks(text):
    """वर्किंग चंकिंग लॉजिक [cite: 2026-02-16]"""
    sentences = re.split('([।!?॥])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        if len(current_chunk) + len(sentence) < 250:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def apply_final_mastering(file_path, amp, pitch_val):
    """इको कम किया गया (-42dB) और क्रिस्टल क्लियर क्लैरिटी [cite: 2026-01-06]"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # ✅ इको सुधार: इसे और कम कर दिया गया है ताकि गूँज न आए
    echo = sound - 42 
    sound = sound.overlay(echo, position=180) 
    
    sound = sound.low_pass_filter(4000)
    final_path = "shriram_perfect_output.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, progress=gr.Progress()):
    # 1. ब्रेन से टेक्स्ट को शुद्ध करना (संस्कृत/हिंदी/इंग्लिश)
    cleaned_text = brain.clean_and_format(text)
    
    # 2. सही प्रोफाइल चुनना (श्लोक मोड या टॉकिंग)
    profile = brain.get_voice_profile(text)
    
    chunks = split_into_chunks(cleaned_text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc="🚀 दिव्य वाणी जनरेट हो रही है...")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🧠 ब्रेन से मिली स्पीड और पेनाल्टी का उपयोग
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=profile['global_speed'], 
            repetition_penalty=15.0, # हकलाहट रोकने के लिए फिक्स्ड
            temperature=0.75, top_p=0.85
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    # मास्टरिंग (पिच 0.96 पर सेट है)
    return apply_final_mastering("combined.wav", 4, profile['global_pitch'])

# 🎨 UI डिजाइन (No Changes to Working Features) [cite: 2026-01-06]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - महाज्ञानी वर्जन (इको फिक्स्ड)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ श्लोक या स्क्रिप्ट लिखें", lines=12, placeholder="उदा: कर्मण्येवाधिकारस्ते...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            btn = gr.Button("दिव्य आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="शुद्ध और साफ़ आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref], out)

demo.launch(share=True)
