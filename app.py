import os
import torch
import gradio as gr
import shutil
import time
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो हाई स्पीड कॉन्फ़िगरेशन [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
# GPU (T4) की पूरी शक्ति का इस्तेमाल सुनिश्चित करना
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True # 🚀 प्रोसेसिंग स्पीड बढ़ाने के लिए

# 📥 100% मैच इंजन लोड (Fast Load)
print(f"⏳ श्रीराम टर्बो इंजन लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    # 🧹 ऑटो-क्लीनर (प्लेयर एरर हटाने के लिए)
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # ⚡ हाई स्पीड चंकिंग (10K कैरेक्टर सपोर्ट)
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    # 🎙️ टर्बो जनरेशन लूप
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=10.0,   # हकलाहट फिक्स
            temperature=pitch,         # 100% मैच (0.80 बेस्ट है)
            top_p=0.85,                
            top_k=50,                  
            enable_text_splitting=False # मैन्युअल स्प्लिटिंग पहले से है, इसे False रखने से रफ़्तार बढ़ती है
        )
        chunk_files.append(name)
    
    # ⚡ हाई स्पीड मर्जिंग (टुकड़ों को बिजली की रफ़्तार से जोड़ना)
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    
    return final_output

# 🎨 श्रीराम वाणी UI (टर्बो लुक)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 100% मैच टर्बो इंजन")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            # सैंपल प्लेयर (फास्ट लोड मोड)
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            # 🎚️ टर्बो कंट्रोल्स
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="आवाज़ की गहराई (Deep Match)", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें (High Speed)", variant="primary")
            
    with gr.Row():
        # फाइनल प्लेयर (ऑटो-रिफ्रेश)
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
