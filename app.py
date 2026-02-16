import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो हाई स्पीड [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 आपका 1000 Epoch वाला खास मॉडल (Hugging Face से सीधा डाउनलोड) [cite: 2026-01-03]
print(f"⏳ आपका 1000 Epoch वाला मॉडल लोड हो रहा है...")
# यहाँ MODEL_CONFIG से सीधा आपके मॉडल की फाइल उठाई जाएगी
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
config_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename="config.json")

# आपके स्पेसिफिक मॉडल को लोड करना (बिना कोड बदले) [cite: 2026-01-06]
tts = TTS(model_path=model_path, config_path=config_path).to(device)

def generate_voice(text, voice_sample, speed, pitch, emotion_scale, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        # 🎙️ 1000% मैच सेटिंग्स (1000 Epoch मॉडल के लिए) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=15.0,   # हकलाहट रोकने के लिए [cite: 2026-01-06]
            temperature=0.65,          # Purity के लिए [cite: 2026-01-06]
            top_p=0.80,                
            top_k=25,                  
            enable_text_splitting=False 
        )
        chunk_files.append(name)
    
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 UI (आपका ओरिजिनल लेआउट)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 1000 Epoch टर्बो इंजन")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ स्क्रिप्ट लिखें", lines=12)
        with gr.Column(scale=1):
            # ✅ वॉइस सैंपल अपलोड फिक्स [cite: 2026-01-06]
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            speed_slider = gr.Slider(label="स्पीड", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="Deep Match", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            emotion_slider = gr.Slider(label="साँस/इमोशन", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider, emotion_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
