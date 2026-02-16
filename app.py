import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download

# ✅ आपके प्रोजेक्ट की फाइलों से इम्पोर्ट
try:
    from app_config import MODEL_CONFIG
    from text_engine import split_into_chunks
    from parallel_processor import combine_chunks
except ImportError:
    print("⚠️ कुछ सहायक फाइलें नहीं मिलीं, कृपया सुनिश्चित करें कि brain.py और अन्य फाइलें मौजूद हैं।")

# ⚡ टर्बो और पाथ सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 इंजन लोड (हगिंग फेस से सीधे आपके रिपो के अनुसार)
print("🚀 श्रीराम वाणी: 100% मैच इंजन लोड हो रहा है...")
try:
    # आपके स्क्रीनशॉट के अनुसार सही रिपो आईडी
    REPO_ID = "Shriramnag/My-Shriram-Voice" 
    MODEL_FILE = "Ramai.pth"
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
except Exception as e:
    print(f"❌ लोड एरर: {e}")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 

    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # लंबी स्क्रिप्ट को छोटे हिस्सों में बांटना [cite: 2026-01-06]
    chunks = split_into_chunks(text) 
    chunk_files = []

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")

        # 🎙️ मास्टर सेटिंग्स: Deep Match और Pitch
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=10.0,   
            temperature=pitch,         
            top_p=0.85,                
            top_k=50,                  
            enable_text_splitting=False 
        )
        chunk_files.append(name)

    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 आपका पुराना शानदार UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 100% मैच टर्बो इंजन")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            ref = gr.Audio(
                label="वॉइस सैंपल यहाँ अपलोड करें", 
                type="filepath", 
                interactive=True,
                visible=True
            )
            
            # स्लाइडर्स जो आपको पसंद थे
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="आवाज़ की गहराई (Deep Match)", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    # पब्लिक लिंक के लिए share=True
    demo.launch(share=True, debug=True, allowed_paths=[os.getcwd(), "/content/"])
