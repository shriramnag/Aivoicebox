import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (CPU से हटाकर पूर्णतः GPU/T4 पर) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
# टर्बो बूस्ट के लिए CUDA सेटिंग्स
torch.backends.cudnn.benchmark = True 
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल इंटीग्रेशन (Hugging Face) [cite: 2026-02-26]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 

print("श्री राम नाग जी, टर्बो इंजन को शुरू किया जा रहा है...")
# आवश्यक ONNX और Config फाइल्स को तेज़ लोड के लिए डाउनलोड करना
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट रोकने के लिए मास्टर टेक्स्ट क्लीनर [cite: 2026-02-20]
def shiv_super_cleaner(text):
    if not text: return ""
    # नंबर फिक्स (शब्दों में) [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # ⚡ स्क्रिप्ट कटर अपडेट: अनचाहे शोर को रोकने के लिए विराम चिन्हों का प्रबंधन
    text = text.replace('.', ',').replace('?', ',').replace('!', ',')
    brain_fix = {"जिंदगी": "ज़िन्दगी", "YouTube": "यूट्यूब", "AI": "ए आई"}
    for k, v in brain_fix.items(): text = text.replace(k, v)
    return text.strip()

# ४. मुख्य इंजन - टर्बो हाई स्पीड + स्क्रिप्ट कटर (LOCKED) [cite: 2026-01-06]
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ अपडेटेड स्क्रिप्ट कटर: वाक्यों को संतुलित लंबाई में काटना ताकि स्पीड बनी रहे
    chunks = [c.strip() for c in re.split(r'[,।॥\n]', p_text) if len(c.strip()) > 1]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"टर्बो स्पीड जनरेशन: भाग {i+1}")
        
        name = f"turbo_chunk_{i}.wav"
        # 🔒 शोर कम करने के लिए Temperature को 0.05 पर सेट किया गया है (Most Stable) [cite: 2026-02-20]
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=1.5, temperature=0.05, top_k=1)
        
        seg = AudioSegment.from_wav(name)
        
        # पिच कंट्रोल
        if pitch_s != 1.0:
            new_rate = int(seg.frame_rate * pitch_s)
            seg = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)

        if use_silence: # साइलेंस रिमूवर (शोर हटाने के लिए थ्रेशोल्ड अपडेटेड) [cite: 2026-01-06]
            try: seg = effects.strip_silence(seg, silence_thresh=-50, padding=100)
            except: pass
            
        combined += seg
        os.remove(name)
        # GPU मेमोरी खाली करना ताकि स्पीड बनी रहे
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: # सिमेट्री क्लीन
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    final_p = "Shiv_AI_v1.5_Turbo_Final.wav"
    combined.export(final_p, format="wav")
    return final_p

# ५. दिव्य UI (टर्बो बटन के साथ) [cite: 2026-02-20]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 टर्बो हाई स्पीड | स्क्रिप्ट कटर | ऑडियो शोर फिक्स [cite: 2026-01-06]")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="यहाँ टाइप करें...")
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ टर्बो सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.8, 1.4, 1.15, label="टर्बो रफ़्तार")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच (Pitch)")
                cln = gr.Checkbox(label="Symmetry Clean (शोर फिक्स)", value=True)
                sln = gr.Checkbox(label="Silence Remover", value=True)
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    out = gr.Audio(label="शिव एआई आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True)
