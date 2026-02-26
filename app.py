import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल इंटीग्रेशन (Hugging Face स्क्रीनशॉट के अनुसार) [cite: 2026-02-26]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 

print("श्री राम नाग जी, ब्रह्मास्त्र लोड हो रहा है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
# स्क्रीनशॉट में दिख रही ज़रूरी फाइलों का लोड सुनिश्चित करना
for f in ["config.json", "tokenizer.json", "speech_encoder.onnx"]:
    hf_hub_download(repo_id=REPO_ID, filename=f)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट रोकने के लिए मास्टर टेक्स्ट क्लीनर [cite: 2026-02-20]
def shiv_super_cleaner(text):
    if not text: return ""
    # नंबर फिक्स (शब्दों में) [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # डॉट को कोमा बनाना ताकि AI न हकलाए [cite: 2026-02-20]
    text = text.replace('.', ',')
    brain_fix = {"जिंदगी": "ज़िन्दगी", "YouTube": "यूट्यूब", "AI": "ए आई"}
    for k, v in brain_fix.items(): text = text.replace(k, v)
    return text.strip()

# ४. मुख्य इंजन - ऑडियो फिक्स और पिच कंट्रोल (LOCKED) [cite: 2026-01-06]
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ वाक्यों का सटीक विभाजन (ताकि ऑडियो न कटे) [cite: 2026-02-20]
    chunks = re.split(r'([,।!?॥\n])', p_text)
    combined = AudioSegment.empty()
    
    valid_chunks = []
    temp_chunk = ""
    for c in chunks:
        if c in [",", "।", "!", "?", "॥", "\n"]:
            valid_chunks.append(temp_chunk + c)
            temp_chunk = ""
        else: temp_chunk += c
    if temp_chunk: valid_chunks.append(temp_chunk)

    for i, chunk in enumerate(valid_chunks):
        if len(chunk.strip()) < 2: continue
        progress((i+1)/len(valid_chunks), desc="शिव एआई शुद्ध आवाज़ बना रहा है...")
        
        if "[pause]" in chunk: combined += AudioSegment.silent(duration=800); continue
        
        name = f"chunk_{i}.wav"
        # 🔒 XTTS सटीक सेटिंग्स (Temperature 0.1 और Top_k 1 से हकलाहट बंद होती है) [cite: 2026-02-20]
        tts.tts_to_file(text=chunk.strip(), speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=1.2, temperature=0.1, top_k=1)
        
        seg = AudioSegment.from_wav(name)
        
        # पिच (Pitch) फिक्स टूल
        if pitch_s != 1.0:
            new_rate = int(seg.frame_rate * pitch_s)
            seg = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)

        if use_silence: # साइलेंस रिमूवर (पैडिंग बढ़ाई गई है ताकि आवाज़ साफ़ रहे) [cite: 2026-01-06]
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=200)
            except: pass
            
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: # एआई बेस सफाई (Symmetry Clean) [cite: 2026-01-06]
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    final_p = "Shiv_AI_v1.5_Fixed.wav"
    combined.export(final_p, format="wav")
    return final_p

# ५. दिव्य UI (वर्ड काउंटर और पिच कंट्रोल के साथ) [cite: 2026-02-20]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 टर्बो हाई स्पीड | ऑडियो फिक्स | हकलाहट मुक्त [cite: 2026-01-06]")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ टूल्स और सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.8, 1.4, 1.15, label="रफ़्तार (Speed)")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच (Pitch - आवाज़ भारी/पतली)")
                cln = gr.Checkbox(label="Symmetry Clean (सफाई)", value=True)
                sln = gr.Checkbox(label="Silence Remover (खामोशी हटाना)", value=True)
            btn = gr.Button("🚀 शुद्ध आवाज जनरेट करें", variant="primary")
            
    out = gr.Audio(label="शिव एआई आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True)
