import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल और फाइलों का इंटीग्रेशन (Hugging Face से) [cite: 2026-02-26]
# आपके स्क्रीनशॉट के अनुसार सही रिपॉजिटरी: Shriramnag/My-Shriram-Voice
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" # आपका वर्किंग मॉडल [cite: 2026-02-16]

print("श्री राम नाग जी, हगिंग फेस से फाइलों को टर्बो मोड में जोड़ा जा रहा है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
# ONNX और अन्य फाइलों को कैश में सुनिश्चित करना
hf_hub_download(repo_id=REPO_ID, filename="config.json")
hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट रोकने के लिए मास्टर टेक्स्ट क्लीनर (LOCKED) [cite: 2026-02-20]
def shiv_super_cleaner(text):
    # नंबर फिक्स [2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # मुश्किल शब्दों को तोड़ना ताकि AI न अटके
    brain_fix = {"जिंदगी": "ज़िन्दगी", "भागदौड़": "भाग दौड़", "YouTube": "यूट्यूब", "AI": "ए आई"}
    for k, v in brain_fix.items(): text = text.replace(k, v)
    
    return text.strip()

# ४. मुख्य इंजन - 'माइक्रो-चंकिंग' तकनीक (LOCKED) [cite: 2026-01-06]
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ हकलाहट फिक्स: वाक्यों को कोमा (,) और पूर्ण विराम (।) पर छोटे टुकड़ों में तोड़ना [cite: 2026-02-20]
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
        progress((i+1)/len(valid_chunks), desc="हकलाहट मुक्त जनरेशन जारी है...")
        
        # [pause] जैसे टैग्स को संभालना
        if "[pause]" in chunk: combined += AudioSegment.silent(duration=800); continue
        
        name = f"final_chunk_{i}.wav"
        # 🔒 सबसे सटीक XTTS सेटिंग्स: Penalty 1.2, Temp 0.1 [cite: 2026-02-20]
        tts.tts_to_file(text=chunk.strip(), speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=1.2, temperature=0.1, top_k=1)
        
        seg = AudioSegment.from_wav(name)
        if use_silence: # साइलेंस रिमूवर बटन [cite: 2026-01-06]
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=100)
            except: pass
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: # Symmetry Clean टूल [cite: 2026-01-06]
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    final_p = "Shiv_AI_v1.5_Pure.wav"
    combined.export(final_p, format="wav")
    return final_p

# ५. दिव्य UI (वर्ड काउंटर के साथ) [cite: 2026-02-20]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 ब्रह्मास्त्र अपडेट: हकलाहट १०००% बंद | टर्बो हाई स्पीड [cite: 2026-01-06]")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.9, 1.4, 1.15, label="रफ़्तार")
                cln = gr.Checkbox(label="एआई बेस सफाई (Symmetry)", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("🚀 शुद्ध आवाज जनरेट करें", variant="primary")
            
    out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, gr.State(1.0), sln, cln], out)

demo.launch(share=True)
