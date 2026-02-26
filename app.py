import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. अल्ट्रा टर्बो 2000% हाई स्पीड सेटअप (LOCKED)
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True 
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. हगिंग फेस मॉडल इंटीग्रेशन
REPO_ID = "Shriramnag/My-Shriram-Voice" 

print("श्री राम नाग जी, शिव AI का शुद्ध इंजन लोड हो रहा है...")

try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
    hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
except:
    print("⚠️ मॉडल फाइल्स पहले से मौजूद हैं।")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. हकलाहट और इंग्लिश शब्दों का सुधार (Transliteration Logic)
def shiv_super_cleaner(text):
    if not text: return ""
    
    # इंग्लिश शब्दों को हिंदी उच्चारण में बदलना ताकि AI न हकलाए [cite: 2026-02-20]
    eng_to_hindi = {
        "Life": "लाइफ", "Simple": "सिंपल", "Dream": "ड्रीम", 
        "Mindset": "माइंडसेट", "Believe": "बिलीव", "Strong": "स्ट्रॉन्ग",
        "Step": "स्टेप", "Fear": "फियर", "Fail": "फेल", "YouTube": "यूट्यूब",
        "AI": "ए आई", "Turbo": "टर्बो", "Speed": "स्पीड"
    }
    
    for eng, hindi in eng_to_hindi.items():
        text = re.sub(rf'\b{eng}\b', hindi, text, flags=re.IGNORECASE)

    # नंबरों को शब्दों में बदलना [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # हकलाहट रोकने के लिए डॉट को कोमा में बदलना [cite: 2026-02-20]
    text = text.replace('.', ',')
    return text.strip()

# ४. मुख्य इंजन - १५०-२०० वर्ड कटर + अनलिमिटेड मोड (LOCKED)
def generate_shiv_v1_5(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    p_text = shiv_super_cleaner(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # १५०-२०० शब्दों का स्मार्ट कटर
    all_words = p_text.split()
    chunks = [" ".join(all_words[i:i+180]) for i in range(0, len(all_words), 180)]

    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"अल्ट्रा टर्बो जनरेशन: भाग {i+1}")
        
        name = f"chunk_{i}.wav"
        # हकलाहट और शोर खत्म करने के लिए Temperature 0.6 पर सेट [cite: 2026-02-20]
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=2.0, temperature=0.6, top_k=50)
        
        seg = AudioSegment.from_wav(name)
        
        if pitch_s != 1.0:
            new_rate = int(seg.frame_rate * pitch_s)
            seg = seg._spawn(seg.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)

        if use_silence:
            try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=200)
            except: pass
            
        combined += seg
        os.remove(name)
        torch.cuda.empty_cache(); gc.collect()

    if use_clean:
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    # आउटपुट फाइल नेम [cite: 2026-02-22]
    final_output_name = "Shri_Ram_Nag_Output.wav"
    combined.export(final_output_name, format="wav")
    return final_output_name

# ५. दिव्य UI (श्री राम नाग संस्करण)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.5 — श्री राम नाग")
    gr.Markdown("### 🔒 टर्बो स्पीड | इंग्लिश हकलाहट फिक्स | अनलिमिटेड स्क्रिप्ट")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12)
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.9, 1.4, 1.15, label="टर्बो रफ़्तार")
                ptch = gr.Slider(0.7, 1.3, 1.0, label="पिच (Pitch)")
                cln = gr.Checkbox(label="Symmetry Clean", value=True)
                sln = gr.Checkbox(label="Silence Remover", value=True)
            btn = gr.Button("🚀 जनरेट और डाउनलोड", variant="primary")
            
    out = gr.Audio(label="श्री राम नाग आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_5, [txt, up_v, git_v, spd, ptch, sln, cln], out)

demo.launch(share=True, debug=True)
