import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल और डिक्शनरी (मस्तिष्क) सेटअप
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

BRAIN_FILE = "shiv_brain.json"

def load_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {"YouTube": "यूट्यूब", "AI": "ए आई", "Technology": "टेक्नोलॉजी"}

def save_brain(brain_data):
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(brain_data, f, ensure_ascii=False, indent=4)

# ३. ऑटो-लर्निंग लॉजिक: स्क्रिप्ट से खुद सीखना
def auto_learn_from_script(text):
    brain = load_brain()
    # स्क्रिप्ट में इंग्लिश शब्दों को खोजना
    eng_words = re.findall(r'\b[a-zA-Z]+\b', text)
    new_learned = False
    
    for word in eng_words:
        if word not in brain:
            # यहाँ हम एक बेसिक रूल लगा रहे हैं, आप बाद में इसे सुधार भी सकते हैं
            # अभी के लिए यह नए शब्दों को रजिस्टर कर लेगा
            brain[word] = word 
            new_learned = True
    
    if new_learned:
        save_brain(brain)

def brain_processor(text):
    brain = load_brain()
    # नंबरों को शब्दों में बदलना [2026-02-20]
    nums = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in nums.items(): text = text.replace(n, w)
    
    # दिमाग से शब्दों का मिलान और सुधार
    for eng, hin in brain.items():
        text = re.sub(r'\b' + eng + r'\b', hin, text, flags=re.IGNORECASE)
    return text.strip()

# ४. जनरेशन और ऑटो-ट्रेनिंग इंजन
def generate_and_learn(text, up_ref, git_ref, speed_s, use_silence, progress=gr.Progress()):
    if not text: return None, "स्क्रिप्ट खाली है!"
    
    # स्टेप १: स्क्रिप्ट से 'सेल्फ-लर्निंग' करना
    auto_learn_from_script(text)
    
    # स्टेप २: टेक्स्ट को साफ़ करना
    clean_text = brain_processor(text)
    
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/" + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    chunks = re.split(r'(?<=[।!?॥.])\s+', clean_text)
    combined = AudioSegment.empty()
    
    

    for i, task in enumerate(chunks):
        if not task.strip(): continue
        progress((i+1)/len(chunks), desc=f"शिव AI सीख रहा है... {i+1}")
        out_name = f"chunk_{i}.wav"
        
        # १०००% शुद्ध सेटिंग्स (LOCKED)
        tts.tts_to_file(text=task, speaker_wav=ref, language="hi", file_path=out_name, 
                        speed=speed_s, repetition_penalty=15.0, temperature=0.01)
        
        combined += AudioSegment.from_wav(out_name)
        os.remove(out_name)
        torch.cuda.empty_cache(); gc.collect()

    final_path = "Shiv_AI_SelfLearned.wav"
    combined.export(final_path, format="wav")
    return final_path, f"✅ एआई ने नई स्क्रिप्ट से सीखा और ऑडियो बनाया।"

# ५. मैन्युअल सुधार टैब
def manual_update_brain(word, correction):
    brain = load_brain()
    brain[word] = correction
    save_brain(brain)
    return f"✅ 'दिमाग' अपडेट हुआ: {word} -> {correction}"

# ६. दिव्य इंटरफ़ेस (v1.2)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.2 — श्री राम नाग")
    gr.Markdown("### 🧠 'महासंगणक' - हर स्क्रिप्ट से खुद सीखने वाला एआई")
    
    with gr.Tabs():
        with gr.TabItem("🎙️ स्क्रिप्ट दें और सिखाएं"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(label="यहाँ स्क्रिप्ट डालें (जितनी ज्यादा स्क्रिप्ट, उतना ज्यादा लर्निंग)", lines=12)
                    spd = gr.Slider(0.9, 1.4, 1.15, label="स्पीड")
                with gr.Column(scale=1):
                    git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
                    up_v = gr.Audio(label="सैंपल अपलोड", type="filepath")
                    btn = gr.Button("🚀 सीखें और जनरेट करें", variant="primary")
            out_audio = gr.Audio(label="शिव AI आउटपुट", type="filepath", autoplay=True)
            out_msg = gr.Markdown()
            
        with gr.TabItem("🧠 मस्तिष्क लाइब्रेरी"):
            gr.Markdown("### यहाँ आप देख सकते हैं कि एआई ने क्या-क्या सीखा है या खुद सुधार सकते हैं:")
            with gr.Row():
                wrong_w = gr.Textbox(label="इंग्लिश शब्द")
                correct_w = gr.Textbox(label="सही हिंदी उच्चारण")
            update_btn = gr.Button("दिमाग में सुधारें")
            update_msg = gr.Markdown()

    btn.click(generate_and_learn, [txt, up_v, git_v, spd, gr.State(True)], [out_audio, out_msg])
    update_btn.click(manual_update_brain, [wrong_w, correct_w], update_msg)

demo.launch(share=True)
