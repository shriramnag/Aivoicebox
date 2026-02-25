import os, torch, gradio as gr, requests, re, gc, wave, struct, math
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Device: {device}")

# Model download
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
print(f"✅ Model: {model_path}")

# XTTS load
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Custom weights inject (agar compatible ho)
try:
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if isinstance(sd, dict):
        tts.synthesizer.tts_model.load_state_dict(sd, strict=False)
        print("✅ Custom Ramai.pth loaded!")
except Exception as e:
    print(f"⚠️  Custom weights skip ({e}) — speaker_wav cloning active")

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ═══════════════════════════════════════════════════════════════
# FIX 1: HINDI NUMBER → WORDS (Stutter fix for mixed script)
# ═══════════════════════════════════════════════════════════════
HINDI_NUMS = {
    '0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार',
    '5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'
}
EN_NUMS = {
    '0':'zero','1':'one','2':'two','3':'three','4':'four',
    '5':'five','6':'six','7':'seven','8':'eight','9':'nine'
}

def replace_numbers(text, lang):
    num_map = HINDI_NUMS if lang == "hi" else EN_NUMS
    # Multi-digit numbers (e.g. 2024 → दो शून्य दो चार)
    def replace_match(m):
        return ' '.join(num_map[d] for d in m.group())
    return re.sub(r'\d+', replace_match, text)

# ═══════════════════════════════════════════════════════════════
# FIX 2: ACCURATE LANGUAGE DETECTION
# Rule: Koi bhi Devanagari = Hindi. Pure English tab hi jab 
# koi Devanagari na ho aur 80%+ English chars hon.
# ═══════════════════════════════════════════════════════════════
def detect_language(text):
    hi = len(re.findall(r'[\u0900-\u097F]', text))
    en = len(re.findall(r'[a-zA-Z]', text))
    if hi > 0:
        return "hi"
    if en > len(text) * 0.5:
        return "en"
    return "hi"

# ═══════════════════════════════════════════════════════════════
# FIX 3: SMART TEXT SPLITTER
# ROOT CAUSE: XTTS drifts after 12-15 words in Hindi.
# Solution: Hard limit of 10 words per chunk.
# Sentence boundaries pe split, phir word limit enforce.
# ═══════════════════════════════════════════════════════════════
MAX_WORDS_PER_CHUNK = 10  # CRITICAL — yeh value 10 se zyada mat karo

def smart_split(text):
    """
    Text ko speech-friendly chunks mein todta hai.
    - Pehle markers split
    - Phir sentence boundaries
    - Phir word limit enforce
    """
    # Step 1: Markers alag karo
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    
    final_chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Markers directly add
        if part in ['[pause]', '[breath]', '[laugh]']:
            final_chunks.append(part)
            continue
        
        # Step 2: Sentence split (Hindi + English punctuation)
        sentences = re.split(r'(?<=[।!?॥])\s+|(?<=[.!?])\s+(?=[A-Z\u0900-\u097F])', part)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            words = sentence.split()
            
            # Step 3: Word limit enforce
            if len(words) <= MAX_WORDS_PER_CHUNK:
                if len(sentence) > 1:
                    final_chunks.append(sentence)
            else:
                # Long sentence ko word limit pe toddo
                chunk_words = []
                for word in words:
                    chunk_words.append(word)
                    # Natural break points pe split prefer karo
                    is_break = (word.endswith(',') or word.endswith('—') or 
                               word.endswith('-') or len(chunk_words) >= MAX_WORDS_PER_CHUNK)
                    if is_break and chunk_words:
                        final_chunks.append(' '.join(chunk_words))
                        chunk_words = []
                if chunk_words:
                    final_chunks.append(' '.join(chunk_words))
    
    return final_chunks

# ═══════════════════════════════════════════════════════════════
# FIX 4: REFERENCE AUDIO PREPROCESSING
# Original voice: Mono, 44100Hz, RMS ~4953
# XTTS speaker encoder: 22050Hz mono prefer karta hai
# ═══════════════════════════════════════════════════════════════
def prepare_reference(ref_path):
    audio = AudioSegment.from_file(ref_path)
    
    # Mono convert
    audio = audio.set_channels(1)
    
    # 22050Hz — XTTS speaker encoder ke liye optimal
    audio = audio.set_frame_rate(22050)
    
    # Normalize
    audio = effects.normalize(audio)
    
    # Min 3 seconds, max 30 seconds (XTTS requirement)
    if len(audio) < 3000:
        audio = audio * (3000 // len(audio) + 1)
    audio = audio[:30000]
    
    clean_path = "ref_prepared.wav"
    audio.export(clean_path, format="wav")
    print(f"✅ Reference prepared: {len(audio)/1000:.1f}s, mono, 22050Hz")
    return clean_path

# ═══════════════════════════════════════════════════════════════
# FIX 5: XTTS PARAMETERS — 100% HAKLAHAT FIX
#
# WRONG (purana):              SAHI (naya):
# temperature=0.15        →   temperature=0.75
# repetition_penalty=20.0 →   repetition_penalty=2.5
# top_k=10                →   top_k=50
# top_p=0.8               →   top_p=0.85
# speed=1.0               →   speed=1.1
#
# WHY:
# - temperature 0.15 = model freeze → loop → haklahat
# - rep_penalty 20 = valid syllables bhi block → broken words
# - top_k 10 = bahut restrict → wrong token choice → stutter
# ═══════════════════════════════════════════════════════════════
def get_tts_params(speed):
    return {
        "temperature": 0.75,          # Natural variance — freeze nahi hoga
        "repetition_penalty": 2.5,    # Sirf real repetition rokna
        "top_k": 50,                  # Enough vocab for natural Hindi
        "top_p": 0.85,                # Balanced sampling
        "speed": speed,
    }

# ═══════════════════════════════════════════════════════════════
# FIX 6: AUDIO OUTPUT — Mono + Loudness Match Original
# Original RMS: ~4953, Peak: 32393/32767
# ═══════════════════════════════════════════════════════════════
import numpy as np

def match_audio_to_original(audio, target_rms=4900):
    """Clone ki awaaz ko original ke level pe laata hai."""
    # Mono enforce
    audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
    
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    current_rms = np.sqrt(np.mean(samples ** 2))
    
    if current_rms > 10:
        gain = min(target_rms / current_rms, 4.0)  # Safety cap
        samples = np.clip(samples * gain, -32767, 32767).astype(np.int16)
        audio = AudioSegment(
            samples.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )
    
    return effects.normalize(audio)

# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ═══════════════════════════════════════════════════════════════
def generate(text, up_ref, git_ref, speed, pitch, use_silence, use_clean, progress=gr.Progress()):
    
    # Step 1: Reference voice prepare karo
    if up_ref:
        ref_path = prepare_reference(up_ref)
    else:
        raw_ref = "ref_raw.wav"
        url = G_RAW + requests.utils.quote(git_ref)
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        with open(raw_ref, "wb") as f:
            f.write(resp.content)
        ref_path = prepare_reference(raw_ref)
    
    # Step 2: Text split
    chunks = smart_split(text)
    total = len(chunks)
    print(f"📝 Total chunks: {total}")
    for i, c in enumerate(chunks):
        print(f"   [{i+1}] '{c[:50]}...' " if len(c)>50 else f"   [{i+1}] '{c}'")
    
    combined = AudioSegment.empty()
    params = get_tts_params(speed)
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/total, desc=f"🎙️ Generating {i+1}/{total}: {chunk[:30]}...")
        
        # Markers
        if chunk == "[pause]":
            combined += AudioSegment.silent(duration=800)
            continue
        elif chunk == "[breath]":
            combined += AudioSegment.silent(duration=300)
            continue
        elif chunk == "[laugh]":
            combined += AudioSegment.silent(duration=100)
            continue
        
        # Language detect
        lang = detect_language(chunk)
        
        # Clean text
        clean = replace_numbers(chunk, lang)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove chars XTTS nahi samajhta (except Devanagari + Latin + basic punct)
        if lang == "hi":
            clean = re.sub(r'[^\u0900-\u097F\s,!?।॥\'"%-]', ' ', clean)
        else:
            clean = re.sub(r'[^a-zA-Z0-9\s,!?.\'"%-]', ' ', clean)
        
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        if len(clean) < 2:
            continue
        
        out_path = f"chunk_{i}.wav"
        
        try:
            tts.tts_to_file(
                text=clean,
                speaker_wav=ref_path,
                language=lang,
                file_path=out_path,
                **params
            )
            
            seg = AudioSegment.from_wav(out_path)
            seg = seg.set_channels(1)  # Mono enforce immediately
            
            if use_silence:
                try:
                    seg = effects.strip_silence(seg, silence_thresh=-42, padding=100)
                except:
                    pass
            
            combined += seg
            # Natural inter-chunk pause (60ms) — robot nahi lagega
            combined += AudioSegment.silent(duration=60)
            
            print(f"   ✅ [{i+1}] '{clean[:30]}' ({lang}) — {len(seg)/1000:.1f}s")
            
        except Exception as e:
            print(f"   ❌ [{i+1}] FAILED: {e}")
            # Retry with simpler params
            try:
                tts.tts_to_file(
                    text=clean, speaker_wav=ref_path, language=lang,
                    file_path=out_path, speed=speed,
                    temperature=0.85, repetition_penalty=1.5, top_k=80
                )
                seg = AudioSegment.from_wav(out_path).set_channels(1)
                combined += seg
                print(f"   ♻️  [{i+1}] Retry success")
            except Exception as e2:
                print(f"   💀 [{i+1}] Retry also failed: {e2}")
        
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
        
        # Memory cleanup every 5 chunks
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final audio fix
    if use_clean:
        combined = match_audio_to_original(combined)
    else:
        combined = combined.set_channels(1).set_frame_rate(44100)
    
    # Cleanup refs
    for f in ["ref_prepared.wav", "ref_raw.wav"]:
        if os.path.exists(f):
            os.remove(f)
    
    final = "Shri_Ram_Nag_Output.wav"
    combined.export(final, format="wav", parameters=["-ar", "44100", "-ac", "1"])
    print(f"✅ Final output: {final} ({len(combined)/1000:.1f}s)")
    return final

# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
js = """function insertTag(tag) { 
    var t = document.querySelector('#script_box textarea'); 
    var s = t.selectionStart; 
    t.value = t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); 
    t.focus(); return t.value; 
}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js) as demo:
    gr.Markdown("""
    # 🚩 शिव AI — श्री राम नाग | Haklahat-Free v3.0
    > ✅ 10-word chunks | ✅ temperature=0.75 | ✅ rep_penalty=2.5 | ✅ Mono output | ✅ Loudness matched
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="स्क्रिप्ट (हिंदी / English / Mixed)",
                lines=12, elem_id="script_box",
                placeholder="यहाँ स्क्रिप्ट लिखें...\nTip: लंबे sentences को [pause] से तोड़ें।"
            )
            wc = gr.Markdown("📊 शब्द: 0")
            txt.change(lambda x: f"📊 शब्द: **{len(x.split()) if x.strip() else 0}**", [txt], [wc])
            
            with gr.Row():
                gr.Button("⏸️ रोके [pause]").click(None, None, txt, js="()=>insertTag('[pause]')")
                gr.Button("💨 सांस [breath]").click(None, None, txt, js="()=>insertTag('[breath]')")
                gr.Button("😊 हँसो [laugh]").click(None, None, txt, js="()=>insertTag('[laugh]')")
        
        with gr.Column(scale=1):
            git_ref = gr.Dropdown(
                choices=["aideva.wav","Joanne.wav"],
                label="📁 GitHub Voice", value="aideva.wav"
            )
            up_ref = gr.Audio(label="🎤 अपनी Voice Upload करें", type="filepath")
            
            with gr.Accordion("⚙️ Settings", open=True):
                spd = gr.Slider(0.9, 1.4, 1.1, step=0.05,
                    label="Speed (1.1 = optimal — 1.0 pe bhi slow tha)")
                ptc = gr.Slider(0.8, 1.1, 0.96, label="Pitch")
                cln = gr.Checkbox(label="✅ Loudness Match + Clean", value=True)
                sln = gr.Checkbox(label="✅ Silence Remover", value=True)
            
            btn = gr.Button("🚀 Generate (Haklahat-Free)", variant="primary", size="lg")
    
    out = gr.Audio(label="🎧 Output", type="filepath", autoplay=True)
    
    btn.click(generate, [txt, up_ref, git_ref, spd, ptc, sln, cln], out)

demo.launch(share=True)
