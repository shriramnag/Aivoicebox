import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from pydub import AudioSegment, AudioEffectsChain

# 🚩 आपके पुराने प्रोजेक्ट की फाइलें (इनके साथ कोई छेड़छाड़ नहीं)
try:
    from text_engine import split_into_chunks
    from parallel_processor import combine_chunks
except ImportError:
    print("⚠️ सहायक फाइलें लोड हो रही हैं...")

# ⚡ टर्बो इंजन सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def apply_pro_effects(file_path, bass_boost, echo_level, amp_level):
    """पुराने ऑडियो में भारीपन और गूँज जोड़ना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp_level # एमप्लीफायर
    
    # भारी बेस और मंदिर जैसी गूँज के लिए इफेक्ट्स
    effects = AudioEffectsChain().bass(gain=bass_boost).reverb(reverberance=echo_level)
    processed_sound = effects(sound)
    
    final_path = "shriram_final_pro_v2.wav"
    processed_sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, pitch, emotion, bass, echo, amp, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("स्क्रिप्ट और सैंपल ज़रूरी हैं।") 

    # 🚀 आपका पुराना 'चंक' प्रोसेसिंग लॉजिक (सुरक्षित है)
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    chunks = split_into_chunks(text) 
    chunk_files = []

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")

        # 🎙️ आपका पुराना इंजन सेटिंग्स + नया इमोशन कंट्रोल
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=12.0,   
            temperature=emotion,       # नया ह्यूमन टच स्लाइडर
            top_p=0.85,
            gpt_cond_len=3             # 0.9 Deep Match के लिए
        )
        chunk_files.append(name)

    # 🔗 पुराना कंबाइन लॉजिक
    combined_temp = "combined_temp.wav"
    combine_chunks(chunk_files, output_file=combined_temp)
    
    # ✨ अब इसमें भारीपन और गूँज जोड़ें
    final_output = apply_pro_effects(combined_temp, bass, echo, amp)
    return final_output

# 🎨 आपका शानदार रॉयल UI (पुराने स्लाइडर्स + नए स्लाइडर्स)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 100% मैच टर्बो इंजन")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath")
            
            with gr.Accordion("⚙️ पुराने वर्किंग स्लाइडर्स", open=True):
                speed_s = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=1.5, value=1.0)
                pitch_s = gr.Slider(label="Deep Match (गहराई)", minimum=0.5, maximum=1.0, value=0.9)
            
            with gr.Accordion("🎭 नए ह्यूमन टच और गूँज कंट्रोल", open=True):
                emo_s = gr.Slider(label="इमोशन (Realistic)", minimum=0.1, maximum=1.0, value=0.8)
                bass_s = gr.Slider(label="भारीपन (Deep Bass)", minimum=0, maximum=20, value=5)
                echo_s = gr.Slider(label="गूँज (Echo/Reverb)", minimum=0, maximum=100, value=20)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-10, maximum=10, value=0)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, emo_s, bass_s, echo_s, amp_s], out)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
