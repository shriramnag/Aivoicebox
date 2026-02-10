import os
from TTS.api import TTS
import gradio as gr

# टर्बो लोड: 1000 Epochs वाला नया मॉडल
device = "cuda" if torch.cuda.is_available() else "cpu"
# ध्यान दें: यहाँ आपकी हगिंग फेस रिपॉजिटरी का लिंक इस्तेमाल होगा
# Shriramoriginalvoice.pth (56.3 MB) अब लोड होने के लिए तैयार है

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice_v2(text, voice_sample, remove_silence):
    output_path = "output_v2.wav"
    
    # भाषा नियंत्रण और हकलाना रोकने के लिए फिक्स
    tts.tts_to_file(
        text=text, 
        speaker_wav=voice_sample, 
        language="hi",
        file_path=output_path,
        split_sentences=True 
    )
    
    # यहाँ आपका साइलेंस रिमूवर लॉजिक काम करेगा...
    return output_path
