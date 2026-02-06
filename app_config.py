import os

class Config:
    # आपके हगिंग फेस मॉडल की सेटिंग्स
    MODEL_NAME = "Shriram_Voice"
    HF_PTH = "https://huggingface.co/Shriramnag/%E0%A4%AE%E0%A4%BE%E0%A4%80%E0%A4%88-%E0%A4%B6%E0%A5%8D%E0%A4%B0%E0%A5%80%E0%A4%B0%E0%A4%BE%E0%A4%AE-%E0%A4%B5%E0%A5%89%E0%A4%87%E0%A4%B8/resolve/main/Shriramoriginalvoice.pth"
    HF_INDEX = "https://huggingface.co/Shriramnag/%E0%A4%AE%E0%A4%BE%E0%A4%80%E0%A4%88-%E0%A4%B6%E0%A5%8D%E0%A4%B0%E0%A5%80%E0%A4%B0%E0%A4%BE%E0%A4%AE-%E0%A4%B5%E0%A5%89%E0%A4%87%E0%A4%B8/resolve/main/added_IVF579_Flat_nprobe_Shriramoriginalvoice_v2.index"
    
    # टर्बो सेटिंग्स
    USE_GPU = True
    SILENCE_THRESHOLD = -45
    MIN_SILENCE_LEN = 400

print("✅ कॉन्फ़िगरेशन सफलतापूर्वक लोड हो गया!")

