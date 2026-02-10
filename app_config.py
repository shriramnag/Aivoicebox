import os

class Config:
    # 1. आपके हगिंग फेस मॉडल की सटीक जानकारी
    MODEL_NAME = "Shriram_Voice_v2"
    
    # Ramai.pth का डायरेक्ट डाउनलोड यूआरएल
    HF_PTH = "https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/Ramai.pth"
    
    # इंडेक्स फाइल का डायरेक्ट डाउनलोड यूआरएल
    HF_INDEX = "https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/added_IVF759_Flat_nprobe_Ramai_Shri_Ram_Voice_Training.index"
    
    # 2. टर्बो और ऑडियो सेटिंग्स
    USE_GPU = True
    SILENCE_THRESHOLD = -45
    MIN_SILENCE_LEN = 400
    
    # 3. हगिंग फेस रिपॉजिटरी आईडी
    REPO_ID = "Shriramnag/My-Shriram-Voice"

print("✅ श्रीराम वाणी कॉन्फ़िगरेशन सफलतापूर्वक अपडेट हो गया!")
