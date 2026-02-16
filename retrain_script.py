import json
import os

def check_training_status():
    if os.path.exists("memory.json"):
        with open("memory.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            count = len(data)
            print(f"🚩 कुल रिकॉर्डेड वाक्य: {count}")
            if count >= 500:
                print("✅ मॉडल को 'महा-शक्तिशाली' बनाने के लिए पर्याप्त डेटा है!")
            else:
                print(f"⏳ अभी {500 - count} वाक्य और चाहिए।")
    else:
        print("❌ memory.json फाइल नहीं मिली।")

if __name__ == "__main__":
    check_training_status()
  
