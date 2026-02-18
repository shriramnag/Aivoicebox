import json
import re

class MahagyaniBrain:
    def __init__(self, sanskrit_file, hindi_file, english_file, prosody_file):
        # फाइलों को लोड करना
        with open(sanskrit_file, 'r', encoding='utf-8') as f:
            self.sanskrit_data = json.load(f)
        with open(hindi_file, 'r', encoding='utf-8') as f:
            self.hindi_data = json.load(f)
        with open(english_file, 'r', encoding='utf-8') as f:
            self.english_data = json.load(f)
        with open(prosody_file, 'r', encoding='utf-8') as f:
            self.prosody_data = json.load(f)

    def get_voice_profile(self, text):
        """टेक्स्ट के आधार पर सही प्रोफाइल चुनना (श्लोक या कहानी) [cite: 2026-02-18]"""
        if "॥" in text or "।" in text:
            return self.prosody_data['voice_profiles']['shlok_mode']
        elif any(c.isalpha() for c in text): # अगर इंग्लिश शब्द हैं
            return self.prosody_data['voice_profiles']['story_mode']
        return self.prosody_data['voice_profiles']['talking_mode']

    def clean_and_format(self, text):
        """तीनों फाइलों के ज्ञान का उपयोग करके टेक्स्ट सुधारना [cite: 2026-02-18]"""
        # 1. संस्कृत श्लोक सुधार (Pronunciation & Stretching)
        s_map = self.sanskrit_data.get('pronunciation_logic', {})
        for key, rule in s_map.items():
            if rule['word'] in text:
                text = text.replace(rule['word'], rule.get('phonetic', rule['word']))

        # 2. हिंदी नंबर और ग्रामर सुधार
        num_map = self.hindi_data.get('number_to_word_map', {})
        for num, word in num_map.items():
            text = text.replace(num, word)

        # 3. इंग्लिश शब्दों के बीच गैप और शुद्ध उच्चारण
        e_map = self.english_data.get('vocabulary_master', {})
        for word, rule in e_map.items():
            if word in text:
                # इंग्लिश शब्द के आगे-पीछे हल्का पॉज़ जोड़ना [cite: 2026-02-18]
                text = text.replace(word, f" {rule['phonetic']} ")

        return text.strip()

    def get_timing_instructions(self):
        """आवाज की गति और पिच के निर्देश भेजना [cite: 2026-02-18]"""
        return self.sanskrit_data['timing_rules']

# यह क्लास अब आपके app.py द्वारा इस्तेमाल की जाएगी
