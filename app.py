from flask import Flask, request, jsonify, send_file
import asyncio
import edge_tts
import os
from langdetect import detect
from fr_af import *

app = Flask(__name__)

# Dossier où les fichiers générés seront sauvegardés
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def synthesize_speech(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def generate_speech(text, voice, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "speech.mp3")
    asyncio.run(synthesize_speech(text, voice, output_file))
    return output_file

def generate_female_and_male_speech(text, gender, output_dir, accent='fr-FR'):
    language = detect(text)
    print("Langue détectée :",language)

    voices = {
        'fr': {
            'fr-FR': {'female': 'fr-FR-DeniseNeural', 'male': 'fr-FR-HenriNeural'},
            'fr-CA': {'female': 'fr-CA-SylvieNeural', 'male': 'fr-CA-AntoineNeural'},
            'fr-AF': {'female': generate_speech_vits_frafF, 'male': generate_speech_your_tts_frafM}
        },
        'en': {'female': 'en-US-EmmaNeural', 'male': 'en-US-AndrewNeural'},
        'es': {'female': 'es-ES-XimenaNeural', 'male': 'es-ES-AlvaroNeural'},
        'de': {'female': 'de-DE-KatjaNeural', 'male': 'de-DE-KillianNeural'},
        'pt': {'female': 'pt-BR-ThalitaNeural', 'male': 'pt-BR-AntonioNeural'},
        'zh-cn': {'female': 'zh-CN-XiaoyiNeural', 'male': 'zh-CN-YunxiNeural'},
        'ko': {'female': 'ko-KR-SunHiNeural', 'male': 'ko-KR-InJoonNeural'},
        'ja': {'female': 'ja-JP-NanamiNeural', 'male': 'ja-JP-KeitaNeural'},
        'hi': {'female': 'hi-IN-SwaraNeural', 'male': 'hi-IN-MadhurNeural'},
        'ms': {'female': 'ms-MY-YasminNeural', 'male': 'ms-MY-OsmanNeural'},
        'id': {'female': 'id-ID-GadisNeural', 'male': 'id-ID-ArdiNeural'},
        'pl': {'female': 'pl-PL-ZofiaNeural', 'male': 'pl-PL-MarekNeural'},
        'ru': {'female': 'ru-RU-SvetlanaNeural', 'male': 'ru-RU-DmitryNeural'},
        'tr': {'female': 'tr-TR-EmelNeural', 'male': 'tr-TR-AhmetNeural'}

    }

    if language == 'fr':
        if accent in voices['fr']:
            if accent == 'fr-AF':
                # Voix personnalisées pour l'accent africain francophone
                if gender.lower() == 'female':
                    return voices['fr']['fr-AF']['female'](text, output_dir)
                elif gender.lower() == 'male':
                    return voices['fr']['fr-AF']['male'](text, output_dir)
            else:
                VOICE_female = voices['fr'][accent]['female']
                VOICE_male = voices['fr'][accent]['male']           
        else:
            VOICE_female = voices['fr']['fr-FR']['female']
            VOICE_male = voices['fr']['fr-FR']['male']
    elif language in voices:
        VOICE_female = voices[language]['female']
        VOICE_male = voices[language]['male']
    else:
        VOICE_female = 'fr-FR-VivienneMultilingualNeural'
        VOICE_male = 'fr-FR-RemyMultilingualNeural'

    if gender.lower() == 'female':
        return generate_speech(text, VOICE_female, output_dir)
    elif gender.lower() == 'male':
        return generate_speech(text, VOICE_male, output_dir)
    else:
        raise ValueError("Le genre doit être 'female' ou 'male'.")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    text = data.get('text')
    gender = data.get('gender', 'female')
    accent = data.get('accent', 'fr-FR')
    output_dir = request.args.get('output_dir', OUTPUT_DIR)  # Dossier où sauvegarder les fichiers MP3
    
    if not text:
        return jsonify({'error': 'Le texte est requis'}), 400
    
    try:
        output_file = generate_female_and_male_speech(text, gender, output_dir, accent)
        return jsonify({'file_path': output_file}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['GET'])
def synthesize_get():
    text = request.args.get('text')
    gender = request.args.get('gender', 'female')
    accent = request.args.get('accent', 'fr-FR')
    
    if not text:
        return jsonify({'error': 'Le texte est requis'}), 400

    try:
        output_file = generate_female_and_male_speech(text, gender, OUTPUT_DIR, accent)
        return send_file(output_file, as_attachment=True)  # Téléchargement direct du fichier
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
