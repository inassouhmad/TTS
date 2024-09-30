import os
import torch
from TTS.api import TTS
from pydub import AudioSegment
import re

# Dictionnaire pour convertir les nombres en mots en français
number_words_french = {
    0: "zéro", 1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf",
    10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze", 15: "quinze", 16: "seize", 17: "dix-sept",
    18: "dix-huit", 19: "dix-neuf", 20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante",
    70: "soixante-dix", 80: "quatre-vingt", 90: "quatre-vingt-dix", 100: "cent", 1000: "mille"
}

# Fonction pour convertir les nombres en mots en français
def number_to_words_french(number):
    if number < 20:
        return number_words_french[number]
    elif number < 100:
        tens, unit = divmod(number, 10)
        return number_words_french[tens * 10] + (" " + number_words_french[unit] if unit else "")
    elif number < 1000:
        hundreds, remainder = divmod(number, 100)
        return (number_words_french[hundreds] + " cent" if hundreds > 1 else "cent") + (" " + number_to_words_french(remainder) if remainder else "")
    elif number < 1000000:
        thousands, remainder = divmod(number, 1000)
        return (number_to_words_french(thousands) + " mille" if thousands > 1 else "mille") + (" " + number_to_words_french(remainder) if remainder else "")
    elif number < 1000000000:
        millions, remainder = divmod(number, 1000000)
        return number_to_words_french(millions) + " million" + (" " + number_to_words_french(remainder) if remainder else "")
    else:
        return str(number)

# Fonction pour remplacer les nombres par leurs équivalents en mots dans un texte
def replace_numbers_with_words_french(text):
    def replace(match):
        number = int(match.group())
        return number_to_words_french(number)

    # Remplacer les nombres dans le texte par leurs équivalents en mots
    result = re.sub(r'\b\d+\b', replace, text)
    return result


# Fonction pour convertir WAV en MP3
def convert_wav_to_mp3(wav_file, mp3_file):
    try:
        # Charger le fichier WAV
        audio = AudioSegment.from_wav(wav_file)
        # Exporter en MP3
        audio.export(mp3_file, format="mp3")
        #print(f"Converted {wav_file} to {mp3_file}")
    except Exception as e:
        print(f"Error converting file: {e}")

# Fonction pour le modèle YourTTS
def generate_speech_your_tts_frafM(text, output_dir):
    converted_text = replace_numbers_with_words_french(text)
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialiser le modèle YourTTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
    
    # Chemin complet du fichier de sortie
    output_file = os.path.join(output_dir, "speech.wav")
    
    # Fichier WAV du locuteur utilisé pour le clonage de voix
    speaker_wav = "afc-gabon_16.06.11_008_read_0101.wav"
    
    # Effectuer la synthèse vocale et sauvegarder dans le fichier
    tts.tts_to_file(converted_text, speaker_wav=speaker_wav, language="fr-fr", file_path=output_file)

    mp3_path = output_file.replace('.wav', '.mp3')
    convert_wav_to_mp3(output_file, mp3_path)
    
    return mp3_path


# Fonction pour le modèle VITS
def generate_speech_vits_frafF(text, output_dir):
    converted_text = replace_numbers_with_words_french(text)
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser le modèle VITS
    tts_model_name = "tts_models/fr/css10/vits"
    api = TTS(tts_model_name)
    
    # Chemin complet du fichier de sortie
    output_file = os.path.join(output_dir, "speech.wav")
    
    # Fichier WAV du locuteur utilisé pour le clonage de voix
    speaker_wav = "v.mpeg.wav"
    
    # Effectuer la synthèse vocale avec clonage de voix et sauvegarder dans le fichier
    api.tts_with_vc_to_file(converted_text, speaker_wav=speaker_wav, file_path=output_file)

    mp3_path = output_file.replace('.wav', '.mp3')
    convert_wav_to_mp3(output_file, mp3_path)
    
    return mp3_path


# Exemple d'utilisation des fonctions

# Modèle YourTTS
"""
text = "salut inass"
output_dir = "C:/Users/Win/Downloads/apiTTS/outputs"

output_your_tts = generate_speech_your_tts_frafM(text, output_dir)
print(f"Fichier YourTTS généré : {output_your_tts}")

# Modèle VITS
output_vits = generate_speech_vits_frafF(text, output_dir)
print(f"Fichier VITS généré : {output_vits}")"""
