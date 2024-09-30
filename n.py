import requests

# URL du serveur et chemin pour télécharger le fichier
url = "https://apitts-1-render.onrender.com/synthesize?text=hello&gender=female"
output_dir = "C:/Users/Win/Documents/"

# Envoyer la requête GET pour télécharger l'audio
response = requests.get(url)

# Chemin de destination pour stocker l'audio sur votre machine locale
output_file = output_dir + "speech.mp3"

# Sauvegarder le fichier téléchargé
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
    print(f"Audio téléchargé et sauvegardé dans {output_file}")
else:
    print(f"Erreur lors du téléchargement : {response.status_code}")
