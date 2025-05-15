# Assistant pédagogique RAG

Un assistant pédagogique IA basé sur la génération augmentée par recherche (RAG) pour gérer, interroger et générer des questions à partir de documents de cours.

## Fonctionnalités

- **Gestion de documents de cours** par matière
- **Prise en charge multi-format** : PDF, DOCX, PPTX, TXT, MD, ODT, ODP et DOC
- **Recherche sémantique** dans les documents de cours
- **Génération de questions de réflexion** sur des concepts spécifiques
- **Interrogation assistée par IA** du contenu des cours
- **Mise à jour intelligente** des documents (ajout, modification, suppression)
- **Traitement intelligent des documents non structurés**

## Prérequis

- Python 3.8 ou supérieur
- Compte Pinecone (pour la base de données vectorielle)
- Compte OpenAI (pour l'IA générative)

## Installation

1. Clonez ce dépôt :
```bash
git clone [URL du dépôt]
cd rag-chatbot
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

Pour la prise en charge complète des fichiers DOC (ancien format Word), vous pourriez avoir besoin d'installer des dépendances supplémentaires selon votre système :

- **Linux** :
```bash
sudo apt-get install antiword poppler-utils tesseract-ocr
```

- **macOS** :
```bash
brew install antiword poppler tesseract
```

3. Créez un fichier `.env` à la racine du projet avec les variables suivantes :
```
OPENAI_API_KEY=votre_clé_api_openai
PINECONE_API_KEY=votre_clé_api_pinecone
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Utilisation

Lancez l'application :
```bash
python main.py
```

### Menu principal

L'assistant propose 4 options principales :

1. **Mettre à jour les documents d'une matière** - Indexe les nouveaux documents, met à jour les documents modifiés et supprime les documents disparus
2. **Générer une question de réflexion** - Crée une question de réflexion sur un concept spécifique
3. **Poser une question sur une matière** - Interroge la base de connaissances sur un sujet précis
4. **Quitter** - Termine l'application

### Organisation des documents

Les documents de cours doivent être organisés selon la structure suivante :
```
cours/
  ├── SYD/                     # Dossier de la matière
  │   ├── document1.md         # Document markdown
  │   ├── presentation.pptx    # Présentation PowerPoint
  │   ├── cours.pdf            # Document PDF
  │   ├── notes.docx           # Document Word
  │   └── ...
  └── [AUTRE_MATIERE]/         # Autre matière
      └── ...
```

### Formats pris en charge

- **Markdown (.md)** - Format recommandé pour une structure optimale
- **PDF (.pdf)** - Texte extrait avec conservation de paragraphes
- **Word (.docx)** - Inclut la détection intelligente des titres
- **PowerPoint (.pptx)** - Chaque diapositive est convertie en section
- **OpenDocument (.odt, .odp)** - Formats LibreOffice/OpenOffice
- **Texte brut (.txt)** - Traité avec découpage par paragraphes
- **Word ancien format (.doc)** - Pris en charge via la bibliothèque textract

## Fonctionnement technique

1. Le système extrait le contenu des documents selon leur format
2. Le texte est traité et divisé en sections, en détectant les titres quand c'est possible ou en utilisant des heuristiques de découpage pour les documents non structurés
3. Ces sections sont converties en vecteurs (embeddings) et stockées dans Pinecone
4. Lors d'une requête, le système :
   - Recherche les sections les plus pertinentes
   - Les combine avec la question pour former un prompt contextualisé
   - Génère une réponse avec le modèle GPT d'OpenAI

## Maintenance

- Le fichier `metadata_cours.json` conserve l'historique des documents indexés
- Le système détecte automatiquement les modifications et suppressions
- Seuls les nouveaux documents ou les documents modifiés sont traités lors des mises à jour

## Personnalisation

Vous pouvez ajouter de nouvelles matières en créant un dossier correspondant dans `cours/` et en y ajoutant des documents dans n'importe quel format pris en charge. 