# Assistant pédagogique RAG

Un assistant pédagogique IA basé sur la génération augmentée par recherche (RAG) pour gérer, interroger et générer des questions à partir de documents de cours.

## Fonctionnalités

- **Gestion de documents de cours** par matière
- **Recherche sémantique** dans les documents de cours
- **Génération de questions de réflexion** sur des concepts spécifiques
- **Interrogation assistée par IA** du contenu des cours
- **Mise à jour intelligente** des documents (ajout, modification, suppression)

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
  │   ├── fichier1.md          # Document au format Markdown
  │   ├── fichier2.md
  │   └── ...
  └── [AUTRE_MATIERE]/         # Autre matière
      └── ...
```

Pour des résultats optimaux, les documents Markdown devraient utiliser des titres de section avec ## et ###.

## Fonctionnement technique

1. Le système divise les documents en sections selon les titres ou par découpage intelligent
2. Ces sections sont converties en vecteurs (embeddings) et stockées dans Pinecone
3. Lors d'une requête, le système :
   - Recherche les sections les plus pertinentes
   - Les combine avec la question pour former un prompt contextualisé
   - Génère une réponse avec le modèle GPT d'OpenAI

## Maintenance

- Le fichier `metadata_cours.json` conserve l'historique des documents indexés
- Le système détecte automatiquement les modifications et suppressions
- Seuls les nouveaux documents ou les documents modifiés sont traités lors des mises à jour

## Personnalisation

Vous pouvez ajouter de nouvelles matières en créant un dossier correspondant dans `cours/` et en y ajoutant des documents au format Markdown ou texte. 