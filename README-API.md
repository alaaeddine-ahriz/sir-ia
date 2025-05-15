# API Le Rhino

Ce projet expose les fonctionnalités du système RAG de gestion de cours via une API REST, permettant d'interagir avec le système via des requêtes HTTP plutôt que via une interface en ligne de commande.

## Installation

1. Assurez-vous d'avoir toutes les dépendances installées:

```bash
pip install fastapi uvicorn pydantic langchain langchain-pinecone langchain-openai pinecone-client pdfplumber python-docx python-pptx odfpy PyPDF2==3.0.1 python-dotenv
```

2. Configurez les variables d'environnement dans un fichier `.env`:

```
PINECONE_API_KEY=votre_clé_api_pinecone
OPENAI_API_KEY=votre_clé_api_openai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Lancement du serveur

Pour lancer le serveur API:

```bash
python app.py
```

Le serveur démarre sur http://localhost:8000 par défaut.

## Documentation de l'API

Une documentation interactive est disponible à l'adresse http://localhost:8000/docs une fois le serveur lancé.

## Points d'accès (Endpoints)

### GET /

Vérifie que l'API est opérationnelle.

**Réponse:**
```json
{
  "success": true,
  "message": "Le Rhino API",
  "data": {
    "status": "online"
  },
  "timestamp": "2023-06-01T12:00:00.000000"
}
```

### GET /matieres

Liste toutes les matières disponibles.

**Réponse:**
```json
{
  "success": true,
  "message": "2 matières trouvées",
  "data": {
    "matieres": ["SYD", "TCP"]
  },
  "timestamp": "2023-06-01T12:00:00.000000"
}
```

### POST /matieres/update

Met à jour l'index pour une matière spécifique.

**Corps de la requête:**
```json
{
  "matiere": "SYD"
}
```

**Réponse:**
```json
{
  "success": true,
  "message": "Matière SYD mise à jour avec succès",
  "data": {
    "matiere": "SYD",
    "updated": true
  },
  "timestamp": "2023-06-01T12:00:00.000000"
}
```

### POST /question

Pose une question sur une matière spécifique.

**Corps de la requête:**
```json
{
  "matiere": "SYD",
  "query": "Expliquez le concept de la virtualisation",
  "output_format": "text",
  "save_output": true
}
```

**Réponse:**
```json
{
  "success": true,
  "message": "Réponse générée avec succès",
  "data": {
    "response": "La virtualisation est un concept qui...",
    "sources": [
      {
        "document": 1,
        "source": "SYD/virtualisation.md",
        "section": "Introduction à la virtualisation",
        "contenu": "La virtualisation est une technologie qui permet..."
      }
    ],
    "matiere": "SYD",
    "query": "Expliquez le concept de la virtualisation"
  },
  "timestamp": "2023-06-01T12:00:00.000000"
}
```

### POST /question/reflection

Génère une question de réflexion sur un concept clé.

**Corps de la requête:**
```json
{
  "matiere": "SYD",
  "concept_cle": "virtualisation",
  "output_format": "json",
  "save_output": true
}
```

**Réponse:**
```json
{
  "success": true,
  "message": "Question de réflexion générée avec succès",
  "data": {
    "question": "{\"question\": \"Quelles sont les implications de la virtualisation sur la sécurité des systèmes d'information?\", \"concepts_abordés\": [\"virtualisation\", \"sécurité\", \"isolation\", \"hyperviseur\"], \"niveau_difficulté\": \"avancé\", \"compétences_visées\": [\"analyse critique\", \"évaluation des risques\", \"conception sécurisée\"], \"éléments_réponse\": [\"Protection par isolation\", \"Vulnérabilités des hyperviseurs\", \"Gestion centralisée des politiques de sécurité\"]}",
    "matiere": "SYD",
    "concept": "virtualisation",
    "format": "json"
  },
  "timestamp": "2023-06-01T12:00:00.000000"
}
```

## Exemples d'utilisation avec curl

### Lister les matières disponibles
```bash
curl -X GET http://localhost:8000/matieres
```

### Mettre à jour une matière
```bash
curl -X POST http://localhost:8000/matieres/update \
  -H "Content-Type: application/json" \
  -d '{"matiere": "SYD"}'
```

### Poser une question
```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"matiere": "SYD", "query": "Expliquez le concept de la virtualisation", "output_format": "text"}'
```

### Générer une question de réflexion
```bash
curl -X POST http://localhost:8000/question/reflection \
  -H "Content-Type: application/json" \
  -d '{"matiere": "SYD", "concept_cle": "virtualisation", "output_format": "json"}'
``` 