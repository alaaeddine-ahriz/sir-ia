"""
Gestion des cours avec RAG - Système de génération de questions
Ce script permet de gérer des documents de cours par matière et de générer des questions de réflexion.
"""

# -----------------------------------------
# Imports et configuration d'environnement
# -----------------------------------------
import os
import time
import uuid
import warnings
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Désactivation des avertissements LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "not-needed"
os.environ["LANGCHAIN_PROJECT"] = "not-needed"
warnings.filterwarnings("ignore", category=Warning, module="langsmith")

# Imports pour le découpage de texte
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Imports pour la base de données vectorielle
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

# Imports pour le système RAG
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.schema import Document

# -----------------------------------------
# Configuration du système
# -----------------------------------------
# Chemin vers le dossier des cours
COURS_DIR = os.path.join(os.getcwd(), "cours")
# Nom de l'index Pinecone
INDEX_NAME = "rag-education"
# Fichier de suivi des mises à jour
METADATA_FILE = os.path.join(os.getcwd(), "metadata_cours.json")

# -----------------------------------------
# Fonctions de gestion des fichiers
# -----------------------------------------
def initialiser_structure_dossiers():
    """
    Initialise la structure des dossiers pour les cours si elle n'existe pas.
    Crée un dossier par matière avec un fichier README explicatif.
    """
    # Créer le dossier principal des cours s'il n'existe pas
    if not os.path.exists(COURS_DIR):
        os.makedirs(COURS_DIR)
        print(f"Dossier principal des cours créé: {COURS_DIR}")
    
    # Liste des matières (à adapter selon vos besoins)
    matieres = ["SYD", "BD", "SECU", "ARCHI", "AI"]
    
    for matiere in matieres:
        matiere_dir = os.path.join(COURS_DIR, matiere)
        if not os.path.exists(matiere_dir):
            os.makedirs(matiere_dir)
            
            # Créer un fichier README explicatif
            readme_path = os.path.join(matiere_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"# Documents de cours pour {matiere}\n\n")
                f.write("Placez ici vos documents de cours pour cette matière.\n")
                f.write("Formats supportés: .md, .txt\n\n")
                f.write("Structure recommandée pour les fichiers markdown:\n")
                f.write("- Utilisez des titres ## pour les sections principales\n")
                f.write("- Chaque fichier doit traiter d'un concept ou d'une notion\n")
            
            print(f"Dossier pour la matière {matiere} créé avec un README explicatif")

def lire_fichiers_matiere(matiere):
    """
    Lit tous les fichiers de cours pour une matière donnée.
    
    Args:
        matiere (str): Identifiant de la matière (ex: "SYD", "BD")
        
    Returns:
        list: Liste des documents avec leur contenu et métadonnées
    """
    matiere_dir = os.path.join(COURS_DIR, matiere)
    
    # Vérifier si le dossier existe
    if not os.path.exists(matiere_dir):
        print(f"Erreur: Le dossier de la matière {matiere} n'existe pas.")
        return []
    
    documents = []
    
    # Extensions supportées
    extensions = ["*.md", "*.txt"]
    
    # Parcourir tous les fichiers avec les extensions supportées
    for ext in extensions:
        for file_path in glob.glob(os.path.join(matiere_dir, "**", ext), recursive=True):
            try:
                # Éviter de traiter les fichiers README
                if os.path.basename(file_path).lower() == "readme.md":
                    continue
                
                # Calculer le hash du fichier pour suivre les modifications
                file_hash = calculer_hash_fichier(file_path)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Métadonnées du document
                relative_path = os.path.relpath(file_path, COURS_DIR)
                metadata = {
                    "source": relative_path,
                    "matiere": matiere,
                    "filename": os.path.basename(file_path),
                    "filetype": os.path.splitext(file_path)[1],
                    "file_hash": file_hash,
                    "updated_at": datetime.now().isoformat()
                }
                
                documents.append({"content": content, "metadata": metadata})
                print(f"Fichier lu: {relative_path}")
                
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {file_path}: {e}")
    
    return documents

def calculer_hash_fichier(file_path):
    """
    Calcule un hash MD5 du contenu d'un fichier pour détecter les modifications.
    
    Args:
        file_path (str): Chemin vers le fichier
        
    Returns:
        str: Hash MD5 du fichier
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
            
    return hash_md5.hexdigest()

def charger_metadata():
    """
    Charge les métadonnées des précédentes mises à jour depuis un fichier JSON.
    
    Returns:
        dict: Métadonnées des documents indexés
    """
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement des métadonnées: {e}")
            return {"matieres": {}}
    else:
        return {"matieres": {}}

def sauvegarder_metadata(metadata):
    """
    Sauvegarde les métadonnées des documents indexés dans un fichier JSON.
    
    Args:
        metadata (dict): Métadonnées à sauvegarder
    """
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métadonnées: {e}")

# -----------------------------------------
# Fonctions de traitement du texte
# -----------------------------------------
def split_document(document):
    """
    Divise un document en sections.
    Utilise des stratégies différentes selon le type de fichier.
    
    Args:
        document (dict): Document avec contenu et métadonnées
        
    Returns:
        list: Liste des sections avec leurs métadonnées
    """
    content = document["content"]
    metadata = document["metadata"]
    
    # Stratégie pour les fichiers markdown: découpage par en-têtes
    if metadata["filetype"] == ".md":
        headers_to_split_on = [
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        
        try:
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
            
            splits = markdown_splitter.split_text(content)
            
            # Si aucun en-tête n'a été trouvé, utiliser la méthode par caractères
            if not splits:
                return split_by_characters(content, metadata)
                
            # Ajouter les métadonnées du document à chaque split
            for split in splits:
                split.metadata.update(metadata)
                
            return splits
            
        except Exception as e:
            print(f"Erreur lors du découpage markdown: {e}")
            # Fallback sur le découpage par caractères
            return split_by_characters(content, metadata)
    
    # Pour les autres types de fichiers: découpage par caractères
    else:
        return split_by_characters(content, metadata)

def split_by_characters(content, metadata):
    """
    Divise un document en chunks par caractères (méthode de secours).
    
    Args:
        content (str): Contenu du document
        metadata (dict): Métadonnées du document
        
    Returns:
        list: Liste des sections avec leurs métadonnées
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    docs = text_splitter.create_documents([content], [metadata])
    return docs

# -----------------------------------------
# Configuration de la base de données vectorielle
# -----------------------------------------
def initialize_pinecone():
    """
    Initialise la connexion à Pinecone et crée un index si nécessaire.
    
    Returns:
        tuple: (client Pinecone, nom de l'index, spécifications)
    """
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
    region = os.environ.get('PINECONE_REGION') or 'us-east-1'
    
    pc = Pinecone(api_key=pinecone_api_key)
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    return pc, INDEX_NAME, spec

def setup_embeddings():
    """
    Initialise le modèle d'embedding pour la vectorisation du texte.
    
    Returns:
        PineconeEmbeddings: Instance configurée du modèle d'embedding
    """
    model_name = 'multilingual-e5-large'
    
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )
    
    return embeddings

def create_or_get_index(pc, index_name, embeddings, spec):
    """
    Crée un nouvel index si nécessaire ou récupère un index existant.
    
    Args:
        pc: Client Pinecone
        index_name (str): Nom de l'index
        embeddings: Modèle d'embedding
        spec: Spécifications de l'index
        
    Returns:
        object: Index Pinecone
    """
    if index_name not in pc.list_indexes().names():
        print(f"Création d'un nouvel index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
        # Attendre que l'index s'initialise
        print("Attente de l'initialisation de l'index...")
        time.sleep(10)
    
    # Afficher l'état de l'index
    print("État actuel de l'index:")
    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    
    # Utiliser un dictionnaire au lieu d'un objet spécifique de Pinecone
    stats_dict = {
        "dimension": index_stats.dimension,
        "index_fullness": index_stats.index_fullness,
        "namespaces": {},
        "total_vector_count": index_stats.total_vector_count
    }
    
    # Ajouter les informations sur les namespaces
    if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
        for ns_name, ns_data in index_stats.namespaces.items():
            stats_dict["namespaces"][ns_name] = {"vector_count": ns_data.vector_count}
    
    print(json.dumps(stats_dict, indent=2))
    print("\n")
    
    return index

def get_matiere_namespace(matiere):
    """
    Génère un namespace standardisé pour une matière.
    
    Args:
        matiere (str): Identifiant de la matière
        
    Returns:
        str: Namespace standardisé
    """
    return f"matiere-{matiere.lower()}"

def upsert_documents(pc, index_name, embeddings, matiere, docs):
    """
    Met à jour l'index vectoriel avec de nouveaux documents ou des documents modifiés.
    
    Args:
        pc: Client Pinecone
        index_name (str): Nom de l'index
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière
        docs (list): Liste des sections de documents
        
    Returns:
        tuple: (store vectoriel, namespace utilisé)
    """
    if not docs:
        print(f"Aucun document à insérer pour la matière {matiere}")
        return None, None
    
    namespace = get_matiere_namespace(matiere)
    
    try:
        print(f"Insertion de {len(docs)} sections de documents dans l'espace '{namespace}'")
        vector_store = PineconeVectorStore.from_documents(
            documents=docs,
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        print("Insertion réussie")
        return vector_store, namespace
    except Exception as e:
        print(f"Erreur lors de l'insertion: {e}")
        return None, None

# -----------------------------------------
# Gestion des mises à jour
# -----------------------------------------
def mettre_a_jour_matiere(pc, index_name, embeddings, matiere):
    """
    Met à jour l'index vectoriel pour une matière spécifique,
    en ne traitant que les fichiers nouveaux ou modifiés.
    
    Args:
        pc: Client Pinecone
        index_name (str): Nom de l'index
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière
        
    Returns:
        bool: True si des mises à jour ont été effectuées, False sinon
    """
    # Charger les métadonnées des précédentes indexations
    metadata = charger_metadata()
    
    # Initialiser la section pour cette matière si elle n'existe pas
    if matiere not in metadata["matieres"]:
        metadata["matieres"][matiere] = {
            "fichiers": {},
            "derniere_mise_a_jour": None
        }
    
    # Lire tous les fichiers de cours pour cette matière
    documents = lire_fichiers_matiere(matiere)
    if not documents:
        print(f"Aucun document trouvé pour la matière {matiere}")
        return False
    
    # Déterminer quels fichiers ont été modifiés ou ajoutés
    fichiers_matieres = metadata["matieres"][matiere]["fichiers"]
    docs_a_traiter = []
    
    for doc in documents:
        source = doc["metadata"]["source"]
        file_hash = doc["metadata"]["file_hash"]
        
        # Vérifier si le fichier est nouveau ou modifié
        if source not in fichiers_matieres or fichiers_matieres[source] != file_hash:
            print(f"Fichier nouveau ou modifié détecté: {source}")
            docs_a_traiter.append(doc)
            # Mettre à jour le hash dans les métadonnées
            fichiers_matieres[source] = file_hash
    
    # Si aucun document n'a été modifié, terminer
    if not docs_a_traiter:
        print(f"Aucune mise à jour nécessaire pour la matière {matiere}")
        return False
    
    # Diviser les documents en sections
    sections = []
    for doc in docs_a_traiter:
        doc_sections = split_document(doc)
        sections.extend(doc_sections)
    
    print(f"Nombre total de sections à indexer: {len(sections)}")
    
    # Mettre à jour l'index vectoriel
    vector_store, namespace = upsert_documents(pc, index_name, embeddings, matiere, sections)
    
    # Mettre à jour les métadonnées
    if vector_store:
        metadata["matieres"][matiere]["derniere_mise_a_jour"] = datetime.now().isoformat()
        sauvegarder_metadata(metadata)
        return True
    
    return False

# -----------------------------------------
# Configuration du système RAG
# -----------------------------------------
def setup_rag_system(index_name, embeddings, matiere, custom_prompt=None):
    """
    Configure le système RAG pour une matière spécifique.
    
    Args:
        index_name (str): Nom de l'index Pinecone
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière
        custom_prompt: Prompt personnalisé facultatif
        
    Returns:
        object: Chaîne de récupération configurée
    """
    namespace = get_matiere_namespace(matiere)
    
    # Créer un store vectoriel pour cette matière
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    # Configurer le prompt de question-réponse
    if custom_prompt:
        retrieval_qa_chat_prompt = custom_prompt
    else:
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Configurer le retriever
    retriever = vector_store.as_retriever()
    
    # Configurer le modèle de langage
    llm = ChatOpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name='gpt-4o-mini',
        temperature=0.0
    )
    
    # Créer la chaîne de traitement des documents
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # Créer la chaîne de récupération
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

def creer_prompt_tuteur(matiere):
    """
    Crée un prompt personnalisé pour le tuteur d'une matière spécifique.
    
    Args:
        matiere (str): Identifiant de la matière
    
    Returns:
        ChatPromptTemplate: Template de prompt pour le tuteur
    """
    TEMPLATE_TUTEUR = """
    Vous êtes un tuteur IA spécialisé dans la matière {matiere}, disposant d'un accès direct aux documents de cours via un système de recherche sémantique (RAG).

    Votre tâche est de générer une seule question de réflexion de niveau avancé, en vous basant strictement sur les extraits suivants:

    {context}

    La question doit :
    1. Solliciter l'analyse critique d'un concept ou d'une relation entre plusieurs notions.
    2. Être formulée de manière claire, concise et précise, avec un vocabulaire académique adapté.
    3. Favoriser une réponse argumentée plutôt qu'une simple définition.

    # Instructions

    - N'utilisez que les passages extraits des documents de cours ci-dessus.
    - Ne proposez qu'une question ouverte et directe en une ligne.
    - N'ajoutez ni explication ni sous-questions.
    - Les questions ne doivent pas demander d'analyse.
    - Référez-vous strictement aux extraits listés.

    Votre question: 
    """
    
    return ChatPromptTemplate.from_template(TEMPLATE_TUTEUR).partial(matiere=matiere)

# -----------------------------------------
# Fonctions d'interrogation
# -----------------------------------------
def interroger_matiere(index_name, embeddings, matiere, query, custom_prompt=None):
    """
    Interroge spécifiquement les documents d'une matière.
    
    Args:
        index_name (str): Nom de l'index Pinecone
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière
        query (str): Question de l'utilisateur
        custom_prompt: Prompt personnalisé facultatif
        
    Returns:
        dict: Réponse du système RAG
    """
    # Configurer le système RAG
    retrieval_chain = setup_rag_system(index_name, embeddings, matiere, custom_prompt)
    
    print(f"\n\n--- Requête pour {matiere}: '{query}' ---")
    
    # Exécuter la requête
    response = retrieval_chain.invoke({"input": query})
    
    # Afficher la réponse
    print(f"\nRéponse: {response['answer']}")
    
    # Afficher les documents sources
    print("\nDocuments sources:")
    for i, doc in enumerate(response["context"]):
        print(f"\nDocument {i+1}:")
        source = doc.metadata.get('source', 'Source inconnue')
        print(f"Source: {source}")
        header = ""
        if "Header 2" in doc.metadata:
            header = doc.metadata["Header 2"]
        elif "Header 3" in doc.metadata:
            header = doc.metadata["Header 3"]
        if header:
            print(f"Section: {header}")
        print(f"Contenu: {doc.page_content[:150]}...")
    
    return response

def generer_question_reflexion(index_name, embeddings, matiere, concept_cle):
    """
    Génère une question de réflexion sur un concept clé dans une matière spécifique.
    
    Args:
        index_name (str): Nom de l'index Pinecone
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière (ex: "SYD", "BD")
        concept_cle (str): Concept sur lequel générer une question
        
    Returns:
        str: Question de réflexion générée
    """
    # Créer le prompt tuteur pour cette matière
    tuteur_prompt = creer_prompt_tuteur(matiere)
    
    # Interroger la matière avec ce prompt
    result = interroger_matiere(
        index_name=index_name, 
        embeddings=embeddings,
        matiere=matiere, 
        query=f"Générer une question de réflexion sur le concept: {concept_cle}",
        custom_prompt=tuteur_prompt
    )
    
    return result["answer"]

# -----------------------------------------
# Fonction principale et utilitaires
# -----------------------------------------
def afficher_matieres_disponibles():
    """
    Affiche la liste des matières disponibles dans le dossier des cours.
    
    Returns:
        list: Liste des matières disponibles
    """
    if not os.path.exists(COURS_DIR):
        print("Le dossier des cours n'existe pas encore.")
        return []
    
    matieres = []
    for item in os.listdir(COURS_DIR):
        item_path = os.path.join(COURS_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            matieres.append(item)
    
    if matieres:
        print("Matières disponibles:")
        for matiere in matieres:
            print(f"- {matiere}")
    else:
        print("Aucune matière n'a été trouvée dans le dossier des cours.")
    
    return matieres

def main():
    """
    Fonction principale qui permet d'utiliser le système interactivement.
    """
    print("=== Système de gestion de cours et génération de questions ===")
    
    # Initialiser la structure des dossiers
    initialiser_structure_dossiers()
    
    # Initialiser Pinecone
    pc, index_name, spec = initialize_pinecone()
    
    # Configurer le modèle d'embedding
    embeddings = setup_embeddings()
    
    # Créer ou récupérer l'index
    index = create_or_get_index(pc, index_name, embeddings, spec)
    
    # Afficher les matières disponibles
    matieres = afficher_matieres_disponibles()
    
    # Menu interactif
    while True:
        print("\nQue souhaitez-vous faire?")
        print("1. Mettre à jour les documents d'une matière")
        print("2. Générer une question de réflexion")
        print("3. Poser une question sur une matière")
        print("4. Quitter")
        
        choix = input("Votre choix (1-4): ")
        
        if choix == "1":
            # Mettre à jour une matière
            if not matieres:
                print("Aucune matière disponible. Veuillez ajouter des documents dans le dossier 'cours'.")
                continue
                
            matiere = input(f"Entrez le nom de la matière à mettre à jour ({', '.join(matieres)}): ").upper()
            if matiere not in matieres:
                print(f"Matière '{matiere}' non trouvée.")
                continue
                
            mettre_a_jour_matiere(pc, index_name, embeddings, matiere)
            
        elif choix == "2":
            # Générer une question de réflexion
            if not matieres:
                print("Aucune matière disponible.")
                continue
                
            matiere = input(f"Entrez le nom de la matière ({', '.join(matieres)}): ").upper()
            if matiere not in matieres:
                print(f"Matière '{matiere}' non trouvée.")
                continue
                
            concept = input("Entrez le concept sur lequel générer une question: ")
            
            try:
                question = generer_question_reflexion(index_name, embeddings, matiere, concept)
                print(f"\nQuestion générée: {question}")
            except Exception as e:
                print(f"Erreur lors de la génération de la question: {e}")
            
        elif choix == "3":
            # Poser une question
            if not matieres:
                print("Aucune matière disponible.")
                continue
                
            matiere = input(f"Entrez le nom de la matière ({', '.join(matieres)}): ").upper()
            if matiere not in matieres:
                print(f"Matière '{matiere}' non trouvée.")
                continue
                
            question = input("Entrez votre question: ")
            
            try:
                interroger_matiere(index_name, embeddings, matiere, question)
            except Exception as e:
                print(f"Erreur lors de la recherche: {e}")
            
        elif choix == "4":
            # Quitter
            print("Au revoir!")
            break
            
        else:
            print("Choix non valide. Veuillez entrer un nombre entre 1 et 4.")

if __name__ == "__main__":
    main() 