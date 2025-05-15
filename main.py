"""
RAG-Chatbot - Système de génération augmentée par récupération
Ce script implémente un système RAG pour la documentation du WonderVector5000
"""

# -----------------------------------------
# Imports et configuration d'environnement
# -----------------------------------------
import os
import time
import uuid
import warnings
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
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Imports pour la base de données vectorielle
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

# Imports pour le système RAG
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# -----------------------------------------
# Données de démonstration
# -----------------------------------------
def get_markdown_document():
    """
    Renvoie le document markdown de démonstration sur le WonderVector5000.
    
    Returns:
        str: Le contenu markdown complet
    """
    return """## Introduction

Welcome to the whimsical world of the WonderVector5000, an astonishing leap into the realms of imaginative technology. This extraordinary device, borne of creative fancy, promises to revolutionize absolutely nothing while dazzling you with its fantastical features. Whether you're a seasoned technophile or just someone looking for a bit of fun, the WonderVector5000 is sure to leave you amused and bemused in equal measure. Let's explore the incredible, albeit entirely fictitious, specifications, setup process, and troubleshooting tips for this marvel of modern nonsense.

## Product overview

The WonderVector5000 is packed with features that defy logic and physics, each designed to sound impressive while maintaining a delightful air of absurdity:

- Quantum Flibberflabber Engine: The heart of the WonderVector5000, this engine operates on principles of quantum flibberflabber, a phenomenon as mysterious as it is meaningless. It's said to harness the power of improbability to function seamlessly across multiple dimensions.

- Hyperbolic Singularity Matrix: This component compresses infinite possibilities into a singular hyperbolic state, allowing the device to predict outcomes with 0% accuracy, ensuring every use is a new adventure.

- Aetherial Flux Capacitor: Drawing energy from the fictional aether, this flux capacitor provides unlimited power by tapping into the boundless reserves of imaginary energy fields.

- Multi-Dimensional Holo-Interface: Interact with the WonderVector5000 through its holographic interface that projects controls and information in three-and-a-half dimensions, creating a user experience that's simultaneously futuristic and perplexing.

- Neural Fandango Synchronizer: This advanced feature connects directly to the user's brain waves, converting your deepest thoughts into tangible actions—albeit with results that are whimsically unpredictable.

- Chrono-Distortion Field: Manipulate time itself with the WonderVector5000's chrono-distortion field, allowing you to experience moments before they occur or revisit them in a state of temporal flux.

## Use cases

While the WonderVector5000 is fundamentally a device of fiction and fun, let's imagine some scenarios where it could hypothetically be applied:

- Time Travel Adventures: Use the Chrono-Distortion Field to visit key moments in history or glimpse into the future. While actual temporal manipulation is impossible, the mere idea sparks endless storytelling possibilities.

- Interdimensional Gaming: Engage with the Multi-Dimensional Holo-Interface for immersive, out-of-this-world gaming experiences. Imagine games that adapt to your thoughts via the Neural Fandango Synchronizer, creating a unique and ever-changing environment.

- Infinite Creativity: Harness the Hyperbolic Singularity Matrix for brainstorming sessions. By compressing infinite possibilities into hyperbolic states, it could theoretically help unlock unprecedented creative ideas.

- Energy Experiments: Explore the concept of limitless power with the Aetherial Flux Capacitor. Though purely fictional, the notion of drawing energy from the aether could inspire innovative thinking in energy research.

## Getting started

Setting up your WonderVector5000 is both simple and absurdly intricate. Follow these steps to unleash the full potential of your new device:

1. Unpack the Device: Remove the WonderVector5000 from its anti-gravitational packaging, ensuring to handle with care to avoid disturbing the delicate balance of its components.

2. Initiate the Quantum Flibberflabber Engine: Locate the translucent lever marked "QFE Start" and pull it gently. You should notice a slight shimmer in the air as the engine engages, indicating that quantum flibberflabber is in effect.

3. Calibrate the Hyperbolic Singularity Matrix: Turn the dials labeled 'Infinity A' and 'Infinity B' until the matrix stabilizes. You'll know it's calibrated correctly when the display shows a single, stable "infinity symbol".

4. Engage the Aetherial Flux Capacitor: Insert the EtherKey into the designated slot and turn it clockwise. A faint humming sound should confirm that the aetherial flux capacitor is active.

5. Activate the Multi-Dimensional Holo-Interface: Press the button resembling a floating question mark to activate the holo-interface. The controls should materialize before your eyes, slightly out of phase with reality.

6. Synchronize the Neural Fandango Synchronizer: Place the neural headband on your forehead and think of the word "Wonder". The device will sync with your thoughts, a process that should take just a few moments.

7. Set the Chrono-Distortion Field: Use the temporal sliders to adjust the time settings. Recommended presets include "Past", "Present", and "Future", though feel free to explore other, more abstract temporal states.

## Troubleshooting

Even a device as fantastically designed as the WonderVector5000 can encounter problems. Here are some common issues and their solutions:

- Issue: The Quantum Flibberflabber Engine won't start.

    - Solution: Ensure the anti-gravitational packaging has been completely removed. Check for any residual shards of improbability that might be obstructing the engine.

- Issue: The Hyperbolic Singularity Matrix displays "infinity infinity".

    - Solution: This indicates a hyper-infinite loop. Reset the dials to zero and then adjust them slowly until the display shows a single, stable infinity symbol.

- Issue: The Aetherial Flux Capacitor isn't engaging.

    - Solution: Verify that the EtherKey is properly inserted and genuine. Counterfeit EtherKeys can often cause malfunctions. Replace with an authenticated EtherKey if necessary.

- Issue: The Multi-Dimensional Holo-Interface shows garbled projections.

    - Solution: Realign the temporal resonators by tapping the holographic screen three times in quick succession. This should stabilize the projections.

- Issue: The Neural Fandango Synchronizer causes headaches.

    - Solution: Ensure the headband is properly positioned and not too tight. Relax and focus on simple, calming thoughts to ease the synchronization process.

- Issue: The Chrono-Distortion Field is stuck in the past.

    - Solution: Increase the temporal flux by 5%. If this fails, perform a hard reset by holding down the "Future" slider for ten seconds."""

# -----------------------------------------
# Fonctions de traitement du texte
# -----------------------------------------
def split_document(markdown_document):
    """
    Divise un document markdown en sections basées sur les en-têtes.
    
    Args:
        markdown_document (str): Document markdown à découper
        
    Returns:
        list: Liste des sections découpées avec leurs métadonnées
    """
    headers_to_split_on = [("##", "Header 2")]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    
    splits = markdown_splitter.split_text(markdown_document)
    
    # Afficher des infos de débogage sur les morceaux créés
    print(f"Nombre de sections: {len(splits)}")
    if splits:
        print(f"Métadonnées de la première section: {splits[0].metadata}")
        print(f"Échantillon de contenu de la première section: {splits[0].page_content[:100]}...")
    else:
        print("ATTENTION: Aucune section n'a été créée!")
        
    return splits

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
    index_name = "rag-getting-started"
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    return pc, index_name, spec

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
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
        # Attendre que l'index s'initialise
        time.sleep(10)
    
    # Afficher l'état de l'index avant l'insertion
    print("État de l'index avant insertion:")
    index_stats = pc.Index(index_name).describe_index_stats()
    print(index_stats)
    print("\n")
    
    return pc.Index(index_name)

def manage_namespace(index, namespace):
    """
    Gère l'espace de noms dans l'index, en créant un nouvel espace si nécessaire.
    
    Args:
        index: Index Pinecone
        namespace (str): Nom de l'espace
        
    Returns:
        str: Nom final de l'espace utilisé
    """
    try:
        namespaces = index.describe_index_stats()['namespaces']
        if namespace in namespaces:
            print(f"Suppression de l'espace de noms existant: {namespace}")
            # On ne peut pas directement supprimer un espace de noms, donc on en crée un nouveau unique
            new_namespace = f"{namespace}-{uuid.uuid4()}"
            print(f"Utilisation du nouvel espace de noms: {new_namespace}")
            return new_namespace
        return namespace
    except Exception as e:
        print(f"Erreur lors de la vérification des espaces de noms: {e}")
        return namespace

def upsert_documents(splits, index_name, embeddings, namespace):
    """
    Insère les documents dans la base de données vectorielle.
    
    Args:
        splits (list): Liste des sections de document à insérer
        index_name (str): Nom de l'index
        embeddings: Modèle d'embedding
        namespace (str): Espace de noms à utiliser
        
    Returns:
        PineconeVectorStore: Store vectoriel avec les documents insérés
    """
    try:
        print(f"Tentative d'insertion de {len(splits)} documents dans l'espace '{namespace}'")
        vector_store = PineconeVectorStore.from_documents(
            documents=splits,
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace
        )
        print("Insertion réussie")
        return vector_store
    except Exception as e:
        print(f"Erreur lors de l'insertion: {e}")
        return None

# -----------------------------------------
# Configuration du système RAG
# -----------------------------------------
def setup_rag_system(vector_store, custom_prompt=None):
    """
    Configure le système RAG avec le retriever et le modèle de langage.
    
    Args:
        vector_store: Base de données vectorielle contenant les documents
        custom_prompt: Prompt personnalisé à utiliser à la place du prompt par défaut
        
    Returns:
        object: Chaîne de récupération configurée
    """
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

# -----------------------------------------
# Fonctions d'interrogation
# -----------------------------------------
def run_rag_query(retrieval_chain, query):
    """
    Exécute une requête sur le système RAG et affiche les résultats.
    
    Args:
        retrieval_chain: Chaîne de récupération configurée
        query (str): Question de l'utilisateur
        
    Returns:
        dict: Réponse complète du système RAG
    """
    print(f"\n\n--- Requête RAG: '{query}' ---")
    
    # Exécuter la requête
    response = retrieval_chain.invoke({"input": query})
    
    # Afficher la réponse
    print(f"\nRéponse: {response['answer']}")
    
    # Afficher les documents sources
    print("\nDocuments sources:")
    for i, doc in enumerate(response["context"]):
        print(f"\nDocument {i+1}:")
        print(f"Section: {doc.metadata.get('Header 2', 'Inconnue')}")
        print(f"Contenu: {doc.page_content[:150]}...")
    
    return response

def verify_vector_store(index, namespace, embeddings):
    """
    Vérifie que les vecteurs existent en effectuant une requête de test.
    
    Args:
        index: Index Pinecone
        namespace (str): Espace de noms à interroger
        embeddings: Modèle d'embedding
    """
    try:
        query = "What are the use cases for WonderVector5000?"
        query_embedding = embeddings.embed_query(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=1,
            namespace=namespace,
            include_metadata=True
        )
        
        print(f"Résultats de la requête de vérification: {results}")
    except Exception as e:
        print(f"Erreur lors de la requête: {e}")

def interroger_matiere(vector_store, matiere, query, custom_prompt=None, index_name=None, embeddings=None):
    """
    Interroge spécifiquement les documents d'une matière.
    
    Args:
        vector_store: Base de données vectorielle
        matiere (str): Identifiant de la matière
        query (str): Question de l'utilisateur
        custom_prompt: Prompt personnalisé facultatif
        index_name (str): Nom de l'index Pinecone (obligatoire si vector_store ne fournit pas cette info)
        embeddings: Modèle d'embedding (obligatoire si vector_store ne fournit pas cette info)
        
    Returns:
        dict: Réponse du système RAG
    """
    namespace = f"matiere-{matiere.lower()}"
    
    # Obtenir le nom de l'index et l'embedding soit à partir des paramètres, soit du vector_store
    if index_name is None:
        if hasattr(vector_store, 'index_name'):
            index_name = vector_store.index_name
        else:
            raise ValueError("Veuillez fournir un index_name, car vector_store ne le fournit pas")
            
    if embeddings is None:
        if hasattr(vector_store, 'embedding'):
            embeddings = vector_store.embedding
        else:
            raise ValueError("Veuillez fournir un embeddings, car vector_store ne le fournit pas")
    
    # Créer un nouveau store avec le namespace spécifique
    matiere_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    # Configurer le système RAG avec ce store
    retrieval_chain = setup_rag_system(matiere_store, custom_prompt)
    
    # Exécuter la requête
    return run_rag_query(retrieval_chain, query)

# Template pour le tuteur
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

def generer_question_reflexion(vector_store, matiere, concept_cle, index_name=None, embeddings=None):
    """
    Génère une question de réflexion sur un concept clé dans une matière spécifique.
    
    Args:
        vector_store: Base de données vectorielle
        matiere (str): Identifiant de la matière (ex: "SYD", "BD")
        concept_cle (str): Concept sur lequel générer une question
        index_name (str): Nom de l'index Pinecone
        embeddings: Modèle d'embedding
        
    Returns:
        str: Question de réflexion générée
    """
    # Créer le prompt tuteur pour cette matière
    tuteur_prompt = creer_prompt_tuteur(matiere)
    
    # Interroger la matière avec ce prompt
    result = interroger_matiere(
        vector_store=vector_store, 
        matiere=matiere, 
        query=f"Générer une question de réflexion sur le concept: {concept_cle}",
        custom_prompt=tuteur_prompt,
        index_name=index_name,
        embeddings=embeddings
    )
    
    return result["answer"]

# -----------------------------------------
# Fonction principale
# -----------------------------------------
def main():
    """
    Fonction principale qui exécute tout le pipeline RAG.
    """
    # Récupérer le document markdown
    markdown_document = get_markdown_document()
    
    # Diviser le document en sections
    splits = split_document(markdown_document)
    
    # Initialiser Pinecone
    pc, index_name, spec = initialize_pinecone()
    
    # Configurer le modèle d'embedding
    embeddings = setup_embeddings()
    
    # Créer ou récupérer l'index
    index = create_or_get_index(pc, index_name, embeddings, spec)
    
    # Gérer l'espace de noms
    namespace = "wondervector5000"
    namespace = manage_namespace(index, namespace)
    
    # Insérer les documents
    vector_store = upsert_documents(splits, index_name, embeddings, namespace)
    
    # Attendre la fin de l'insertion
    time.sleep(10)
    
    # Vérifier l'état de l'index après l'insertion
    print("État de l'index après insertion:")
    print(pc.Index(index_name).describe_index_stats())
    print("\n")
    
    # Vérifier que les vecteurs existent
    verify_vector_store(index, namespace, embeddings)
    
    # Configurer le système RAG
    retrieval_chain = setup_rag_system(vector_store)
    
    # Exécuter des requêtes de test
    run_rag_query(retrieval_chain, "What are the features of the WonderVector5000?")
    run_rag_query(retrieval_chain, "How do I troubleshoot the Quantum Flibberflabber Engine?")

def initialiser_matiere(pc, index_name, embeddings, matiere, documents):
    """
    Initialise ou met à jour l'espace vectoriel pour une matière spécifique.
    
    Args:
        pc: Client Pinecone
        index_name (str): Nom de l'index
        embeddings: Modèle d'embedding
        matiere (str): Identifiant de la matière (ex: "SYD", "BD", etc.)
        documents (list): Liste des documents à traiter pour cette matière
        
    Returns:
        tuple: (store vectoriel, namespace utilisé)
    """
    # Créer un namespace spécifique à la matière
    namespace = f"matiere-{matiere.lower()}"
    
    # Récupérer l'index
    index = pc.Index(index_name)
    
    # Gérer le namespace (vérifier s'il existe déjà)
    namespace = manage_namespace(index, namespace)
    
    # Traiter et insérer les documents
    splits = []
    for doc in documents:
        if isinstance(doc, str):
            # Si c'est un texte brut, le découper
            doc_splits = split_document(doc)
            splits.extend(doc_splits)
        elif hasattr(doc, 'page_content'):
            # Si c'est déjà un Document LangChain
            splits.append(doc)
        else:
            print(f"Format de document non supporté: {type(doc)}")
    
    # Insérer les documents dans l'espace vectoriel
    vector_store = upsert_documents(splits, index_name, embeddings, namespace)
    
    return vector_store, namespace

# Exemple d'utilisation du système avec plusieurs matières
def demo_rag_multi_matieres():
    """
    Démontre l'utilisation du système RAG avec plusieurs matières.
    """
    # Initialiser Pinecone
    pc, index_name, spec = initialize_pinecone()
    
    # Configurer le modèle d'embedding
    embeddings = setup_embeddings()
    
    # Créer ou récupérer l'index
    index = create_or_get_index(pc, index_name, embeddings, spec)
    
    # Documents pour la matière SYD (exemple)
    docs_syd = [
        "## Concepts fondamentaux en SYD\n\nLes systèmes distribués sont caractérisés par leur capacité à fonctionner sur plusieurs nœuds interconnectés...",
        "## Synchronisation dans les systèmes distribués\n\nLa synchronisation est un défi majeur dans les systèmes distribués en raison des délais de communication variables..."
    ]
    
    # Documents pour la matière BD (exemple)
    docs_bd = [
        "## Normalisation des bases de données\n\nLa normalisation est un processus de conception qui vise à minimiser la redondance et à améliorer l'intégrité des données...",
        "## Transactions et ACID\n\nLes propriétés ACID garantissent la fiabilité des transactions dans les systèmes de gestion de bases de données..."
    ]
    
    # Initialiser les matières
    store_syd, ns_syd = initialiser_matiere(pc, index_name, embeddings, "SYD", docs_syd)
    store_bd, ns_bd = initialiser_matiere(pc, index_name, embeddings, "BD", docs_bd)
    
    # Attendre que l'indexation soit terminée
    time.sleep(5)
    
    # Interroger chaque matière
    print("\n=== Questions pour SYD ===")
    interroger_matiere(
        vector_store=store_syd, 
        matiere="SYD", 
        query="Quels sont les défis de la synchronisation?",
        index_name=index_name,
        embeddings=embeddings
    )
    
    print("\n=== Questions pour BD ===")
    interroger_matiere(
        vector_store=store_bd, 
        matiere="BD", 
        query="Expliquez les propriétés ACID",
        index_name=index_name,
        embeddings=embeddings
    )
    
    # Générer une question de réflexion
    print("\n=== Génération de question de réflexion pour SYD ===")
    question = generer_question_reflexion(store_syd, "SYD", "synchronisation", index_name, embeddings)
    print(f"Question générée: {question}")

# Exécuter la fonction principale si le script est exécuté directement
if __name__ == "__main__":
    # Choisir quelle démonstration exécuter
    # main()  # Démonstration standard avec WonderVector5000
    demo_rag_multi_matieres()  # Démonstration avec plusieurs matières
    # main()  # Par défaut, exécuter la démo standard
