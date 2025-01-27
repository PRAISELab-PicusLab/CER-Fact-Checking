import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from openai import OpenAI
import re
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from newsplease import NewsPlease
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu
# https://fbe.unimelb.edu.au/newsroom/fake-news-in-the-age-of-covid-19

# Percorsi dei file
embeddings_file = r"data\abstract_embeddings.npy"
pmid_file = r"data\pmids.npy"
faiss_index_file = r"data\faiss_index.index"
file_path = r'data\parte_205.csv'
#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
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Carica i dati
data = pd.read_csv(file_path)

# Inizializza il modello
model = SentenceTransformer('all-MiniLM-L6-v2')  # Puoi cambiare modello se necessario

def get_article_data(url):
    try:
        # Effettua una richiesta HTTP all'URL specificato
        response = requests.get(url)
        # Verifica che la risposta sia positiva (status code 200)
        response.raise_for_status()

        # Estrae il contenuto HTML della pagina
        html_content = response.text

        # Utilizza NewsPlease per analizzare il contenuto dell'articolo
        article = NewsPlease.from_html(html_content, url=url)

        # Restituisci i dati strutturati dell'articolo
        return {
            "title": article.title,
            "authors": article.authors,
            "date_publish": article.date_publish,
            "content": article.maintext,
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Errore durante il recupero dell'URL: {e}"}
    except Exception as e:
        return {"error": f"Errore durante l'elaborazione dell'articolo: {e}"}

def extract_and_split_claims(claims):
    """Estrae e divide i claims da un testo"""
    start_index = claims.find("Claim 1:")
    if start_index != -1:
        claims = claims[start_index:]

    claim_lines = claims.strip().split("\n\n")

    claims_dict = {}
    for i, claim in enumerate(claim_lines, start=1):
        claims_dict[f"Claim_{i}"] = claim

    for var_name, claim_text in claims_dict.items():
        globals()[var_name] = claim_text

    return claims_dict

def extract_label_and_score(result):
    """
    Estrae la prima label e il primo score da una stringa data.

    Parametri:
        result (str): Stringa contenente i dati (formato JSON-like).
    
    Ritorna:
        tuple: (predicted_label, score_label), dove:
               - predicted_label (str) è la prima label trovata.
               - score_label (float) è il primo score trovato.
    """
    # Estrarre la prima label
    label_match = re.search(r"'labels': \['(.*?)'", result)
    predicted_label = label_match.group(1) if label_match else None

    # Estrarre il primo score
    score_match = re.search(r"'scores': \[(\d+\.\d+)", result)
    score_label = float(score_match.group(1)) if score_match else None

    return predicted_label, score_label

# Funzione per estrarre e pulire le frasi da un elenco di stringhe
def clean_phrases(phrases, pattern):
    """
    Pulisce le frasi eliminando informazioni aggiuntive e restituendo un dizionario.

    Args:
    - phrases (list): Lista di stringhe contenenti frasi con riferimenti nel formato
      "testo della frase" (abstract_N).
    - pattern (str): Pattern per estrarre il testo della frase e il riferimento all'abstract.

    Returns:
    - list: Lista di dizionari con "text" e "abstract".
    """
    cleaned_phrases = []
    for phrase in phrases:
        matches = re.findall(pattern, phrase)
        cleaned_phrases.extend([{"text": match[0], "abstract": f"abstract_{match[1]}"} for match in matches])
    return cleaned_phrases

# Funzione per evidenziare le frasi
def highlight_phrases(abstract_text, phrases, color, label):
    """
    Evidenzia le frasi corrispondenti con un background colorato specifico, con possibilità di invertire i colori.

    Args:
    - abstract_text (str): Testo dell'abstract.
    - phrases (list): Lista di dizionari contenenti frasi e abstract di riferimento.
    - color (str): Colore per evidenziare il background delle frasi.
    - label (str): Se "false", inverte i colori; se "true" o "NEI", non fa niente.

    Returns:
    - str: Abstract con le frasi evidenziate.
    """
    # Inverti i colori se label è "false"
    if label.lower() == "false":
        color = "lightgreen" if color == "red" else color

    # Evidenzia le frasi con il background colorato specificato
    for phrase in phrases:
        abstract_text = re.sub(
            re.escape(phrase["text"]),
            f'<span style="background-color: {color}; font-weight: bold; border: 1px solid black; border-radius: 5px;">{phrase["text"]}</span>',
            abstract_text,
            flags=re.IGNORECASE
        )
    return abstract_text

def parse_response(response):
    # Valori di default per ogni campo
    first_label = "Non trovato"
    justification = "Non trovato"
    supporting = "Non trovato"
    refusing = "Non trovato"
    notes = "Non trovato"

    # Pattern per ciascun campo
    patterns = {
        "first_label": r"Label:\s*(.*?)\n",
        "justification": r"Justification:\s*(.*?)(?=\nSupporting sentences)",
        "supporting": r"Supporting sentences from abstracts:\n(.*?)(?=\nRefusing sentences)",
        "refusing": r"Refusing sentences from abstracts:\n(.*?)(?=\nNote:)",
        "notes": r"Note:\s*(.*)"
    }

    # Parsing dei campi
    if match := re.search(patterns["first_label"], response, re.DOTALL):
        first_label = match.group(1).strip()
    if match := re.search(patterns["justification"], response, re.DOTALL):
        justification = match.group(1).strip()
    if match := re.search(patterns["supporting"], response, re.DOTALL):
        supporting = [{"text": sentence.strip(), "abstract": f"abstract_{i+1}"} for i, sentence in enumerate(match.group(1).strip().split('\n'))]
    if match := re.search(patterns["refusing"], response, re.DOTALL):
        refusing = [{"text": sentence.strip(), "abstract": f"abstract_{i+1}"} for i, sentence in enumerate(match.group(1).strip().split('\n'))]
    if match := re.search(patterns["notes"], response, re.DOTALL):
        notes = match.group(1).strip()

    # Restituisce le variabili separate
    return first_label, justification, supporting, refusing, notes

def load_embeddings(embeddings_file, pmid_file, faiss_index_file):
    if not (os.path.exists(embeddings_file) and os.path.exists(pmid_file) and os.path.exists(faiss_index_file)):
        raise FileNotFoundError("Uno o più file salvati non sono stati trovati.")

    # Carica embeddings e PMID
    embeddings = np.load(embeddings_file)
    pmids = np.load(pmid_file, allow_pickle=True)

    # Carica l'indice FAISS
    index = faiss.read_index(faiss_index_file)

    print("Embeddings e indice FAISS caricati con successo.")
    return embeddings, pmids, index

def retrieve_top_abstracts(claim, model, index, pmids, data, top_k=5):
    claim_embedding = model.encode([claim])
    faiss.normalize_L2(claim_embedding)  # Normalizza l'embedding della claim
    distances, indices = index.search(claim_embedding, top_k)

    results = []
    for j, i in enumerate(indices[0]):
        pmid = pmids[i]
        abstract_text = data[data['PMID'] == pmid]['AbstractText'].values[0]
        distance = distances[0][j]
        results.append((abstract_text, pmid, distance))

    return results

def generate_justification(query, justification):
    # Frase di giustificazione e classi
    Class = ["True", "False","NEI"]

    # Preparare la stringa di giustificazione
    justification_text = (
        f'Justification: "{justification}"'
    )

    # Limitare la lunghezza della giustificazione a 512 caratteri
    max_length = 512
    if len(justification_text) > max_length:
        justification_text = justification_text[:max_length]

    # Generare output con il classificatore
    output = zeroshot_classifier(
        query,
        Class,
        hypothesis_template=f"The claim is '{{}}' for: {justification_text}",
        multi_label=False
    )

    # Preparare la giustificazione finale
    final_justification = f'{output}.'

    return final_justification


# Title
st.set_page_config(page_title="CER - Combining Evidence and Reasoning Demo", layout="wide", initial_sidebar_state="collapsed")

# horizontal menu
page = option_menu(None, ["Single claim check", "Page check"], 
    icons=['check', 'ui-checks'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
print(page)


st.markdown("<h1 style='text-align: center; color: black;'>✔️✨ CER - Biomedical Fact Checker</h1>", unsafe_allow_html=True)

# Sidebar per la navigazione
st.sidebar.title("🔬 Combining Evidence and Reasoning Demo")
st.sidebar.caption("🔍 Fact-check biomedical claims using scientific evidence and reasoning.")
st.sidebar.markdown("---")
st.sidebar.caption("#### ℹ️ About")
st.sidebar.caption("This is a demo application for fact-checking biomedical claims using scientific evidence and reasoning. It uses a combination of language models, scientific literature, and reasoning to provide explanations for the predictions.")

# Carica embeddings e indice FAISS una sola volta
if 'embeddings_loaded' not in st.session_state:
    embeddings, pmids, index = load_embeddings(embeddings_file, pmid_file, faiss_index_file)
    st.session_state.embeddings = embeddings
    st.session_state.pmids = pmids
    st.session_state.index = index
    st.session_state.embeddings_loaded = True
else:
    embeddings = st.session_state.embeddings
    pmids = st.session_state.pmids
    index = st.session_state.index

# Mantieni la claim nello stato della sessione
if 'claim' not in st.session_state:
    st.session_state.claim = ""

if 'top_abstracts' not in st.session_state:
    st.session_state.top_abstracts = []

if page == "Single claim check":
    st.subheader("Single claim check")
    st.caption("✨ Enter a single claim to fact-check and hit the button to see the results! 🔍")

    # # Interfaccia utente
    # st.text_input(
    #     "Claim to fact-check:",
    #     value=st.session_state.claim,
    #     key="claim",  # Associa il valore allo stato della sessione
    #     on_change=lambda: None  # Funzione vuota per il binding
    # )
    st.session_state.claim = st.text_input("Claim to fact-check:")

    if st.button("✨ Fact Check"):

        if st.session_state.claim:
            top_abstracts = retrieve_top_abstracts(st.session_state.claim, model, index, pmids, data, top_k=5)
            st.session_state.top_abstracts = top_abstracts  # Salva i risultati

            st.markdown("### **Results**")

            with st.container():
                for i, (abstract, pmid, distance) in enumerate(st.session_state.top_abstracts, 1):
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    globals()[f"abstract_{i}"] = abstract
                    globals()[f"reference_{i}"] = pubmed_url
                    globals()[f"distance_{i}"] = distance
                prompt_template = f'''[INST] '''

                with st.spinner('🔍 We are checking...'):
                    try:
                        # Preleva la domanda dal DataFrame
                        query = st.session_state.claim

                        # Costruisci il prompt
                        prompt_template = f'''[INST]  <<SYS>>

                        You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

                        Use the information in Context

                        elaborate the context to generate a new information.

                        Use only the knowledge in Context to answer.

                        Answer describing in a scentific way. Be formal during the answer. Use the third person.

                        Answer without mentioning the context. Use it but don't refer to it in the text

                        to answer, use max 300 word

                        Create a Justification from the sentences given.

                        Use the structure: Justification: .... (don't use the word context)

                        Write as an online doctor to create the justification.

                        After, give some sentences from Context from scientific papers: that supports the label and reject the label

                        Supporting sentences from abstracts:
                        information sentence from abstract_1:
                        information sentence from abstract_2: 
                        ..
                        Refusing sentences from abstracts:
                        information sentence from abstract_1:
                        information sentence from abstract_2: 
                        ..
                        Add where it comes from (abstract_1, abstract_2, abstract_3, abstract_4, abstract_5)

                        with the answer, gives a line like: "Label:". Always put Label as first. After Label, give the justification
                        The justification will be always given as Justification:
                        Label can be yes, no, NEI, where yes: claim is true. no: claim is false. NEI: not enough information.
                        The Label will be chosen with a voting system of support/refuse before
                        <<SYS>>

                        Question: {query} [/INST]
                        Context from scientific papers: {abstract_1} ; {abstract_2} ; {abstract_3} ; {abstract_4} ; {abstract_5} [/INST]
                        '''

                        # Chiamata API
                        completion = client.chat.completions.create(
                        model="meta/llama-3.1-405b-instruct",
                        messages=[{"role": "user", "content": prompt_template}],
                        temperature=0.1,
                        top_p=0.7,
                        max_tokens=1024,
                        stream=True
                        )

                        # Raccogli la risposta
                        Risposta = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                Risposta += chunk.choices[0].delta.content

                        # Debug: Controlla la risposta
                        #st.write(Risposta)

                    except Exception as e:
                        st.write(f"Error processing index: {e}")

                with st.spinner('🤔💬 Justifying the check...'):
                    # Esegui il parsing e separa le variabili
                    zeroshot_classifier = pipeline(
                    "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
                    )
                    first_label, justification, supporting, refusing, notes = parse_response(Risposta)
                    
                with st.spinner('🕵️‍♂️📜 We are finding evidence...'):      
                    result = generate_justification(st.session_state.claim, justification)
                    predicted_label, score_label = extract_label_and_score(result)

                    if predicted_label == "True":
                        color = f"rgba(0, 204, 0, {score_label})"  # Verde
                    elif predicted_label == "False":
                        color = f"rgba(204, 0, 0, {score_label})"  # Rosso
                    elif predicted_label == "NEI":
                        color = f"rgba(255, 255, 0, {score_label})"  # Giallo
                    else:
                        color = "black"  # Default color

                    st.caption(f"📝 The Claim: {st.session_state.claim}")
                    #st.markdown(f'{st.session_state.claim} is <span style="color: {color}; font-weight: bold;">{predicted_label}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Prediction of claim:** <span style='color: {color}; font-weight: bold;'>{predicted_label}</span>", unsafe_allow_html=True)

                    st.markdown("### **Justification**")
                    st.markdown(f'<p> {justification}</p>', unsafe_allow_html=True)
                    abstracts = {
                        "abstract_1": abstract_1,
                        "abstract_2": abstract_2,
                        "abstract_3": abstract_3,
                        "abstract_4": abstract_4,
                        "abstract_5": abstract_5
                    }
                    supporting_texts = []
                    for item in supporting:
                        try:
                            supporting_texts.append(item["text"])
                        except (TypeError, KeyError):
                            continue

                    refusing_text = []
                    for item in refusing:
                        try:
                            refusing_text.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    pattern = r'"\s*(.*?)\s*"\s*\(abstract_(\d+)\)'
                    #st.write(supporting)
                    #st.write(supporting_texts)
                    supporting = clean_phrases(supporting_texts, pattern)
                    #st.write(supporting)
                    refusing = clean_phrases(refusing_text, pattern)
                    processed_abstracts = {}
                    for abstract_name, abstract_text in abstracts.items():
                        # Evidenzia frasi di supporto in verde
                        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, supporting_matches, "lightgreen", predicted_label)
                        
                        # Evidenzia frasi di rifiuto in rosso
                        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, refusing_matches, "red", predicted_label)
                        
                        # Aggiungi solo abstract che hanno frasi evidenziate in verde
                        if supporting_matches:
                            # Aggiungi la reference se esiste una variabile corrispondente
                            reference_variable = f"reference_{abstract_name.split('_')[1]}"  # Genera il nome della variabile
                            if reference_variable in globals():  # Controlla se la variabile esiste
                                reference_value = globals()[reference_variable]
                                abstract_text += f"<br><br><strong>🔗 Reference:</strong> {reference_value}"
                            
                            # Aggiungi l'abstract processato
                            processed_abstracts[abstract_name] = abstract_text

                    # Itera sugli abstract processati ed elimina duplicati
                    seen_contents = set()  # Insieme per tracciare contenuti già visti
                    evidence_counter = 1
                    # Visualizza i risultati degli abstract processati con expander numerati
                    st.markdown("### **Scientific Evidence**")
                    # Aggiungi una legenda per i colori
                    legend_html = """
                    <div style="display: flex; flex-direction: column; align-items: flex-start;">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: lightgreen; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Positive Evidence</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Negative Evidence</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Dubious Evidence</div>
                    </div>
                    </div>
                    """
                    col1, col2 = st.columns([0.8, 0.2])
                    
                    with col1:
                        if processed_abstracts:
                            tabs = st.tabs([f"Scientific Evidence {i}" for i in range(1, len(processed_abstracts) + 1)])
                            for tab, (name, content) in zip(tabs, processed_abstracts.items()):
                                if content not in seen_contents:  # Aggiungi solo se non è già stato visto
                                    seen_contents.add(content)
                                    with tab:
                                        # Usa `st.write` per visualizzare HTML direttamente
                                        st.write(content, unsafe_allow_html=True)
                        else:
                            st.markdown("No relevant Scientific Evidence found")
                    
                    with col2:
                        st.caption("Legend")
                        st.markdown(legend_html, unsafe_allow_html=True)

elif page == "Page check":
    st.subheader("Page check")
    st.caption("✨ Enter a URL to fact-check the health-related claims on the page and hit the button to see the results! 🔍")
    url = st.text_input("URL to fact-check:")

    if st.button("✨ Fact Check") and url:

        st.session_state.true_count = 0
        st.session_state.false_count = 0
        st.session_state.nei_count = 0

        with st.spinner('🌐🔍 Extracting claims...'):
            article_data = get_article_data(url)

            try:
                # Costruisci il prompt
                prompt_template = f'''[INST]  <<SYS>>

                You are an expert scraper. Your task is to extract from the url health related question.

                the url from extract the context and the clam is: {article_data}

                Create simple claim of single sentence.

                Dont's use *

                Give just the claim. Don't write other things

                Extract only health related claim.

                Rank eventual claim like:

                Claim 1:
                Claim 2:
                Claim 3:
                
                Use always this structure.
                Start every claim with "Claim " followed by the number

                The number of claims may go from 1 to a max of 5 

                The claims have to be always health related

                
                '''

                # Chiamata API
                completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt_template}],
                temperature=0.1,
                top_p=0.7,
                max_tokens=1024,
                stream=True
                )

                # Raccogli la risposta
                Answer = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        Answer += chunk.choices[0].delta.content

                # Debug: Controlla la risposta
                print(f"{Answer}")

            except Exception as e:
                print(f"Error {e}")

            claims_dict = extract_and_split_claims(Answer)

        # Visualizza le claim su Streamlit con expander
        st.markdown("### **Claims Extracted**")
        st.caption("🔍 Here are the health-related claims extracted from the page:")
        cols = st.columns(3)
        for i, (claim_key, claim_text) in enumerate(claims_dict.items(), 1):
            col = cols[(i - 1) % 3]
            with col.expander(f"Claim {i} 📝", expanded=True):
                st.write(claim_text)

        st.markdown("### **Results**")
        st.caption("🔍 Here are the results for the extracted claims:")
        for claim_key, claim_text in claims_dict.items():
            st.session_state.claim = claim_text
            if st.session_state.claim:
                top_abstracts = retrieve_top_abstracts(st.session_state.claim, model, index, pmids, data, top_k=5)
                st.session_state.top_abstracts = top_abstracts  # Salva i risultati

            with st.expander(f"✔️ **Results for {claim_key}**", expanded=True):
                for i, (abstract, pmid, distance) in enumerate(st.session_state.top_abstracts, 1):
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    globals()[f"abstract_{i}"] = abstract
                    globals()[f"reference_{i}"] = pubmed_url
                    globals()[f"distance_{i}"] = distance
                prompt_template = f'''[INST] '''

                with st.spinner('🔍 We are checking...'):
                    try:
                        # Preleva la domanda dal DataFrame
                        query = st.session_state.claim

                        # Costruisci il prompt
                        prompt_template = f'''[INST]  <<SYS>>

                        You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

                        Use the information in Context

                        elaborate the context to generate a new information.

                        Use only the knowledge in Context to answer.

                        Answer describing in a scentific way. Be formal during the answer. Use the third person.

                        Answer without mentioning the context. Use it but don't refer to it in the text

                        to answer, use max 300 word

                        Create a Justification from the sentences given.

                        Use the structure: Justification: .... (don't use the word context)

                        Write as an online doctor to create the justification.

                        After, give some sentences from Context from scientific papers: that supports the label and reject the label

                        Supporting sentences from abstracts:
                        information sentence from abstract_1:
                        information sentence from abstract_2: 
                        ..
                        Refusing sentences from abstracts:
                        information sentence from abstract_1:
                        information sentence from abstract_2: 
                        ..
                        Add where it comes from (abstract_1, abstract_2, abstract_3, abstract_4, abstract_5)

                        with the answer, gives a line like: "Label:". Always put Label as first. After Label, give the justification
                        The justification will be always given as Justification:
                        Label can be yes, no, NEI, where yes: claim is true. no: claim is false. NEI: not enough information.
                        The Label will be chosen with a voting system of support/refuse before
                        <<SYS>>

                        Question: {query} [/INST]
                        Context from scientific papers: {abstract_1} ; {abstract_2} ; {abstract_3} ; {abstract_4} ; {abstract_5} [/INST]
                        '''

                        # Chiamata API
                        completion = client.chat.completions.create(
                            model="meta/llama-3.1-405b-instruct",
                            messages=[{"role": "user", "content": prompt_template}],
                            temperature=0.1,
                            top_p=0.7,
                            max_tokens=1024,
                            stream=True
                        )

                        # Raccogli la risposta
                        Risposta = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                Risposta += chunk.choices[0].delta.content

                        # Debug: Controlla la risposta
                        print("{Nuova answer}")

                    except Exception as e:
                            st.write(f"Error processing index: {e}")

                with st.spinner('🤔💬 Justifying the check...'):
                    # Esegui il parsing e separa le variabili
                    zeroshot_classifier = pipeline(
                    "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
                    )
                    first_label, justification, supporting, refusing, notes = parse_response(Risposta)
                    
                with st.spinner('🕵️‍♂️📜 We are finding evidence...'):      
                    result = generate_justification(st.session_state.claim, justification)
                    predicted_label, score_label = extract_label_and_score(result)

                    # Incrementa i contatori in base al predicted_label
                    if predicted_label == "True":
                        color = f"rgba(0, 204, 0, {score_label})"  # Verde
                        st.session_state.true_count += 1
                    elif predicted_label == "False":
                        color = f"rgba(204, 0, 0, {score_label})"  # Rosso
                        st.session_state.false_count += 1
                    elif predicted_label == "NEI":
                        color = f"rgba(255, 255, 0, {score_label})"  # Giallo
                        st.session_state.nei_count += 1
                    else:
                        color = "black"  # Default color

                    st.caption(f"📝 {st.session_state.claim}")
                    #st.markdown(f'{st.session_state.claim} is <span style="color: {color}; font-weight: bold;">{predicted_label}</span>', unsafe_allow_html=True)
                    st.markdown(f"**Prediction of claim:** <span style='color: {color}; font-weight: bold;'>{predicted_label}</span>", unsafe_allow_html=True)
                    
                    st.markdown("### **Justification**")
                    st.markdown(f'<p> {justification}</p>', unsafe_allow_html=True)
                    abstracts = {
                    "abstract_1": abstract_1,
                    "abstract_2": abstract_2,
                    "abstract_3": abstract_3,
                    "abstract_4": abstract_4,
                    "abstract_5": abstract_5
                    }
                    supporting_texts = []
                    for item in supporting:
                        try:
                            supporting_texts.append(item["text"])
                        except (TypeError, KeyError):
                            continue

                    refusing_text = []
                    for item in refusing:
                        try:
                            refusing_text.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    pattern = r'"\s*(.*?)\s*"\s*\(abstract_(\d+)\)'
                    supporting = clean_phrases(supporting_texts, pattern)
                    refusing = clean_phrases(refusing_text, pattern)
                    processed_abstracts = {}
                    for abstract_name, abstract_text in abstracts.items():
                        # Evidenzia frasi di supporto in verde
                        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, supporting_matches, "lightgreen", predicted_label)
                        
                        # Evidenzia frasi di rifiuto in rosso
                        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, refusing_matches, "red", predicted_label)
                        
                        # Aggiungi solo abstract che hanno frasi evidenziate in verde
                        if supporting_matches:
                            # Aggiungi la reference se esiste una variabile corrispondente
                            reference_variable = f"reference_{abstract_name.split('_')[1]}"  # Genera il nome della variabile
                            if reference_variable in globals():  # Controlla se la variabile esiste
                                reference_value = globals()[reference_variable]
                                abstract_text += f"<br><br><strong>🔗 Reference:</strong> {reference_value}"
                            
                            # Aggiungi l'abstract processato
                            processed_abstracts[abstract_name] = abstract_text

                    # Itera sugli abstract processati ed elimina duplicati
                    seen_contents = set()  # Insieme per tracciare contenuti già visti
                    evidence_counter = 1
                    # Visualizza i risultati degli abstract processati con expander numerati
                    st.markdown("### **Scientific Evidence**")
                    # Aggiungi una legenda per i colori
                    legend_html = """
                    <div style="display: flex; flex-direction: column; align-items: flex-start;">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: lightgreen; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Positive Evidence</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: red; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Negative Evidence</div>
                    </div>
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 10px; border-radius: 5px;"></div>
                    <div>Dubious Evidence</div>
                    </div>
                    </div>
                    """
                    col1, col2 = st.columns([0.8, 0.2])
                    
                    with col1:
                        if processed_abstracts:
                            tabs = st.tabs([f"Scientific Evidence {i}" for i in range(1, len(processed_abstracts) + 1)])
                            for tab, (name, content) in zip(tabs, processed_abstracts.items()):
                                if content not in seen_contents:  # Aggiungi solo se non è già stato visto
                                    seen_contents.add(content)
                                    with tab:
                                        # Usa `st.write` per visualizzare HTML direttamente
                                        st.write(content, unsafe_allow_html=True)
                        else:
                            st.markdown("No relevant Scientific Evidence found")
                    
                    with col2:
                        st.caption("Legend")
                        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("### **Page Summary**")
        st.caption("📊 Here is a summary of the results for the extracted claims:")

        # Labels e colori
        labels = ['True', 'False', 'NEI']
        colors = ['green', 'red', 'yellow']

        # Dati
        sizes = [
            st.session_state.true_count,
            st.session_state.false_count,
            st.session_state.nei_count
        ]

        # Configurazione del grafico a torta
        options = {
            "tooltip": {"trigger": "item"},
            "legend": {"top": "5%", "left": "center"},
            "series": [
            {
                "name": "Document Status",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2,
                },
                "label": {"show": True, "position": "center"},
                "emphasis": {
                "label": {"show": True, "fontSize": "20", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": [
                {"value": sizes[0], "name": labels[0], "itemStyle": {"color": colors[0]}},
                {"value": sizes[1], "name": labels[1], "itemStyle": {"color": colors[1]}},
                {"value": sizes[2], "name": labels[2], "itemStyle": {"color": colors[2]}},
                ],
            }
            ],
        }

        # Visualizzazione con Streamlit
        st1, st2 = st.columns([0.6, 0.4])

        with st1:
            st.markdown("#### The page is :")
            true_count = st.session_state.true_count
            false_count = st.session_state.false_count
            nei_count = st.session_state.nei_count

            if true_count > 0 and false_count == 0:
                reliability = '<span style="color: darkgreen; font-weight: bold;">Highly Reliable</span>'
            elif true_count > false_count:
                reliability = '<span style="color: lightgreen; font-weight: bold;">Moderately Reliable</span>'
            elif true_count == 0:
                reliability = '<span style="color: darkred; font-weight: bold;">Not Reliable at All</span>'
            elif false_count > true_count:
                reliability = '<span style="color: lightcoral; font-weight: bold;">Not Reliable</span>'
            elif (true_count == false_count) or (nei_count > true_count and nei_count > false_count and true_count != 0 and false_count != 0):
                reliability = '<span style="color: yellow; font-weight: bold;">NEI</span>'
            else:
                reliability = '<span style="color: black; font-weight: bold;">Completely Reliable</span>'

            st.caption(f"### {reliability}")
            st.markdown(f"The page is considered {reliability} because it contains {true_count} true claims, {false_count} false claims, and {nei_count} claims with not enough information.", unsafe_allow_html=True)
        
            with st.popover("ℹ️ Understanding the Truthfulness Ratings"):
                st.markdown("""
                The reliability of the page is determined based on the number of true and false claims extracted from the page.
                - If the page contains only true claims, it is considered **Highly Reliable**.
                - If the page has more true claims than false claims, it is considered **Moderately Reliable**.
                - If the page contains only false claims, it is considered **Not Reliable at All**.
                - If the page has an equal number of true and false claims, it is considered **NEI**.
                """)

        with st2:
            st_echarts(
            options=options, height="500px",
            )
