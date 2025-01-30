import re
import os
import faiss
import whisper
import ffmpeg
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from newsplease import NewsPlease
from streamlit_echarts import st_echarts
from streamlit_option_menu import option_menu

# NEWS to check
# https://fbe.unimelb.edu.au/newsroom/fake-news-in-the-age-of-covid-19                          True Claim
# https://newssalutebenessere.altervista.org/covid-19-just-a-simple-flue-or-something-else/     False Claim

###### CONFIGURATIONS ######
# Debug mode
debug = False

# File paths
embeddings_file = r"data\abstract_embeddings.npy"
pmid_file = r"data\pmids.npy"
faiss_index_file = r"data\faiss_index.index"
file_path = r'data\parte_205.csv'

# Initialize OpenAI API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=st.secrets.nvidia.api_key
)

# Load data
data = pd.read_csv(file_path)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_article_data(url):
    """
    Extracts article data from a specified URL.
    
    Args:
        url (str): URL of the article to analyze.
    
    Returns:
        dict: Structured article data, including: title, authors, publication date, and content.
    """
    try:
        # Make an HTTP request to the specified URL
        response = requests.get(url)
        # Check if the request was successful (i.e., status code 200)
        response.raise_for_status()

        # Extract the HTML content from the response
        html_content = response.text

        # Use NewsPlease to extract structured data from the HTML content
        article = NewsPlease.from_html(html_content, url=url)

        # Return the structured article data
        return {
            "title": article.title,
            "authors": article.authors,
            "date_publish": article.date_publish,
            "content": article.maintext,
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Error during URL retrieval: {e}"}

    except Exception as e:
        return {"error": f"Error processing the article: {e}"}


def extract_and_split_claims(claims):
    """
    Extracts and splits claims from a given string.
    
    Args:
        claims (str): String containing claims.
    
    Returns:
        dict: Dictionary containing the extracted claims.
    """
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
    Extracts the predicted label and score from the result string.
    
    Args:
        result (str): String containing the prediction result.
    
    Returns:
        tuple: Predicted label and score.
    """
    # Extract the predicted label
    label_match = re.search(r"'labels': \['(.*?)'", result)
    predicted_label = label_match.group(1) if label_match else None

    # Extract the score
    score_match = re.search(r"'scores': \[(\d+\.\d+)", result)
    score_label = float(score_match.group(1)) if score_match else None

    return predicted_label, score_label


def clean_phrases(phrases, pattern):
    """
    Clean and extract phrases from a list of strings using a specified pattern.
    
    Args:
        phrases (list): List of strings containing phrases.
        pattern (str): Regular expression pattern to extract phrases.
    
    Returns:
        list: List of cleaned phrases as dictionaries with text and abstract keys
    """
    cleaned_phrases = []

    for phrase in phrases:
        matches = re.findall(pattern, phrase)
        cleaned_phrases.extend([{"text": match[0], "abstract": f"abstract_{match[1]}"} for match in matches])

    return cleaned_phrases


def highlight_phrases(abstract_text, phrases, color, label):
    """
    Highlight phrases in the abstract text with the specified background color.
    
    Args:
        abstract_text (str): Text of the abstract to highlight.
        phrases (list): List of phrases to highlight.
        color (str): Background color to use for highlighting.
        label (str): Predicted label for the claim.
    
    Returns:
        str: Abstract text with highlighted phrases.
    """
    # Switch colors if the label is "False"
    if label.lower() == "false":
        color = "lightgreen" if color == "red" else color

    # Highlight each phrase in the abstract text
    for phrase in phrases:
        abstract_text = re.sub(
            re.escape(phrase["text"]),
            f'<span style="background-color: {color}; font-weight: bold; border: 1px solid black; border-radius: 5px;">{phrase["text"]}</span>',
            abstract_text,
            flags=re.IGNORECASE
        )

    return abstract_text


def parse_response(response):
    """
    Parse the response from the model and extract the fields.
    
    Args:
        response (str): Response string from the model.
    
    Returns:
        tuple: Extracted fields from the response.
    """
    # Initial values for the fields
    first_label = "Non trovato"
    justification = "Non trovato"
    supporting = "Non trovato"
    refusing = "Non trovato"
    notes = "Non trovato"

    # Regular expression patterns for extracting fields
    patterns = {
        "first_label": r"Label:\s*(.*?)\n",
        "justification": r"Justification:\s*(.*?)(?=\nSupporting sentences)",
        "supporting": r"Supporting sentences from abstracts:\n(.*?)(?=\nRefusing sentences)",
        "refusing": r"Refusing sentences from abstracts:\n(.*?)(?=\nNote:)",
        "notes": r"Note:\s*(.*)"
    }

    # Extract the fields using regular expressions
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

    # Return the extracted fields
    return first_label, justification, supporting, refusing, notes


def load_embeddings(embeddings_file, pmid_file, faiss_index_file, debug=False):
    """
    Load embeddings, PMIDs, and FAISS index from the specified files.
    
    Args:
        embeddings_file (str): File path for the embeddings.
        pmid_file (str): File path for the PMIDs.
        faiss_index_file (str): File path for the FAISS index.
    
    Returns:
        tuple: Tuple containing the embeddings, PMIDs, and FAISS index.
    """
    # Check if the files exist
    if not (os.path.exists(embeddings_file) and os.path.exists(pmid_file) and os.path.exists(faiss_index_file)):
        raise FileNotFoundError("One or more files not found. Please check the file paths.")

    # Load the embeddings and PMIDs
    embeddings = np.load(embeddings_file)
    pmids = np.load(pmid_file, allow_pickle=True)

    # Load the FAISS index
    index = faiss.read_index(faiss_index_file)

    if debug:
        print("Embeddings, PMIDs, and FAISS index loaded successfully.")

    return embeddings, pmids, index


def retrieve_top_abstracts(claim, model, index, pmids, data, top_k=5):
    """
    Retrieve the top abstracts from the FAISS index for a given claim.
    
    Args:
        claim (str): Claim to fact-check.
        model (SentenceTransformer): Sentence transformer model for encoding text.
        index (faiss.IndexFlatIP): FAISS index for similarity search.
        pmids (np.ndarray): Array of PMIDs for the abstracts.
        data (pd.DataFrame): DataFrame containing the abstract data.
        top_k (int): Number of top abstracts to retrieve.
    
    Returns:
        list: List of tuples containing the abstract text, PMID, and distance.
    """
    # Encode the claim using the SentenceTransformer model
    claim_embedding = model.encode([claim])
    faiss.normalize_L2(claim_embedding)  # Normalize the claim embedding (with L2 norm)
    distances, indices = index.search(claim_embedding, top_k)

    # Retrieve the top abstracts based on the indices
    results = []
    for j, i in enumerate(indices[0]):
        pmid = pmids[i]
        abstract_text = data[data['PMID'] == pmid]['AbstractText'].values[0]
        distance = distances[0][j]
        results.append((abstract_text, pmid, distance))

    return results


def generate_justification(query, justification):
    """
    Generate a justification for the claim using the Zero-Shot Classification model.
    
    Args:
        query (str): Claim to fact-check.
        justification (str): Justification for the claim.
    
    Returns:
        str: Final justification for the claim.
    """
    # Define the classes for the Zero-Shot Classification model
    Class = ["True", "False","NEI"]

    # Generate the justification text
    justification_text = (
        f'Justification: "{justification}"'
    )

    # Limit the justification text to a maximum length
    max_length = 512
    if len(justification_text) > max_length:
        justification_text = justification_text[:max_length]

    # Generate the final justification using the Zero-Shot Classification model
    output = zeroshot_classifier(
        query,
        Class,
        hypothesis_template=f"The claim is '{{}}' for: {justification_text}",
        multi_label=False
    )

    # Prepare the final justification text
    final_justification = f'{output}.'

    return final_justification


def llm_reasoning_template(query):
    """
    Generate a template for the prompt used for justification generation by the LLM model.
    
    Args:
        query (str): Claim to fact-check.
    
    Returns:
        str: Reasoning template for the claim.
    """
    llm_reasoning_prompt = f"""<<SYS>> [INST]

    You are a helpful, respectful and honest Doctor. Always answer as helpfully as possible using the context text provided.

    Use the information in Context.

    Elaborate the Context to generate a new information.

    Use only the knowledge in Context to answer.

    Answer describing in a scentific way. Be formal during the answer. Use the third person.

    Answer without mentioning the Context. Use it but don't refer to it in the text.

    To answer, use max 300 word.

    Create a Justification from the sentences given.

    Use the structure: Justification: The claim is (label) because... (don't use the word "context")

    Write as an online doctor to create the Justification.

    After, give some sentences from Context from scientific papers: that supports the label and reject the label.

    Supporting sentences from abstracts:
    information sentence from abstract_1: 
    information sentence from abstract_2: 
    ..
    Refusing sentences from abstracts:
    information sentence from abstract_1: 
    information sentence from abstract_2: 
    ..
    Add where it comes from (abstract_1, abstract_2, abstract_3, abstract_4, abstract_5)

    With the answer, gives a line like: "Label:". Always put Label as first. After Label, give the Justification.
    The justification will be always given as Justification: 
    Label can be yes, no, NEI, where yes: claim is true. no: claim is false. NEI: not enough information.
    The Label will be chosen with a voting system of support/refuse before.

    [/INST] <</SYS>>

    [INST] Question: {query} [/INST]
    [INST] Context from scientific papers: 
    """

    return llm_reasoning_prompt


def claim_detection_template(full_text):
    """
    Generate a template for the prompt used for claim detection by the LLM model.
    
    Args:
        full_text (str): Full text to analyze.
    
    Returns:
        str: Template for claim detection.
    """
    claim_detection_prompt = f"""<<SYS>> [INST]

    Your task is to extract from the text potential health related question to verify their veracity.

    The context extracted from the online where to take the claim is: {full_text}

    Create simple claim of single sentence from the context.

    Dont's use *

    Give just the claim. Don't write other things.

    Extract only health related claim.

    Rank eventual claim like:

    Claim 1:
    Claim 2:
    Claim 3:
    
    Use always this structure.
    Start every claim with "Claim " followed by the number.

    The number of claims may go from 1 to a max of 5.

    The claims have to be always health related. [/INST] <</SYS>>
    """

    return claim_detection_prompt


# Page and Title Configuration
st.set_page_config(page_title="CER - Combining Evidence and Reasoning Demo", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center; color: inherit;'>✔️✨ CER - Biomedical Fact Checker</h1>", unsafe_allow_html=True)

# Horizontal option menu for selecting the page
page = option_menu(None, ["Single claim check", "Page check", "Video check"], 
    icons=['check', 'ui-checks'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

# Sidebar Configuration
st.sidebar.title("🔬 Combining Evidence and Reasoning Demo")
st.sidebar.caption("🔍 Fact-check biomedical claims using scientific evidence and reasoning.")
st.sidebar.markdown("---")
st.sidebar.caption("#### ℹ️ About")
st.sidebar.caption("This is a demo application for fact-checking biomedical claims using scientific evidence and reasoning. It uses a combination of language models, scientific literature, and reasoning to provide explanations for the predictions.")

# Load embeddings, PMIDs, and FAISS index
if 'embeddings_loaded' not in st.session_state:
    embeddings, pmids, index = load_embeddings(embeddings_file, pmid_file, faiss_index_file, debug)
    st.session_state.embeddings = embeddings
    st.session_state.pmids = pmids
    st.session_state.index = index
    st.session_state.embeddings_loaded = True
else:
    embeddings = st.session_state.embeddings
    pmids = st.session_state.pmids
    index = st.session_state.index

# Check if the claim and top_abstracts are in the session state
if 'claim' not in st.session_state:
    st.session_state.claim = ""

if 'top_abstracts' not in st.session_state:
    st.session_state.top_abstracts = []


#### Single claim check PAGE ####
if page == "Single claim check":
    st.subheader("Single claim check")
    st.caption("✨ Enter a single claim to fact-check and hit the button to see the results! 🔍")

    st.session_state.claim = st.text_input("Claim to fact-check:")

    if st.button("✨ Fact Check"):

        if st.session_state.claim:
            # Retrieve the top abstracts for the claim
            top_abstracts = retrieve_top_abstracts(st.session_state.claim, model, index, pmids, data, top_k=5)
            st.session_state.top_abstracts = top_abstracts

            st.markdown("### **Results**")

            with st.container():
                for i, (abstract, pmid, distance) in enumerate(st.session_state.top_abstracts, 1):
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    globals()[f"abstract_{i}"] = abstract
                    globals()[f"reference_{i}"] = pubmed_url
                    globals()[f"distance_{i}"] = distance

                with st.spinner('🔍 We are checking...'):
                    try:
                        # Retrieve the question from the DataFrame
                        query = st.session_state.claim

                        # Generate the reasoning template
                        prompt_template = llm_reasoning_template(query)

                        # Add the abstracts to the prompt
                        for i in range(1, len(st.session_state.top_abstracts)):
                            prompt_template += f"{globals()[f'abstract_{i}']} ; "
                        prompt_template += f"{globals()[f'abstract_{i+1}']} [/INST]"

                        # Call the API
                        completion = client.chat.completions.create(
                            model="meta/llama-3.1-405b-instruct",
                            messages=[{"role": "user", "content": prompt_template}],
                            temperature=0.1,
                            top_p=0.7,
                            max_tokens=1024,
                            stream=True
                        )

                        # Collect the response
                        answer = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                answer += chunk.choices[0].delta.content

                        # Debug: Check the answer
                        if debug:
                            print(f"{answer}")

                    except Exception as e:
                        st.write(f"Error processing index: {e}")

                with st.spinner('🤔💬 Justifying the check...'):
                    # Perform parsing and separate variables
                    zeroshot_classifier = pipeline(
                        "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
                    )
                    first_label, justification, supporting, refusing, notes = parse_response(answer)

                with st.spinner('🕵️‍♂️📜 We are finding evidence...'):
                    # Generate the justification for the claim
                    result = generate_justification(st.session_state.claim, justification)
                    predicted_label, score_label = extract_label_and_score(result)

                    if predicted_label == "True":
                        color = f"rgba(0, 204, 0, {score_label})"  # Green
                    elif predicted_label == "False":
                        color = f"rgba(204, 0, 0, {score_label})"  # Red
                    elif predicted_label == "NEI":
                        color = f"rgba(255, 255, 0, {score_label})"  # Yellow
                    else:
                        color = "black"  # Default color

                    # Calculate the confidence score
                    confidence = f"{score_label * 100:.2f}%"
                    st.caption(f"📝 The Claim: {st.session_state.claim}")
                    st.markdown(
                        f"**Prediction of claim:** Most likely <span style='color: {color}; font-weight: bold;'>{predicted_label}</span> with a confidence of <span style='color: {color}; font-weight: bold;'>{confidence}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("### **Justification**")
                    st.markdown(f'<p> {justification}</p>', unsafe_allow_html=True)

                    # Extract the abstracts and references
                    abstracts = {}
                    for i in range(1, len(st.session_state.top_abstracts) + 1):
                        abstracts[f"abstract_{i}"] = globals()[f"abstract_{i}"]

                    pattern = r'"\s*(.*?)\s*"\s*\(abstract_(\d+)\)'

                    supporting_texts = []
                    for item in supporting:
                        try:
                            supporting_texts.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    supporting = clean_phrases(supporting_texts, pattern)

                    refusing_text = []
                    for item in refusing:
                        try:
                            refusing_text.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    refusing = clean_phrases(refusing_text, pattern)

                    if debug:
                        print(supporting)
                        print(refusing)

                    processed_abstracts = {}
                    for abstract_name, abstract_text in abstracts.items():
                        # Highlight supporting phrases in green
                        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, supporting_matches, "lightgreen", predicted_label)
                        
                        # Highlight refusing phrases in red
                        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, refusing_matches, "red", predicted_label)
                        
                        # Add only if supporting matches are found
                        if supporting_matches:
                            # Add the reference if a corresponding variable exists
                            reference_variable = f"reference_{abstract_name.split('_')[1]}"
                            if reference_variable in globals():
                                reference_value = globals()[reference_variable]
                                abstract_text += f"<br><br><strong>🔗 Reference:</strong> {reference_value}"
                            
                            # Add the processed abstract
                            processed_abstracts[abstract_name] = abstract_text

                    # Iterate over the processed abstracts and remove duplicates
                    seen_contents = set()  # Set to track already seen contents
                    evidence_counter = 1

                    # Display the results of the processed abstracts with numbered expanders
                    st.markdown("### **Scientific Evidence**")

                    # Add a legend for the colors
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
                                if content not in seen_contents:  # Check for duplicates
                                    seen_contents.add(content)
                                    with tab:
                                        # Switch colors if the label is "False"
                                        if predicted_label.lower() == "false":
                                            content = content.replace("background-color: lightgreen", "background-color: tempcolor")
                                            content = content.replace("background-color: red", "background-color: lightgreen")
                                            content = content.replace("background-color: tempcolor", "background-color: red")
                                        
                                        # Use `st.write` to display HTML directly
                                        st.write(content, unsafe_allow_html=True)
                        else:
                            st.markdown("No relevant Scientific Evidence found")

                    with col2:
                        st.caption("Legend")
                        st.markdown(legend_html, unsafe_allow_html=True)


#### Web page check PAGE ####
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
                # Retrieve the claims from the article data
                prompt_template = claim_detection_template(article_data)

                # Call the API
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-405b-instruct",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.1,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True
                )

                # Collect the response
                answer = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        answer += chunk.choices[0].delta.content

                # Debug: Controlla la risposta
                print(f"{answer}")

            except Exception as e:
                print(f"Error {e}")

            claims_dict = extract_and_split_claims(answer)

        # Display the extracted claims
        st.markdown("### **Claims Extracted**")
        st.caption("🔍 Here are the health-related claims extracted from the page:")
        cols = st.columns(3)
        for i, (claim_key, claim_text) in enumerate(claims_dict.items(), 1):
            col = cols[(i - 1) % 3]
            with col.expander(f"Claim {i} 📝", expanded=True):
                st.write(claim_text)

        # Display the results for the extracted claims
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

                with st.spinner('🔍 We are checking...'):
                    try:
                        # Retrieve the question from the DataFrame
                        query = st.session_state.claim

                        # Generate the reasoning template
                        prompt_template = llm_reasoning_template(query)

                        # Add the abstracts to the prompt
                        for i in range(1, len(st.session_state.top_abstracts)):
                            prompt_template += f"{globals()[f'abstract_{i}']} ; "
                        prompt_template += f"{globals()[f'abstract_{i+1}']} [/INST]"

                        # Call the API
                        completion = client.chat.completions.create(
                            model="meta/llama-3.1-405b-instruct",
                            messages=[{"role": "user", "content": prompt_template}],
                            temperature=0.1,
                            top_p=0.7,
                            max_tokens=1024,
                            stream=True
                        )

                        # Collect the response
                        answer = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                answer += chunk.choices[0].delta.content

                        # Debug: Check the answer
                        if debug:
                            print(f"{answer}")

                    except Exception as e:
                            st.write(f"Error processing index: {e}")

                with st.spinner('🤔💬 Justifying the check...'):
                    # Perform parsing and separate variables
                    zeroshot_classifier = pipeline(
                        "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
                    )
                    first_label, justification, supporting, refusing, notes = parse_response(answer)
                
                with st.spinner('🕵️‍♂️📜 We are finding evidence...'):
                    # Generate the justification for the claim
                    result = generate_justification(st.session_state.claim, justification)
                    predicted_label, score_label = extract_label_and_score(result)

                    # Update the counts based on the predicted label
                    if predicted_label == "True":
                        color = f"rgba(0, 204, 0, {score_label})"  # Green
                        st.session_state.true_count += 1
                    elif predicted_label == "False":
                        color = f"rgba(204, 0, 0, {score_label})"  # Red
                        st.session_state.false_count += 1
                    elif predicted_label == "NEI":
                        color = f"rgba(255, 255, 0, {score_label})"  # Yellow
                        st.session_state.nei_count += 1
                    else:
                        color = "black"  # Default color

                    confidence = f"{score_label * 100:.2f}%" 
                    st.caption(f"📝 The Claim: {st.session_state.claim}")
                    st.markdown(
                        f"**Prediction of claim:** Most likely <span style='color: {color}; font-weight: bold;'>{predicted_label}</span> with a confidence of <span style='color: {color}; font-weight: bold;'>{confidence}</span>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("### **Justification**")
                    st.markdown(f'<p> {justification}</p>', unsafe_allow_html=True)
                    
                    abstracts = {}
                    for i in range(1, len(st.session_state.top_abstracts) + 1):
                        abstracts[f"abstract_{i}"] = globals()[f"abstract_{i}"]
                    
                    pattern = r'"\s*(.*?)\s*"\s*\(abstract_(\d+)\)'
                    
                    supporting_texts = []
                    for item in supporting:
                        try:
                            supporting_texts.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    supporting = clean_phrases(supporting_texts, pattern)

                    refusing_text = []
                    for item in refusing:
                        try:
                            refusing_text.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    refusing = clean_phrases(refusing_text, pattern)

                    if debug:
                        print(supporting)
                        print(refusing)

                    processed_abstracts = {}
                    for abstract_name, abstract_text in abstracts.items():
                        # Highlight supporting phrases in green
                        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, supporting_matches, "lightgreen", predicted_label)
                        
                        # Highlight refusing phrases in red
                        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, refusing_matches, "red", predicted_label)
                        
                        # Add only if supporting matches are found
                        if supporting_matches:
                            # Add the reference if a corresponding variable exists
                            reference_variable = f"reference_{abstract_name.split('_')[1]}"
                            if reference_variable in globals():
                                reference_value = globals()[reference_variable]
                                abstract_text += f"<br><br><strong>🔗 Reference:</strong> {reference_value}"
                            
                            # Add the processed abstract
                            processed_abstracts[abstract_name] = abstract_text

                    # Iterate over the processed abstracts and remove duplicates
                    seen_contents = set()  # Set to track already seen contents
                    evidence_counter = 1
                    
                    # Display the results of the processed abstracts with numbered expanders
                    st.markdown("### **Scientific Evidence**")
                    
                    # Add a legend for the colors
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
                                if content not in seen_contents:  # Check for duplicates
                                    seen_contents.add(content)
                                    with tab:
                                        # Switch colors if the label is "False"
                                        if predicted_label.lower() == "false":
                                            content = content.replace("background-color: lightgreen", "background-color: tempcolor")
                                            content = content.replace("background-color: red", "background-color: lightgreen")
                                            content = content.replace("background-color: tempcolor", "background-color: red")
                                        
                                        # Use `st.write` to display HTML directly
                                        st.write(content, unsafe_allow_html=True)
                        else:
                            st.markdown("No relevant Scientific Evidence found")
                    
                    with col2:
                        st.caption("Legend")
                        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("### **Page Summary**")
        st.caption("📊 Here is a summary of the results for the extracted claims:")

        # Labels and Colors
        labels = ['True', 'False', 'NEI']
        colors = ['green', 'red', 'yellow']

        # Sizes of the pie chart
        sizes = [
            st.session_state.true_count,
            st.session_state.false_count,
            st.session_state.nei_count
        ]

        # Configure the Pie Chart Options
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

        # Display the Pie Chart
        st1, st2 = st.columns([0.6, 0.4])

        with st1:
            st.markdown("#### The page is :")
            true_count = st.session_state.true_count
            false_count = st.session_state.false_count
            nei_count = st.session_state.nei_count

            if true_count > 0 and false_count == 0:
                reliability = '<span style="color: darkgreen; font-weight: bold;">Highly Reliable</span>'
            elif true_count > false_count:
                reliability = '<span style="color: lightgreen; font-weight: bold;">Fairly Reliable</span>'
            elif true_count == 0:
                reliability = '<span style="color: darkred; font-weight: bold;">Strongly Considered Unreliable</span>'
            elif false_count > true_count:
                reliability = '<span style="color: lightcoral; font-weight: bold;">Unlikely to be Reliable</span>'
            elif (true_count == false_count) or (nei_count > true_count and nei_count > false_count and true_count != 0 and false_count != 0):
                reliability = '<span style="color: yellow; font-weight: bold;">NEI</span>'
            else:
                reliability = '<span style="color: black; font-weight: bold;">Completely Reliable</span>'

            st.markdown(f"The page is considered {reliability} because it contains {true_count} true claims, {false_count} false claims, and {nei_count} claims with not enough information.", unsafe_allow_html=True)

            with st.popover("ℹ️ Understanding the Truthfulness Ratings"):
                st.markdown("""
                The reliability of the page is determined based on the number of true and false claims extracted from the page.
                - If the page contains only true claims, it is considered **Highly Reliable**.
                - If the page has more true claims than false claims, it is considered **Fairly Reliable**.
                 -If the page has more false claims than true claims, it is considered **Unlikely to be Reliable**.
                - If the page contains only false claims, it is considered **Strongly Considered Unreliable**.
                - If the page has an equal number of true and false claims, it is considered **NEI**.
                """)

        with st2:
            st_echarts(
            options=options, height="500px",
            )


#### Video check PAGE ####
elif page == "Video check":
    st.subheader("Video claim check")
    st.caption("✨ Upload a video to fact-check and hit the button to see the results! 🔍")

    video = st.file_uploader("Choose a video...", type=["mp4"])
    video_box, text_box = st.columns([0.6, 0.4])
    if video is not None:
        with video_box:
            with st.expander("▶️ See uploaded video", expanded=False):
                st.video(video)

    if st.button("✨ Fact Check") and video is not None:
        with st.spinner('🎥🔄 Processing video...'):
            # Save the video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video.read())
                temp_video_path = temp_video.name
            
            # Extract the audio from the video
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            ffmpeg.input(temp_video_path).output(temp_audio_path, acodec="pcm_s16le", ar=16000, ac=1).run(overwrite_output=True)
            
            # Transcribe the audio
            model1 = whisper.load_model("small")
            result = model1.transcribe(temp_audio_path)
            
            # Extract the final text
            transcribed_text = result["text"]
            with text_box:
                with st.expander("📝 Transcribed Text", expanded=False):
                    st.caption("🔍 Here is the transcribed text from the uploaded video:")
                    container = st.container(height=322)
                    container.write(transcribed_text)

        st.session_state.true_count = 0
        st.session_state.false_count = 0
        st.session_state.nei_count = 0

        with st.spinner('🌐🔍 Extracting claims from video...'):
            try:
                # Retrieve the claims from the video
                prompt_template = claim_detection_template(transcribed_text)

                # Call the API
                completion = client.chat.completions.create(
                    model="meta/llama-3.1-405b-instruct",
                    messages=[{"role": "user", "content": prompt_template}],
                    temperature=0.1,
                    top_p=0.7,
                    max_tokens=1024,
                    stream=True
                )

                # Collect the response
                answer = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        answer += chunk.choices[0].delta.content

                # Debug: Check the answer
                if debug:
                    print(f"{answer}")

            except Exception as e:
                print(f"Error {e}")

            claims_dict = extract_and_split_claims(answer)

        # Display the extracted claims
        st.markdown("### **Claims Extracted**")
        st.caption("🔍 Here are the health-related claims extracted from the video:")
        cols = st.columns(3)
        for i, (claim_key, claim_text) in enumerate(claims_dict.items(), 1):
            col = cols[(i - 1) % 3]
            with col.expander(f"Claim {i} 📝", expanded=True):
                st.write(claim_text)

        # Display the results for the extracted claims
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

                with st.spinner('🔍 We are checking...'):
                    try:
                        # Retrieve the question from the DataFrame
                        query = st.session_state.claim

                        # Generate the reasoning template
                        prompt_template = llm_reasoning_template(query)

                        # Add the abstracts to the prompt
                        for i in range(1, len(st.session_state.top_abstracts)):
                            prompt_template += f"{globals()[f'abstract_{i}']} ; "
                        prompt_template += f"{globals()[f'abstract_{i+1}']} [/INST]"

                        # Call the API
                        completion = client.chat.completions.create(
                            model="meta/llama-3.1-405b-instruct",
                            messages=[{"role": "user", "content": prompt_template}],
                            temperature=0.1,
                            top_p=0.7,
                            max_tokens=1024,
                            stream=True
                        )

                        # Collect the response
                        answer = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                answer += chunk.choices[0].delta.content

                        # Debug: Check the answer
                        if debug:
                            print(f"{answer}")

                    except Exception as e:
                            st.write(f"Error processing index: {e}")

                with st.spinner('🤔💬 Justifying the check...'):
                    # Perform parsing and separate variables
                    zeroshot_classifier = pipeline(
                        "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
                    )
                    first_label, justification, supporting, refusing, notes = parse_response(answer)
                    
                with st.spinner('🕵️‍♂️📜 We are finding evidence...'):
                    # Generate the justification for the claim
                    result = generate_justification(st.session_state.claim, justification)
                    predicted_label, score_label = extract_label_and_score(result)

                    # Update the counts based on the predicted label
                    if predicted_label == "True":
                        color = f"rgba(0, 204, 0, {score_label})"  # Green
                        st.session_state.true_count += 1
                    elif predicted_label == "False":
                        color = f"rgba(204, 0, 0, {score_label})"  # Red
                        st.session_state.false_count += 1
                    elif predicted_label == "NEI":
                        color = f"rgba(255, 255, 0, {score_label})"  # Yellow
                        st.session_state.nei_count += 1
                    else:
                        color = "black"  # Default color

                    confidence = f"{score_label * 100:.2f}%" 
                    st.caption(f"📝 The Claim: {st.session_state.claim}")
                    st.markdown(
                        f"**Prediction of claim:** Most likely <span style='color: {color}; font-weight: bold;'>{predicted_label}</span> with a confidence of <span style='color: {color}; font-weight: bold;'>{confidence}</span>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("### **Justification**")
                    st.markdown(f'<p> {justification}</p>', unsafe_allow_html=True)
                    
                    abstracts = {}
                    for i in range(1, len(st.session_state.top_abstracts) + 1):
                        abstracts[f"abstract_{i}"] = globals()[f"abstract_{i}"]

                    pattern = r'"\s*(.*?)\s*"\s*\(abstract_(\d+)\)'

                    supporting_texts = []
                    for item in supporting:
                        try:
                            supporting_texts.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    supporting = clean_phrases(supporting_texts, pattern)

                    refusing_text = []
                    for item in refusing:
                        try:
                            refusing_text.append(item["text"])
                        except (TypeError, KeyError):
                            continue
                    refusing = clean_phrases(refusing_text, pattern)
                    
                    processed_abstracts = {}
                    for abstract_name, abstract_text in abstracts.items():
                        # Highlight supporting phrases in green
                        supporting_matches = [phrase for phrase in supporting if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, supporting_matches, "lightgreen", predicted_label)
                        
                        # Highlight refusing phrases in red
                        refusing_matches = [phrase for phrase in refusing if phrase["abstract"] == abstract_name]
                        abstract_text = highlight_phrases(abstract_text, refusing_matches, "red", predicted_label)
                        
                        if supporting_matches:
                            # Add the reference if a corresponding variable exists
                            reference_variable = f"reference_{abstract_name.split('_')[1]}"
                            if reference_variable in globals():
                                reference_value = globals()[reference_variable]
                                abstract_text += f"<br><br><strong>🔗 Reference:</strong> {reference_value}"
                            
                            # Add the processed abstract
                            processed_abstracts[abstract_name] = abstract_text

                    # Iterate over the processed abstracts and remove duplicates
                    seen_contents = set()  # Set to track already seen contents
                    evidence_counter = 1
                    
                    # Display the results of the processed abstracts with numbered expanders
                    st.markdown("### **Scientific Evidence**")
                    
                    # Add a legend for the colors
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
                                if content not in seen_contents:  # Check for duplicates
                                    seen_contents.add(content)
                                    with tab:
                                        # Switch colors if the label is "False"
                                        if predicted_label.lower() == "false":
                                            content = content.replace("background-color: lightgreen", "background-color: tempcolor")
                                            content = content.replace("background-color: red", "background-color: lightgreen")
                                            content = content.replace("background-color: tempcolor", "background-color: red")
                                        
                                        # Use `st.write` to display HTML directly
                                        st.write(content, unsafe_allow_html=True)
                        else:
                            st.markdown("No relevant Scientific Evidence found")
                    
                    with col2:
                        st.caption("Legend")
                        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("### **Video Summary**")
        st.caption("📊 Here is a summary of the results for the extracted claims:")

        # Labels and Colors
        labels = ['True', 'False', 'NEI']
        colors = ['green', 'red', 'yellow']

        # Sizes of the pie chart
        sizes = [
            st.session_state.true_count,
            st.session_state.false_count,
            st.session_state.nei_count
        ]

        # Configure the Pie Chart Options
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

        # Display the Pie Chart
        st1, st2 = st.columns([0.6, 0.4])

        with st1:
            st.markdown("#### The Video is :")
            true_count = st.session_state.true_count
            false_count = st.session_state.false_count
            nei_count = st.session_state.nei_count

            if true_count > 0 and false_count == 0:
                reliability = '<span style="color: darkgreen; font-weight: bold;">Highly Reliable</span>'
            elif true_count > false_count:
                reliability = '<span style="color: lightgreen; font-weight: bold;">Fairly Reliable</span>'
            elif true_count == 0:
                reliability = '<span style="color: darkred; font-weight: bold;">Strongly Considered Unreliable</span>'
            elif false_count > true_count:
                reliability = '<span style="color: lightcoral; font-weight: bold;">Unlikely to be Reliable</span>'
            elif (true_count == false_count) or (nei_count > true_count and nei_count > false_count and true_count != 0 and false_count != 0):
                reliability = '<span style="color: yellow; font-weight: bold;">NEI</span>'
            else:
                reliability = '<span style="color: black; font-weight: bold;">Completely Reliable</span>'

            st.markdown(f"The video is considered {reliability} because it contains {true_count} true claims, {false_count} false claims, and {nei_count} claims with not enough information.", unsafe_allow_html=True)
        
            with st.popover("ℹ️ Understanding the Truthfulness Ratings"):
                st.markdown("""
                The reliability of the video is determined based on the number of true and false claims extracted from the video.
                - If the video contains only true claims, it is considered **Highly Reliable**.
                - If the video has more true claims than false claims, it is considered **Fairly Reliable**.
                - If the video has more false claims than true claims, it is considered **Unlikely to be Reliable**.
                - If the video contains only false claims, it is considered **Strongly Considered Unreliable**.
                - If the video has an equal number of true and false claims, it is considered **NEI**.
                """)

        with st2:
            st_echarts(
            options=options, height="500px",
            )

