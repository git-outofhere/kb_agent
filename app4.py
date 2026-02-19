###################################
# app4.py  KB Agent: 07/02/2025
###################################
import base64
import os
import glob
import re
import csv
import pickle
import html as html_module
import networkx as nx
import streamlit as st
from typing import List
from pathlib import Path
from datetime import datetime

# ---------------------------
# LangChain & local LLM imports
# ---------------------------
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Vector store + embeddings from local
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Local LLM (Ollama)
from langchain_community.chat_models import ChatOllama

# For triple extraction with a local LLM chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


#######################################
# 1) LOCAL TRIPLE EXTRACTION SETUP
#######################################
triple_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Extract all entity and relationship triples from the text below. "
        "Output each triple on its own line in the format: (Subject, Predicate, Object). "
        "If there is no clear relationship, skip.\n\n"
        "Text: {text}\n\n"
        "Triples:"
    ),
)

local_llm_for_triples = ChatOllama(
    #model="deepseek-r1",  # 7B model
    model="deepseek-r1:32b",
    temperature=0.0,
    top_k=5,
    top_p=0.8
)
triple_extraction_chain = LLMChain(llm=local_llm_for_triples, prompt=triple_extraction_prompt)


#######################################
# 2) BUILD KNOWLEDGE GRAPH FROM CHUNKS
#######################################
def build_knowledge_graph_from_chunks(chunks: List[Document], extraction_chain: LLMChain) -> nx.DiGraph:
    """
    For each text chunk, call the triple extraction chain and parse its output
    to add edges (with a 'relation' attribute) into a directed graph.
    """
    G = nx.DiGraph()
    for doc in chunks:
        text = doc.page_content
        try:
            result = extraction_chain.run(text=text)
        except Exception as e:
            print(f"Error extracting triples: {e}")
            continue
        # Assume each line of the result is a triple like: (Subject, Predicate, Object)
        lines = result.strip().splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                content = line[1:-1]
                parts = content.split(",")
                if len(parts) == 3:
                    subj = parts[0].strip()
                    pred = parts[1].strip()
                    obj = parts[2].strip()
                    if subj and obj:
                        G.add_edge(subj, obj, relation=pred)
    return G


#######################################
# 3) EMBEDDING FUNCTION
#######################################
embedding_function = OllamaEmbeddings(
    model="nomic-embed-text",
    show_progress=True
)


###################################################
# Helpers to load docs, build graph & vector store
###################################################
def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Reads all PDFs and DOCXs from folder_path and returns them as a list of Documents.
    """
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    docx_paths = glob.glob(os.path.join(folder_path, "*.docx"))
    all_docs = []
    for pdf_file in pdf_paths:
        loader = PyPDFLoader(pdf_file)
        all_docs.extend(loader.load())
    for docx_file in docx_paths:
        loader = Docx2txtLoader(docx_file)
        all_docs.extend(loader.load())
    return all_docs


def build_graph_and_vectorstore(folder_paths: List[str]):
    """
    1) Load text files from each folder in folder_paths as Document nodes.
    2) Split into chunks.
    3) Build or load a knowledge graph via triple extraction.
    4) Build or load a Chroma vector store.
    Returns: (knowledge_graph, vectorstore, chunks)
    """
    raw_docs: List[Document] = []
    # Load .txt files as whole-document nodes
    for folder in folder_paths:
        for txt_path in glob.glob(os.path.join(folder, "*.txt")):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            raw_docs.append(Document(page_content=text,
                                     metadata={"source": os.path.basename(txt_path)}))

    if not raw_docs:
        print("No documents found in:", folder_paths)
        return None, None, []

    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    persist_directory = "/home/shiraoka/projects/kb_agent/vector_store4"
    kg_file = Path(persist_directory) / "knowledge_graph.pkl"
    chroma_index_path = Path(persist_directory) / "chroma-collections.parquet"

    if chroma_index_path.exists():
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_directory
        )

    if kg_file.exists():
        with open(kg_file, "rb") as f:
            G = pickle.load(f)
    else:
        G = build_knowledge_graph_from_chunks(chunks, triple_extraction_chain)
        with open(kg_file, "wb") as f:
            pickle.dump(G, f)

    return G, vectorstore, chunks


#########################################
# 4) GRAPH-BASED RETRIEVAL
#########################################
def graph_retrieve(G: nx.DiGraph, query: str, top_k=3) -> List[str]:
    triple_texts = []
    for subj, obj, attrs in G.edges(data=True):
        relation = attrs.get('relation', '')
        triple_str = f"({subj}, {relation}, {obj})"
        score = sum(term in (subj + relation + obj).lower() for term in query.lower().split())
        triple_texts.append((triple_str, score))
    triple_texts.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in triple_texts[:top_k] if t[1] > 0]


#########################################
# 5) EMBEDDING-BASED RETRIEVAL
#########################################
def embedding_retrieve(vectorstore, query: str, top_k=3) -> List[str]:
    results = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


#########################################
# 6) HYBRID RETRIEVAL
#########################################
def hybrid_retrieve(G: nx.DiGraph, vectorstore, query: str, top_k=3) -> str:
    top_triples = graph_retrieve(G, query, top_k=top_k)
    top_chunks = embedding_retrieve(vectorstore, query, top_k=top_k)
    merged = ""
    if top_triples:
        merged += "### Relationship Facts:\n"
        for t in top_triples:
            merged += f"- {t}\n"
        merged += "\n"
    if top_chunks:
        merged += "### Embedding-based Fragments:\n"
        for i, chunk in enumerate(top_chunks, 1):
            merged += f"Fragment {i}:\n{chunk}\n\n"
    return merged


def extract_links(text: str, exclude: str | None = None) -> List[str]:
    pattern = re.compile(r"https?://[^\s<>'\"]+")
    seen = set()
    links = []
    for raw in pattern.findall(text or ""):
        cleaned = raw.rstrip('.,)')
        
        if cleaned and cleaned not in seen and cleaned != exclude:
            seen.add(cleaned)
            links.append(cleaned)
    return links


#########################################
# 7) FINAL ANSWER WITH LOCAL LLM
#########################################
def answer_with_local_llm(query: str, context: str) -> str:
    prompt = (
        "You are an AI assistant with knowledge from a knowledge graph and semantically "
        "retrieved text chunks. Using ONLY the information below, answer the question.\n\n"
        f"Context:\n{context}\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    llm = ChatOllama(model="deepseek-r1", temperature=0.2, top_k=5, top_p=0.8)
    return llm.predict(prompt)


#########################################
# 8) STREAMLIT UI
#########################################
st.set_page_config(page_title="Member Service Knowledge Base", layout="wide")




# Top header decoration
from streamlit.components.v1 import html
with st.container():
    html("""
    <script>
    var decoration = window.parent.document.querySelectorAll('[data-testid="stDecoration"]')[0];
    decoration.style.height = "2.0rem";
    decoration.style.right = "-50px";
    decoration.innerText = "Powered by Alli AI";
    decoration.style.fontWeight = "bold";
    decoration.style.display = "flex";
    decoration.style.fontSize = "1em";
    decoration.style.backgroundColor = "transparent";
    decoration.style.backgroundImage = "none";
    decoration.style.backgroundSize = "unset";
    </script>
    """, width=0, height=0)

# Custom styling
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { margin-top: -180px !important; padding-top: 0 !important; }
    .block-container { background-color: #FFFFFF !important; }
    [data-testid="stHeader"] { background-color: #FFFFFF !important; }
    [data-testid="stTextInput"] input { background-color: white !important; color: black !important; }
    [data-testid="stSidebar"] { background-color:#6495ED; border-right:2px solid black !important; }
    [data-testid="stSidebar"] * { color: white; }
    [data-testid="stSidebar"]::-webkit-scrollbar { width: 12px; }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb { background-color:white; border-radius:6px; border:3px solid #333333; }
    [data-testid="stAppViewContainer"] { border-left:2px solid black !important; }

    /* make the text inside all sidebar text_areas black */
    [data-testid="stSidebar"] .stTextArea textarea {
        color: black !important;
    } 
    
    /* Make the sidebar scrollbar wider */
    [data-testid="stSidebar"] ::-webkit-scrollbar {
        background-color: #FFFFFF !important;
        width: 25px !important;
    }
    
    .kb-highlight {
        background-color: #ffd7a4 !important;  /* very light yellow */
        color: inherit;
    }

    [data-testid="stSidebar"] .sidebar-article-wrapper {
        max-height: calc(100vh - 220px);
        overflow-y: auto;
        padding-right: 12px;
        scrollbar-width: thin;
    }

    [data-testid="stSidebar"] .sidebar-article-wrapper::-webkit-scrollbar {
        width: 14px;
    }

    [data-testid="stSidebar"] .sidebar-article-wrapper::-webkit-scrollbar-thumb {
        background: #ffffff;
        border-radius: 6px;
        border: 3px solid #333333;
    }


   </style>
    """, unsafe_allow_html=True
)

def log_query(question: str, answer: str, agent_reasoning: str, retrieval_context: str):
    log_file = "/home/shiraoka/projects/kb_agent/log/gquery_log.csv"
    fieldnames = ["timestamp", "question", "answer", "agent_reasoning", "retrieval_context"]
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "agent_reasoning": agent_reasoning,
            "retrieval_context": retrieval_context
        })

@st.cache_resource
def init_resources(folder_paths: List[str]):
    G, vectordb, _ = build_graph_and_vectorstore(folder_paths)
    return G, vectordb

def main():
    st.title("Knowledge Base Agent")

    text_dirs = [
        "/home/shiraoka/projects/kb_agent/data/kbcontents/renamed_clean_text",
        "/home/shiraoka/projects/kb_agent/data/kbcontents/eoc"
    ]
    G, vectordb = init_resources(text_dirs)
    
    if G is None:
        st.error("No valid documents found in the specified folders. Please check your data paths.")
        return

    # Initialize session_state keys
    if "final_answer" not in st.session_state:
        st.session_state["final_answer"] = ""
    if "reasoning_text" not in st.session_state:
        st.session_state["reasoning_text"] = ""
    if "retrieval_context" not in st.session_state:
        st.session_state["retrieval_context"] = ""
    if "fragment1_text" not in st.session_state:
        st.session_state["fragment1_text"] = ""
    if "fragment1_source" not in st.session_state:
        st.session_state["fragment1_source"] = ""
    if "fragment1_fulltext" not in st.session_state:
        st.session_state["fragment1_fulltext"] = ""
    if "reference_link" not in st.session_state:
        st.session_state["reference_link"] = ""
    if "retrieved_links" not in st.session_state:
        st.session_state["retrieved_links"] = []
    
    # Add KB Agent title
    img_path = "/home/shiraoka/projects/kb_agent/data/ai_bot.png"
    img_b64  = base64.b64encode(open(img_path, "rb").read()).decode()
    st.markdown(f"""<h2 style='margin-top:-20px;'>KBase Agent<img src="data:image/png;base64,{img_b64}" width="70" style="margin-left:10px;"/></h2>""", unsafe_allow_html=True)


    user_query = st.text_input(
        "Please enter your specific question including necessary context:",
        key="user_query"
    )
    
    if st.button("Submit Question"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                # 1) Hybrid retrieve & store context
                context = hybrid_retrieve(G, vectordb, user_query, top_k=3)
                st.session_state["retrieval_context"] = context

                # 2) Try to fetch the top embedding match for the sidebar file
                docs = vectordb.similarity_search(user_query, k=3)
                if docs:
                    first_doc = docs[0]
                    src = first_doc.metadata.get("source", "")
                    st.session_state["fragment1_source"] = src
                    content = ""
                    for folder in text_dirs:
                        p = Path(folder) / src
                        if p.exists():
                            content = p.read_text(encoding="utf-8")
                            break

                    special_sources = {"2024_Medi-Cal_EOC_EN_clean.txt", "IHSS_EOC_ENG.txt"}
                    if src in special_sources:
                        relevant_chunks = [doc.page_content for doc in docs if doc.metadata.get("source") == src]
                        chunk_text = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else first_doc.page_content
                        st.session_state["fragment1_fulltext"] = chunk_text
                        content_for_links = chunk_text
                    else:
                        st.session_state["fragment1_fulltext"] = content
                        content_for_links = content

                    reference_match = re.search(r"https?://[^\s<>'\"]+", content_for_links or "")
                    reference_link = reference_match.group(0).rstrip('.,)') if reference_match else ""
                    st.session_state["reference_link"] = reference_link
                    st.session_state["retrieved_links"] = extract_links(content_for_links, exclude=reference_link)
                else:
                    # no match at all
                    st.session_state["fragment1_source"]   = ""
                    st.session_state["fragment1_fulltext"] = ""
                    st.session_state["retrieved_links"] = []
                    st.session_state["reference_link"] = ""

                # 4) Generate final answer
                final_answer = answer_with_local_llm(user_query, context)
                
                reasoning = ""
                match = re.search(r"<think>(.*?)</think>", final_answer, re.DOTALL)
                if match:
                    reasoning = match.group(1).strip()
                    final_answer = re.sub(r"<think>.*?</think>", "", final_answer, flags=re.DOTALL).strip()
                
                st.session_state["final_answer"] = final_answer
                st.session_state["reasoning_text"] = reasoning
                log_query(user_query, final_answer, reasoning, context)

    st.markdown("## Answer")
    st.write(st.session_state["final_answer"])

    if st.session_state.get("retrieved_links"):
        st.markdown("### Related Links")
        for url in st.session_state["retrieved_links"]:
            st.markdown(f"- [{url}]({url})")

    if st.session_state["reasoning_text"]:
        with st.sidebar.expander("### View Agent Reasoning"):
            st.write(st.session_state["reasoning_text"])
    
    # ----- SHOW TOP MATCHING FILES in app by user keywords -----
    # extract keywords (no articles/prepositions)
    stopwords = {
    "a", "an", "the", "i", "we", "you", "he", "she", "it", "they", "me", "my", "mine",
    "in", "on", "at", "for", "to", "from", "with", "of", "by", "about", "as", "into",
    "out", "up", "down", "over", "under", "off", "again", "further",
    "and", "or", "but", "so", "yet", "because", "although", "though", "while",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "not", "no", "yes", "if", "than", "then", "just", "only", "also", "very", "too", "more", "most", "some", "such", "each", "every", "either", "neither",
    "all", "any", "both", "few", "many", "much", "one", "two", "three"
    }
    
    query_words = re.findall(r"\w+", user_query.lower())
    keywords = [w for w in query_words if w not in stopwords]

    def _keyword_variants(term: str) -> set[str]:
        term = term.lower()
        variants = {term}
        if len(term) > 3 and term.endswith("ies"):
            variants.add(term[:-3] + "y")
        if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            variants.add(term[:-1])
        if len(term) > 3 and not term.endswith("s"):
            variants.add(term + "s")
        return {v for v in variants if len(v) > 1}


    def highlight_keywords(text: str, terms: list[str]) -> str:
        if not text:
            return ""
        if not terms:
            return html_module.escape(text)

        # --- Config knobs ---
        # Base-word suffixes we allow (kept short to avoid overmatch)
        COMMON_SUFFIXES = ("s", "es", "ed", "ing", "al", "ally")
        # Acronyms that may appear as compounds with underscores (e.g., CTC_NMT)
        # You can leave this empty and it will auto-detect ALL-CAPS 2–6 char tokens in `terms`.
        ACRONYM_WHITELIST = set()

        # Normalize inputs
        raw_terms = [t.strip() for t in terms if t and t.strip()]
        if not raw_terms:
            return html_module.escape(text)

        # Partition: acronyms vs. normal words
        acronyms = {t for t in raw_terms if t.isupper() and 2 <= len(t) <= 6}
        if ACRONYM_WHITELIST:
            acronyms |= ACRONYM_WHITELIST
        normals  = {t.lower() for t in raw_terms if not (t.isupper() and 2 <= len(t) <= 6)}

        # Build patterns
        pieces = []

        # 1) Acronyms:
        #   - exact token: \bCTC\b
        #   - underscore compound: \bCTC(?:_[A-Za-z0-9]+)+\b  (matches CTC_NMT, CTC_NMT_V2, etc.)
        for ac in sorted(acronyms, key=len, reverse=True):
            esc = re.escape(ac)
            pieces.append(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])")                 # exact CTC
            pieces.append(rf"(?<![A-Za-z0-9]){esc}(?:_[A-Za-z0-9]+)+(?![A-Za-z0-9])")  # CTC_*

        # 2) Base words (length >= 4 only → avoids “pa”, “ed” noise):
        #   - exact token: (?<![A-Za-z])instruction(?![A-Za-z])
        #   - limited suffixes: instruction(?:s|es|ed|ing|al|ally)
        #   - hyphen/underscore compounds: instruction(?:[-_][A-Za-z0-9]+)+
        for t in sorted(normals, key=len, reverse=True):
            if len(t) < 4:
                # too short → skip to avoid random matches
                continue
            esc = re.escape(t)
            suffixes = "|".join(COMMON_SUFFIXES)
            # exact or with limited suffixes
            pieces.append(rf"(?<![A-Za-z]){esc}(?:{'' if not suffixes else '(?:' + suffixes + ')'})?(?![A-Za-z])")
            # compounds: instruction_guide / instruction-related
            pieces.append(rf"(?<![A-Za-z]){esc}(?:[-_][A-Za-z0-9]+)+(?![A-Za-z0-9])")

        if not pieces:
            return html_module.escape(text)

        pattern = re.compile("|".join(pieces), re.IGNORECASE)

        highlighted_parts: list[str] = []
        last_index = 0
        for match in pattern.finditer(text):
            start, end = match.span()
            if start < last_index:
                # Overlapping match; skip to avoid nested spans.
                continue
            highlighted_parts.append(html_module.escape(text[last_index:start]))
            highlighted_parts.append(
                f"<span class='kb-highlight'>{html_module.escape(match.group(0))}</span>"
            )
            last_index = end

        highlighted_parts.append(html_module.escape(text[last_index:]))
        return "".join(highlighted_parts)


    def linkify_and_highlight(text: str, terms: list[str]) -> str:
        if not text:
            return ""

        pattern = re.compile(r"https?://[^\s<>'\"]+")
        pieces: list[str] = []
        last_index = 0

        for match in pattern.finditer(text):
            start, end = match.span()
            before = text[last_index:start]
            if before:
                pieces.append(highlight_keywords(before, terms))

            url = match.group(0)
            cleaned = url.rstrip('.,)')
            trailing = url[len(cleaned):]
            anchor = (
                f"<a href=\"{html_module.escape(cleaned)}\" target=\"_blank\">"
                f"{html_module.escape(cleaned)}</a>"
            )
            pieces.append(anchor)

            if trailing:
                pieces.append(highlight_keywords(trailing, terms))

            last_index = end

        if last_index < len(text):
            pieces.append(highlight_keywords(text[last_index:], terms))

        return "".join(pieces).replace("\n", "<br/>")

    # score files by keyword frequency
    scores = []
    clean_dir = "/home/shiraoka/projects/kb_agent/data/kbcontents/renamed_clean_text"
    for path in glob.glob(os.path.join(clean_dir, "*.txt")):
        # skip the concatenated all.txt file
        if os.path.basename(path) == "all.txt":
          continue 
        text = Path(path).read_text(encoding="utf-8").lower()
        score = sum(text.count(kw) for kw in keywords)
        if score > 0:
            scores.append((os.path.basename(path), score))

    # top 10
    top_files = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

    st.subheader("Top Matching Articles")
    if top_files:
        for fname, cnt in top_files:
            with st.expander(f"{fname} — {cnt} hits"):
                file_text = Path(os.path.join(clean_dir, fname)).read_text(encoding="utf-8")
                st.text_area("Contents", file_text, height=300)
    else:
        st.write("No matching files found.")
    
    
    comment_text = st.text_input(
        "", placeholder="Please enter any comment(s) or suggestion(s) here", key="comment_input"
    )
    if st.button("Send Comment"):
        comment_log_file = "/home/shiraoka/projects/kb_agent/log/comment_log.csv"
        file_exists = os.path.exists(comment_log_file)
        with open(comment_log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "comment"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), comment_text])
        st.success("Comment added!")
        
    # Spacer so the full-file expander is lower and visible
    st.sidebar.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)

        # Sidebar full source file (always visible)
    if st.session_state["fragment1_fulltext"]:
        st.sidebar.markdown(f"### 📄 KB File ({st.session_state['fragment1_source']})")
        raw = st.session_state["fragment1_fulltext"]
        html = linkify_and_highlight(raw, keywords)
        st.sidebar.markdown(
            f"""
            <div class="sidebar-article-wrapper">
                {html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown(
            """
            <script>
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"] section');
            if (sidebar) { sidebar.scrollTop = 0; }
            </script>
            """,
            unsafe_allow_html=True,
        )


    # display CCAH image at the bottom
    st.sidebar.image(
    "/home/shiraoka/projects/kb_agent/data/ccah.png",
    caption=None,
    width=500 
    )     

if __name__ == "__main__":
    main()
