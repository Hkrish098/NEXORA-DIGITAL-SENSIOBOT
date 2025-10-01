# app.py - Final Version with All Features, Including Feedback

import os
import re
import pickle
import logging
from typing import List
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document

# --- Import the action tools from db.py ---
from db import create_support_ticket, find_ticket_by_id

# --- 1. Setup and Configuration ---
load_dotenv()
st.set_page_config(page_title="SentioBot | Nexora Support", page_icon="💡", layout="centered")

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- (CSS, Pydantic Models, and all previous functions remain the same) ---
USER_AVATAR = "https://api.dicebear.com/7.x/adventurer/svg?seed=user"
ASSISTANT_AVATAR = "https://api.dicebear.com/7.x/bottts/svg?seed=sentiobot&backgroundColor=00ffff"

def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #0E1117; }
            [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            .stTextInput>div>div>input { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; }
            .stExpander { background-color: #161B22; border-radius: 8px; border: 1px solid #30363D; }
            .stChatMessage { animation: fadeIn 0.5s; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            div[data-testid="stHorizontalBlock"] button {
                background-color: #161B22; border: 1px solid #30363D; color: #C9D1D9;
                padding: 8px 12px; border-radius: 8px; transition: background-color 0.3s;
            }
            div[data-testid="stHorizontalBlock"] button:hover { background-color: #30363D; color: #FFFFFF; }
        </style>
    """, unsafe_allow_html=True)

load_css()

class Citation(BaseModel):
    source_id: int = Field(description="The integer index of the source document that supports the claim.")
    claim: str = Field(description="The specific claim from the answer supported by this source.")

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The final answer to the user's question, in Markdown.")
    citations: List[Citation] = Field(description="A list of all claims and their source IDs.")

@st.cache_resource(show_spinner="Initializing SentioBot...")
def get_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db_path, store_path, parent_list_path = "vector_db", "parent_docstore", "parents.pkl"
    if not all(os.path.exists(p) for p in [db_path, store_path, parent_list_path]):
        st.error("Data stores not found. Please run your data ingestion script first.")
        st.stop()
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    byte_store = LocalFileStore(store_path)
    store = create_kv_docstore(byte_store)
    with open(parent_list_path, 'rb') as f:
        all_parent_docs = pickle.load(f)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    parent_retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id", child_splitter=child_splitter)
    bm25_retriever = BM25Retriever.from_documents(all_parent_docs)
    bm25_retriever.k = 10
    ensemble_retriever = EnsembleRetriever(retrievers=[parent_retriever, bm25_retriever], weights=[0.5, 0.5])
    llm = get_llm()
    multiquery_retriever = MultiQueryRetriever.from_llm(retriever=ensemble_retriever, llm=llm)
    return multiquery_retriever

@st.cache_resource(show_spinner=False)
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Google API key is not set. Please add it to your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.1)

def format_answer_with_citations(response_obj: AnswerWithCitations, sources: list) -> str:
    formatted_answer = response_obj.answer
    cited_source_ids = sorted(list(set(c.source_id for c in response_obj.citations)))
    citation_markers = "".join(f" [{i+1}]" for i in range(len(cited_source_ids)))
    if citation_markers:
        formatted_answer += f" {citation_markers}"
    citation_references = []
    for i, source_id in enumerate(cited_source_ids):
        if 1 <= source_id <= len(sources):
            source_doc = sources[source_id - 1]
            source_name = source_doc.metadata.get('source', 'N/A')
            section_name = source_doc.metadata.get('section_title', 'N/A')
            citation_references.append(f"[{i+1}] {source_name} | Section: {section_name}")
    if citation_references:
        formatted_answer += "\n\n---\n**Sources:**\n" + "\n".join(citation_references)
    return formatted_answer

def format_docs_with_ids(docs: List[Document]) -> str:
    formatted = [f"---\nSource ID: {i+1}\nContent: {doc.page_content}\n---" for i, doc in enumerate(docs)]
    return "\n\n".join(formatted)

@st.cache_data(show_spinner=False)
def get_query_category(query: str) -> str:
    llm = get_llm()
    classification_prompt_text = """Your task is to analyze the user's query and determine their primary goal. Respond with ONLY ONE of the following two categories:
    1. `create_a_ticket`: Choose this ONLY if the user's primary goal is to get help for a broken/malfunctioning device.
    2. `answer_a_question`: This is the default. Choose this if the user is asking for information, REGARDLESS of context.
    --- EXAMPLES ---
    User Query: "My Thermosmart Pro is not working. Is this covered by the warranty?" -> Category: answer_a_question
    User Query: "I'm stuck on step 3 of the guide for my broken bulb." -> Category: answer_a_question
    User Query: "My lightbulb is broken, I need help." -> Category: create_a_ticket
    ---
    User Query: {user_query} -> Category:"""
    prompt = ChatPromptTemplate.from_template(classification_prompt_text)
    chain = prompt | llm
    try:
        response = chain.invoke({"user_query": query})
        category = response.content.strip().lower()
        return "create_a_ticket" if "create_a_ticket" in category else "answer_a_question"
    except Exception as e:
        return "answer_a_question"

def get_rag_chain(_retriever):
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnswerWithCitations)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "Given a chat history, formulate a standalone question."), MessagesPlaceholder("chat_history"), ("human", "{input}")])
    history_aware_retriever = create_history_aware_retriever(llm, _retriever, contextualize_q_prompt)
    qa_system_prompt = """## Persona and Role
You are SentioBot... [Your detailed RAG prompt]"""
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "Question: {input}\nContext:\n{context}")])
    rag_chain = (RunnablePassthrough.assign(context=history_aware_retriever)
                 .assign(answer=(RunnablePassthrough.assign(context=lambda x: format_docs_with_ids(x["context"])) | qa_prompt | structured_llm)))
    return rag_chain

# --- Main Application Initialization ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "feedback" not in st.session_state: st.session_state.feedback = {}
if "show_ticket_form" not in st.session_state: st.session_state.show_ticket_form = False
if "last_problem_query" not in st.session_state: st.session_state.last_problem_query = ""
if "last_query_processed" not in st.session_state: st.session_state.last_query_processed = None

retriever = get_retriever()
rag_chain = get_rag_chain(retriever)

st.title("SentioBot: Your Nexora Electronics Expert")

if not st.session_state.chat_history:
    st.session_state.chat_history.append(AIMessage(content="Hello! I am SentioBot. How can I assist you with your Nexora devices today?"))

# --- RE-INTEGRATED FEEDBACK LOGIC HERE ---
for i, msg in enumerate(st.session_state.chat_history):
    avatar = ASSISTANT_AVATAR if isinstance(msg, AIMessage) else USER_AVATAR
    with st.chat_message(msg.type, avatar=avatar):
        st.markdown(msg.content)
        if isinstance(msg, AIMessage) and i > 0:
            feedback_key = f"feedback_{i}"
            if feedback_key not in st.session_state.feedback:
                cols = st.columns([1, 1, 10])
                if cols[0].button("👍", key=f"up_{i}", use_container_width=True):
                    st.session_state.feedback[feedback_key] = "up"
                    st.rerun()
                if cols[1].button("👎", key=f"down_{i}", use_container_width=True):
                    st.session_state.feedback[feedback_key] = "down"
                    st.rerun()
            else:
                st.caption(f"✓ Feedback submitted as '{st.session_state.feedback[feedback_key]}'")

            if sources := msg.additional_kwargs.get("sources"):
                with st.expander("View All Retrieved Sources"):
                    for source in sources:
                        st.info(f"**Source:** `{source.metadata.get('source', 'N/A')}`\n\n**Section:** `{source.metadata.get('section_title', 'N/A')}`")

if "user_query" not in st.session_state: st.session_state.user_query = ""
def set_query_from_button(query): st.session_state.user_query = query
if len(st.session_state.chat_history) <= 1:
    cols = st.columns(3)
    if cols[0].button("Reset Thermosmart Pro?"): set_query_from_button("How to reset Thermosmart Pro?"); st.rerun()
    if cols[1].button("Warranty Period?"): set_query_from_button("What is the warranty period?"); st.rerun()
    if cols[2].button("Power Sync Safety?"): set_query_from_button("Is the Power Sync plug safe?"); st.rerun()

user_query = st.chat_input("Ask me about Nexora products...")

if user_query and not st.session_state.get("show_ticket_form", False):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.last_query_processed = None # Allow the new query to be processed
    st.rerun()

if st.session_state.chat_history and st.session_state.chat_history[-1].type == 'human' and st.session_state.last_query_processed != st.session_state.chat_history[-1].content:
    query = st.session_state.chat_history[-1].content
    st.session_state.last_query_processed = query
    
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        match = re.search(r"NEX-\d{5}", query, re.IGNORECASE)
        if match:
            with st.spinner(f"Searching for ticket..."):
                details = find_ticket_by_id(match.group(0).upper())
                answer = f"I found the details for ticket **{details['TicketID']}**:\n- Customer: {details['CustomerName']}\n- Status: {details['Status']}" if details else f"I could not find ticket **{match.group(0).upper()}**."
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.rerun()
        else:
            with st.spinner("Analyzing your request..."):
                category = get_query_category(query)
            if category == "create_a_ticket":
                with st.spinner("Searching for troubleshooting..."):
                    response_dict = rag_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
                    response_obj = response_dict.get("answer")
                    sources = response_dict.get("context", [])
                    answer = "I'm sorry, I couldn't find specific troubleshooting steps, but I can create a support ticket for you."
                    if isinstance(response_obj, AnswerWithCitations) and response_obj.answer:
                        answer = "I found some potential troubleshooting steps for you:\n\n" + format_answer_with_citations(response_obj, sources)
                    st.session_state.chat_history.append(AIMessage(content=answer, additional_kwargs={"sources": sources}))
                    st.session_state.show_ticket_form = True
                    st.session_state.last_problem_query = query
                    st.rerun()
            else: # answer_a_question
                with st.spinner("Searching for an answer..."):
                    response_dict = rag_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
                    response_obj = response_dict.get("answer")
                    sources = response_dict.get("context", [])
                    if isinstance(response_obj, AnswerWithCitations):
                        final_answer = format_answer_with_citations(response_obj, sources)
                    else:
                        final_answer = "Sorry, I couldn't generate a valid answer at this time. Please try rephrasing."
                    st.session_state.chat_history.append(AIMessage(content=final_answer, additional_kwargs={"sources": sources}))
                    st.rerun()

if st.session_state.get("show_ticket_form", False):
    st.info("If those steps didn't help, I can create a support ticket.")
    with st.form("ticket_form"):
        name = st.text_input("Your Name", key="user_name")
        email = st.text_input("Your Email", key="user_email")
        if st.form_submit_button("Submit Ticket"):
            if not name or not email:
                st.error("Please provide both name and email.")
            else:
                with st.spinner("Creating ticket..."):
                    problem = st.session_state.last_problem_query
                    product = "Nexora Device"
                    if "secureview" in problem.lower(): product = "SecureView Camera"
                    elif "thermosmart" in problem.lower(): product = "Thermosmart Pro"
                    ticket_id = create_support_ticket(name, email, product, problem)
                    answer = f"Thank you, **{name}**. I've created ticket **{ticket_id}** for you."
                    st.success("Ticket created!")
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    st.session_state.show_ticket_form = False
                    st.session_state.last_query_processed = None
                    st.rerun()