import os
import time
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Any
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Cache frequently used computations
@st.cache_resource
def init_embeddings():
    """Initialize and cache embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder="/tmp/hf_cache"
    )
@st.cache_data
def load_pdf_document(_file):
    """Cache PDF loading operations - only first page"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(_file.getvalue())
        loader = PyPDFLoader(tmp_file.name)
        # Load all pages but only return the first one
        pages = loader.load()
        return [pages[0]] if pages else []

class SessionState:
    """Handle session state initialization"""
    @staticmethod
    def init_session_state():
        default_states = {
            'mode': 'home',
            'processing_times': [],
            'chat_history': [],
            'vectorstorage': None,
            'error': None
        }
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

class EnvironmentManager:
    """Manage environment variables and API keys"""
    REQUIRED_ENV_VARS = {
        'GROQ_API_KEY': 'Groq API key',
       # 'HF_TOKEN': 'HuggingFace token'
    }

    @staticmethod
    def check_environment() -> bool:
        """Verify all required environment variables are set"""
        missing_vars = []
        for var, description in EnvironmentManager.REQUIRED_ENV_VARS.items():
            if not os.getenv(var):
                missing_vars.append(f"{description} ({var})")
        
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            st.info("Please create a .env file with the following variables:\n" + 
                   "\n".join([f"{var}=your_{var.lower()}" for var in EnvironmentManager.REQUIRED_ENV_VARS]))
            return False
        return True

class LLMManager:
    """Handle LLM initialization and operations"""
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.llm = self._init_llm()
        self.prompt = self._init_prompt()

    def _init_llm(self):
        """Initialize LLM with optimized settings"""
        return ChatGroq(
            temperature=0.1,
            groq_api_key=self.api_key,
            model_name="mixtral-8x7b-32768",
            max_tokens=4096,
            top_p=0.9,
            streaming=True
        )

    def _init_prompt(self):
        """Initialize prompt template with improved system message"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a precise and helpful assistant. When answering questions:
                         - Use the provided context accurately
                         - Be concise but thorough
                         - If the answer isn't in the context, say so
                         - Cite specific parts of the context when relevant"""),
            ("user", "Context: {context}\nQuestion: {input}"),
            ("assistant", "I'll provide a precise answer based on the context provided.")
        ])

class VectorStoreManager:
    """Manage vector store operations"""
    def __init__(self):
        self.embeddings = init_embeddings()

    def create_vector_store(self, files) -> bool:
        """Create vector store from first page of uploaded files"""
        try:
            documents = []
            for file in files:
                first_page = load_pdf_document(file)
                documents.extend(first_page)
            
            if documents:
                st.session_state.vectorstorage = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                return True
            else:
                st.session_state.error = "No valid documents found to process."
                return False
        except Exception as e:
            st.session_state.error = f"Error creating vector store: {str(e)}"
            return False

class QueryProcessor:
    """Handle query processing and response generation"""
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    def process_rag_query(self, query: str) -> Tuple[Optional[dict], float]:
        """Process RAG queries with improved error handling"""
        try:
            document_chain = create_stuff_documents_chain(
                self.llm_manager.llm,
                self.llm_manager.prompt
            )
            retriever = st.session_state.vectorstorage.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            retriever_chain = create_retrieval_chain(retriever, document_chain)
            
            start_time = time.process_time()
            response = retriever_chain.invoke({'input': query})
            return response, time.process_time() - start_time
        except Exception as e:
            st.session_state.error = f"Query processing error: {str(e)}"
            return None, 0

    def process_qa_query(self, query: str) -> Tuple[Optional[str], float]:
        """Process direct QA queries"""
        try:
            start_time = time.process_time()
            response = self.llm_manager.llm.invoke(query)
            return str(response), time.process_time() - start_time
        except Exception as e:
            st.session_state.error = f"Query processing error: {str(e)}"
            return None, 0

def display_sidebar_metrics():
    """Display performance metrics in sidebar"""
    with st.sidebar:
        st.title("Performance Metrics")
        if st.session_state.processing_times:
            avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            st.metric("Total Queries", len(st.session_state.processing_times))
            
            if st.button("Clear History"):
                st.session_state.processing_times = []
                st.session_state.chat_history = []
                st.rerun()

def display_home():
    """Display home page with mode selection"""
    st.title('Document Analysis & Q&A System')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 RAG Document Query", use_container_width=True):
            st.session_state.mode = 'rag'
            st.rerun()
            
    with col2:
        if st.button("💭 Q&A Chatbot", use_container_width=True):
            st.session_state.mode = 'qa'
            st.rerun()

def display_rag(vector_store_manager: VectorStoreManager, query_processor: QueryProcessor):
    """Display RAG interface"""
    st.title('📚 RAG Document Query System')
    if st.button("← Back to Home"):
        st.session_state.mode = 'home'
        st.rerun()
        
    st.write("Upload PDFs and ask questions about their content.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Initialize Vector Embeddings"):
        if vector_store_manager.create_vector_store(uploaded_files):
            st.success("Vector embeddings created successfully!")
    
    user_prompt = st.text_input(
        "Enter your question about the documents:",
        placeholder="What would you like to know?"
    )
    
    if user_prompt and st.session_state.vectorstorage:
        response, processing_time = query_processor.process_rag_query(user_prompt)
        
        if response:
            st.write("### Answer")
            st.write(response['answer'])
            st.info(f"Processing time: {processing_time:.2f} seconds")
            
            with st.expander("Document Similarity Search Results"):
                for i, doc in enumerate(response['context'], 1):
                    st.markdown(f"**Relevant Extract {i}:**")
                    st.write(doc.page_content)
                    st.divider()
    elif user_prompt:
        st.warning("Please initialize vector embeddings first!")

def display_qa(llm_manager: LLMManager):
    """Display Q&A interface"""
    st.title('💭 Q&A Chatbot')
    if st.button("← Back to Home"):
        st.session_state.mode = 'home'
        st.rerun()
    
    query_processor = QueryProcessor(llm_manager)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_prompt = st.chat_input("Ask me anything!")
    
    if user_prompt:
        # Display user message
        with st.chat_message("user"):
            st.write(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response, processing_time = query_processor.process_qa_query(user_prompt)
            if response:
                st.write(response)
                st.session_state.processing_times.append(processing_time)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.caption(f"Processed in {processing_time:.2f} seconds")

def main():
    # Initialize session state
    SessionState.init_session_state()
    
    # Display sidebar metrics
    display_sidebar_metrics()
    
    # Check environment variables
    if not EnvironmentManager.check_environment():
        return

    # Initialize managers
    llm_manager = LLMManager()
    vector_store_manager = VectorStoreManager()
    query_processor = QueryProcessor(llm_manager)

    # Display interface based on mode
    if st.session_state.mode == 'home':
        display_home()
    elif st.session_state.mode == 'rag':
        display_rag(vector_store_manager, query_processor)
    elif st.session_state.mode == 'qa':
        display_qa(llm_manager)

    # Display any errors
    if st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None

if __name__ == "__main__":
    main()