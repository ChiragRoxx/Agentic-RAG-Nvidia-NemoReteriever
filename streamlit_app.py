import streamlit as st
import sys
import os
import time
import random
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_agent import RAGAgent
from src.nvidia_embeddings import NVIDIAEmbeddings

# Try to import agentic capabilities
try:
    from src.agentic_rag import AgenticRAGAgent, AGENTIC_AI_AVAILABLE
except ImportError:
    AGENTIC_AI_AVAILABLE = False
    AgenticRAGAgent = None

# Page configuration
st.set_page_config(
    page_title="RAG Assistant - For the hydrogen and Renewable Energy Resources",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-color: #7046f8;
        --secondary-color: #00b4d8;
        --accent-color: #ff4081;
        --dark-bg: #11131e;
        --light-bg: #f8f9fa;
        --success-color: #4caf50;
        --warning-color: #ff9800;
        --error-color: #f44336;
    }

    /* Improve overall text readability */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    }
    
    /* Responsive design adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem !important;
        }
    }
    
    /* Beautiful Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #7046f8 0%, #00b4d8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .main-header:hover {
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .main-header h1 {
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Enhanced Chat Messages */
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-message:hover {
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .user-message {
        background-color: #f0f7ff;
        border-left: 4px solid #7046f8;
    }
    
    .assistant-message {
        background-color: #f9f4ff;
        border-left: 4px solid #00b4d8;
    }
    
    /* Beautiful Source Cards */
    .source-card {
        background-color: #fff9f0;
        border: none;
        border-left: 3px solid #ff9800;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.7rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateX(2px);
    }
    
    /* Stylish Metric Cards */
    .metric-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.6rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    /* Status Indicators with Animation */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        position: relative;
    }
    
    .status-online {
        background-color: #4caf50;
        box-shadow: 0 0 0 rgba(76, 175, 80, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(76, 175, 80, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
        }
    }
    
    .status-offline {
        background-color: #f44336;
    }
    
    /* Agentic Mode Indicators */
    .agentic-badge {
        background: linear-gradient(135deg, #7046f8 0%, #00b4d8 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
        display: inline-block;
    }
    
    .agentic-tools {
        background-color: #f0f8ff;
        border-left: 3px solid #00b4d8;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .mode-indicator {
        font-size: 1.1rem;
        margin-left: 5px;
    }
    
    /* Beautiful Buttons */
    button {
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Card-like Elements */
    .stExpander {
        border: none !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12) !important;
    }
    
    /* Better Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0 20px;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #7046f8 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_agent():
    """Initialize the RAG agent (cached for performance)"""
    try:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")

        if not api_key:
            st.error("❌ NVIDIA_API_KEY not found in environment variables")
            st.info("Please check your .env file and ensure the API key is set correctly.")
            return None

        # Test NVIDIA API connection first
        with st.spinner("🔌 Testing NVIDIA API connection..."):
            try:
                embeddings = NVIDIAEmbeddings(api_key)
                if not embeddings.test_connection():
                    st.error("❌ Failed to connect to NVIDIA API")
                    return None
            except Exception as e:
                st.error(f"❌ NVIDIA API connection failed: {str(e)}")
                return None

        # Initialize RAG agent
        with st.spinner("🤖 Initializing RAG Agent..."):
            rag_agent = RAGAgent(docs_folder, api_key)

        # Setup knowledge base
        with st.spinner("📚 Loading knowledge base..."):
            if rag_agent.setup_knowledge_base():
                st.success("✅ RAG system initialized successfully!")
                return rag_agent
            else:
                st.error("❌ Failed to setup knowledge base")
                st.info("Please ensure PDF files are in the Data/Docs folder.")
                return None

    except Exception as e:
        st.error(f"❌ Failed to initialize RAG agent: {str(e)}")
        st.exception(e)
        return None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Assistant - NVIDIA NemoRetriever</h1>
        <p>AI-Powered Document Q&A System with Advanced Retrieval</p>
        <div style="margin-top: 10px; font-size: 0.8rem; opacity: 0.8;">Powered by NemoRetriever & LLaMA 3.1</div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(rag_agent):
    """Display the sidebar with system information"""
    st.sidebar.markdown("## 📊 System Status")
    
    if rag_agent:
        # System status
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center; background-color: #f0f7ff; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
            <span class="status-indicator status-online"></span>
            <strong>System Online</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Get knowledge base stats
        stats = rag_agent.get_knowledge_base_stats()
        
        st.sidebar.markdown("### 📚 Knowledge Base")
        
        # Display metrics in a more appealing way
        col1, col2 = st.sidebar.columns(2)
        col1.metric("📄 Text Chunks", stats.get('document_count', 0))
        col2.metric("📚 PDF Files", stats.get('pdf_files_available', 0))
        
        # Model information with nicer formatting
        st.sidebar.markdown("""
        <div style="background-color: #f9f4ff; padding: 15px; border-radius: 10px; margin: 15px 0;">
            <h3 style="margin-top: 0; font-size: 1.1rem;">🤖 AI Models</h3>
            <div style="margin: 8px 0;"><strong>Embedding:</strong> nvidia/nv-embed-v1</div>
            <div style="margin: 8px 0;"><strong>LLM:</strong> meta/llama-3.1-8b-instruct</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Agentic AI Mode toggle
        st.sidebar.markdown("### 🧠 AI Mode")
        
        if AGENTIC_AI_AVAILABLE:
            agentic_mode = st.sidebar.toggle(
                "🚀 Agentic AI Mode",
                value=st.session_state.get('agentic_mode', False),
                help="Enable autonomous AI agents with multi-step reasoning, web search, and analysis capabilities."
            )
            st.session_state['agentic_mode'] = agentic_mode
            
            if agentic_mode:
                st.sidebar.success("🚀 Agentic AI Mode: ENABLED")
                st.sidebar.markdown("""
                <div style="background-color: #e8f5e8; padding: 10px; border-radius: 8px; margin: 10px 0;">
                    <small>
                    <strong>Capabilities:</strong><br>
                    • Multi-step reasoning<br>
                    • Web search integration<br>
                    • Information analysis<br>
                    • Tool-based responses
                    </small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.info("📚 Standard RAG Mode: ACTIVE")
        else:
            st.sidebar.warning("🔧 Agentic AI: Not Available")
            st.sidebar.markdown("""
            <div style="background-color: #fff4e6; padding: 10px; border-radius: 8px; margin: 10px 0;">
                <small>
                Install NVIDIA Agent Intelligence Toolkit:<br>
                <code>pip install aiqtoolkit[langchain]</code>
                </small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state['agentic_mode'] = False
        

            
    else:
        st.sidebar.markdown("""
        <div style="display: flex; align-items: center; background-color: #fff0f0; padding: 15px; border-radius: 8px;">
            <span class="status-indicator status-offline"></span>
            <strong>System Offline</strong>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.error("RAG system not available")

def display_chat_interface(rag_agent):
    """Display the main chat interface"""
    st.markdown("## 💬 Ask Your Question")
    
    # Add comprehensive mode toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        comprehensive_mode = st.toggle(
            "🔍 Comprehensive Mode", 
            value=False,
            help="Enable for more detailed, complete responses with extended context. May take longer but provides thorough answers."
        )
        
    if comprehensive_mode:
        st.info("🔍 **Comprehensive Mode Enabled**: Responses will be more detailed and complete, using extended context from your documents.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your RAG Assistant powered by NVIDIA NemoRetriever. I can help you find information from your document collection. What would you like to know?",
            "sources": [],
            "processing_time": 0
        })
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if message.get("sources"):
                display_sources(message["sources"], message.get("processing_time", 0))
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {prompt}
        </div>
        """, unsafe_allow_html=True)
        
        if rag_agent:
            try:
                # Check if agentic mode is enabled
                agentic_mode = st.session_state.get('agentic_mode', False)
                
                # Show loading spinner with appropriate message
                if agentic_mode:
                    loading_message = "🧠 Agentic AI processing (multi-step reasoning)..."
                elif comprehensive_mode:
                    loading_message = "🔍 Searching documents comprehensively..."
                else:
                    loading_message = "🔍 Searching documents..."
                    
                with st.spinner(loading_message):
                    # Get response based on mode
                    if agentic_mode and AGENTIC_AI_AVAILABLE and AgenticRAGAgent:
                        # Initialize agentic agent if not already done
                        if 'agentic_agent' not in st.session_state:
                            try:
                                api_key = os.getenv("NVIDIA_API_KEY")
                                st.session_state['agentic_agent'] = AgenticRAGAgent(
                                    rag_agent=rag_agent,
                                    api_key=api_key,
                                    enable_web_search=True,
                                    enable_analysis=True
                                )
                            except Exception as e:
                                st.error(f"Failed to initialize agentic agent: {str(e)}")
                                agentic_mode = False
                        
                        if agentic_mode and 'agentic_agent' in st.session_state:
                            response = st.session_state['agentic_agent'].ask_question(prompt)
                        else:
                            # Fallback to comprehensive mode
                            response = rag_agent.ask_comprehensive_question(prompt, max_context_chunks=12)
                    elif comprehensive_mode:
                        response = rag_agent.ask_comprehensive_question(prompt, max_context_chunks=12)
                    else:
                        response = rag_agent.ask_question(prompt)

                # Validate response
                if not response or not response.answer:
                    st.error("❌ Failed to get a response. Please try again.")
                    return

                # Add assistant response with enhanced metadata for agentic responses
                message_data = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.source_documents,
                    "processing_time": response.processing_time
                }
                
                # Add agentic-specific metadata if available
                if hasattr(response, 'is_agentic') and response.is_agentic:
                    message_data.update({
                        "is_agentic": True,
                        "agent_reasoning": getattr(response, 'agent_reasoning', ''),
                        "tools_used": getattr(response, 'tools_used', []),
                        "intermediate_steps": getattr(response, 'intermediate_steps', [])
                    })
                
                st.session_state.messages.append(message_data)

                # Display assistant response with mode indicator
                mode_indicator = ""
                if hasattr(response, 'is_agentic') and response.is_agentic:
                    mode_indicator = " 🧠"
                elif comprehensive_mode:
                    mode_indicator = " 🔍"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant{mode_indicator}:</strong><br>
                    {response.answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display agentic information if available
                if hasattr(response, 'is_agentic') and response.is_agentic:
                    if hasattr(response, 'tools_used') and response.tools_used:
                        tools_str = ", ".join(response.tools_used)
                        st.markdown(f"""
                        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 3px solid #00b4d8;">
                            <small><strong>🔧 Tools Used:</strong> {tools_str}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if hasattr(response, 'agent_reasoning') and response.agent_reasoning:
                        with st.expander("🧠 Agent Reasoning", expanded=False):
                            st.write(response.agent_reasoning)

                # Show a visual divider with response time
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 20px 0; opacity: 0.7;">
                    <div style="flex-grow: 1; height: 1px; background-color: #ddd;"></div>
                    <div style="margin: 0 10px; font-size: 0.8rem;">Response generated in {response.processing_time:.2f}s</div>
                    <div style="flex-grow: 1; height: 1px; background-color: #ddd;"></div>
                </div>
                """, unsafe_allow_html=True)

                # Display sources
                display_sources(response.source_documents, response.processing_time)

            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                    "sources": [],
                    "processing_time": 0
                })

        else:
            st.error("❌ RAG system not available. Please check the system status.")
            st.info("Try refreshing the page or contact support if the issue persists.")

def display_sources(source_documents, processing_time, confidence_scores=None):
    """Display source documents in an elegant format"""
    if not source_documents:
        st.info("ℹ️ No specific sources found for this response.")
        return

    # Create unique identifier for this display instance
    import time
    unique_id = f"{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"  # Microseconds + random for uniqueness

    # Create a container for sources
    with st.container():
        st.markdown("---")

        # Header with metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("📄 Sources Found", len(source_documents))
        with col3:
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                st.metric("🎯 Avg. Relevance", f"{avg_confidence:.2f}")

        st.markdown("### 📚 Sources & References")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Document Sources", "📊 Source Analysis", "🔍 Quick Search"])

        with tab1:
            for i, doc in enumerate(source_documents, 1):
                source_file = doc.metadata.get("source_file", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                chunk_id = doc.metadata.get("chunk_id", "Unknown")

                # Clean up filename for display
                display_name = source_file.replace("_", " ").replace("-", " ").title()
                if display_name.endswith(".pdf"):
                    display_name = display_name[:-4]

                # Confidence indicator
                confidence_indicator = ""
                if confidence_scores and i <= len(confidence_scores):
                    score = confidence_scores[i-1]
                    if score > 0.8:
                        confidence_indicator = "🟢 High Relevance"
                    elif score > 0.6:
                        confidence_indicator = "🟡 Medium Relevance"
                    else:
                        confidence_indicator = "🔴 Low Relevance"

                with st.expander(f"📖 Source {i}: {display_name} (Page {page}) {confidence_indicator}"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**📁 File**: {source_file}")
                        st.markdown(f"**📄 Page**: {page}")
                        st.markdown(f"**🔢 Chunk ID**: {chunk_id}")
                        if confidence_scores and i <= len(confidence_scores):
                            st.markdown(f"**🎯 Relevance Score**: {confidence_scores[i-1]:.3f}")

                    with col2:
                        # Quick actions with unique keys
                        if st.button(f"📋 Copy Text {i}", key=f"copy_{i}_{unique_id}"):
                            st.code(doc.page_content, language="text")

                    st.markdown("**📝 Content Preview**:")
                    # Better text formatting
                    preview_text = doc.page_content[:800]
                    if len(doc.page_content) > 800:
                        preview_text += "..."

                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 3px solid #2a5298;">
                        {preview_text}
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            # Create source distribution chart
            source_files = [doc.metadata.get("source_file", "Unknown") for doc in source_documents]
            file_counts = {}
            for file in source_files:
                # Clean filename for chart
                clean_name = file.replace("_", " ").replace("-", " ")
                if clean_name.endswith(".pdf"):
                    clean_name = clean_name[:-4]
                file_counts[clean_name] = file_counts.get(clean_name, 0) + 1

            if file_counts:
                # Pie chart for source distribution
                fig_pie = px.pie(
                    values=list(file_counts.values()),
                    names=list(file_counts.keys()),
                    title="📊 Source Document Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True, key=f"source_distribution_pie_{unique_id}")

                # Bar chart for page distribution
                pages = [doc.metadata.get("page", 0) for doc in source_documents]
                if pages:
                    fig_bar = px.histogram(
                        x=pages,
                        title="📄 Page Distribution of Sources",
                        labels={'x': 'Page Number', 'y': 'Number of Sources'},
                        color_discrete_sequence=['#2a5298']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key=f"page_distribution_bar_{unique_id}")

        with tab3:
            st.markdown("#### 🔍 Search Within Sources")
            search_term = st.text_input("Search for specific terms in the source documents:", key=f"search_sources_{unique_id}")

            if search_term:
                matches = []
                for i, doc in enumerate(source_documents, 1):
                    if search_term.lower() in doc.page_content.lower():
                        # Find the context around the search term
                        content = doc.page_content.lower()
                        index = content.find(search_term.lower())
                        start = max(0, index - 100)
                        end = min(len(content), index + len(search_term) + 100)
                        context = doc.page_content[start:end]

                        matches.append({
                            'source': i,
                            'file': doc.metadata.get("source_file", "Unknown"),
                            'page': doc.metadata.get("page", "Unknown"),
                            'context': context
                        })

                if matches:
                    st.success(f"Found {len(matches)} matches for '{search_term}':")
                    for match in matches:
                        st.markdown(f"""
                        **Source {match['source']}** - {match['file']} (Page {match['page']})
                        > ...{match['context']}...
                        """)
                else:
                    st.info(f"No matches found for '{search_term}' in the source documents.")

def display_document_stats(rag_agent):
    """Display detailed document statistics"""
    if not rag_agent:
        st.error("RAG system not available")
        return

    stats = rag_agent.get_knowledge_base_stats()

    # Overview metrics in a clean card layout
    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 20px;">
    <h3 style="margin-top: 0;">📈 Knowledge Base Overview</h3>
    <p style="color: #666; margin-bottom: 15px; font-size: 0.9rem;">
        <strong>Note:</strong> PDFs are split into smaller text chunks for better retrieval. 
        Each chunk contains ~1500 characters with 300 character overlap for context preservation.
    </p>
    """, unsafe_allow_html=True)
    
    # Responsive metric layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f9f4ff 0%, #f3e5f5 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #7046f8;">{stats.get('document_count', 0)}</div>
            <div style="font-size: 0.9rem;">Document Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #1976d2;">{stats.get('pdf_files_available', 0)}</div>
            <div style="font-size: 0.9rem;">PDF Files</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold; color: #4caf50;">{stats.get('document_count', 0)}</div>
            <div style="font-size: 0.9rem;">Searchable Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        index_status = "Ready" if stats.get('index_exists') else "Not Found"
        status_color = "#4caf50" if stats.get('index_exists') else "#f44336"
        bg_gradient = "linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%)" if stats.get('index_exists') else "linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)"
        
        st.markdown(f"""
        <div style="background: {bg_gradient}; padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: bold; color: {status_color};">{index_status}</div>
            <div style="font-size: 0.9rem;">Index Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Document types in a visually appealing grid
    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 20px;">
        <h3 style="margin-top: 0;">📖 Document Types & Use Cases</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 15px;">
    """, unsafe_allow_html=True)
    
    doc_categories = {
        "Research Papers": "📚",
        "Technical Documentation": "🔧",
        "Legal Documents": "⚖️",
        "Corporate Policies": "🏢",
        "Training Materials": "🎓",
        "User Manuals": "📖",
        "Compliance Documents": "✅",
        "Reports & Analysis": "📊",
        "Contracts & Agreements": "📝",
        "Specifications": "🔍",
        "Academic Papers": "🎓",
        "Reference Materials": "📚"
    }

    for category, emoji in doc_categories.items():
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; display: flex; align-items: center;">
            <div style="font-size: 1.5rem; margin-right: 10px;">{emoji}</div>
            <div style="font-size: 0.9rem;"><strong>{category}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Add detailed breakdown section
    if stats.get('avg_chunks_per_pdf') and stats.get('file_details'):
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 20px;">
            <h3 style="margin-top: 0;">📊 Document Processing Details</h3>
        """, unsafe_allow_html=True)
        
        # Show processing summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Avg Chunks per PDF", f"{stats.get('avg_chunks_per_pdf', 0)}")
        with col2:
            st.metric("💾 Total Size", f"{stats.get('total_size_mb', 0)} MB")
        with col3:
            st.metric("⚙️ Chunk Size", "1500 chars")
        
        # Show individual file breakdown
        st.markdown("#### 📋 Individual PDF Files:")
        for i, file_detail in enumerate(stats.get('file_details', []), 1):
            estimated_chunks = int((file_detail['size_mb'] * 1024 * 1024) / 1500)  # Rough estimate
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid #7046f8;">
                <strong>{i}. {file_detail['name']}</strong><br>
                📄 Size: {file_detail['size_mb']} MB | 📊 Est. chunks: ~{estimated_chunks}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # System health in a card layout
    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
        <h3 style="margin-top: 0;">🔧 System Health</h3>
    """, unsafe_allow_html=True)
    
    health_col1, health_col2 = st.columns(2)

    with health_col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
            <h4 style="margin-top: 0; font-size: 1.1rem;">API Status</h4>
        """, unsafe_allow_html=True)
        
        if rag_agent:
            st.success("🟢 NVIDIA API Connected")
            st.success("🟢 Vector Database Loaded")
            st.success("🟢 LLM Model Ready")
        else:
            st.error("🔴 System Offline")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with health_col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
            <h4 style="margin-top: 0; font-size: 1.1rem;">Performance Metrics</h4>
        """, unsafe_allow_html=True)
        
        # Performance metrics with visual indicators
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #4caf50; margin-right: 10px;"></div>
            <div><strong>Embedding Dimension:</strong> 4096</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #2196f3; margin-right: 10px;"></div>
            <div><strong>Average Query Time:</strong> 2-8 seconds</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #ff9800; margin-right: 10px;"></div>
            <div><strong>Storage:</strong> Local FAISS Index</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'agentic_mode' not in st.session_state:
        st.session_state['agentic_mode'] = False
    
    # Display header
    display_header()

    # Initialize RAG agent
    rag_agent = initialize_rag_agent()

    # Display sidebar
    display_sidebar(rag_agent)

    # Create tabs for different views with custom styling
    st.markdown("""
    <style>
        /* Custom styling for the main tabs */
        .big-tabs [data-baseweb="tab-list"] {
            gap: 12px;
            margin-bottom: 0.8rem;
        }
        
        .big-tabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 0 25px;
            font-size: 1.05rem;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .big-tabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7046f8 0%, #00b4d8 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Apply custom class to tabs container
    tabs_html = """<div class="big-tabs">"""
    st.markdown(tabs_html, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Document Statistics"])

    with tab1:
        # Main content area - more responsive layout
        st.markdown("""
        <div style="display: flex; gap: 2%; flex-wrap: wrap;">
            <div style="flex: 3; min-width: 300px;">
        """, unsafe_allow_html=True)
        
        # Chat interface 
        display_chat_interface(rag_agent)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Quick tips sidebar (becomes bottom section on mobile)
        st.markdown("""
        <div style="flex: 1; min-width: 250px;">
        """, unsafe_allow_html=True)
        st.markdown("### 💡 Quick Tips")
        st.info("""
        **Sample Questions:**
        • What is the main topic of the documents?
        • Summarize the key points
        • What are the requirements mentioned?
        • How does [concept A] relate to [concept B]?
        • What are the benefits described?
        • Explain the process for [specific topic]
        
        **💡 Pro Tip:** Enable "Comprehensive Mode" for:
        • Complete lists, steps, or recommendations
        • Detailed analysis requiring multiple sources
        • Questions where you need thorough, untruncated answers
        """)



        # Advanced features with card layout
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin: 15px 0;">
            <h3 style="margin-top: 0; font-size: 1.2rem;">🛠️ Actions</h3>
        """, unsafe_allow_html=True)

        action_cols = st.columns(2)
        
        # Export chat history with enhanced button
        with action_cols[0]:
            if st.button("📥 Export Chat", use_container_width=True):
                if st.session_state.messages:
                    chat_export = []
                    for msg in st.session_state.messages:
                        chat_export.append({
                            "timestamp": datetime.now().isoformat(),
                            "role": msg["role"],
                            "content": msg["content"],
                            "sources_count": len(msg.get("sources", [])),
                            "processing_time": msg.get("processing_time", 0)
                        })

                    import json
                    export_data = json.dumps(chat_export, indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=export_data,
                        file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No chat history to export")

        # Clear chat button
        with action_cols[1]:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # Knowledge base rebuild
        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        # System information
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 15px; border-radius: 10px; margin-top: 10px;">
            <h4 style="margin-top: 0; font-size: 1rem;">This AI assistant is powered by:</h4>
            <ul style="margin-bottom: 0; padding-left: 20px;">
                <li><strong>NVIDIA</strong> embedding models</li>
                <li><strong>Meta LLaMA</strong> language model</li>
                <li><strong>FAISS</strong> vector database</li>
                <li><strong>1,869</strong> legal document chunks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat statistics with visual formatting
        if st.session_state.messages:
            st.markdown("### 📈 Session Stats")
            user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
            
            stats_cols = st.columns(2)
            
            with stats_cols[0]:
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: bold; color: #7046f8;">{len(user_messages)}</div>
                    <div style="font-size: 0.9rem;">Questions Asked</div>
                </div>
                """, unsafe_allow_html=True)
                
            with stats_cols[1]:
                st.markdown(f"""
                <div style="background-color: #f9f4ff; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: bold; color: #00b4d8;">{len(assistant_messages)}</div>
                    <div style="font-size: 0.9rem;">Responses Given</div>
                </div>
                """, unsafe_allow_html=True)

            if assistant_messages:
                avg_time = sum(m.get("processing_time", 0) for m in assistant_messages) / len(assistant_messages)
                st.markdown(f"""
                <div style="background-color: #fff8e1; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px;">
                    <div style="font-size: 1.8rem; font-weight: bold; color: #ff9800;">{avg_time:.2f}s</div>
                    <div style="font-size: 0.9rem;">Avg Response Time</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Close the responsive container divs
        st.markdown("</div></div>", unsafe_allow_html=True)

    with tab2:
        # Document statistics page with enhanced visuals
        if rag_agent:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <h2>📊 Document Statistics</h2>
                <p style="opacity: 0.7;">Overview of your knowledge base and system performance</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_document_stats(rag_agent)
        else:
            st.error("RAG system not available. Please check your configuration and try again.")

if __name__ == "__main__":
    main()
