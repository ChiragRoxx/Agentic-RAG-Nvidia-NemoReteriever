"""
Test script for Agentic RAG integration
Tests both standard RAG and Agentic AI modes
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_agent import RAGAgent
from src.nvidia_embeddings import NVIDIAEmbeddings

# Try to import agentic capabilities
try:
    from src.agentic_rag import AgenticRAGAgent, AGENTIC_AI_AVAILABLE
    print("✅ Agentic AI capabilities available")
except ImportError as e:
    print(f"❌ Agentic AI not available: {e}")
    AGENTIC_AI_AVAILABLE = False
    AgenticRAGAgent = None


def test_nvidia_embeddings():
    """Test NVIDIA embeddings connection"""
    print("\n🔧 Testing NVIDIA Embeddings...")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return False
    
    try:
        embeddings = NVIDIAEmbeddings(api_key)
        if embeddings.test_connection():
            print("✅ NVIDIA Embeddings connection successful")
            return True
        else:
            print("❌ NVIDIA Embeddings connection failed")
            return False
    except Exception as e:
        print(f"❌ NVIDIA Embeddings error: {str(e)}")
        return False


def test_standard_rag():
    """Test standard RAG functionality"""
    print("\n📚 Testing Standard RAG Agent...")
    
    try:
        api_key = os.getenv("NVIDIA_API_KEY")
        docs_folder = "Data/Docs"
        
        # Initialize RAG agent
        rag_agent = RAGAgent(docs_folder, api_key)
        
        # Test knowledge base setup
        if rag_agent.setup_knowledge_base():
            print("✅ Knowledge base setup successful")
        else:
            print("⚠️  Knowledge base setup failed (may be due to no PDFs)")
        
        # Test basic question
        test_question = "What is the main topic of the documents?"
        response = rag_agent.ask_question(test_question)
        
        print(f"✅ Standard RAG test successful")
        print(f"   Question: {test_question}")
        print(f"   Answer length: {len(response.answer)} characters")
        print(f"   Sources found: {len(response.source_documents)}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        
        return True, rag_agent
        
    except Exception as e:
        print(f"❌ Standard RAG test failed: {str(e)}")
        return False, None


def test_agentic_rag(rag_agent):
    """Test agentic RAG functionality"""
    print("\n🧠 Testing Agentic RAG Agent...")
    
    if not AGENTIC_AI_AVAILABLE:
        print("❌ Agentic AI not available - skipping test")
        return False
    
    try:
        api_key = os.getenv("NVIDIA_API_KEY")
        
        # Initialize agentic agent
        agentic_agent = AgenticRAGAgent(
            rag_agent=rag_agent,
            api_key=api_key,
            enable_web_search=True,
            enable_analysis=True
        )
        
        # Test agent status
        status = agentic_agent.get_agent_status()
        print(f"   Agent Status: {status}")
        
        # Test capabilities
        capabilities = agentic_agent.test_agentic_capabilities()
        print(f"   Capabilities Test: {capabilities}")
        
        # Test simple question
        test_question = "What are the key topics in the documents?"
        response = agentic_agent.ask_question(test_question)
        
        print(f"✅ Agentic RAG test successful")
        print(f"   Question: {test_question}")
        print(f"   Answer length: {len(response.answer)} characters")
        print(f"   Is agentic: {getattr(response, 'is_agentic', False)}")
        print(f"   Tools used: {getattr(response, 'tools_used', [])}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Agentic RAG test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🚀 RAG System Integration Test")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Test results
    results = {}
    
    # Test NVIDIA embeddings
    results['embeddings'] = test_nvidia_embeddings()
    
    # Test standard RAG
    results['standard_rag'], rag_agent = test_standard_rag()
    
    # Test agentic RAG if available
    if results['standard_rag'] and rag_agent:
        results['agentic_rag'] = test_agentic_rag(rag_agent)
    else:
        results['agentic_rag'] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Overall status
    if all(results.values()):
        print("\n🎉 All tests passed! System is fully functional.")
    elif results['embeddings'] and results['standard_rag']:
        print("\n✅ Core system functional. Agentic AI mode may need setup.")
    else:
        print("\n❌ Some core tests failed. Check your configuration.")
    
    # Next steps
    print("\n📋 Next Steps:")
    if not results['embeddings']:
        print("   1. Check your NVIDIA_API_KEY in .env file")
        print("   2. Verify your internet connection")
    
    if not results['standard_rag']:
        print("   1. Ensure you have PDF files in Data/Docs folder")
        print("   2. Check that all dependencies are installed")
    
    if not results['agentic_rag'] and AGENTIC_AI_AVAILABLE:
        print("   1. Agentic agent initialization failed - check logs")
    elif not AGENTIC_AI_AVAILABLE:
        print("   1. Install agentic AI: pip install aiqtoolkit[langchain]")
        print("   2. Optional: pip install duckduckgo-search")


if __name__ == "__main__":
    main()
