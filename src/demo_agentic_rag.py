"""
Demo Agentic RAG Implementation
A simplified agentic AI demonstration that works without the full NVIDIA Agent Intelligence Toolkit
"""

import os
import logging
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from langchain.schema import Document
    from langchain.tools import Tool as LangChainTool
    from langchain.agents import initialize_agent, AgentType
    from langchain_community.tools import DuckDuckGoSearchRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Agentic features will be limited.")

from .rag_agent import RAGAgent, RAGResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Demo implementation - works without full AIQ toolkit
AGENTIC_AI_AVAILABLE = True  # Enable demo mode


@dataclass
class AgenticResponse:
    """Enhanced response from Demo Agentic RAG agent"""
    answer: str
    source_documents: List[Document] = None
    confidence_scores: Optional[List[float]] = None
    query: str = ""
    processing_time: float = 0.0
    agent_reasoning: str = ""
    tools_used: List[str] = None
    intermediate_steps: List[Dict[str, Any]] = None
    is_agentic: bool = True


class DemoRAGTool:
    """RAG as a tool for the demo agentic system"""
    
    def __init__(self, rag_agent: RAGAgent):
        self.rag_agent = rag_agent
        
    def search_documents(self, query: str) -> str:
        """Search the knowledge base for relevant information"""
        try:
            response = self.rag_agent.ask_question(query)
            
            # Format the response for the agent
            result = f"Knowledge Base Answer: {response.answer}\n\n"
            
            if response.source_documents:
                result += "Sources Found:\n"
                for i, doc in enumerate(response.source_documents[:3], 1):
                    source_file = doc.metadata.get("source_file", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    result += f"  {i}. {source_file} (Page {page}): {preview}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"RAG tool error: {str(e)}")
            return f"Error searching knowledge base: {str(e)}"


class DemoWebSearchTool:
    """Demo web search tool"""
    
    def __init__(self):
        try:
            if LANGCHAIN_AVAILABLE:
                self.search = DuckDuckGoSearchRun()
            else:
                self.search = None
        except Exception as e:
            logger.warning(f"Web search tool not available: {str(e)}")
            self.search = None
    
    def search_web(self, query: str) -> str:
        """Search the web for additional information"""
        if not self.search:
            return "Web search is not available. Install duckduckgo-search for web search capabilities."
        
        try:
            # Limit search results to avoid overwhelming the agent
            results = self.search.run(query)
            # Truncate results if too long
            if len(results) > 1000:
                results = results[:1000] + "..."
            return f"Web search results for '{query}':\n{results}"
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Error searching web: {str(e)}"


class DemoAnalysisTool:
    """Demo tool for analytical reasoning"""
    
    def analyze_information(self, information: str) -> str:
        """Analyze and synthesize information"""
        try:
            # Simple analysis framework
            analysis = f"""
Information Analysis for: {information[:100]}{'...' if len(information) > 100 else ''}

Key Patterns Identified:
• Length: {len(information)} characters
• Contains technical terms: {'Yes' if any(term in information.lower() for term in ['technology', 'system', 'process', 'method']) else 'No'}
• Mentions numbers: {'Yes' if any(char.isdigit() for char in information) else 'No'}
• Information density: {'High' if len(information) > 500 else 'Medium' if len(information) > 200 else 'Low'}

Analysis Summary:
This information appears to be {'technical documentation' if 'system' in information.lower() else 'general content'} 
with {'detailed' if len(information) > 300 else 'basic'} level of detail.

Recommendation: Use this information to {'support technical decisions' if 'system' in information.lower() else 'provide general guidance'}.
"""
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis tool error: {str(e)}")
            return f"Error analyzing information: {str(e)}"


class DemoAgenticRAGAgent:
    """Demo Enhanced RAG Agent with simplified autonomous reasoning capabilities"""
    
    def __init__(
        self,
        rag_agent: RAGAgent,
        api_key: str,
        enable_web_search: bool = True,
        enable_analysis: bool = True
    ):
        """
        Initialize Demo Agentic RAG Agent
        
        Args:
            rag_agent: Base RAG agent
            api_key: NVIDIA API key
            enable_web_search: Whether to enable web search capability
            enable_analysis: Whether to enable analysis capabilities
        """
        self.rag_agent = rag_agent
        self.api_key = api_key
        self.enable_web_search = enable_web_search
        self.enable_analysis = enable_analysis
        
        # Initialize tools
        self.rag_tool = DemoRAGTool(rag_agent)
        self.web_search_tool = DemoWebSearchTool() if enable_web_search else None
        self.analysis_tool = DemoAnalysisTool() if enable_analysis else None
        
        # Simple reasoning patterns
        self.reasoning_patterns = {
            'comparison': ['compare', 'versus', 'vs', 'difference', 'similarities'],
            'analysis': ['analyze', 'analysis', 'evaluate', 'assessment'],
            'summary': ['summarize', 'summary', 'overview', 'key points'],
            'current': ['latest', 'current', 'recent', 'today', 'now', '2024', '2025'],
            'detailed': ['detailed', 'comprehensive', 'complete', 'thorough']
        }
        
        logger.info("Demo Agentic RAG Agent initialized successfully!")
    
    def _analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the question to determine intent and required tools"""
        question_lower = question.lower()
        
        intent = {
            'requires_web_search': False,
            'requires_analysis': False,
            'requires_comparison': False,
            'complexity': 'simple',
            'tools_needed': ['knowledge_base']
        }
        
        # Check for patterns that indicate need for external search
        current_indicators = any(pattern in question_lower for pattern in self.reasoning_patterns['current'])
        if current_indicators:
            intent['requires_web_search'] = True
            intent['tools_needed'].append('web_search')
        
        # Check for analysis needs
        analysis_indicators = any(pattern in question_lower for pattern in self.reasoning_patterns['analysis'])
        if analysis_indicators:
            intent['requires_analysis'] = True
            intent['tools_needed'].append('analysis')
        
        # Check for comparison needs
        comparison_indicators = any(pattern in question_lower for pattern in self.reasoning_patterns['comparison'])
        if comparison_indicators:
            intent['requires_comparison'] = True
            intent['complexity'] = 'complex'
        
        # Determine complexity
        if len(question_lower.split()) > 15 or any(word in question_lower for word in ['and', 'also', 'additionally']):
            intent['complexity'] = 'complex'
        
        return intent
    
    def _execute_reasoning_chain(self, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain of reasoning steps based on the question intent"""
        reasoning_steps = []
        collected_information = []
        tools_used = []
        
        # Step 1: Always search knowledge base first
        reasoning_steps.append("1. Searching internal knowledge base for relevant information")
        kb_result = self.rag_tool.search_documents(question)
        collected_information.append(("knowledge_base", kb_result))
        tools_used.append("knowledge_base_search")
        
        # Step 2: Web search if needed
        if intent['requires_web_search'] and self.web_search_tool:
            reasoning_steps.append("2. Searching web for current/additional information")
            web_result = self.web_search_tool.search_web(question)
            collected_information.append(("web_search", web_result))
            tools_used.append("web_search")
        
        # Step 3: Analysis if needed
        if intent['requires_analysis'] and self.analysis_tool:
            reasoning_steps.append("3. Analyzing collected information for insights")
            all_info = "\n".join([info[1] for info in collected_information])
            analysis_result = self.analysis_tool.analyze_information(all_info)
            collected_information.append(("analysis", analysis_result))
            tools_used.append("information_analysis")
        
        # Step 4: Synthesis
        reasoning_steps.append("4. Synthesizing information to provide comprehensive answer")
        
        return {
            'reasoning_steps': reasoning_steps,
            'collected_information': collected_information,
            'tools_used': tools_used
        }
    
    def _synthesize_final_answer(self, question: str, reasoning_result: Dict[str, Any]) -> str:
        """Synthesize a final answer from all collected information"""
        
        # Extract information from different sources
        kb_info = ""
        web_info = ""
        analysis_info = ""
        
        for source, info in reasoning_result['collected_information']:
            if source == "knowledge_base":
                kb_info = info
            elif source == "web_search":
                web_info = info
            elif source == "analysis":
                analysis_info = info
        
        # Create a synthesized answer
        answer_parts = []
        
        # Start with knowledge base information
        if "Knowledge Base Answer:" in kb_info:
            kb_answer = kb_info.split("Knowledge Base Answer:")[1].split("Sources Found:")[0].strip()
            answer_parts.append(f"Based on the available documents: {kb_answer}")
        
        # Add web information if available
        if web_info and "Web search results" in web_info:
            answer_parts.append(f"\nAdditional current information: {web_info[:300]}...")
        
        # Add analysis if available
        if analysis_info and "Analysis Summary:" in analysis_info:
            analysis_summary = analysis_info.split("Analysis Summary:")[1].split("Recommendation:")[0].strip()
            answer_parts.append(f"\nAnalysis: {analysis_summary}")
        
        # Combine all parts
        if answer_parts:
            final_answer = "\n".join(answer_parts)
        else:
            final_answer = "I was able to search various sources, but couldn't find specific information to answer your question."
        
        return final_answer
    
    def ask_question(self, question: str) -> AgenticResponse:
        """
        Ask a question using demo agentic AI capabilities
        
        Args:
            question: The question to ask
            
        Returns:
            AgenticResponse with comprehensive answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze question intent
            intent = self._analyze_question_intent(question)
            
            # Step 2: Execute reasoning chain
            reasoning_result = self._execute_reasoning_chain(question, intent)
            
            # Step 3: Synthesize final answer
            final_answer = self._synthesize_final_answer(question, reasoning_result)
            
            # Step 4: Create reasoning explanation
            reasoning_explanation = "Demo Agentic AI Reasoning Process:\n"
            for step in reasoning_result['reasoning_steps']:
                reasoning_explanation += f"  {step}\n"
            
            reasoning_explanation += f"\nIntent Analysis: {intent}\n"
            reasoning_explanation += f"Tools Used: {', '.join(reasoning_result['tools_used'])}"
            
            # Step 5: Extract source documents from knowledge base search
            source_docs = []
            if hasattr(self.rag_tool.rag_agent, 'last_response') and self.rag_tool.rag_agent.last_response:
                source_docs = self.rag_tool.rag_agent.last_response.source_documents or []
            
            processing_time = time.time() - start_time
            
            return AgenticResponse(
                answer=final_answer,
                source_documents=source_docs,
                query=question,
                processing_time=processing_time,
                agent_reasoning=reasoning_explanation,
                tools_used=reasoning_result['tools_used'],
                intermediate_steps=[{
                    'step': i+1,
                    'action': step,
                    'result': 'completed'
                } for i, step in enumerate(reasoning_result['reasoning_steps'])],
                is_agentic=True
            )
                
        except Exception as e:
            logger.error(f"Demo agentic agent error: {str(e)}")
            
            # Fallback to basic RAG
            try:
                basic_response = self.rag_agent.ask_question(question)
                return AgenticResponse(
                    answer=f"Demo agentic mode encountered an error. Fallback response: {basic_response.answer}",
                    source_documents=basic_response.source_documents,
                    confidence_scores=basic_response.confidence_scores,
                    query=question,
                    processing_time=time.time() - start_time,
                    agent_reasoning=f"Error in demo agentic mode: {str(e)}. Fell back to basic RAG.",
                    tools_used=["basic_rag_fallback"],
                    intermediate_steps=[],
                    is_agentic=False
                )
            except Exception as fallback_error:
                return AgenticResponse(
                    answer=f"I apologize, but I encountered errors in both agentic and basic modes. Error: {str(e)}",
                    source_documents=[],
                    query=question,
                    processing_time=time.time() - start_time,
                    agent_reasoning=f"Multiple errors: {str(e)}, {str(fallback_error)}",
                    tools_used=[],
                    intermediate_steps=[],
                    is_agentic=False
                )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about the demo agentic capabilities"""
        return {
            "agentic_ai_available": True,
            "demo_mode": True,
            "using_full_aiq_toolkit": False,
            "agent_initialized": True,
            "tools_available": 3,
            "web_search_enabled": self.enable_web_search,
            "analysis_enabled": self.enable_analysis,
            "tools": ["knowledge_base_search", "web_search", "information_analysis"],
            "reasoning_patterns": len(self.reasoning_patterns)
        }
    
    def test_agentic_capabilities(self) -> Dict[str, bool]:
        """Test various demo agentic capabilities"""
        results = {}
        
        try:
            # Test RAG tool
            test_rag = self.rag_tool.search_documents("test query")
            results["rag_tool_works"] = "Error" not in test_rag
            
            # Test web search if enabled
            if self.web_search_tool and self.enable_web_search:
                test_web = self.web_search_tool.search_web("hydrogen energy")
                results["web_search_works"] = "not available" not in test_web.lower()
            else:
                results["web_search_works"] = False
            
            # Test analysis tool if enabled
            if self.analysis_tool and self.enable_analysis:
                test_analysis = self.analysis_tool.analyze_information("test data")
                results["analysis_works"] = "Error" not in test_analysis
            else:
                results["analysis_works"] = False
            
            # Test intent analysis
            intent = self._analyze_question_intent("What are the latest trends in hydrogen technology?")
            results["intent_analysis_works"] = intent is not None
            
            results["demo_mode_active"] = True
                
        except Exception as e:
            logger.error(f"Error testing demo agentic capabilities: {str(e)}")
            results["error"] = str(e)
        
        return results


# Alias for compatibility with the full implementation
AgenticRAGAgent = DemoAgenticRAGAgent
