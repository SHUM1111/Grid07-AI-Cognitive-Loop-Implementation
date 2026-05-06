
"""
Phase 2: Autonomous Content Engine using LangGraph with structured outputs
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Mock search tool
@tool
def mock_search_xsearch(query: str) -> str:
    """
    Mock search tool that returns hardcoded news based on keywords.
    
    Args:
        query: Search query string
    
    Returns:
        Recent news headlines related to the query
    """
    query_lower = query.lower()
    
    news_data = {
        "crypto": "Bitcoin hits new all-time high at $75,000 amid regulatory ETF approvals from SEC. Ethereum layer-2 solutions see record adoption.",
        "ai": "OpenAI announces GPT-5 with 10x improved reasoning capabilities. Major tech companies pledge $10B for AI safety research.",
        "ml": "DeepMind's new algorithm solves protein folding for all known diseases. ML models now match human radiologists in cancer detection.",
        "space": "SpaceX successfully launches Starship to Mars orbit. NASA confirms water ice deposits on lunar south pole.",
        "economy": "Federal Reserve signals rate cuts in Q2 2025. Unemployment falls to 3.8% as tech sector leads job growth.",
        "privacy": "New EU regulations impose $50B fines for data breaches. Privacy-focused browsers see 200% user growth.",
        "climate": "Global carbon emissions drop 15% due to renewable adoption. New carbon capture tech removes 1M tons annually.",
        "default": "Markets show mixed signals as tech earnings beat expectations. Global economic outlook remains cautiously optimistic."
    }
    
    for keyword, news in news_data.items():
        if keyword in query_lower:
            return f"📰 Search results for '{query}': {news}"
    
    return f"📰 Search results for '{query}': {news_data['default']}"


# Define state structure
class GraphState(TypedDict):
    """State for the LangGraph workflow"""
    bot_id: str
    bot_persona: str
    topic: str
    search_query: str
    search_results: str
    post_content: str


# Define structured output model
class BotPost(BaseModel):
    """Structured output for bot posts"""
    bot_id: str = Field(description="The ID of the bot creating the post")
    topic: str = Field(description="The main topic of the post")
    post_content: str = Field(description="The 280-character opinionated post")


class ContentEngine:
    """LangGraph-based autonomous content engine"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # Initialize LLM (supports OpenAI, Groq, or local Ollama)
        self.llm = self._setup_llm(model_name)
        self.llm_with_structure = self.llm.with_structured_output(BotPost)
        self.graph = self._build_graph()
    
    def _setup_llm(self, model_name: str):
        """Setup LLM - supports OpenAI, Groq, or fallback to mock for demo"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key and model_name.startswith("gpt"):
            return ChatOpenAI(model=model_name, temperature=0.7)
        else:
            # Fallback mock for demonstration
            print("⚠️  No API key found. Using Mock LLM for demonstration.")
            return MockLLM()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("decide_search", self.decide_search_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("draft_post", self.draft_post_node)
        
        # Add edges
        workflow.set_entry_point("decide_search")
        workflow.add_edge("decide_search", "web_search")
        workflow.add_edge("web_search", "draft_post")
        workflow.add_edge("draft_post", END)
        
        return workflow.compile()
    
    def decide_search_node(self, state: GraphState) -> GraphState:
        """Node 1: LLM decides what topic to post about and formats search query"""
        print(f"\n🤖 Bot {state['bot_id']} deciding on topic...")
        
        prompt = f"""You are {state['bot_persona']}

Decide what interesting, controversial, or relevant topic you want to post about TODAY.
Consider current trends, your personality, and what would engage your audience.

Return a SEARCH QUERY that would find real-world news about this topic.
The query should be specific (2-5 words) for news search.

Example: "AI replacing jobs", "crypto regulations", "climate tech innovation"

Your response MUST be just the search query, nothing else."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a social media bot deciding what to post."),
            HumanMessage(content=prompt)
        ])
        
        search_query = response.content.strip()
        topic = search_query.split()[0]  # Simplified topic extraction
        
        print(f"  📍 Decided topic: {topic}")
        print(f"  🔍 Search query: {search_query}")
        
        return {
            **state,
            "topic": topic,
            "search_query": search_query
        }
    
    def web_search_node(self, state: GraphState) -> GraphState:
        """Node 2: Execute search to get real-world context"""
        print(f"\n🌐 Searching for: {state['search_query']}")
        
        search_results = mock_search_xsearch.invoke({"query": state['search_query']})
        print(f"  📊 Results: {search_results[:150]}...")
        
        return {
            **state,
            "search_results": search_results
        }
    
    def draft_post_node(self, state: GraphState) -> GraphState:
        """Node 3: Generate opinionated post with structured output"""
        print(f"\n✍️  Drafting post for Bot {state['bot_id']}...")
        
        prompt = f"""You are {state['bot_persona']}

CONTEXT FROM SEARCH:
{state['search_results']}

TASK: Create a highly opinionated, engaging social media post (MAX 280 CHARACTERS) about {state['topic']}.

Requirements:
- Express YOUR personality strongly (agree/disagree/ridicule/praise based on your persona)
- Reference the search context
- Be controversial or provocative to drive engagement
- NO hashtags, NO emojis (except maybe one if appropriate)
- Must be under 280 characters"""

        try:
            # Attempt structured output
            result = self.llm_with_structure.invoke([
                SystemMessage(content="You are a social media bot. Generate structured posts."),
                HumanMessage(content=prompt)
            ])
            
            post_content = result.post_content
            bot_id = result.bot_id
            
        except:
            # Fallback to regular invocation for mock LLM
            response = self.llm.invoke([
                SystemMessage(content="Generate a social media post. Return ONLY the post text, nothing else."),
                HumanMessage(content=prompt)
            ])
            post_content = response.content[:280]
            bot_id = state['bot_id']
        
        # Ensure post fits 280 characters
        if len(post_content) > 280:
            post_content = post_content[:277] + "..."
        
        print(f"  📝 Generated ({len(post_content)} chars): \"{post_content}\"")
        
        # Return as JSON-strict structure
        return {
            **state,
            "post_content": post_content
        }
    
    def generate_post(self, bot_id: str, bot_persona: str) -> Dict[str, Any]:
        """Generate a post for the given bot"""
        print("\n" + "=" * 60)
        print(f"PHASE 2: Generating Post for {bot_id}")
        print("=" * 60)
        
        initial_state = {
            "bot_id": bot_id,
            "bot_persona": bot_persona,
            "topic": "",
            "search_query": "",
            "search_results": "",
            "post_content": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        
        # Return strict JSON output
        output = {
            "bot_id": final_state["bot_id"],
            "topic": final_state["topic"],
            "post_content": final_state["post_content"]
        }
        
        print("\n📦 Final JSON Output:")
        print(json.dumps(output, indent=2))
        
        return output


class MockLLM:
    """Mock LLM for demonstration when no API key is available"""
    
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        
        # Extract user query
        user_msg = messages[-1].content if messages else ""
        
        class Response:
            content = ""
        
        if "search query" in user_msg.lower():
            if "tech" in user_msg.lower() or "ai" in user_msg.lower():
                Response.content = "AI replacing software engineers"
            elif "crypto" in user_msg.lower():
                Response.content = "Bitcoin price surge"
            else:
                Response.content = "tech market trends"
        
        elif "post" in user_msg.lower() or "280" in user_msg.lower():
            if "Bot A" in str(messages):
                Response.content = "AI isn't replacing devs, it's supercharging them! The future is now 🚀"
            elif "Bot B" in str(messages):
                Response.content = "More AI hype from tech bros trying to justify their billion-dollar grift."
            else:
                Response.content = "Markets are bullish on AI. My portfolio is up 40% this quarter alone."
        
        else:
            Response.content = "tech innovation"
        
        return Response


def run_phase2_demo():
    """Demo Phase 2 functionality"""
    personas = {
        "Bot A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration.",
        "Bot B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI and billionaires.",
        "Bot C": "I strictly care about markets, interest rates, trading algorithms, and making money."
    }
    
    engine = ContentEngine()
    results = []
    
    for bot_id, persona in personas.items():
        result = engine.generate_post(bot_id, persona)
        results.append(result)
        print()
    
    return results


if __name__ == "__main__":
    run_phase2_demo()
