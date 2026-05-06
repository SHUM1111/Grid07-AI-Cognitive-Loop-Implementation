
"""
Grid07 Assignment - Complete Execution
Runs all three phases sequentially
"""

from phase1_router import run_phase1_demo
from phase2_langgraph import run_phase2_demo
from phase3_rag_defense import run_phase3_demo

def main():
    print("\n" + "=" * 60)
    print("GRID07 AI COGNITIVE LOOP - COMPLETE ASSIGNMENT")
    print("=" * 60)
    
    # Phase 1
    print("\n" + "🎯 PHASE 1: Vector Persona Matching")
    print("-" * 40)
    phase1_results = run_phase1_demo()
    
    # Phase 2
    print("\n\n" + "🤖 PHASE 2: LangGraph Content Engine")
    print("-" * 40)
    phase2_results = run_phase2_demo()
    
    # Phase 3
    print("\n\n" + "🛡️ PHASE 3: RAG Combat Engine")
    print("-" * 40)
    phase3_results = run_phase3_demo()
    
    print("\n" + "=" * 60)
    print("✅ ASSIGNMENT COMPLETE")
    print("=" * 60)
    
    return {
        "phase1": phase1_results,
        "phase2": phase2_results,
        "phase3": phase3_results
    }


if __name__ == "__main__":
    main()
