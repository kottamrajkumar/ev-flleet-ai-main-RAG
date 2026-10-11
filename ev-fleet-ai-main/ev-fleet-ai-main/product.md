# EV Fleet AI Assistant

**Goal:** A unified chat interface that routes user queries to the appropriate specialized agent.

**Features:**
- **Intelligent Routing:** Classifies queries as Database (live rider/vehicle records), Document (Transcil offerings, models, pricing, policies), or General.
- **Database Agent (Text-to-SQL):** Connects to a MySQL database (`ev_fleet`) to answer queries about specific riders and live vehicle status.
- **Document Agent (RAG):** Uses ChromaDB and Re-ranking to answer questions from uploaded company documents (products, plans, partnerships).
- **General Agent:** Direct LLM interaction for queries outside the scope of DB or Docs.

**Tech Stack:** LangChain, LangGraph, Streamlit, ChromaDB, FlashRank (for re-ranking), MySQL, Google Gemini API.
