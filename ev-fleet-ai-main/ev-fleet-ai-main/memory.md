# Project Structure

```
/ev-fleet-ai
│-- app.py                 # Streamlit UI entry point
│-- graph.py               # LangGraph state and workflow compilation
│-- agents.py              # Node functions (router, sql, rag, general)
│-- db_setup.py            # Script to initialize MySQL schema & dummy data
│-- document_loader.py     # ChromaDB ingestion logic with FlashRank
│-- requirements.txt       # Dependencies
│-- .env                   # Environment variables
```
