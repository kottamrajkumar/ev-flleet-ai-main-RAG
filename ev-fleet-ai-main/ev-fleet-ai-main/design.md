# UI & System Design

- **UI Framework:** Streamlit.
- **Layout:**
  - Main panel: Chat interface using `st.chat_message` to maintain conversation history.
  - Sidebar: File uploader to ingest `.txt` or `.pdf` company documents into ChromaDB. MySQL credentials and the Gemini API key are loaded from `.env`.
- **UX Flow:** User types a query -> UI displays a spinner ("Thinking...") -> Router determines path -> Specialized agent processes -> Final response is streamed/displayed.
