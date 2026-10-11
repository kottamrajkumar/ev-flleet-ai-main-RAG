# Coding Rules

- Use standard Python typing (PEP 484).
- Use `langchain-google-genai` for integrating Gemini.
- **Models:** For the LLM, use `"gemini-1.5-flash"` (maps from the requested latest stable flash model). Assign it to a variable `GEMINI_MODEL="gemini-1.5-flash"`. For embeddings, strictly use `"models/gemini-embedding-001"`.
- Error handling: Gracefully catch MySQL connection errors and empty Vector Store queries.
