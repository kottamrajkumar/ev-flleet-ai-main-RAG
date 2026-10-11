"""Streamlit chat UI for the EV Fleet AI Assistant."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agents import AppConfig, _plain_text
from document_loader import ingest_uploaded_bytes
from graph import GraphState, build_graph

load_dotenv()

st.set_page_config(page_title="EV Fleet AI Assistant", page_icon="⚡", layout="wide")


def _config_from_env() -> AppConfig:
    return {
        "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
        "mysql_host": os.getenv("MYSQL_HOST", "localhost"),
        "mysql_port": os.getenv("MYSQL_PORT", "3306"),
        "mysql_user": os.getenv("MYSQL_USER", "root"),
        "mysql_password": os.getenv("MYSQL_PASSWORD", ""),
        "mysql_database": os.getenv("MYSQL_DATABASE", "ev_fleet"),
    }


def _get_graph(config: AppConfig) -> Any:
    if "_compiled_graph" not in st.session_state:
        st.session_state._compiled_graph = build_graph(config)
    return st.session_state._compiled_graph


def main() -> None:
    st.title("EV Fleet AI Assistant")
    st.caption("Routes questions to database, document, or general agents.")

    config = _config_from_env()

    st.sidebar.header("Company documents")
    uploaded = st.sidebar.file_uploader(
        "Upload .txt or .pdf",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded and config["google_api_key"]:
        for file in uploaded:
            key = f"ingested_{file.name}_{file.size}"
            if key not in st.session_state:
                with st.sidebar.status(f"Ingesting {file.name}..."):
                    chunks = ingest_uploaded_bytes(
                        file.name,
                        file.getvalue(),
                        config["google_api_key"],
                    )
                st.session_state[key] = chunks
                st.sidebar.success(f"Ingested {file.name} ({chunks} chunks).")
    elif uploaded and not config["google_api_key"]:
        st.sidebar.warning("Set GOOGLE_API_KEY in the .env file before ingesting documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("category"):
                st.caption(f"Routed to: {message['category']}")

    prompt = st.chat_input("Ask about riders, vehicles, or company documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not config["google_api_key"]:
        answer = "Set GOOGLE_API_KEY in the .env file to start chatting."
        category = ""
    else:
        with st.spinner("Thinking..."):
            graph = _get_graph(config)
            result: GraphState = graph.invoke({"question": prompt})
        answer = _plain_text(result.get("final_answer")) or "No response was generated."
        category = result.get("category", "")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "category": category}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        if category:
            st.caption(f"Routed to: {category}")


if __name__ == "__main__":
    main()
