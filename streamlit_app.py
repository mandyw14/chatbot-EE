from openai import OpenAI
import streamlit as st
import pandas as pd
import re

st.title("Dataset Chatbot")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DATASET_PATH = "Dimensions-Publication.csv"


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # Create clickable DOI links if the dataset has a DOI column
    if "DOI" in df.columns:
        df["Article Link"] = df["DOI"].apply(
            lambda doi: f"https://doi.org/{doi}"
            if pd.notna(doi) and str(doi).strip() != ""
            else ""
        )

    return df


def search_dataset(query, df, max_results=12):
    query_terms = re.findall(r"\w+", query.lower())

    if not query_terms:
        return pd.DataFrame()

    searchable_df = df.fillna("").astype(str)

    def row_score(row):
        text = " ".join(row.values).lower()
        return sum(1 for term in query_terms if term in text)

    scored = searchable_df.copy()
    scored["_score"] = searchable_df.apply(row_score, axis=1)

    results = scored[scored["_score"] > 0]
    results = results.sort_values("_score", ascending=False)

    return results.drop(columns=["_score"]).head(max_results)


df = load_dataset()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about the dataset"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    results = search_dataset(prompt, df)

    if results.empty:
        dataset_context = "NO RESULTS FOUND IN THE DATASET."
        result_note = "The dataset search returned 0 results."
    else:
        dataset_context = results.to_markdown(index=False)
        result_note = f"The dataset search returned {len(results)} result(s)."

system_prompt = f"""
You are a chatbot that answers questions using ONLY the provided dataset search results for scientific claims.

Core rules:
- Do not use papers that are not included in the dataset search results.
- Do not invent authors, titles, journals, dates, DOIs, findings, links, or conclusions.
- Scientific claims must be supported by the dataset search results.
- If the search returns few results, say so clearly.
- If the dataset does not contain enough information, say that clearly.
- Never create article links. Only use links that exist in the dataset search results.

Interactive explanation rules:
- You may define general scientific or technical terms in plain language.
- When defining a term, clearly label it as "General definition" rather than as a dataset finding.
- Do not cite or imply outside papers when giving definitions.
- Keep definitions brief and accessible.
- You may ask one helpful follow-up question if the user's query is broad or unclear.
- You may suggest related searches based only on terms, authors, topics, conditions, or interventions visible in the dataset results.

When discussing papers:
- Include the publication title.
- Include the year if available.
- Include the authors if available.
- Include a clickable article link whenever an article link, DOI link, PubMed link, URL, or Dimensions link exists in the dataset.
- Format article links using Markdown: [View article](URL)

Suggested response format:

### Summary
Briefly answer the user's question using only the dataset results.

### General definition, if helpful
Define any technical term the user may need to understand.

### Relevant papers from the dataset
1. Paper title  
   Authors:  
   Year:  
   Dataset-supported finding or relevance:  
   Link:  

### Search limitations
State if the search returned few results or if the dataset is limited.

### You could also ask
Suggest 1–3 related searches using only terms found in the dataset results.

Search result note:
{result_note}

Dataset search results:
{dataset_context}
"""

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                *[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            ],
            stream=True,
        )

        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
