from openai import OpenAI
import streamlit as st
import pandas as pd

st.title("Evidence Explorer")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Load dataset from GitHub repo
@st.cache_data
def load_data():
    return pd.read_csv(
        "data/branchout_dataset.csv"
    )

dataset = load_data()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input(
    "Ask about neurological research..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Search your dataset
    matches = dataset[
        dataset.apply(
            lambda row:
            row.astype(str)
            .str.contains(
                prompt,
                case=False
            ).any(),
            axis=1
        )
    ]


    dataset_context = (
        matches.head(10)
        .to_string()
    )


    system_prompt = f"""
You are Evidence Explorer.

Use two sources:

1. Branch Out dataset:
{dataset_context}

2. Consensus MCP:
Use Consensus for scientific evidence.

Rules:
- Identify if information came from Branch Out data.
- Identify broader scientific evidence separately.
- Explain evidence strength.
"""


    with st.chat_message("assistant"):

        response = client.responses.create(

            model="gpt-4.1-mini",

            instructions=system_prompt,

            input=prompt
        )

        answer = response.output_text

        st.write(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
