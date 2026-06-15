from openai import OpenAI
import streamlit as st

st.title("Evidence Explorer")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input(
    "Ask about brain health evidence..."
):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.write(prompt)


    instructions = """
You are Evidence Explorer.

You help users understand neurological research.

Always:
1. Search the uploaded Branch Out dataset first.
2. Use Consensus MCP for scientific evidence.
3. Clearly separate:
   - Branch Out funded research
   - broader scientific evidence
4. Summarize evidence strength.
5. Avoid medical recommendations.
"""

    response = client.responses.create(

        model="gpt-4.1-mini",

        instructions=instructions,

        input=prompt,

        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [
                    VECTOR_STORE_ID
                ]
            }
        ]

    )

    answer = response.output_text


    with st.chat_message("assistant"):
        st.write(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
