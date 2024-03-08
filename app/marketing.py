import requests
import streamlit as st

st.title("营销文案 Bot v0.1.0")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False


for message in st.session_state.messages:
    role = message["from"]
    role = {"gpt": "assistant", "human": "user"}[role]
    with st.chat_message(role):
        st.markdown(message["value"])


def post():
    json = st.session_state.messages
    return requests.post("http://127.0.0.1:5000/chat", json=json, stream=True)


def disable_input():
    st.session_state["disabled"] = True


def enable_input():
    st.session_state["disabled"] = False


prompt = st.chat_input("输入关键词", on_submit=disable_input)

if prompt:
#    st.session_state.messages.append({"from": "human", "value": prompt})
	# single turn only
    st.session_state.messages = [{"from": "human", "value": prompt}]
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in post():
            full_response += response.decode("utf-8")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    enable_input()
    st.session_state.messages= [{"from": "human", "value": prompt}, {"from": "gpt", "value": full_response}]
