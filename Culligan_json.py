import bs4
import os
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from chainlit.input_widget import Select, Switch, Slider
import chainlit as cl
import json
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

llm = None  # Placeholder for the LLM instance

def update_llm(settings):
    global llm
    model = settings.get("Model", "gpt-3.5-turbo")
    temperature = settings.get("Temperature", 1.0)
    streaming = settings.get("Streaming", True)

    llm = ChatOpenAI(model=model, temperature=temperature, streaming=streaming)

# Initialize the LLM with default settings before using it
default_settings = {
    "Model": "gpt-3.5-turbo",
    "Temperature": 0.2,
    "Streaming": True,
}
update_llm(default_settings)

# Load JSON data from file
def load_json_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

json_data = load_json_data("product_url.json")

# Extract text from JSON data
def extract_text_from_json(json_data):
    texts = []
    for product in json_data.get("products", []):
        product_info = f"""
        Product Name: {product.get('product_name')}
        Description: {product.get('description', '')}
        Price: {product.get('price')}
        Additional Info: {product.get('additional_info', '')}
        Technical Specifications: {product.get('technical_specifications', '')}
        Useful Information: {product.get('useful_information', '')}
        """
        texts.append(product_info)
    return texts

file = "culligan.txt"
with open(file, "r") as f:
    culligan = f.read()

data = extract_text_from_json(json_data) + [culligan]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = [split for text in data for split in text_splitter.split_text(text)]

# Construct retriever
vectorstore = Chroma.from_texts(splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

guidelines_text = """
1. Greet and assess needs
2. suggest products
3. offer details or quotes
4. verify compatibility, make sure it is compatible
5. inquire about additional services
6. and finalize the quote.
7. Never handle more than one step at a time
"""

# Answer question
system_prompt = (
    "You are a friendly and knowledgeable assistant for Culligan, specializing in water filters, sinks, and related products. "
    "Your name is Tony. "
    "Use the following pieces of retrieved context to answer the question and help the customer find the ideal product for their needs with the end goal of providing a quote. "
    "Here are some guidelines for our conversation, follow this order: " + guidelines_text + " "
    "NEVER ask more than one question at a time."
    "Cover the following topics: "
    "1. What type of product are you looking for (e.g., water filter, sink, faucet)? "
    "2. Do you have any specific requirements or preferences (e.g., size, style, features)?"
    "3. If interested, we can give you more details on the product or proceed with a quote."
    "4. Perfect! To verify compatibility specify your current system."
    "5. Do you need professional installation services? "
    "Once you have the necessary information, provide a detailed quote including product costs and installation fees, if applicable. "
    "If customer is undecided or unsatisfied master sales techniques by understanding"
    "and solving customer problems, providing unique insights, "
    "building strong relationships, leveraging psychological principles of persuasion, "
    "and confidently guiding conversations to close deals effectively"
    "If you don't know the answer or price is missing, say that you don't know, and advise customer to" 
    "call +39 800857025 from 8:30 to 17:30 Monday to Friday"
    "or visit https://www.culligan.it/ufficio/assistenza/"
    "Be as concise as possible. Write one sentence if possible, and never go beyond 5 sentences."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Chat settings
@cl.on_chat_start
async def chat_settings():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4"],
                initial_index=0
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1
            ),
        ]
    ).send()
    update_llm(settings)  # Apply settings to the LLM

# Starters
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Che rubinetti offrite?",
            message="Che tiplogie di rubinetti offrite a Culligan?",
            icon="https://static.vecteezy.com/system/resources/previews/014/237/107/non_2x/intelligent-system-idea-software-development-gradient-icon-vector.jpg"
        ),
        cl.Starter(
            label="Qual'e il rubinetto meno caro?",
            message="Trova il rubinetto meno caro offerto da Culligan, mostra il prezzo e una descrizione del prodotto",
            icon="https://www.creativefabrica.com/wp-content/uploads/2022/07/25/Html-File-Line-Gradient-Icon-Graphics-34809782-1-1-580x387.jpg"
        ),
        cl.Starter(
            label="Avete un rubinetto erogatore di acqua frizzante? ",
            message="Spiega in termini layman che cos'e un rubinetto di acqua frizzante e come funziona",
            icon="https://png.pngtree.com/png-clipart/20230920/original/pngtree-book-pixel-perfect-gradient-linear-ui-icon-red-color-illustration-vector-png-image_12806820.png"
        ),
        cl.Starter(
            label="Cosa fa Culligan",
            message="Di cosa si occupa Culligan? quali sosno i prodotti principali che offre? Cosa distingue Culligan?",
            icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfpV1cOOZDFNr2atjHgVNWedyrzJCQ1q2AUw&s"
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    await chat_settings()  # Ensure chat settings are sent on chat start
    cl.user_session.set("session_id", str(uuid.uuid4()))  # Add session ID generation

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(conversational_rag_chain.stream)(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], configurable={"session_id": session_id}),
    ):
        # Ensure chunk is serializable before sending
        if isinstance(chunk, str):
            await msg.stream_token(chunk)
        elif isinstance(chunk, dict):
            await msg.stream_token(chunk.get("answer", ""))
    
    await msg.send()

# Handling settings update
@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    update_llm(settings)

if __name__ == "__main__":
    cl.run()
