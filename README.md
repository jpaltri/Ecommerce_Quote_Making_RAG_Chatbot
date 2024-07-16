# Culligan Quote-Making Chatbot

This project features a Retrieval-Augmented Generation (RAG) assistant designed to provide product information and quotes for Culligan's water filters, sinks, and related products. There are two versions of the chatbot: one that loads data from web links (`Culligan_web`) and another from a JSON file (`Culligan_json`). Both versions utilize advanced language models to deliver accurate and relevant information to customers.

## Features

### Memory Management
- **Session Management**: The chatbot uses a stateful approach to manage chat history, ensuring context is preserved throughout a user's session.
- **History Awareness**: The chatbot can reformulate user questions based on previous interactions within the same session, improving the relevance and accuracy of responses.

### User Interface
- **Custom Chat Profiles**: Users can select from different chat profiles representing various language models, allowing them to choose the model that best suits their needs.
- **Interactive Settings**: Users can customize their experience by selecting the model, enabling or disabling token streaming, and adjusting the temperature for response generation.
- **Predefined Starters**: The chatbot includes predefined starter prompts to help users quickly get answers to common questions related to Culligan products.

### Retrieval-Augmented Generation (RAG)
- **Document Integration**: The chatbot can load and process content from PDF documents, web links, or JSON files, using them as a knowledge base for answering user queries.
- **Advanced Retrieval**: The chatbot employs RAG techniques to enhance its responses by retrieving relevant information before generating answers, ensuring responses are both contextually relevant and backed by accurate data.

## Key Differences Between Versions

### Culligan_web
- **Data Source**: Loads data from web links.
- **Flexibility**: More convenient and flexible as it can fetch the latest information directly from the web.
- **Susceptibility**: More susceptible to hallucinations due to the variability and unpredictability of web content.

### Culligan_json
- **Data Source**: Loads data from a JSON file.
- **Accuracy**: More accurate and less susceptible to hallucinations as the data is predefined and controlled.
- **Convenience**: Less convenient than web scraping as it requires manual updates to the JSON file.

## Setup

### Prerequisites
- Python 3.7+
- `pip` (Python package installer)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/jpaltrin/Quote_Making_RAG_Chatbot.git
    cd Quote_Making_RAG_Chatbot
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Create a `.env` file in the project root directory and add your environment variables:**

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langchain_api_key
    ```

### Usage

1. **Run the chatbot:**

    ```sh
    python Culligan_web.py  # For the web version
    python Culligan_json.py  # For the JSON version
    ```

2. **Interact with the chatbot through the provided UI.**

## Project Structure

- `Culligan_web.py`: Script for the web-based version of the chatbot.
- `Culligan_json.py`: Script for the JSON-based version of the chatbot.
- `requirements.txt`: Lists the dependencies required for the project.
- `.env`: Contains environment variables (not included in the repository for security reasons).
- `README.md`: Project documentation.
- `product_url.json`: JSON file containing product data for the JSON-based version.

