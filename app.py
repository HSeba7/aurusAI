import sys
import threading
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QCoreApplication
import os
import openai

load_dotenv()  

app = Flask(__name__)

CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

openai.api_key = os.getenv("OPENAI_API_KEY")


plugin_path = os.path.join(os.path.dirname(sys.executable), "Qt5/plugins")
QCoreApplication.addLibraryPath(plugin_path)

message_history = []

@app.route('/new_chat', methods=['POST'])
def new_chat():
    global message_history
    message_history = []  # Reset the message history
    return jsonify({"message": "Chat reset successfully"}), 200

# @app.route('/chat', methods=['POST'])
# def chats():
#     # Get user input text from the request
#     data = request.get_json()
#     user_message = data.get('message', '')
#     message_history.append({"role": "user", "content": user_message})
#     def generate():
#         # Send the user's input to ChatGPT using OpenAI API with streaming
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=message_history,
#             # messages=[{"role": "user", "content": user_message}],
#             stream=True
#         )
#         assistant_message = "" 
#         # Stream the response to the client in chunks
#         for chunk in response:
            
#             if 'choices' in chunk:
#                 text_chunk = chunk['choices'][0]['delta'].get('content', '')
#                 assistant_message += text_chunk 
#                 yield text_chunk
#         message_history.append({"role": "assistant", "content": assistant_message})


        
#     return Response(generate(), content_type='text/event-stream')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    print(data)
    user_message = data.get('message', '')

    from langchain_openai.embeddings import OpenAIEmbeddings  # Still uses langchain_openai
    from langchain_community.vectorstores import Pinecone
    from langchain.chains.question_answering import load_qa_chain
    from langchain_community.chat_models import ChatOpenAI


    embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
    index = Pinecone.from_existing_index(index_name='chatbot', embedding=embeddings)

    def retrieve_query(query,k=1):
        matching_results=index.similarity_search(query,k=k)
        return matching_results

    
    llm=ChatOpenAI(model_name="gpt-4o")
    chain=load_qa_chain(llm,chain_type="stuff")

    ## Search answers from VectorDB
    def retrieve_answers(query):
        doc_search=retrieve_query(query)
        input_data = {
            "input_documents": doc_search,
            "question": query
        }
        response=chain.invoke(input=input_data)
        answer = response.get("output_text", "No answer found.")
        return answer.strip()

    answer = retrieve_answers(user_message)
    print("================",answer)

    return Response(answer)


def run_flask():
    # Start Flask app on a different thread
    app.run(host='127.0.0.1', port=5000, debug=False)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window icon (ensure the path to your .ico file is correct)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "templates", "images","logo-img.ico")
        self.setWindowIcon(QIcon(icon_path))
        # self.setWindowTitle("HTML Chat Interface")
        self.resize(690, 780)

        # Create a QWebEngineView to display HTML content
        self.browser = QWebEngineView()

        # Get the absolute file path for the HTML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        html_file = os.path.join(current_dir, "templates", "index.html")

        # Load the local HTML file into the browser
        self.browser.setUrl(QUrl.fromLocalFile(html_file))

        # Set the browser as the central widget
        self.setCentralWidget(self.browser)


def start_app():
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.setDaemon(True)
    flask_thread.start()

    # Start PyQt application
    qt_app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec_())


if __name__ == '__main__':
    start_app()
