# AI-Powered Local Document Assistant

A locally hosted AI-powered document assistant that enables researchers to interact with PDFs, DOCX, and Markdown files using natural language.  
Built with Retrieval-Augmented Generation (RAG), it supports multi-modal retrieval of text, images, and tables while ensuring data privacy, boosting research efficiency, and reducing redundant work.

---

## ✨ Features
- 🔒 **Privacy-first**: Runs locally, no cloud dependency.
- 📄 **Multi-format support**: Works with PDF, DOCX, and Markdown files.
- 🖼️ **Multi-modal retrieval**: Extracts and queries text, images, and tables.
- 🗂️ **Document scoping**: Query specific files or collections.
- 🧠 **Persistent memory**: Stores embeddings and chat history across sessions.
- 🎙️ **Voice-enabled responses**: Optional text-to-speech integration.
- 🐳 **Containerized deployment**: Easy setup with Docker.

---

## 📚 Tech Stack
- **Backend**: Python, LangChain  
- **Vector Store**: ChromaDB  
- **Embeddings**: HuggingFace sentence-transformers (tested with Azure OpenAI as well)  
- **Frontend**: Streamlit + Web Speech API  
- **Parsing**: PyMuPDF, python-docx, camelot  
- **Deployment**: Docker  

---

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.10  
- Docker (optional, for containerized deployment)  

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Create a folder named Notes under the project root. This will serve as your document vault.

5. Run the application:
streamlit run app.py

6. Upload documents and start querying them through the browser UI.

🧪 Example Workflow

Upload one or more PDFs, DOCX, or Markdown files.

The system extracts and embeds text, tables, and images into ChromaDB.

Ask natural language questions (e.g., “Summarize key findings from page 5”).

Receive contextual answers with text, tables, or images displayed.

Optionally, listen to answers with voice playback.

🌱 Sustainability Impact

Reduces redundant experimental work → fewer wasted resources.

Local deployment minimizes carbon footprint compared to cloud queries.

Promotes paperless workflows and knowledge reuse.

🔮 Roadmap

Domain-specific fine-tuning for materials science.

Multilingual support (French, German, Chinese, etc.).

Advanced query chaining and document comparison.

User feedback loop for improving accuracy.

Integration with internal tools (e.g., SharePoint, ELN systems).

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and open a pull request.
