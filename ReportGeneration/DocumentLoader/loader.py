from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def document_loader():
    loader = DirectoryLoader(
        path='ReportGeneration\KnowledgeBase',
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )

    docs = loader.load()

    return docs

