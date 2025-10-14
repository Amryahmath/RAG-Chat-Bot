import sys

def try_import(name, alt=None):
    try:
        module = __import__(name)
        print(f"{name} OK:", getattr(module, '__version__', 'version unknown'))
    except Exception as e:
        print(f"{name} import failed:", type(e).__name__, e)
        if alt:
            try:
                __import__(alt)
                print(f"Alternate import {alt} OK")
            except Exception as e2:
                print(f"Alternate import {alt} failed:", type(e2).__name__, e2)

if __name__ == '__main__':
    try_import('PyPDF2')
    # langchain submodules
    try:
        from langchain.document_loaders import PyPDFLoader
        print('langchain.document_loaders.PyPDFLoader OK')
    except Exception as e:
        print('PyPDFLoader import failed:', type(e).__name__, e)

    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print('langchain.text_splitter OK')
    except Exception as e:
        print('text_splitter import failed:', type(e).__name__, e)

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print('langchain_google_genai OK')
    except Exception as e:
        print('langchain_google_genai import failed:', type(e).__name__, e)

    print('\nPython executable:', sys.executable)
