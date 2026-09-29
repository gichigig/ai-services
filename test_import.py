import modal
app = modal.App("test-import")
image = modal.Image.debian_slim(python_version="3.11").apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libsndfile1", "python3-dev", "build-essential").pip_install_from_requirements("/Users/macbook/Downloads/bruv-schools/ai-service/requirements.txt")
@app.function(image=image)
def test_import():
    try:
        import faiss
        print("faiss imported successfully")
    except Exception as e:
        print("faiss import failed:", e)
        import traceback; traceback.print_exc()
    try:
        from sentence_transformers import SentenceTransformer
        print("sentence_transformers imported successfully")
    except Exception as e:
        print("sentence_transformers import failed:", e)
        import traceback; traceback.print_exc()

@app.local_entrypoint()
def main():
    test_import.remote()
