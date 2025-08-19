'''
Entry point for cloud hosted streamlit server
'''
import os
from dotenv import load_dotenv
from tinydb import TinyDB
from medpseg.streamlit_server import streamlit_server


if __name__ == "__main__":
    load_dotenv()
    
    try:
        TINYDB_PATH = os.environ["TINYDB_PATH"]
    except KeyError as e:
        print(f"Error loading database: {e}. Please setup .env file with TINYDB_PATH. Check example.env for reference.")
    
    os.makedirs(os.path.dirname(TINYDB_PATH), exist_ok=True)
    db = TinyDB(TINYDB_PATH)
    print(f"Database initialized at: {TINYDB_PATH}")
    streamlit_server(db)
