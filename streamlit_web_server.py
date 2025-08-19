'''
Entry point for cloud hosted streamlit server
'''
import os
from tinydb import TinyDB
from medpseg.streamlit_server import streamlit_server


if __name__ == "__main__":
    TINYDB_PATH = os.path.join(os.environ["HOME"], "medpseg", "db", "medpseg_db.json")
    os.makedirs(os.path.dirname(TINYDB_PATH), exist_ok=True)
    db = TinyDB(TINYDB_PATH)
    streamlit_server(db)
