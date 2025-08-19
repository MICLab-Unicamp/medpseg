'''
Entry point for cloud hosted streamlit server
'''
import os
from tinydb import TinyDB
from medpseg.streamlit_server import streamlit_server


if __name__ == "__main__":
    TINYDB_PATH = os.path.join("db", "medpseg_db.json")
    db = TinyDB(TINYDB_PATH)
    streamlit_server(db)
