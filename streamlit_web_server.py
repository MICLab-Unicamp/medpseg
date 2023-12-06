'''
Entry point for cloud hosted streamlit server
'''
import subprocess
from medpseg.streamlit_server import streamlit_server


if __name__ == "__main__":
    subprocess.run(["pip" , "install", "medpseg"])
    streamlit_server()