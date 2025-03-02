'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Streamlit wrapper over medpseg compiled CLI
'''
import os
import io
import site
import time
import glob
import zipfile
import tempfile
import subprocess
import streamlit as st
import pandas as pd
import streamlit_scrollable_textbox as stx
from st_aggrid import AgGrid
from streamlit.logger import get_logger


LOGGER = get_logger(__name__)
TMP_DIR = tempfile.gettempdir()


if os.name == "nt":
    ICON_PNG = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "icon.png")
    ICON_ORIGINAL_PNG = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "icon_original.png")
    DEMO_FILE = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "coronacases_100_003.png")
    VOL_RESP = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "respiratory.gif")
    VOL_DIS = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "diseased.gif")
else:
    ICON_PNG = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "icon.png")
    ICON_ORIGINAL_PNG = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "icon_original.png")
    DEMO_FILE = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "coronacases_100_003.png")
    VOL_RESP = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "respiratory.gif")
    VOL_DIS = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "diseased.gif")

@st.cache_data(show_spinner=False)
def run(input_path: str, output_path: str) -> str:
    '''
    Wrapper around MEDPSeg executable, monitoring output and reporting in real time
    Also returns the full output when execution is finished
    '''
    process = subprocess.Popen(["medpseg_cpu", 
                                "-i", input_path, 
                                "-o", output_path], stdout=subprocess.PIPE)
    subbody = st.empty()
    output_buffer = []
    process_running = True
    while process_running:
        output = process.stdout.readline().strip().decode()
        if output == '' and process.poll() is not None:
            process_running = False
        else:
            with subbody.container():
                st.write(output)
            output_buffer.append(output)
        print(f"Process running... Writing to browser: {output}")
        time.sleep(0.1)
    print("Process finished.")

    rc = process.poll()
    if rc != 0:
        st.write(f"internal error: {rc}")

    subbody = st.empty()

    return '\n'.join(output_buffer)

def render_outputs(ID: str, dl_button):
    '''
    Render outputs that contain ID string in their name, and most recent report
    '''
    # Map MEDPSeg output names to readable captions
    CAPTION_MAP = {"airway_only": "Airway (red)",
                   "airway": "Airway (red) over input",
                   "all_segmentations": "All segmentations (airway in light blue, pulmonary artery in yellow) over input",
                   "all_segmentations_only": "All segmentations (airway in light blue, pulmonary artery in yellow)",
                   "consolidation": "Consolidation (red) over input",
                   "consolidation_only": "Consolidation (red)",
                   "findings": "Consolidation + GGO (red) over input",
                   "findings_only": "Consolidation + GGO (red)",
                   "ggo": "GGO (red) over input",
                   "ggo_only": "GGO (red)",
                   "lung": "Lung (red) over input",
                   "lung_only": "Lung (red)",
                   "medpseg_reverse_engineered": "Reverse engineered input. If this looks wrong, check instructions above upload widget.",
                   "vessel": "Pulmonary artery over input",
                   "vessel_only": "Pulmonary artery"}

    output_files = sorted(glob.glob(os.path.join(TMP_DIR, f"*{ID}*.png")))
    st.write("Output images:")
    captions = []
    for output_file in output_files:
        map_idx = os.path.basename(output_file).split(f"{ID}_")[-1].replace(".png", '')
        caption = CAPTION_MAP[map_idx]
        captions.append(caption)
    st.image(output_files, caption=captions)
    st.write("Output report:")
    report_file_path = sorted(glob.glob(os.path.join(TMP_DIR, "*run_statistics*.csv")),
                              key=os.path.getmtime,
                              reverse=True)[0]
    report_file = pd.read_csv(report_file_path)
    AgGrid(report_file)
    st.write("WARNING: The report above is not accurate for 2D images, just a sample of what fields are involved when running in actual volumetric images.")

    # Zip outputs
    master_buffer = io.BytesIO()
    with zipfile.ZipFile(master_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as master_zip:
        output_files.append(report_file_path)
        for output_file in output_files:
            with open(output_file, 'rb') as f:
                master_zip.writestr(os.path.basename(output_file), f.read())

    dl_button.download_button(f"Processing done! Click to download all outputs as ZIP", master_buffer, f"{ID}_outputs.zip")
    st.write("Output download link available on the sidebar.")

def run_image(input_file: io.BytesIO, _dl_button):
    '''
    From UploadedFile(io.BytesIO) run MEDPSeg internally, calling renders of output artifacts
    '''
    # Make input bytes into temporary file
    st.write(f"Processing the following image: {input_file.name}")
    st.image(input_file)
    st.write("MEDPSeg Output:")
    fmt = input_file.name.split('.')[-1]
    suffix = f".{fmt}"
    with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=suffix) as temp_input_file:
        temp_input_file.write(input_file.getbuffer())
        ID = os.path.basename(temp_input_file.name).replace(suffix, '')
        print(temp_input_file.name)
        print(ID)
        io_block = st.empty()
        with io_block.container():
            io_text = run(temp_input_file.name, TMP_DIR)
            stx.scrollableTextbox(io_text, height=200)
    
    render_outputs(ID, _dl_button)

def demo(_dl_button):
    '''
    Runs demo when nothing has ben given as input
    '''
    st.header("Upload a image using the widget in the sidebar to start processing on you data!")
    with st.expander("Sample run...", expanded=True):
        st.write("Running MEDPSeg --help. Output:")
        help = st.empty()
        output = subprocess.run(["medpseg_cpu", "--help"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        with help.container():
            stx.scrollableTextbox(output.stdout.decode(), height=200)

        st.write(f"Testing MEDPSeg integrity in demo {os.path.basename(DEMO_FILE)} image.")
        st.image(DEMO_FILE, caption="Demo input coronacases_100_003.png")
        st.write("Output:")
        integrity = st.empty()
        with integrity.container():
            test_output = run(DEMO_FILE,
                                TMP_DIR)
        
        with integrity.container():
            stx.scrollableTextbox(test_output, height=200)

        render_outputs("coronacases_100_003", _dl_button)

def guide_text():
    '''
    Writes all the introductory text and page layout with st.write
    '''
    st.write("MEDPSeg supports predicting over .png uint8 images. However, make sure the image was derived from original CT scans!")
    with st.expander("How to derive a .png image from original CT scans?", expanded=False):
        st.write("Given a X original CT scan numpy ndarray with intensities in Hounsfield Units (HU), input .png image X_img should be produced as follows:")
        st.write("1) Clip to the [-1024, 600] range: X_clip = np.clip(x, -1024, 600)")
        st.write("2) Min-max normalize: X_norm = (X_clip - (-1024))/(600 -(-1024))")
        st.write("3) Bring to uint8 representation: X_img = (X_norm*255).astype(np.uint8)")
        st.write("Finally, save X_img as a .png file. We recommend using the imageio library.")
        st.write("If you want to customize the expected MIN and MAX values on the HU clip process, use the CLI interface available in our GitHub.")

def streamlit_server():
    '''
    Main entrypoint for streamlit code
    '''
    st.set_page_config(
        page_title="MEDPSeg Demo",
        layout="wide",
        page_icon=ICON_ORIGINAL_PNG
    )
    st.sidebar.image(ICON_ORIGINAL_PNG, width=100)
    st.sidebar.header("MEDPseg")
    inference_mode = st.sidebar.selectbox("Inference mode", ["Select one", "2D Inference", "3D Inference"])
    inference_3d = inference_mode == "3D Inference"
    inference_2d = inference_mode == "2D Inference"
    
    st.write("# MEDPSeg (https://github.com/MICLab-Unicamp/medpseg)")
    st.write("## Welcome to the MEDPSeg online demo!")
    if not inference_2d and not inference_3d:
        st.write("Select an inference mode in the sidebar.")
    elif inference_3d:
        st.write("### Online 3D Inference coming soon!")
        st.write("This streamlit demo deploy only operates over slices saved as 2D images, for demostration purposes.")
        st.write("To setup MEDPSeg on your machine for volumetric inference, check our README at https://github.com/MICLab-Unicamp/medpseg")
        st.write("When using MEDPSeg locally with a CUDA enbaled GPU and NifT volumetric scans, you can generate results as shown below in 1 minute (transparent green: lung, blue: pulmonary artery, red: airway, yellow: opacities, purple: consolidations):")
        coll, colr = st.columns(2)
        with coll:
            st.image(VOL_DIS, caption="MEDPSeg results in coronacases_003 CT from the CoronaCases dataset.")
        with colr:
            st.image(VOL_RESP, caption="MEDPSeg results in ID 013 contrast enhanced CT from the PARSE dataset.")
    elif inference_2d:
        st.write("### 2D Inference")
        guide_text()
        input_file = st.sidebar.file_uploader("Upload a 2D .png/.jpg here for processing.", type=[".png", ".jpg", ".jpeg"])
        dl_button = st.sidebar.empty()
        sample = st.button("Click here to process a sample image...")
        st.write("Or upload you own image using the sidebar upload widget!")
        
        if sample:
            dl_button.write(f"Processing sample image... Scroll down to check output logs. Results will be available for download here. You can interrupt at any time by uploading your image.")
            demo(dl_button)
        elif input_file is not None:
            dl_button.write(f"Processing {input_file.name}... Scroll down to check output logs. Results will be available for download here.")
            run_image(input_file, dl_button)

    st.sidebar.write("This demo is possible thanks to the support of [NeuralMind](https://neuralmind.ai/).")
    st.sidebar.write("Check our [paper](https://arxiv.org/abs/2312.02365) to learn more about MEDPSeg.")

if __name__ == "__main__":
    streamlit_server()
