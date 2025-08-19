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
import uuid
import site
import time
import glob
import zipfile
import tempfile
import subprocess
import datetime
import hashlib
import psutil
import streamlit as st
import pandas as pd
import streamlit_scrollable_textbox as stx
from st_aggrid import AgGrid
from tinydb import TinyDB, Query
from multiprocessing import Process, Queue
from streamlit.logger import get_logger


LOGGER = get_logger(__name__)
TMP_DIR = tempfile.gettempdir()

def get_client_ip():
    '''
    Get the client IP address from Streamlit request headers
    '''
    try:
        # Try to get IP from various headers (for proxy scenarios)
        headers = st.context.headers if hasattr(st, 'context') and hasattr(st.context, 'headers') else {}
        
        # Check common headers for client IP (in order of preference)
        ip_headers = [
            'x-forwarded-for',
            'x-real-ip', 
            'cf-connecting-ip',  # Cloudflare
            'x-client-ip',
            'remote-addr'
        ]
        
        for header in ip_headers:
            if header in headers:
                ip = headers[header]
                # x-forwarded-for can contain multiple IPs, take the first one
                if ',' in ip:
                    ip = ip.split(',')[0].strip()
                if ip and ip != 'unknown':
                    return ip
        
        # Fallback: try to get from session info
        try:
            session_info = st.runtime.get_instance().get_session_info()
            if session_info and hasattr(session_info, 'client'):
                return session_info.client.request.remote_ip
        except:
            pass
            
        return 'unknown'
        
    except Exception as e:
        print(f"Warning: Could not get client IP: {e}")
        return 'unknown'


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

class StreamlitProcessor(Process):
    def __init__(self, UID: str, input_path: str, output_path: str, db: TinyDB, input_filename: str):
        super().__init__()
        self.db = db
        self.UID = UID
        self.input_path = input_path
        self.output_path = output_path
        self.input_filename = input_filename
        self.running = False
        self.logs = ""
        self.start_time = datetime.datetime.now()
        
        # Start DB entry for async processing
        self.db.table('processings').insert({
            'uid': self.UID,
            'timestamp': self.start_time.isoformat(),
            'input_filename': self.input_filename,
            'output_path': None,
            'volumetric': True,
            'status': 'processing',
            'logs': 'Processing started...',
            'client_ip': get_client_ip()
        })

    def update_logs_in_db(self):
        '''
        Update the logs field in the database with current self.logs content
        '''
        try:
            from tinydb import Query
            self.db.table('processings').update({
                'logs': self.logs
            }, Query().uid == self.UID)
        except Exception as e:
            # Don't let database update failures crash the processing
            print(f"Warning: Failed to update logs in database: {e}")

    def run(self):
        self.running = True
        
        try:
            process = subprocess.Popen([os.getenv("MEDPSEG_CMD", "medpseg_cpu"),
                                        "--disable_lobe",
                                        "-i", self.input_path, 
                                        "-o", self.output_path], stdout=subprocess.PIPE)
            while process.poll() is None:
                output = process.stdout.readline().strip().decode()
                if output == '' and process.poll() is not None:
                    self.running = False
                else:
                    self.logs += output + "\n"
                    # Update logs in database every time we add new output
                    self.update_logs_in_db()
                    
            self.running = False

            # Check if process completed successfully
            rc = process.poll()
            end_time = datetime.datetime.now()
            
            if rc == 0:
                # Update DB entry for successful completion
                from tinydb import Query
                self.db.table('processings').update({
                    'end_timestamp': end_time.isoformat(),
                    'status': 'completed',
                    'processing_time_seconds': (end_time - self.start_time).total_seconds(),
                    'logs': self.logs,
                    'output_path': self.output_path
                }, Query().uid == self.UID)
            else:
                # Update DB entry for failed completion
                from tinydb import Query
                self.db.table('processings').update({
                    'end_timestamp': end_time.isoformat(),
                    'status': 'failed',
                    'error': f'Process exited with code {rc}',
                    'processing_time_seconds': (end_time - self.start_time).total_seconds(),
                    'logs': self.logs
                }, Query().uid == self.UID)

            return self.logs
            
        except Exception as e:
            # Update DB entry for exception
            end_time = datetime.datetime.now()
            error_message = str(e)
            from tinydb import Query
            self.db.table('processings').update({
                'end_timestamp': end_time.isoformat(),
                'status': 'failed',
                'error': error_message,
                'processing_time_seconds': (end_time - self.start_time).total_seconds(),
                'logs': self.logs
            }, Query().uid == self.UID)
            
            self.running = False
            raise e


@st.cache_data(show_spinner=False)
def run(input_path: str, output_path: str) -> str:
    '''
    Wrapper around MEDPSeg executable, monitoring output and reporting in real time
    Also returns the full output when execution is finished
    '''
    process = subprocess.Popen([os.getenv("MEDPSEG_CMD", "medpseg_cpu"), 
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

def render_outputs(ID: str, dl_button, output_path: str):
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

    output_files = sorted(glob.glob(os.path.join(output_path, f"*{ID}*.png")))
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

def cleanup_orphaned_processing_entries(db: TinyDB) -> int:
    '''
    Check for database entries with status 'processing' but no medpseg_cpu process running.
    Since we only have one processing slot, if any 'processing' entries exist but no 
    medpseg_cpu is running, all those entries are orphaned and should be marked as failed.
    '''
    processings_table = db.table('processings')
    
    # Find all entries with status 'processing'
    processing_entries = processings_table.search(Query().status == 'processing')
    
    if not processing_entries:
        return 0
    
    # Check if any medpseg_cpu process is running
    medpseg_running = False
    try:
        CMD = os.getenv("MEDPSEG_CMD", "medpseg_cpu")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == CMD or (
                    proc.info['cmdline'] and 
                    any(cmd.endswith(CMD) or cmd == CMD for cmd in proc.info['cmdline'])
                ):
                    medpseg_running = True
                    print(f"Found running {CMD} process: PID {proc.info['pid']}")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"Warning: Could not check running processes: {e}")
        # If we can't check processes, don't clean anything to be safe
        return 0
    
    print(f"Found {len(processing_entries)} database entries with 'processing' status")
    print(f"medpseg_cpu process running: {medpseg_running}")
    
    # If no medpseg_cpu process is running but we have 'processing' entries, they are orphaned
    cleaned_count = 0
    if not medpseg_running and processing_entries:
        print("No medpseg_cpu process running - cleaning all 'processing' entries")
        for entry in processing_entries:
            end_time = datetime.datetime.now()
            processings_table.update({
                'end_timestamp': end_time.isoformat(),
                'status': 'failed',
                'error': 'Process was terminated or crashed (cleanup)',
                'processing_time_seconds': (end_time - datetime.datetime.fromisoformat(entry['timestamp'])).total_seconds()
            }, Query().uid == entry['uid'])
            cleaned_count += 1
            print(f"Marked orphaned entry as failed: {entry['uid']}")
    
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} orphaned processing entries")
    
    return cleaned_count

def run_image(input_file: io.BytesIO, _dl_button, volumetric: bool, db: TinyDB):
    '''
    From UploadedFile(io.BytesIO) run MEDPSeg internally, calling renders of output artifacts.
    Also logs each processing to TinyDB "processings" table with timestamp, input, and logs.
    '''
    # Initialize logging variables
    start_time = datetime.datetime.now()
    processing_logs = ""

    # Generate UID for this processing
    UID = str(uuid.uuid4())
    
    # Make input bytes into temporary file
    st.write(f"Processing the following image: {input_file.name}.")
    st.info(f"Save this UID: {UID} to get results later with the UID checker in the sidebar!")
    st.warning("Outputs are not guaranteed to be available days after processing.")
    
    st.write("MEDPSeg Output:")
    if volumetric:
        if ".nii.gz" in input_file.name:
            fmt = "nii.gz"
        elif ".nii" in input_file.name:
            fmt = "nii"
        else:
            st.write("Unsupported file format. Please upload a .nii.gz or .nii file.")
            # Log failed processing
            db.table('processings').insert({
                'uid': UID,
                'timestamp': start_time.isoformat(),
                'input_filename': input_file.name,
                'output_path': None,
                'volumetric': volumetric,
                'status': 'failed',
                'error': 'Unsupported file format',
                'logs': 'Unsupported file format. Please upload a .nii.gz or .nii file.',
                'client_ip': get_client_ip()
            })
            return
    else:
        fmt = input_file.name.split('.')[-1]
    suffix = f".{fmt}"

    try:
        output_path = os.path.join(TMP_DIR, f"{UID}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if volumetric:
            # Before starting new 3D processing, cleanup any orphaned 'processing' entries
            st.write("🔍 Checking for orphaned processing entries...")
            cleaned_count = cleanup_orphaned_processing_entries(db)
            if cleaned_count > 0:
                st.warning(f"⚠️ Found and cleaned {cleaned_count} orphaned processing entries from previous crashed processes.")
            else:
                st.success("✅ No orphaned entries found. System is clean.")
            
            # Check if a medpseg_cpu process is already running (single slot limitation)
            medpseg_running = False
            try:
                CMD = os.getenv("MEDPSEG_CMD", "medpseg_cpu")
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] == CMD or (
                            proc.info['cmdline'] and 
                            any(CMD in str(cmd) for cmd in proc.info['cmdline'])
                        ):
                            medpseg_running = True
                            cmdline_str = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'N/A'
                            print(f"Debug: Found running process - Name: {proc.info['name']}, PID: {proc.info['pid']}, Command: {cmdline_str}")
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
            except Exception as e:
                st.error(f"Could not check running processes: {e}")
                return
            
            if medpseg_running:
                st.error(f"❌ Cannot start new 3D processing: Another {CMD} process is already running.")
                st.info("💡 Only one 3D processing can run at a time due to hardware limitations.")
                st.info("🔍 Check the status of your existing processing using the UID checker in the sidebar.")
                return
            
            # Save uploaded file to temporary path for volumetric processing
            temp_input_path = os.path.join(TMP_DIR, f"{UID}_input{suffix}")
            with open(temp_input_path, 'wb') as f:
                f.write(input_file.getbuffer())
            
            # Start new processing
            processor = StreamlitProcessor(UID=UID, input_path=temp_input_path, output_path=output_path, db=db, input_filename=input_file.name)
            processor.start()
            
            # For volumetric processing, we show a message and don't wait for completion
            st.write(f"🚀 Volumetric processing started with UID: **{UID}**")
            st.write("⏳ This may take several minutes. Check back later using your UID in the sidebar.")
            st.info(f"💡 Save this UID to check your results: **{UID}**")
        else:
            # This one is fast, runs in the streamlit "server" thread
            with tempfile.NamedTemporaryFile(dir=TMP_DIR, suffix=suffix) as temp_input_file:
                temp_input_file.write(input_file.getbuffer())
                ID = os.path.basename(temp_input_file.name).replace(suffix, '')
                print(temp_input_file.name)
                print(ID)
                io_block = st.empty()
                with io_block.container():
                    io_text = run(temp_input_file.name, output_path)
                    processing_logs = io_text
                    stx.scrollableTextbox(io_text, height=200)
        
            render_outputs(ID, _dl_button, output_path)
            
            # Log successful processing
            end_time = datetime.datetime.now()
            db.table('processings').insert({
                'uid': UID,
                'timestamp': start_time.isoformat(),
                'end_timestamp': end_time.isoformat(),
                'input_filename': input_file.name,
                'volumetric': volumetric,
                'status': 'completed',
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'logs': processing_logs,
                'output_id': ID,
                'output_path': output_path,
                'client_ip': get_client_ip()
            })
            
    except Exception as e:
        # Log failed processing (only for 2D processing, volumetric is handled by StreamlitProcessor)
        if not volumetric:
            end_time = datetime.datetime.now()
            error_message = str(e)
            st.error(f"Processing failed: {error_message}")
            
            db.table('processings').insert({
                'uid': UID,
                'timestamp': start_time.isoformat(),
                'end_timestamp': end_time.isoformat(),
                'input_filename': input_file.name,
                'output_path': None,
                'volumetric': volumetric,
                'status': 'failed',
                'error': error_message,
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'logs': processing_logs,
                'client_ip': get_client_ip()
            })
        else:
            # For volumetric processing, just show the error but don't log (StreamlitProcessor handles it)
            st.error(f"Failed to start volumetric processing: {str(e)}")

def demo(_dl_button, db: TinyDB):
    '''
    Runs demo when nothing has ben given as input
    Also logs the demo processing to TinyDB "processings" table.
    '''
    # Initialize logging variables for demo
    start_time = datetime.datetime.now()
    UID = str(uuid.uuid4())
    
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
        
        try:
            with integrity.container():
                test_output = run(DEMO_FILE, TMP_DIR)
            
            with integrity.container():
                stx.scrollableTextbox(test_output, height=200)

            render_outputs("coronacases_100_003", _dl_button)
            
            # Log successful demo processing
            end_time = datetime.datetime.now()
            db.table('processings').insert({
                'uid': UID,
                'timestamp': start_time.isoformat(),
                'end_timestamp': end_time.isoformat(),
                'input_filename': os.path.basename(DEMO_FILE),
                'volumetric': False,
                'status': 'completed',
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'logs': test_output,
                'output_id': "coronacases_100_003",
                'output_path': None,
                'is_demo': True,
                'client_ip': get_client_ip()
            })
            
        except Exception as e:
            # Log failed demo processing
            end_time = datetime.datetime.now()
            error_message = str(e)
            st.error(f"Demo processing failed: {error_message}")
            
            db.table('processings').insert({
                'uid': UID,
                'timestamp': start_time.isoformat(),
                'end_timestamp': end_time.isoformat(),
                'input_filename': os.path.basename(DEMO_FILE),
                'volumetric': False,
                'status': 'failed',
                'error': error_message,
                'processing_time_seconds': (end_time - start_time).total_seconds(),
                'logs': '',
                'output_path': None,
                'is_demo': True,
                'client_ip': get_client_ip()
            })

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

def check_admin_password(password: str) -> bool:
    '''
    Check if the provided password matches the admin password
    Using a simple hash for demo purposes - in production use proper authentication
    '''
    # Simple password for demo - in production, use environment variables and proper hashing
    admin_password_hash = "581f55407aecdcc11652e0fbc253fed70d7e38c2db0f9ef94991562f70d1f571"
    provided_hash = hashlib.sha256(password.encode()).hexdigest()
    return provided_hash == admin_password_hash

def display_admin_panel(db: TinyDB):
    '''
    Display admin panel with full database contents
    '''
    st.header("🔐 Admin Panel")
    
    # Get all processing entries
    processings_table = db.table('processings')
    all_entries = processings_table.all()
    
    if not all_entries:
        st.write("No processing entries found in database.")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(all_entries)
    
    # Display summary statistics
    st.subheader("📊 Processing Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processings", len(all_entries))
    
    with col2:
        completed_count = len([e for e in all_entries if e.get('status') == 'completed'])
        st.metric("Completed", completed_count)
    
    with col3:
        failed_count = len([e for e in all_entries if e.get('status') == 'failed'])
        st.metric("Failed", failed_count)
    
    with col4:
        processing_count = len([e for e in all_entries if e.get('status') == 'processing'])
        st.metric("In Progress", processing_count)
    
    # Display processing type breakdown
    volumetric_count = len([e for e in all_entries if e.get('volumetric', False)])
    demo_count = len([e for e in all_entries if e.get('is_demo', False)])
    
    st.subheader("📈 Processing Types")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2D Processings", len(all_entries) - volumetric_count)
    with col2:
        st.metric("3D Processings", volumetric_count)
    with col3:
        st.metric("Demo Runs", demo_count)
    
    # Display full database table
    st.subheader("🗄️ Full Database Contents")
    
    # Sort by timestamp (newest first)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    
    # Display with AgGrid for better interaction
    AgGrid(df, height=400, fit_columns_on_grid_load=True)
    
    # Option to download database as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Database as CSV",
        data=csv,
        file_name=f"medpseg_database_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def check_processing_status(uid: str, db: TinyDB):
    '''
    Check processing status for a given UID and provide download if available
    '''
    if not uid.strip():
        st.warning("Please enter a UID to check status.")
        return
    
    # Search for the UID in database
    processings_table = db.table('processings')
    entries = processings_table.search(Query().uid == uid.strip())
    
    if not entries:
        st.error(f"No processing found with UID: {uid}")
        return
    
    entry = entries[0]  # Should be unique
    
    st.success(f"Processing found for UID: {uid}")
    
    # Display processing information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Filename:**", entry.get('input_filename', 'N/A'))
        st.write("**Processing Type:**", "3D Volumetric" if entry.get('volumetric', False) else "2D Image")
        st.write("**Status:**", entry.get('status', 'Unknown'))
        if entry.get('is_demo', False):
            st.write("**Type:** Demo Run")
    
    with col2:
        st.write("**Started:**", entry.get('timestamp', 'N/A'))
        if 'end_timestamp' in entry:
            st.write("**Completed:**", entry.get('end_timestamp'))
        if 'processing_time_seconds' in entry:
            st.write("**Duration:**", f"{entry.get('processing_time_seconds'):.2f} seconds")
        if 'error' in entry:
            st.write("**Error:**", entry.get('error'))
    
    # Display logs if available
    if entry.get('logs'):
        st.subheader("📋 Processing Logs")
        stx.scrollableTextbox(entry.get('logs'), height=200)
    
    # Check for downloadable outputs
    status = entry.get('status')
    if status == 'completed':
        # Check if outputs are available
        output_path = entry.get('output_path')
        output_id = entry.get('output_id')
        
        if output_path and os.path.exists(output_path):
            st.success("✅ Outputs are available for download!")
            
            # Get all files in the output directory
            all_files = []
            image_files = []
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
                    # Keep track of image files for preview
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(file_path)
            
            if all_files:
                # Create download zip with entire output folder
                master_buffer = io.BytesIO()
                with zipfile.ZipFile(master_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as master_zip:
                    for file_path in all_files:
                        # Use relative path from output_path as the archive name
                        archive_name = os.path.relpath(file_path, output_path)
                        with open(file_path, 'rb') as f:
                            master_zip.writestr(archive_name, f.read())
                
                st.download_button(
                    f"📥 Download Results for {uid}",
                    master_buffer,
                    f"{uid}_outputs.zip",
                    mime="application/zip"
                )
                
                st.write(f"📊 {len(all_files)} files ready for download")
                
                # Show preview of images if available
                if image_files:
                    st.subheader("🖼️ Output Preview")
                    st.image(image_files[:3], caption=[os.path.basename(f) for f in image_files[:3]], width=200)
                    if len(image_files) > 3:
                        st.write(f"... and {len(image_files) - 3} more images in the download.")
            else:
                st.warning("⚠️ Processing completed but output folder is empty.")
        else:
            st.warning(f"⚠️ Sorry, you took too long to check and your outputs have been deleted from temporary storage {output_path}.")
    
    elif status == 'processing':
        st.info("⏳ Processing is still in progress. Please check back later.")
    
    elif status == 'failed':
        st.error("❌ Processing failed. Check the error message and logs above.")
    
    else:
        st.warning(f"⚠️ Unknown processing status: {status}")

def check_processing_status_sidebar(uid: str, db: TinyDB):
    '''
    Check processing status for a given UID and display results in sidebar
    '''
    if not uid.strip():
        st.sidebar.warning("Please enter a UID to check status.")
        return
    
    # Search for the UID in database
    processings_table = db.table('processings')
    entries = processings_table.search(Query().uid == uid.strip())
    
    if not entries:
        st.sidebar.error(f"No processing found with UID: {uid}")
        return
    
    entry = entries[0]  # Should be unique
    
    st.sidebar.success(f"✅ Processing found!")
    
    # Display basic processing information in sidebar
    st.sidebar.write("**Input:**", entry.get('input_filename', 'N/A'))
    st.sidebar.write("**Type:**", "3D Volumetric" if entry.get('volumetric', False) else "2D Image")
    
    status = entry.get('status', 'Unknown')
    if status == 'completed':
        st.sidebar.success(f"**Status:** ✅ {status.title()}")
        if 'processing_time_seconds' in entry:
            st.sidebar.write(f"**Duration:** {entry.get('processing_time_seconds'):.1f}s")
    elif status == 'processing':
        st.sidebar.info(f"**Status:** ⏳ {status.title()}")
        st.sidebar.write("Still processing... Check back later.")
    elif status == 'failed':
        st.sidebar.error(f"**Status:** ❌ {status.title()}")
        if 'error' in entry:
            st.sidebar.write(f"**Error:** {entry.get('error')}")
    else:
        st.sidebar.warning(f"**Status:** ⚠️ {status}")
    
    if entry.get('is_demo', False):
        st.sidebar.info("🎭 Demo Run")
    
    # For completed processing, check if outputs are available for download
    if status == 'completed':
        output_path = entry.get('output_path')
        
        if output_path and os.path.exists(output_path):
            # Get all files in the output directory
            all_files = []
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            if all_files:
                st.sidebar.success("📥 Results ready for download!")
                
                # Create download zip with entire output folder
                master_buffer = io.BytesIO()
                with zipfile.ZipFile(master_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as master_zip:
                    for file_path in all_files:
                        # Use relative path from output_path as the archive name
                        archive_name = os.path.relpath(file_path, output_path)
                        with open(file_path, 'rb') as f:
                            master_zip.writestr(archive_name, f.read())
                
                st.sidebar.download_button(
                    f"📥 Download Results",
                    master_buffer,
                    f"{uid}_outputs.zip",
                    mime="application/zip"
                )
                
                st.sidebar.write(f"📊 {len(all_files)} files ready")
            else:
                st.sidebar.warning("⚠️ Output folder is empty")
        else:
            st.sidebar.warning("⚠️ Results expired (deleted from temp storage)")
    
    check_processing_status(uid, db)

def streamlit_server(db: TinyDB):
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
    
    # Reserve space for file uploader
    file_uploader_placeholder = st.sidebar.empty()

    # UID Status Checker
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Check Processing Status")
    uid_input = st.sidebar.text_input("Enter your UID:", placeholder="e.g., 12345678-1234-1234-1234-123456789abc")
    if st.sidebar.button("Check Status"):
        if uid_input.strip():
            # Display status in sidebar below the input
            check_processing_status_sidebar(uid_input, db)
        else:
            st.sidebar.error("Please enter a UID")
    
    st.write("# MEDPSeg (https://github.com/MICLab-Unicamp/medpseg)")
    st.write("## Welcome to the MEDPSeg online demo!")
    if not inference_2d and not inference_3d:
        st.write("Select an inference mode in the sidebar.")
    elif inference_3d:
        if st.session_state.get('processing_3d_started', False):
            st.info("✅ 3D processing started! Use the UID checker to monitor progress.")
        else:
            st.write("### Online CPU-based 3D Inference")
            st.write("3D inference is running in the CPU for this demo.")
            st.write("To setup MEDPSeg on your machine for GPU accelerated volumetric inference, check our README at https://github.com/MICLab-Unicamp/medpseg")
            st.write("When using MEDPSeg locally with a CUDA enbaled GPU and NifT volumetric scans, you can generate results as shown below in 1 minute (transparent green: lung, blue: pulmonary artery, red: airway, yellow: opacities, purple: consolidations):")
            st.write("To request a (slow) online 3D inference, use the sidebar upload widget.")
            coll, colr = st.columns(2)
            with coll:
                st.image(VOL_DIS, caption="MEDPSeg results in coronacases_003 CT from the CoronaCases dataset.")
            with colr:
                st.image(VOL_RESP, caption="MEDPSeg results in ID 013 contrast enhanced CT from the PARSE dataset.")

        with file_uploader_placeholder.container():
            # Use a processing flag to control uploader behavior
            if not st.session_state.get('processing_3d_started', False):
                input_file = st.file_uploader("Upload a 3D .nii.gz here for processing.", type=[".nii.gz", ".nii"])
            else:
                input_file = None
                st.info("✅ 3D processing started! Use the UID checker to monitor progress.")
                if st.button("🔄 Upload Another File", key="reset_3d_upload"):
                    st.session_state['processing_3d_started'] = False
                    st.rerun()
        
        dl_button = st.sidebar.empty()
        if input_file is not None:
            dl_button.write(f"Processing {input_file.name}... Scroll down to check output logs. Results will be available for download here.")
            st.session_state['processing_3d_started'] = True
            run_image(input_file, dl_button, volumetric=True, db=db)
    elif inference_2d:
        st.write("### 2D Inference")
        guide_text()
        with file_uploader_placeholder.container():
            # Use a processing flag to control uploader behavior
            if not st.session_state.get('processing_2d_started', False):
                input_file = st.file_uploader("Upload a 2D .png/.jpg here for processing.", type=[".png", ".jpg", ".jpeg"])
            else:
                input_file = None
                st.info("✅ 2D processing completed! Upload another file if needed.")
                if st.button("🔄 Upload Another File", key="reset_2d_upload"):
                    st.session_state['processing_2d_started'] = False
                    st.rerun()
        
        dl_button = st.sidebar.empty()
        sample = st.button("Click here to process a sample image...")
        st.write("Or upload you own image using the sidebar upload widget!")
        
        if sample:
            dl_button.write(f"Processing sample image... Scroll down to check output logs. Results will be available for download here. You can interrupt at any time by uploading your image.")
            demo(dl_button, db)
        elif input_file is not None:
            dl_button.write(f"Processing {input_file.name}... Scroll down to check output logs. Results will be available for download here.")
            st.session_state['processing_2d_started'] = True
            run_image(input_file, dl_button, volumetric=False, db=db)

    st.sidebar.write("This demo is possible thanks to the support of [NeuralMind](https://neuralmind.ai/).")
    st.sidebar.write("Check our [paper](https://arxiv.org/abs/2312.02365) to learn more about MEDPSeg.")
    
    # Admin Panel in expander at the bottom
    with st.sidebar.expander("🔐 Admin Panel"):
        admin_password = st.text_input("Admin Password:", type="password", placeholder="Enter admin password", key="admin_password")
        if st.button("Access Admin Panel", key="admin_button"):
            if admin_password and check_admin_password(admin_password):
                st.session_state.admin_authenticated = True
                st.success("✅ Admin access granted!")
            elif admin_password:
                st.error("❌ Invalid password")
                st.session_state.admin_authenticated = False
            else:
                st.error("Please enter password")
    
    # Show admin panel if authenticated
    if st.session_state.get('admin_authenticated', False):
        with st.container():
            display_admin_panel(db)

if __name__ == "__main__":
    streamlit_server()
