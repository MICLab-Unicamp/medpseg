'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab) 
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

# External imports
import os
import site
import time
import glob
import argparse
import threading
import numpy as np
import multiprocessing as mp
import tkinter as tk
from queue import Empty
from PIL import ImageTk, Image
from typing import List, Union
from tkinter import messagebox
from tkinter import HORIZONTAL, VERTICAL, Tk, Text, PhotoImage, Canvas, NW
from tkinter.ttk import Progressbar, Button, Label, Style, Scrollbar
from tkinter.filedialog import askopenfilename, askdirectory
from torchvision.transforms import InterpolationMode, Resize

# Internal imports
from medpseg.pipeline import pipeline
from medpseg.monitoring import get_stats
from medpseg.utils import DummyTkIntVar

# Set global variable for icon.png location
if os.name == "nt":
    ICON_ORIGINAL_PNG = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "icon_original.png")
    ICON_PNG = os.path.join(site.getsitepackages()[1], "medpseg", "assets", "icon.png")
else:
    ICON_ORIGINAL_PNG = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "icon_original.png")
    ICON_PNG = os.path.join(site.getsitepackages()[0], "medpseg", "assets", "icon.png")
    
# Default title global variable
DEFAULT_TITLE = "Modified EfficientDet for Polymorphic Pulmonary Segmentation (MEDPSeg)"


def file_dialog(dir: bool = False):
    '''
    Simple Tkinter GUI to chose files or folders if dir is True
    '''
    Tk().withdraw() 
    if dir:
        filename = askdirectory()
    else:
        filename = askopenfilename()  

    if isinstance(filename, str) and len(filename) > 0:
        return filename
    else: 
        return None


def alert_dialog(msg: str, title: str = DEFAULT_TITLE):
    '''
    Shows msg as an alert dialog
    '''
    Tk().withdraw()
    messagebox.showinfo(title, msg)


def error_dialog(msg: str, title: str = DEFAULT_TITLE):
    '''
    Shows msg as an error dialog
    '''
    Tk().withdraw()
    messagebox.showerror(title, msg)


def confirm_dialog(msg: str, title: str = DEFAULT_TITLE):
    '''
    Simple confirmation dialog
    '''
    Tk().withdraw()
    MsgBox = messagebox.askquestion(title, msg)
    if MsgBox == 'yes':
        return True
    else:
        return False


class MainWindow(threading.Thread):
    '''
    Abstracts the GUI management as a Thread.
    '''
    def __init__(self, args: argparse.Namespace, info_q: mp.Queue):
        '''
        args: command line arguments
        info_q: queue communication highway with worker threads
        '''
        super().__init__()
        self.args = args
        self.info_q = info_q

        # Runlist stores lists of files to run in pipeline
        self.runlist: List[str] = None

        # Self pipeline stores instance of pipeline class which can implement many trained models
        self.pipeline: object = None

        # Simple resize transform to be used for displaying images in the GUI
        self.resizer = Resize((512, 512), interpolation=InterpolationMode.NEAREST)

        # When input and output folders are given through CLI, do not invoke the GUI
        self.cli = self.args.input_folder is not None and self.args.output_folder is not None
        
        if self.cli:
            print("Input and output given through CLI, not invoking GUI.")
            assert os.path.exists(self.args.input_folder), f"Input folder {self.args.input_folder} doesn't exist."
            self.input_path = self.args.input_folder
            
            # Standins for variables that usually come from the GUI
            self.lobe_seg = DummyTkIntVar(value=int(not self.args.disable_lobe))
            self.display = DummyTkIntVar(value=0)
            self.act = DummyTkIntVar(value=0)
            self.post = DummyTkIntVar(value=int(self.args.post))

            # Storage for loading bars of general and iteration progress
            self.general_progress = {}
            self.iter_progress = {}

            # Populate list of files to run pipeline using CLI
            self.populate_runlist()

        # Start GUI thread
        self.start()
        
    def output_selection(self):
        '''
        Selects where the output will be. Can already be populated by CLI.
        '''
        if self.args.output_folder is None:
            alert_dialog("Double click (enter) the folder where you want to save the segmentation output.")
            output_folder = file_dialog(dir=True)
            if output_folder is None:
                # None means closed dialog most likely
                return None
            self.write_to_textbox(f"\nWill save outputs in {output_folder}.\n")
        else:
            output_folder = self.args.output_folder

        self.output_folder = output_folder

        self.write_to_textbox("\nClick start processing to start.")
    
    def start_processing(self):
        '''
        Launches pipeline thread with arguments filled by CLI or GUI

        Pipeline communication thread communicates with pipeline through que info_q for progress feedback
        '''
        # Only run if self.runlist is populated with input files
        if self.runlist is None:
            self.write_to_textbox("\nPlease load a folder or image before starting processing.\n")
            return
        
        # Separate thread for heavy processing. Threads for using less ram. Multiprocessing might be faster.
        # This shares computation with MainWindow GUI thread, however the overhead of GUI is hopefully worth using less RAM
        self.pipeline = threading.Thread(target=pipeline, args=(self.runlist, 
                                                                self.args.batch_size, 
                                                                self.output_folder,
                                                                bool(self.display.get()),
                                                                self.info_q,
                                                                self.args.cpu,
                                                                self.args.win_itk_path,
                                                                self.args.linux_itk_path,
                                                                self.args.debug,
                                                                bool(self.act.get()),
                                                                bool(self.post.get()),
                                                                self.args.min_hu,
                                                                self.args.max_hu,
                                                                self.args.slicify,
                                                                bool(self.lobe_seg.get()),
                                                                self.cli))
        
        # Start thread for communication between pipeline and GUI
        self.pipeline_comms_thread = threading.Thread(target=self.pipeline_comms)                                                                
        self.pipeline_comms_thread.start()

        # Start pipeline
        self.pipeline.start()

    def display_slice(self, slice: np.ndarray):
        '''
        Displays a given slice array coming from pipeline
        '''
        if not self.cli:
            pil_image = self.resizer(Image.fromarray((slice*255).astype(np.uint8)))
            self.img = ImageTk.PhotoImage(pil_image)
            self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def pipeline_comms(self):
         '''
         Communication between GUI and pipeline through Queue
         '''
         while True:
            try:
                info = self.info_q.get()
                msg_is_exception = isinstance(info, Exception)
                if info is None or msg_is_exception:
                    if msg_is_exception:
                        error_msg = f"ERROR: {str(info.__class__.__name__)}: {info}"
                        self.write_to_textbox(f"ERROR: {str(info.__class__.__name__)}: {info}")
                        self.write_to_textbox(f"Aborting processing! This was most likely an error regarding the data input or missing weights before doing pip install.")
                        self.write_to_textbox(f"Feel free to create an issue in our GitHub.")
                        if not self.cli:
                            alert_dialog(error_msg)
                    self.write_to_textbox("Closing worker thread...")
                    self.pipeline.join()
                    self.write_to_textbox("Done.")
                    self.runlist = None
                    self.pipeline = None
                    self.set_icon()
                    return
                elif isinstance(info, tuple):
                    try:
                        if info[0] == "write":
                            self.write_to_textbox(str(info[1]))
                        elif info[0] == "iterbar":
                            self.iter_progress['value'] = int(info[1])
                        elif info[0] == "generalbar":
                            self.general_progress['value'] = int(info[1])
                        elif info[0] == "slice":
                            self.display_slice(info[1])
                        elif info[0] == "icon":
                            self.set_icon()
                    except Exception as e:
                        self.write_to_textbox(f"Malformed pipeline message: {e}. Please create an issue on github.")
                        quit()
            except Empty:
                pass

    def parse_medical_files(self, input_path: Union[List[str], str]) -> List[str]:
        '''
        Look for .nii files or .dcm series in self.input_path
        to populate self.runlist. Input can be either a single file or a folder of files.

        Searchs for .nii and .dcm extensions accordingly
        '''
        if os.path.isdir(input_path):
            # Given path is a dir, look for nift or dcm files
            self.write_to_textbox(f"Searching {input_path} for files...")
            runlist = glob.glob(os.path.join(input_path, "*.nii")) + glob.glob(os.path.join(input_path, "*.nii.gz"))
            if len(runlist) == 0:
                self.write_to_textbox("Did not find NifT files. Looking for .dcm series...")
                runlist = glob.glob(os.path.join(input_path, "*.dcm"))
                dcms = [os.path.basename(x) for x in runlist]
                if len(runlist) > 0:
                    self.write_to_textbox(f"Found {dcms} DCM inside folder {input_path}.")
                    runlist = [runlist]
                else:
                    self.write_to_textbox("Could not find valid image or folder of images!")
                    runlist = None
        else:
            # If given path is just a file, the run list is a list with the file itself
            runlist = [self.input_path]
        
        return runlist

    def txt_file_to_runlist(self, input_path: str) -> List[str]:
        '''
        Parses .txt file in input_path to populate runlist
        Assume each line contains a path
        '''
        print("Detected .txt file, attempting to parse...")
        try:
            runlist = []
            with open(input_path, 'r') as text_input_file:
                for line in text_input_file:
                    runlist.append(line.strip())
        except Exception as e:
            print(f"Error trying to parse .txt input file: {e}, attempting to use input path as single input.")
            runlist = [input_path]  

        return runlist 

    def populate_runlist(self):
        '''
        Routine to parse input path into runlist
        '''
        self.runlist = None

        if not self.cli:
            # Reset loading bars
            self.general_progress['value'] = 0
            self.iter_progress['value'] = 0

        if self.input_path is None:
            pass
        # Check if path exists and is a .nii file or dcm series folder or file, and parse a runlist
        elif os.path.exists(self.input_path) and (self.input_path.endswith(".nii") or self.input_path.endswith(".nii.gz") or self.input_path.endswith(".dcm") or os.path.isdir(self.input_path)):
            self.runlist = self.parse_medical_files(self.input_path)
        # It can also be an image file for slice-wise running.
        elif os.path.exists(self.input_path) and (self.input_path.endswith(".png") or self.input_path.endswith(".jpg") or self.input_path.endswith(".jpeg")):
            self.runlist = [self.input_path]
        # For a .txt file parse it 
        elif os.path.isfile(self.input_path) and self.input_path.endswith('.txt'):
            self.runlist = self.txt_file_to_runlist(self.input_path)
        
        if self.runlist is None:
            error_str = "No valid volume or folder given, please give a nift volume, dcm volume, dcm series folder, folder with NifTs, or a preprocessed grayscale 8-bit (0-255) .png/.jpg/.jpeg. Note we do not support running over a folder of .png/.jpg/.jpeg"
            if self.cli:
                raise ValueError(error_str)
            else:
                alert_dialog(error_str)
        else:
            self.write_to_textbox(f"Runlist: {self.runlist}.\n{len(self.runlist)} volumes detected.")

    def write_to_textbox(self, s):
        '''
        Writes to GUI textbox or prints in CLI mode
        '''
        if self.cli:
            print(s)
        else:
            self.T.insert(tk.END, f"\n{s}\n")
            self.T.see(tk.END)
        
    def load_file(self):
        '''
        Selects a file using graphical user interface
        '''
        self.input_path = file_dialog(dir=False)
        if self.input_path is None:
            # Cancel pipeline if nothing is given
            return
        self.populate_runlist()
        if self.runlist is not None:
            self.output_selection()
    
    def load_folder(self):
        '''
        Selects a folder using graphical user interface
        '''
        alert_dialog("Double click (enter) the directory with the input files.")
        self.input_path = file_dialog(dir=True)
        if self.input_path is None:
            # Cancel pipeline if nothing is given
            return
        self.populate_runlist()
        if self.runlist is not None:
            self.output_selection()
    
    def on_closing(self):
        '''
        Event when X GUI button is clicked
        '''
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.pipeline is not None and self.pipeline.is_alive():
                self.write_to_textbox("Closing...")
                self.pipeline.join()
                self.write_to_textbox("Done.")
            self.ws.quit()

    def monitoring_loop(self):
        '''
        Loop executing the self.monitoring function, to be run in a Thread
        '''
        while True:
            self.monitoring()
            time.sleep(0.1)

    def monitoring(self):
        '''
        Gets hardware stats
        '''
        stats = get_stats()
        for key, value in stats.items():
            getattr(self, key)['value'] = value

    def set_icon(self):
        '''
        Sets the program Icon as the image in the image visualizer
        '''
        if not self.cli:
            self.img = ImageTk.PhotoImage(self.resizer(Image.open(ICON_PNG)))
            self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def run(self):
        '''
        Design intent:
            - Plain window, with image/folder loading button and start processing button. 
            - TQDM progress somehow reflected on gui bar
            - Text box with debug output
            - Run in thread

        Runs when GUI thread is called, even in CLI mode
        '''
        if self.cli:
            # In CLI mode, output might have not exist yet, create it
            if self.args.output_folder is not None:
                os.makedirs(self.args.output_folder, exist_ok=True)
                self.write_to_textbox(f"Results will be in the '{self.args.output_folder}' folder")
            self.output_folder = self.args.output_folder

            # All arguments are already defined in CLI, just start processing!
            self.start_processing() 
            self.pipeline.join()
        else:
            # Here all GUI elements are initialized. WS is the "top-level window"
            self.ws = Tk()
            icon = PhotoImage(file=ICON_ORIGINAL_PNG)
            self.ws.iconphoto(False, icon)
            self.ws.title(DEFAULT_TITLE)
            self.ws.geometry('1800x600')  # default for 1080p, could be adjustable in the future

            # Canvas for showing Images
            self.canvas = Canvas(self.ws, width=512, height=512)
            self.canvas.pack(side='right')
            self.set_icon()

            # Text output, the text box!
            scroll = Scrollbar(self.ws)
            self.T = Text(self.ws, height=20, width=60, font=("Sans", 14), yscrollcommand=scroll.set)        
            scroll.config(command=self.T.yview)
            scroll.pack(side='right', fill='y')
            self.T.pack(side='top', fill='both')

            # Initialization text, including initializing output
            self.write_to_textbox(f"Welcome to MEDPSeg! {DEFAULT_TITLE}")
            self.write_to_textbox('Check "Display" to attempt to display results using ITKSnap (default off).')
            self.write_to_textbox('The "Post" option selects the largets connected component of airway and vessel segmentation. This can cause problems in low-resolution scans (default off).')
            self.write_to_textbox('The "Save act." option saves activations and attention maps on the output folder, uses more RAM (default off).')
            self.write_to_textbox('The "Lobe seg." option performs 3D lobe segmentation with a trained VNet. Includes per lobe reports in the output .csv sheet. Adds around 1 minute more processing time per scan, using a GPU (default off).')
            if self.args.output_folder is not None:
                os.makedirs(self.args.output_folder, exist_ok=True)
                self.write_to_textbox(f"Results will be in the '{self.args.output_folder}' folder")
            if self.args.cpu:
                self.write_to_textbox(f"Forcing CPU usage. Prediction might take a while.")
            
            # Initialize loading bars, general progress and iteration progress.
            general_progress = Label(self.ws, text="General Progress")
            general_progress.pack(side='bottom')
            self.general_progress = Progressbar(self.ws, orient=HORIZONTAL, length=600, mode='determinate')
            self.general_progress.pack(side='bottom', fill='x')
            iter_progress = Label(self.ws, text="Processing Progress")
            iter_progress.pack(side='bottom')
            self.iter_progress = Progressbar(self.ws, orient=HORIZONTAL, length=600, mode='determinate')
            self.iter_progress.pack(side='bottom', fill='x')

            # Monitoring bars for CPU, GPU, RAM and VRAM
            cpu_label = Label(self.ws, text="CPU")
            cpu_label.pack(side='left')
            self.cpu = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.cpu.pack(side='left')
            
            gpu_label = Label(self.ws, text="GPU")
            gpu_label.pack(side='left')
            self.gpu = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.gpu.pack(side='left')
            
            ram_label = Label(self.ws, text="RAM")
            ram_label.pack(side='left')
            self.cpu_ram = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.cpu_ram.pack(side='left')
            
            vram_label = Label(self.ws, text="VRAM")
            vram_label.pack(side='left')
            self.gpu_ram = Progressbar(self.ws, orient=VERTICAL, length=60, mode='determinate')
            self.gpu_ram.pack(side='left')

            # Initializing checkboxes for GUI arguments
            self.display = tk.IntVar(value=0)
            c1 = tk.Checkbutton(self.ws, text='Display', variable=self.display, onvalue=1, offvalue=0, state='active')
            c1.config(font=("Sans", "14"))
            c1.pack(side='left')

            self.act = tk.IntVar(value=0)
            c3 = tk.Checkbutton(self.ws, text='Save act.', variable=self.act, onvalue=1, offvalue=0, state='active')
            c3.config(font=("Sans", "14"))
            c3.pack(side='left')

            self.post = tk.IntVar(value=0)
            c4 = tk.Checkbutton(self.ws, text='Post', variable=self.post, onvalue=1, offvalue=0, state='active')
            c4.config(font=("Sans", "14"))
            c4.pack(side='left')

            self.lobe_seg = tk.IntVar(value=0)
            c5 = tk.Checkbutton(self.ws, text='Lobe seg.', variable=self.lobe_seg, onvalue=1, offvalue=0, state='active')
            c5.config(font=("Sans", "14"))
            c5.pack(side='left')

            # Initializing buttons for actions (load and start processing)
            boldStyle = Style ()
            boldStyle.configure("Bold.TButton", font = ('Sans','10','bold'))
            Button(self.ws, text='Start processing', command=self.start_processing, style="Bold.TButton").pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)        
            Button(self.ws, text='Load image ', command=self.load_file).pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)
            Button(self.ws, text='Load folder', command=self.load_folder).pack(side='right', ipady=10, pady=10, ipadx=5, padx=5)
            
            # Register on closing window event
            self.ws.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start hardware monitoring daemon
            monitoring = threading.Thread(target=self.monitoring_loop)
            monitoring.daemon = True
            monitoring.start()

            # Start GUI mainloop. Since this is running in a Thread there is no blocking of other related processes.
            self.ws.mainloop()
    