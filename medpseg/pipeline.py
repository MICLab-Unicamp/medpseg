'''
Copyright (c) Diedre Carmo, Medical Imaging Computing Lab (MICLab)
https://miclab.fee.unicamp.br/
https://github.com/MICLab-Unicamp/medpseg
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Main processing pipeline. Big function that does every pipeline step. 
I tried to keep it specific to the current version processing, but some old unused paths might still be in here
'''
import os
import datetime
import numpy as np
import subprocess
import imageio
import site
import pandas
from typing import List
import SimpleITK as sitk
from collections import defaultdict
from multiprocessing import Queue

# Still here from V1
from medpseg.image_read import read_preprocess

# Still here from V2
from medpseg.seg_pipeline import SegmentationPipeline

# V3
from medpseg.poly_pipeline import PolySegmentationPipeline, get_connected_components

# Lober addition
from medpseg.lober_monai import LoberModule
from monai.transforms import SaveImaged

# V4: Reorganization, documentation, removal of unused code
import traceback
from medpseg import check_weight


def int_to_8bit_rgb(arrays: np.ndarray, backgrounds: np.ndarray, save_path: str, slicify: bool):
    '''
    Transforms a (1, x, y) int16 ndarray into a uint8 rgb
    mapped to a color_map and saves as png at save_path with background in background
    Only runs in pseudo volumes
    '''
    COLOR_MAP = [[0, 0, 0], # black
                 [255, 0, 0], # red
                 [0, 255, 0], # green
                 [0, 0, 255], # blue
                 [0, 255, 255], # light blue
                 [255, 255, 0], # yellow
                 [255, 0, 255]] # purple
    # Check if background is a pseudo-3d single slice volume
    if slicify:
        directory = save_path.replace(".png", '')
        os.makedirs(directory, exist_ok=True)

    if backgrounds is not None and (backgrounds.shape[0] == 1 or slicify):
        for j, (array, background) in enumerate(zip(arrays, backgrounds)):
            rgb_array = np.zeros_like(array, shape=(array.shape[-2], array.shape[-1], 3), dtype=np.uint8)
            rgb_array_only = np.zeros_like(array, shape=(array.shape[-2], array.shape[-1], 3), dtype=np.uint8)
            rgb_array[:, :, 0] = background
            rgb_array[:, :, 1] = background
            rgb_array[:, :, 2] = background

            for i, color_map in enumerate(COLOR_MAP):
                if i == 0:
                    # Skip black 
                    continue
                mask = array == i
                rgb_array[mask] = np.array(color_map)
                rgb_array_only[mask] = np.array(color_map)

            if slicify:
                # Save with background
                imageio.imwrite(os.path.join(directory, os.path.basename(save_path.replace(".png", f"_{j}.png"))), rgb_array)
                # Save without background
                imageio.imwrite(os.path.join(directory, os.path.basename(save_path.replace(".png", f"_{j}_only.png"))), rgb_array_only)
            else:
                # Save with background
                imageio.imwrite(save_path, rgb_array)
                # Save without background
                imageio.imwrite(save_path.replace(".png", "_only.png"), rgb_array_only)


def pipeline(runlist: List[str], 
             batch_size: int, 
             output_path: str, 
             display: bool, 
             info_q: Queue, 
             cpu: bool, 
             windows_itksnap_path: str, 
             linux_itksnap_path: str, 
             debug: bool, 
             act: bool,
             post: bool,
             min_hu: int = -1024,
             max_hu: int = 600,
             slicify: bool = False,
             lobe_seg: bool = True,
             cli: bool = True):  
    
    try:
        # General exception wrapper, sends it through info_q and quits if anything goes wrong
        if debug:
            pkg_path = 'medpseg'
        else:
            pkg_path = os.path.join(site.getsitepackages()[int((os.name=="nt")*1)], "medpseg")
        
        assert len(runlist) > 0, "No file found on given input path."

        # MEDPSeg initialization
        poly_weight = os.path.join(pkg_path, "poly_medseg_25d_fix.ckpt")
        check_weight(poly_weight)
        poly_model = PolySegmentationPipeline(weight=poly_weight,
                                              batch_size=batch_size, cpu=cpu, output_dir=output_path, post=post, cli=cli)

        # We still use the poly_lung part of the old pipeline
        poly_lung_weight = os.path.join(pkg_path, "poly_lung.ckpt")
        check_weight(poly_lung_weight)
        model = SegmentationPipeline(best_3d=poly_lung_weight,  # Only this weight is still being used, rest is None
                                     best_25d=None,
                                     best_25d_raw=None,
                                     airway=None,
                                     parse=None,
                                     batch_size=batch_size,
                                     cpu=cpu,
                                     n=None)
        
        # Load Jean's lobe segmentor, and its needed monai_saver
        if lobe_seg:
            lober_weight = os.path.join(pkg_path, "lober.ckpt")
            check_weight(lober_weight)
            lober = LoberModule.load_from_checkpoint(lober_weight, map_location="cpu")
            monai_saver = SaveImaged(keys=["image"],
                                     meta_keys=["image_meta_dict"],
                                     output_ext=".nii.gz",
                                     output_dir=output_path,
                                     output_dtype=np.uint8,
                                     output_postfix="lobes",
                                     separate_folder=False)

        info_q.put(("write", "Succesfully initialized all models"))
        runlist_len = len(runlist)

        # Output CSV will hold report information
        output_csv = defaultdict(list)

        # Iterate and operate on each input file separately
        for i, run in enumerate(runlist):
            info_q.put(("write", f"Loading and Pre-processing {run}..."))

            # Statistics
            non_medical_format = False
            if isinstance(run, list):
                ID = os.path.basename(os.path.dirname(run[0]))
            elif run.endswith((".nii", ".nii.gz", ".dcm")):
                ID = os.path.basename(run).replace(".nii", '').replace(".gz", '').replace(".dcm", '')
            elif run.endswith((".png", ".jpg", ".jpeg")):
                # Assume normalized with -1024/600 clip and 0-1. Give biiig warning.
                info_q.put(("write", "WARNING: Processing image file, will assume 0-255 uint8 values mapped from HU clipped to -1024/600 and 0-1 min max normalization"))
                info_q.put(("write", "WARNING: Results might be wrong if the correct normalization wasn't performed before hand. To avoid this, use the original NifT or DICOM files."))
                info_q.put(("write", "WARNING: For best results, make sure this image is an axial slice, with the bed in the bottom part of the image."))
                ID = os.path.basename(run).replace(".png", '').replace(".jpg", '').replace(".jpeg", '')
                non_medical_format = True

            info_q.put(("write", f"ID {ID}..."))

            if non_medical_format:
                # If non medical format, save as medical format in temporary directory
                data = imageio.imread(run)
                
                # Adjust number of dimensions and axes
                if data.ndim == 2:
                    data = np.expand_dims(data, 0)
                elif data.ndim == 3:
                    channel_last = np.argmin(data.shape) != 0
                    if channel_last:
                        data = data.transpose(2, 0, 1)
                else:
                    raise ValueError("Unsupported number of dimensions in non medical format.")
            
                # Adjust number of channels
                C = data.shape[0]
                new_data = np.zeros_like(data, shape=(1, data.shape[-2], data.shape[-1]))
                if C == 1:
                    new_data = data
                elif C == 4:
                    new_data = data[:3].mean(axis=0, keepdims=True)
                elif C == 3:
                    new_data = data.mean(axis=0, keepdims=True)
                else:
                    raise ValueError(f"Unsupported number of image channels: {C}")
                
                data = new_data
                
                # Revert supposed preprocessing
                data = data.astype(np.float32)/255.0
                data = data*(max_hu - min_hu) + min_hu
                data = np.round(data).astype(np.int16)
                nslices = 3

                original_image = sitk.GetImageFromArray(data)
                spacing = original_image.GetSpacing()
                directions = original_image.GetDirection()
                origin = original_image.GetOrigin()
                run = os.path.join(output_path, f"{ID}_medpseg_reverse_engineered.nii.gz")
                sitk.WriteImage(original_image, run)

                info_q.put(("write", f"WARNING: Adjusted 2D input to pseudo-3D image shape: {data.shape}. Intensities were reversed engineered, check correctnes in {run}."))

            if not non_medical_format and lobe_seg:
                # Perform lobe segmentation
                info_q.put(("write", "Performing lobe segmentation pipeline..."))
                lobe_dict = lober.predict(run, cpu=cpu)
                info_q.put(("write", "Segmentation done, lobe segmentation cleanup..."))
        
                # Save raw lobe
                monai_saver(lobe_dict)
                del lobe_dict

                info_q.put(("write", "Lobe segmentation pipeline finished."))
            else:
                info_q.put(("write", "Lobe segmentation disabled"))

            data, _, origin, spacing, directions, original_image = read_preprocess(run, norm=True)
            nslices = data.shape[0]
            
            # Batchfy read volume (5 dimensions)
            data = data.unsqueeze(0).unsqueeze(0)

            # Direction array 
            dir_array = np.asarray(directions)

            # Initialize ggo and consolidation
            ggo, consolidation, left_right_label, left_lung_volume, right_lung_volume = None, None, None, None, None

            if not non_medical_format:
                # Perform Left and Right poly lung segmentation
                left_right_label, left_lung_volume, right_lung_volume = model(input_volume=data, spacing=spacing, tqdm_iter=info_q, minimum_return=False, act=False, lung_only=True)
            else:
                info_q.put(("write", "Left and right lung localization doesn't work in 2D inputs."))
                slice_background: np.ndarray = (data.numpy()[0][0]*255).astype(np.uint8)
                background_path = os.path.join(output_path, ID + "_medpseg_reverse_engineered.png")
                imageio.imwrite(background_path, slice_background[0])

            # Perform MEDPSeg segmentation
            poly_out, airway, parse = poly_model(data, info_q, 8 if act else None, original_image, ID, dir_array, act)

            # Lung ensemble deactivated, LR lung is polymorphic unet, lung is medpseg aggregation
            lung_25d_activations = poly_out[1:].sum(axis=0) 
            lung = (lung_25d_activations > 0.5).astype(np.int32)
            del lung_25d_activations

            if lung.max() == 0:
                info_q.put(("write", f"\nCritical WARNING: Lung not found in image. Are you sure this image is of a chest CT scan?\n"))
            
            info_q.put(("write", "Post-processing, saving outputs and report."))
            lung = get_connected_components(lung, return_largest=2)[0].astype(np.uint8)
            
            # Activations and Post processing                    
            poly_out = poly_out.argmax(axis=0).astype(np.uint8)

            # Left right localizaiton processing
            if left_right_label is not None:
                left_right_label = left_right_label.argmax(axis=0).astype(np.uint8)

                # Save left and right masks. np.uint8 has the same memory footprint as np.bool
                left_lung = (left_right_label == 1).astype(np.uint8)
                right_lung = (left_right_label == 2).astype(np.uint8)
            else:
                # Running in 2D mode
                left_lung, right_lung = None, None

            # Filter by lung defined by summing the whole polymorphic output
            poly_out = poly_out * lung  # filter by lung ensemble
            
            # Separate unhealthy and lesion separations into 3 arrays
            covid = (poly_out > 1).astype(np.uint8)
            ggo = (poly_out == 2).astype(np.uint8)
            consolidation = (poly_out == 3).astype(np.uint8)
            
            # 90 % progress on iteration loading bar
            info_q.put(("iterbar", 90))

            # Use voxel volume to calculate volume in mm3, with subsequent involvement calculations.
            # VOI: Volume of Involvement, POI: Percentage of Involvement
            voxvol = (spacing[0]*spacing[1]*spacing[2])/1e+6  # voxel volume in Liters
            airway_volume = round(airway.sum()*voxvol, 3)
            parse_volume = round(parse.sum()*voxvol, 3)

            # Volumes not rounded for precision with operations
            lung_volume = lung.sum()
            ggo_volume = ggo.sum()
            consolidation_volume = consolidation.sum()
            covid_volume = covid.sum()

            # What is the % of the lung involved by unhealthy tissue? Main POI metric.
            try:
                lung_ocupation = round((covid_volume/lung_volume)*100, 2)
            except:
                lung_ocupation = None
            
            # Left VOI
            try:
                left_f_v = round((covid*left_lung).sum()*voxvol, 3)
            except:
                left_f_v = None
            
            # Right VOI
            try:
                right_f_v = round((covid*right_lung).sum()*voxvol, 3)
            except:
                right_f_v = None

            # GGO volume (rounded)
            try:
                ggo_vol = round(ggo_volume*voxvol, 3)
            except:
                ggo_vol = None

            # Left GGO VOI
            try:
                left_ggo_vol = round((ggo*left_lung).sum()*voxvol, 3)
            except:
                left_ggo_vol = None
            
            # Right GGO VOI
            try:
                right_ggo_vol = round((ggo*right_lung).sum()*voxvol, 3)
            except:
                right_ggo_vol = None

            # GGO POI
            try:
                ggo_occupation = round((ggo_volume/lung_volume)*100, 2)
            except:
                ggo_occupation = None

            # Left GGO POI
            try:
                left_ggo_oc = round((left_ggo_vol/left_lung_volume)*100, 2)
            except:
                left_ggo_oc = None
            
            # Right GGO POI
            try:
                right_ggo_oc = round((right_ggo_vol/right_lung_volume)*100, 2)
            except:
                right_ggo_oc = None

            # Consolidation volume rounded)
            try:
                consolidation_vol = round(consolidation_volume*voxvol, 3)
            except:
                consolidation_vol = None

            # Left consolidation volume
            try:
                left_consolidation_vol = round((consolidation*left_lung).sum()*voxvol, 3)
            except:
                left_consolidation_vol = None
            
            # Right consolidation volume
            try:
                right_consolidation_vol = round((consolidation*right_lung).sum()*voxvol, 3)
            except:
                right_consolidation_vol = None
            
            # Consolidation POI
            try:
                consolidation_occupation = round((consolidation_volume/lung_volume)*100, 2)
            except:
                consolidation_occupation = None
            
            # Left consolidaiton POI
            try:
                left_consolidation_oc = round((left_consolidation_vol/left_lung_volume)*100, 2)
            except:
                left_consolidation_oc = None
            
            # Right consolidation POI
            try:
                right_consolidation_oc = round((right_consolidation_vol/right_lung_volume)*100, 2)
            except:
                right_consolidation_oc = None

            # Findings volume(rounded)
            try:
                f_v = round(covid_volume*voxvol, 3)
            except:
                f_v = None

            # Left findings POI
            try:
                l_o = round(left_f_v*100/left_lung_volume, 2)
            except:
                l_o = None

            # Right findings POI
            try:
                r_o = round(right_f_v*100/right_lung_volume, 2)
            except:
                r_o = None

            # Total lung volume added post hoc for exception treatment
            try:
                total_lung_volume = left_lung_volume + right_lung_volume
            except:
                total_lung_volume = None

            output_csv["Path"].append(run)
            output_csv["ID"].append(ID)
            
            output_csv["Lung Volume (L)"].append(total_lung_volume)
            output_csv["Left Lung Volume (L)"].append(left_lung_volume)
            output_csv["Right Lung Volume (L)"].append(right_lung_volume)
            
            output_csv["Airway Volume (L)"].append(airway_volume)
            output_csv["Vessel Volume (L)"].append(parse_volume)

            output_csv["Lung Findings Volume (L)"].append(f_v)
            output_csv["Left Lung Findings Volume (L)"].append(left_f_v)
            output_csv["Right Lung Findings Volume (L)"].append(right_f_v)
            output_csv["POI (%)"].append(lung_ocupation)
            output_csv["Left POI (%)"].append(l_o)
            output_csv["Right POI (%)"].append(r_o)
            
            output_csv["GGO Volume (L)"].append(ggo_vol)
            output_csv["GGO Left Volume (L)"].append(left_ggo_vol)
            output_csv["GGO Right Volume (L)"].append(right_ggo_vol)
            output_csv["GGO POI (%)"].append(ggo_occupation)
            output_csv["GGO Left POI (%)"].append(left_ggo_oc)
            output_csv["GGO Right POI (%)"].append(right_ggo_oc)
            
            output_csv["Consolidation Volume (L)"].append(consolidation_vol)
            output_csv["Consolidation Left Volume (L)"].append(left_consolidation_vol)
            output_csv["Consolidation Right Volume (L)"].append(right_consolidation_vol)
            output_csv["Consolidation POI (%)"].append(consolidation_occupation)
            output_csv["Consolidation Left POI (%)"].append(left_consolidation_oc)
            output_csv["Consolidation Right POI (%)"].append(right_consolidation_oc)

            output_csv["Voxel volume (mm³)"].append(voxvol)
            output_csv["NSlices"].append(nslices)

            # Undo lungmask style flips
            if len(dir_array) == 9:
                airway = np.flip(airway, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                parse = np.flip(parse, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                
                lung = np.flip(lung, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                covid = np.flip(covid, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                
                try:
                    left_right_label = np.flip(left_right_label, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                    ggo = np.flip(ggo, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                    consolidation = np.flip(consolidation, np.where(dir_array[[0,4,8]][::-1]<0)[0]).copy()
                except:
                    pass
            
            # The "eye candy" file, everything!
            merged = lung + ggo + consolidation*2
            merged[airway==1] = 4
            merged[parse==1] = 5
            
            ## Create save paths and save ##
            # output_input_path = os.path.join(output_path, ID + "_input.nii.gz")
            output_lung_path = os.path.join(output_path, ID + "_lung.nii.gz")
            output_lr_lung_path = os.path.join(output_path, ID + "_lr_lung.nii.gz")
            output_findings_path = os.path.join(output_path, ID + "_findings.nii.gz")
            output_merged_path = os.path.join(output_path, ID + "_all_segmentations.nii.gz")
            output_ggo_path = os.path.join(output_path, ID + "_ggo.nii.gz")
            output_consolidation_path = os.path.join(output_path, ID + "_consolidation.nii.gz")
            output_airway = os.path.join(output_path, ID + "_airway.nii.gz")
            output_parse = os.path.join(output_path, ID + "_vessel.nii.gz")

            # Create images
            # Reverting spacing back for saving
            spacing = spacing[::-1]
            writer = sitk.ImageFileWriter()

            # If slicify, slice_background is the whole original volume.
            # This is independent of non_medical_format processing 
            if slicify:
                slice_background = (data.squeeze().numpy()*255).astype(np.uint8)
            elif not non_medical_format:
                slice_background = None

            # Save airway image
            airway_image = sitk.GetImageFromArray(airway)
            airway_image.SetDirection(directions)
            airway_image.SetOrigin(origin)
            airway_image.SetSpacing(spacing)
            writer.SetFileName(output_airway)
            writer.Execute(airway_image)
            int_to_8bit_rgb(airway, slice_background, output_airway.replace(".nii.gz", ".png"), slicify)
            del airway
            
            # Save lung image
            lung_image = sitk.GetImageFromArray(lung)
            lung_image.SetDirection(directions)
            lung_image.SetOrigin(origin)
            lung_image.SetSpacing(spacing)
            writer.SetFileName(output_lung_path)
            writer.Execute(lung_image)
            int_to_8bit_rgb(lung, slice_background, output_lung_path.replace(".nii.gz", ".png"), slicify)
            del lung

            # Save parse image
            parse_image = sitk.GetImageFromArray(parse)
            parse_image.SetDirection(directions)
            parse_image.SetOrigin(origin)
            parse_image.SetSpacing(spacing)
            writer.SetFileName(output_parse)
            writer.Execute(parse_image)
            int_to_8bit_rgb(parse, slice_background, output_parse.replace(".nii.gz", ".png"), slicify)
            del parse

            try:
                # Save lr lung image
                lr_lung_image = sitk.GetImageFromArray(left_right_label)
                lr_lung_image.SetDirection(directions)
                lr_lung_image.SetOrigin(origin)
                lr_lung_image.SetSpacing(spacing)
                writer.SetFileName(output_lr_lung_path)
                writer.Execute(lr_lung_image)
                del left_right_label
            except Exception as e:
                print(f"Left right lung localization not running.")

            # Save findings image
            covid_image = sitk.GetImageFromArray(covid)
            covid_image.SetDirection(directions)
            covid_image.SetOrigin(origin)
            covid_image.SetSpacing(spacing)
            writer.SetFileName(output_findings_path)
            writer.Execute(covid_image)
            int_to_8bit_rgb(covid, slice_background, output_findings_path.replace(".nii.gz", ".png"), slicify)
            del covid

            # Save lung and findings image
            merged_image = sitk.GetImageFromArray(merged)
            merged_image.SetDirection(directions)
            merged_image.SetOrigin(origin)
            merged_image.SetSpacing(spacing)
            writer.SetFileName(output_merged_path)
            writer.Execute(merged_image)
            int_to_8bit_rgb(merged, slice_background, output_merged_path.replace(".nii.gz", ".png"), slicify)
            del merged

            try:
                # GGO image
                ggo_image = sitk.GetImageFromArray(ggo)
                ggo_image.SetDirection(directions)
                ggo_image.SetOrigin(origin)
                ggo_image.SetSpacing(spacing)
                writer.SetFileName(output_ggo_path)
                writer.Execute(ggo_image)
                int_to_8bit_rgb(ggo, slice_background, output_ggo_path.replace(".nii.gz", ".png"), slicify)
                del ggo

                # Consolidation image
                consolidation_image = sitk.GetImageFromArray(consolidation)
                consolidation_image.SetDirection(directions)
                consolidation_image.SetOrigin(origin)
                consolidation_image.SetSpacing(spacing)
                writer.SetFileName(output_consolidation_path)
                writer.Execute(consolidation_image)
                int_to_8bit_rgb(consolidation, slice_background, output_consolidation_path.replace(".nii.gz", ".png"), slicify)
                del consolidation
            except Exception as e:
                print(f"GGO and Consolidation not present {e}.")

            try:
                # Filter lobe by saved lung. Hopefully reading will be aligned.
                lung_image = sitk.ReadImage(output_lung_path)
                lung = sitk.GetArrayFromImage(lung_image)
                output_lobes_path = output_lung_path.replace("_lung.nii.gz", "_lobes.nii.gz")
                lobes_image = sitk.ReadImage(output_lobes_path)
                lobes = sitk.GetArrayFromImage(lobes_image)
                if post:
                    info_q.put(("write", f"Filtering lobes by lung segmentation."))
                    lobes = lobes*lung
                int_to_8bit_rgb(lobes, slice_background, output_lobes_path.replace(".nii.gz", ".png"), slicify)
                lobes_image_fixed = sitk.GetImageFromArray(lobes)
                lobes_image_fixed.CopyInformation(lobes_image)
                sitk.WriteImage(lobes_image_fixed, output_lobes_path)
                del lung

                # Use filtered lobes and re-read covid to compute POI per lobe
                covid_image = sitk.ReadImage(output_findings_path)
                covid = sitk.GetArrayFromImage(covid_image)
            except Exception as e:
                print(f"Lobe segmentation not being done in this run.")

            # Output_csv logs, Lobe statistics.
            for c, lobe_str in enumerate(["LUL", "LLL", "RUL", "RML", "RLL"]):
                try:
                    lobe = (lobes == (c + 1)).astype(np.uint8)
                    lobe_volume = lobe.sum()
                    try:
                        lobe_poi = round(((covid*lobe).sum()/lobe_volume)*100, 2)
                    except:
                        lobe_poi = None
                except:
                    lobe_poi = None
                output_csv[f"{lobe_str} POI (%)"].append(lobe_poi)
            
            if not non_medical_format:
                # If those are still instantiated delete
                try:
                    del lobes
                except Exception:
                    pass

                try:
                    del covid
                except Exception:
                    pass
            
            # Lobe statistics done, everything saved!
            info_q.put(("iterbar", 100))
            info_q.put(("write", f"Processing finished {run}."))

            if display:
                # ITKSnap
                info_q.put(("write", "Displaying results with itksnap.\nClose itksnap windows to continue."))
                try:
                    itksnap_output_path = output_merged_path
                    if os.name == "nt":
                        subprocess.Popen([windows_itksnap_path, "-g", run, "-s", itksnap_output_path])
                    else:
                        subprocess.Popen([linux_itksnap_path, "-g", run, "-s", itksnap_output_path])
                    
                except Exception as e:
                    info_q.put(("write", "Error displaying results. Do you have itksnap installed?"))
                    print(e)
                
            info_q.put(("generalbar", (100*i+100)/runlist_len))
            info_q.put(("write", f"{i+1} volumes processed out of {runlist_len}.\nResult are on the {output_path} folder."))
        uid = str(datetime.datetime.today()).replace('.', '').replace(':', '').replace('-', '').replace(' ', '')

        output_csv_path = os.path.join(output_path, f"medseg_run_statistics_{uid}.csv")
        pandas.DataFrame.from_dict(output_csv).to_csv(output_csv_path)
        info_q.put(("write", f"Sheet with pulmonary involvement statistics saved in {output_csv_path}."))
        info_q.put(None)
    except Exception as e:
        traceback.print_exc()
        info_q.put(e)
        return
    