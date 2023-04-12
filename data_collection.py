import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np

root_folder_path = "455_LungCT\LungCT-Diagnosis"

output_folder_path = "Saved_DCM_Files"

# Create a dictionary to store the pixel arrays by patient ID
patient_pixel_arrays = {}

# Loop through all the subfolders in the root folder
for subdir, dirs, files in os.walk(root_folder_path):

    # Loop through all the files in the subfolder
    for file in files:

        # Check if the file is a .dcm file
        if file.endswith('.dcm'):

            # Read the DICOM file using pydicom
            file_path = os.path.join(subdir, file)
            dicom_image = dicom.dcmread(file_path)

            # Get the patient ID from the DICOM metadata
            patient_id = dicom_image.PatientID

            # Get the pixel array from the DICOM image
            pixel_array = dicom_image.pixel_array

            # Check if the patient ID is already in the dictionary
            if patient_id in patient_pixel_arrays:
                patient_pixel_arrays[patient_id].append(pixel_array)
            else:
                patient_pixel_arrays[patient_id] = [pixel_array]

            # Close the DICOM image to free up memory

# Loop through the patient IDs in the dictionary
for patient_id in patient_pixel_arrays:

    # Create a subfolder for the patient ID in the output folder
    patient_folder_path = os.path.join(output_folder_path, patient_id)
    os.makedirs(patient_folder_path, exist_ok=True)

    # Loop through the pixel arrays for the patient ID
    for index, pixel_array in enumerate(patient_pixel_arrays[patient_id]):
        
        # Save the pixel array as a numpy file to the patient folder with an index in the filename
        output_file_path = os.path.join(patient_folder_path, f"{patient_id}_{index}.npy")
        np.save(output_file_path, pixel_array)
