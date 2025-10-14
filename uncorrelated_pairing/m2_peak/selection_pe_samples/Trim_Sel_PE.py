import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

m1_min = 20
m2_min = 3
m2_max = 50
p_cut = 0.5

input_filename = "/mnt/ceph/users/sroy1/GWTC4/Selection_Samples_With_Mock_PE.h5"
output_filename = "Trimmed_Selection_File.h5"

with h5py.File(input_filename, 'r') as h5f:

    injections_group = h5f["injections"]
    injections = pd.DataFrame({col: injections_group[col][...] for col in injections_group.keys()})

    selected_rows = []
    pe_group = h5f["injections-pe"]
        
    for i in tqdm(range(len(injections))):
        m1 = np.array(pe_group[f"Source_Frame_m1{i}"])
        m2 = np.array(pe_group[f"Source_Frame_m2{i}"])

        mask = (m1 > m1_min) & (m2 > m2_min) & (m2 < m2_max)
        frac = mask.sum() / len(m1)

        if frac > p_cut:
            selected_rows.append(i)

    injections = injections.iloc[selected_rows].reset_index(drop=True)

with h5py.File(input_filename, "r") as infile:
    info_group = infile["info"]
    
    with h5py.File(output_filename, "w") as outfile:
        new_info_group = outfile.create_group("info")
        for name, dataset in info_group.items():
            new_info_group.create_dataset(name, data=dataset[()])

        df_group = outfile.create_group("injections")
        for col in injections.columns:
            df_group.create_dataset(col, data=injections[col].values)