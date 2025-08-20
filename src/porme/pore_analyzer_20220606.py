# %% defs
# -*- coding: utf-8 -*-
# Code best viewed at 100 characters per line
"""
# SPDX-License-Identifier: MIT

Created on Thu Jun 17 16:04:00 2021

@author: José I. Contreras Raggio
"""
from copy import deepcopy as copy
from multiprocessing import Pool
from functools import partial
import pickle
import traceback
import time
import os
import pandas as pd
# import cv2
import analyzer_funcs as f
from analyzer_funcs import kill_cv2, show


# %% Global init
if __name__ == "__main__":
    if "__file__" in dir():
        og_path = os.path.dirname(__file__)
    else:
        og_path = os.getcwd()

    # Especify max amount of concurrent processes to run
    # This should only be changed if you have too many cores, somehow
    # Else just leave it set to None, which will use the amount of cores in the system by default
    max_processes = None
    _start = time.time()
    if "paths.txt" not in os.listdir():
        print("Please create a 'paths.txt' file and add the desired folder paths to analyze there")
        print("Add each folder path in a different line")
        input("Press any key to exit...")
        exit()
    with open("paths.txt", "r") as _paths_file:
        PATHS = _paths_file.read().splitlines()
    if len(PATHS) == 0:
        print("No paths were found in paths.txt")
        print("Add each folder path in a different line")
        input("Press any key to exit...")
        exit()

    PATHS = [os.path.normpath(_path) for _path in PATHS]
    paths = []
    for _root in PATHS:
        for _PATH, _dirs, __ in os.walk(_root, topdown=False):
            for _name in _dirs:
                _path = os.path.join(_PATH, _name)
                if any([cond in _path.lower() for cond in ("fail", "other", "weird")]):
                    continue
                elif _path[-1].lower() in ("l", "r"):
                    paths.append(os.path.normpath(_path))
                    print(_path)
    # paths = paths[0:16]

    CROP_DIMS, GLOBAL_COUNT = f.analyzer(paths[0], fast=True, verbose=False)
    data = []

# %% Processing
    print(f"Processing {len(paths)} image sets")
    # print("More information available on Anaconda prompt")
    print("This process can take a while...")
    with Pool(processes=max_processes) as _exe:
        _analyzer = partial(f.analyzer, CROP=CROP_DIMS, COUNT=GLOBAL_COUNT)
        for _num, _data in enumerate(_exe.imap(_analyzer, paths)):
            data.append(_data)
            print(f"Finished {paths[_num]} ({_num+1}/{len(paths)})")

    _df = pd.DataFrame
    _paths2 = [res for res in data if not isinstance(res[0], _df)]
    _paths2 = [res[2] for res in _paths2 if res[0] == "crop retry"]
    _path_v = [path for path in _paths2]

    while len(_paths2):
        print("------------")
        print(f"{len(_paths2)} samples remaining...")
        print("------------")
        CROP_DIMS = None
        CROP_DIMS, GLOBAL_COUNT = f.analyzer(_paths2[0], fast=True, verbose=False)

        with Pool(processes=max_processes) as _exe:
            data2 = []
            _analyzer = partial(f.analyzer, CROP=CROP_DIMS, COUNT=GLOBAL_COUNT)
            for _num, _data in enumerate(_exe.imap(_analyzer, _paths2)):
                data2.append(_data)
                print(f"Finished {_paths2[_num]} ({_num+1}/{len(_paths2)})")

        for _count, _res in enumerate(data2):
            if isinstance(_res[0], _df) or _res[0] != "crop retry":
                data[paths.index(_paths2[_count])] = _res

        _paths2 = [res for res in data2 if not isinstance(res[0], _df)]
        _paths2 = [res[2] for res in _paths2 if res[0] == "crop retry"]
        _path_v = [path for path in _paths2]

    _end = time.time()
    print(f"Process took {round(_end - _start)} seconds")
    print("----------")

# %% Saving
    print("Do you wish to save the .csv files?")
    _save = input("[Y]/n: ").lower()
    while _save not in ("y", "n", ""):
        _save = input("y/n: ").lower()

    if _save in ("y", ""):
        while True:
            try:
                savepath = os.path.join(og_path, "Processed CSV")
                print("Save files to")
                print(savepath)
                print("?")

                _save = input("[Y]/n: ").lower()
                while _save not in ("y", "n", ""):
                    _save = input("y/n: ").lower()

                if _save == "n":
                    savepath = ""
                    while not os.path.exists(savepath):
                        print("Input an existing folder path")
                        print("Note that several CSVs may be written:")
                        savepath = input("Path: ")


                for _num, dataf in enumerate(data):
                    if isinstance(dataf[0], pd.DataFrame):
                        dataf[0].to_csv(
                            os.path.join(savepath, ("-").join(paths[_num].split("\\")[-2:]) + ".csv"),
                                         index=False)

            except PermissionError:
                print("Permission denied for one or more files")
                print("Close all currently open files in the folder and try again")
                print("----------")
            except OSError:
                print("The savepath appears to be non-existant")
                print("Select a new path")
                print("----------")
            except Exception:
                print("There was an unknown error saving the files:")
                print("----------")
                traceback.print_exc()
                print("----------")
                print("Do you wish to save a pickled version of the data?")
                print("It can be opened with Python to further investigation")

                _save = input("[Y]/n: ").lower()
                while _save not in ("y", "n", ""):
                    _save = input("y/n: ").lower()
                if _save == "n":
                    print("Closing program in 5 seconds...")
                    print("Press Ctrl+C to abort")
                    exit()

                with open("params_exp.pkl", "wb") as file:
                    pickle.dump(data, file)
                print(f"File saved as 'params_exp.pkl' in {os.getcwd()}")
                break
            else:
                print("Files saved successfully")
                break

    print("----------")
    input("Press any key to finish...")

# %% only for debugging purposes
# if __name__ == "__main__":
#     # _analyzer = partial(f.analyzer, CROP=CROP_DIMS, COUNT=GLOBAL_COUNT)
#     res = f.analyzer(
#         paths[56],
#         # fast=_fastflag,
#         CROP=CROP_DIMS,
#         COUNT=GLOBAL_COUNT,
#         verbose=True,
#     )

# # # # f.analyzer(paths[100], CROP=CROP_DIMS, COUNT=GLOBAL_COUNT, verbose=True)
