# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 17:33:22 2021

@author: pami
"""


import os

PATH = input("Drag parent folder: \n").strip("'")

paths = []
for PATH, dirs, __ in os.walk(PATH, topdown=False):
    for name in dirs:
        if "cylinder" in name and "fail" not in name.lower():
            paths.append((os.path.join(PATH, name)))

for path in paths:
    files = [file for file in os.listdir(path) if "png" in file]
    if files == []:
        continue

    print(f"Fixing {path}...")
    try:
        os.mkdir(os.path.join(path, "L"))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(path, "R"))
    except FileExistsError:
        pass

    for file in files:
        try:
            if "(2)" in file:
                os.rename(os.path.join(path, file),
                          os.path.join(path, "R", file))
            else:
                os.rename(os.path.join(path, file),
                          os.path.join(path, "L", file))
        except FileNotFoundError:
            pass
