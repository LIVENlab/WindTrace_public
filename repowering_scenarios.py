


import re
import sys
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import WindTrace_onshore
from typing import Optional, Literal, Tuple
import bw2data as bd
import bw2io as bi
import consts




spold_files = (r"C:\Users\1439891\OneDrive - UAB\Documentos\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets")
bd.projects.set_current("lci_model")
bi.bw2setup()
if "cutoff391" not in bd.databases:
    ei = bi.SingleOutputEcospold2Importer(spold_files, "cutoff391", use_mp=False)
    ei.apply_strategies()
    ei.write_database()
cutoff391 = bd.Database("cutoff391")
if 'new_db' not in bd.databases:
    new_db = bd.Database('new_db')
    new_db.register()
new_db = bd.Database('new_db')

#load scenarios


for scenario in scenarios:
    lci_repowering()
    if scenario[]
        life_extension
    if scenario []
