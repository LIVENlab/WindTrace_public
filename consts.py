import os

# When possible, RER or Europe without Switzerland locations have been selected
MATERIALS_EI_ACTIVITY_CODES = {
    'Low alloy steel':
        {'name': 'market for steel, low-alloyed', 'location': 'GLO', 'reference product': 'steel, low-alloyed'},
    'Low alloy steel_foundations':
        {'name': 'market for reinforcing steel', 'location': 'GLO', 'reference product': 'reinforcing steel'},
    'Chromium steel':
        {'name': 'market for steel, chromium steel 18/8', 'location': 'GLO', 'reference product': 'steel, chromium steel 18/8'},
    'Chromium steel_foundations':
        {'name': 'market for steel, chromium steel 18/8', 'location': 'GLO', 'reference product': 'steel, chromium steel 18/8'},
    'Cast iron':
        {'name': 'market for cast iron', 'location': 'GLO', 'reference product': 'cast iron'},
    'Aluminium':
        {'name': 'market for aluminium, wrought alloy', 'location': 'GLO', 'reference product': 'aluminium, wrought alloy'},
    'Copper':
        {'name': 'market for copper, cathode', 'location': 'GLO', 'reference product': 'copper, cathode'},
    'Copper_foundations':
        {'name': 'market for copper, cathode', 'location': 'GLO', 'reference product': 'copper, cathode'},
    'Epoxy resin':
        {'name': 'market for epoxy resin, liquid', 'location': 'RER', 'reference product': 'epoxy resin, liquid'},
    'Rubber':
        {'name': 'market for synthetic rubber', 'location': 'GLO', 'reference product': 'synthetic rubber'},
    'PUR':
        {'name': 'market for polyurethane, rigid foam', 'location': 'RER', 'reference product': 'polyurethane, rigid foam'},
    'PVC':
        {'name': 'market for polyvinylchloride, bulk polymerised', 'location': 'GLO', 'reference product': 'polyvinylchloride, bulk polymerised'},
    'PE':
        {'name': 'market for polyethylene, high density, granulate', 'location': 'GLO', 'reference product': 'polyethylene, high density, granulate'},
    'Fiberglass':
        {'name': 'market for glass fibre reinforced plastic, polyamide, injection moulded', 'location': 'GLO', 'reference product': 'glass fibre reinforced plastic, polyamide, injection moulded'},
    'electronics':
        {'name': 'market for electronics, for control units', 'location': 'GLO', 'reference product': 'electronics, for control units'},
    'Electrics':
        {'name': 'cable production, unspecified', 'location': 'GLO', 'reference product': 'cable, unspecified'},
    'Lubricating oil':
        {'name': 'market for lubricating oil', 'location': 'RER', 'reference product': 'lubricating oil'},
    'Ethyleneglycol':
        {'name': 'market for ethylene glycol', 'location': 'GLO', 'reference product': 'ethylene glycol'},
    'Praseodymium':
        {'name': 'market for praseodymium oxide', 'location': 'GLO', 'reference product': 'praseodymium oxide'},
    'Neodymium':
        {'name': 'market for neodymium oxide', 'location': 'GLO', 'reference product': 'neodymium oxide'},
    'Dysprosium':
        {'name': 'market for dysprosium oxide', 'location': 'GLO', 'reference product': 'dysprosium oxide'},
    'Terbium':
        {'name': 'market for terbium oxide', 'location': 'GLO', 'reference product': 'terbium oxide'},
    'Boron':
        {'name': 'market for boron carbide', 'location': 'GLO', 'reference product': 'boron carbide'},
    'Concrete_foundations':
        {'name': 'market group for concrete, normal strength', 'location': 'GLO', 'reference product': 'concrete, normal strength'}
}

EOL_EI_ACTIVITY_CODES = {
    'Low alloy steel':
        {'landfill':
             {'name': 'treatment of scrap steel, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'},
         'incineration':
             {'name': 'treatment of scrap steel, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'}},
    'Low alloy steel_foundations':
        {'landfill':
             {'name': 'treatment of scrap steel, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'},
         'incineration':
             {'name': 'treatment of scrap steel, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'}},
    'Chromium steel':
        {'landfill':
             {'name': 'treatment of scrap steel, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'},
         'incineration':
             {'name': 'treatment of scrap steel, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'}},
    'Chromium steel_foundations':
        {'landfill':
             {'name': 'treatment of scrap steel, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'},
         'incineration':
             {'name': 'treatment of scrap steel, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'}},
    'Cast iron':
        {'landfill':
             {'name': 'treatment of scrap steel, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'},
         'incineration': {'name': 'treatment of scrap steel, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap steel'}},
    'Aluminium':
        {'landfill':
             {'name': 'treatment of waste aluminium, sanitary landfill', 'location': 'CH', 'reference product': 'waste aluminium'},
         'incineration':
             {'name': 'treatment of scrap aluminium, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap aluminium'}},
    'Copper':
        {'landfill':
             {'name': 'treatment of copper slag, residual material landfill', 'location': 'GLO', 'reference product': 'copper slag'},
         'incineration':
             {'name': 'treatment of scrap copper, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap copper'}},
    'Copper_foundations':
        {'landfill':
             {'name': 'treatment of copper slag, residual material landfill', 'location': 'GLO', 'reference product': 'copper slag'},
         'incineration':
             {'name': 'treatment of scrap copper, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'scrap copper'}},
    'Epoxy resin':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': 'treatment of waste polyurethane, municipal incineration', 'location': 'CH', 'reference product': 'waste polyurethane'}},
    'Rubber':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': 'treatment of waste rubber, unspecified, municipal incineration', 'location': 'Europe without Switzerland', 'reference product': 'waste rubber, unspecified'}},
    'PUR':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': 'treatment of waste polyurethane, municipal incineration', 'location': 'CH', 'reference product': 'waste polyurethane'}},
    'PVC':
        {'landfill':
             {'name': 'treatment of waste polyvinylchloride, sanitary landfill', 'location': 'CH', 'reference product': 'waste polyvinylchloride'},
         'incineration':
             {'name': 'treatment of waste polyvinylchloride, municipal incineration', 'location': 'CH', 'reference product': 'waste polyvinylchloride'}},
    'PE':
        {'landfill':
             {'name': 'treatment of waste polyethylene, sanitary landfill', 'location': 'CH', 'reference product': 'waste polyethylene'},
         'incineration':
             {'name': 'treatment of waste polyethylene, municipal incineration', 'location': 'CH', 'reference product': 'waste polyethylene'}},
    'Fiberglass':
        {'landfill':
             {'name': 'treatment of waste glass, sanitary landfill', 'location': 'GLO', 'reference product': 'waste glass'},
         'incineration':
             {'name': 'treatment of waste glass, municipal incineration', 'location': 'CH', 'reference product': 'waste glass'}},
    'electronics':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Electrics':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Lubricating oil':
        {'landfill':
             {'name': None, 'location': None, 'reference product': None},
         'incineration':
             {'name': 'treatment of waste mineral oil, hazardous waste incineration, with energy recovery', 'location': 'Europe without Switzerland', 'reference product': 'waste mineral oil'}},
    'Praseodymium':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Neodymium':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Dysprosium':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Terbium':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Boron':
        {'landfill':
             {'name': 'treatment of inert waste, sanitary landfill', 'location': 'Europe without Switzerland', 'reference product': 'inert waste'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}},
    'Concrete_foundations':
        {'landfill':
             {'name': 'treatment of waste concrete, inert material landfill', 'location': 'Europe without Switzerland', 'reference product': 'waste concrete'},
         'incineration':
             {'name': None, 'location': None, 'reference product': None}}
}


MATERIAL_PROCESSING_EI_ACTIVITY_CODES = {
    'Copper':
        {'name': 'market for wire drawing, copper', 'location': 'GLO', 'reference product': 'wire drawing, copper'},
    'Aluminium':
        {'name': 'market for sheet rolling, aluminium', 'location': 'GLO', 'reference product': 'sheet rolling, aluminium'},
    'Chromium steel':
        {'name': 'market for sheet rolling, chromium steel', 'location': 'GLO', 'reference product': 'sheet rolling, chromium steel'},
    'Steel_tower_rolling':
        {'name': 'market for section bar rolling, steel', 'location': 'GLO', 'reference product': 'section bar rolling, steel'},
    'Steel_tower_welding':
        {'name': 'market for welding, arc, steel', 'location': 'GLO', 'reference product': 'welding, arc, steel'},
    'Cast iron':
        {'name': 'market for section bar rolling, steel', 'location': 'GLO', 'reference product': 'section bar rolling, steel'},
    'Zinc coating':
        {'name': 'zinc coating, pieces', 'location': 'RER', 'reference product': 'zinc coat, pieces'},
    'PVC':
        {'name': 'extrusion, plastic film', 'location': 'RER', 'reference product': 'extrusion, plastic film'},
    'PE':
        {'name': 'extrusion, plastic film', 'location': 'RER', 'reference product': 'extrusion, plastic film'}}

MANUFACTURER_LOC = {'Vestas': {1: {'country': 'DK', 'location': (54.8291253662034, 11.1212115492466)},
                               2: {'country': 'DK', 'location': (56.0227841143509, 8.38093375265438)},
                               3: {'country': 'DK', 'location': (56.0883528380133, 8.23514084205859)},
                               4: {'country': 'DK', 'location': (55.4688414575748, 10.5463625927981)},
                               5: {'country': 'DK', 'location': (56.0874350071926, 8.23751960633171)},
                               6: {'country': 'DE', 'location': (53.9353019822524, 10.8436417188164)},
                               7: {'country': 'IT', 'location': (40.5044941341608, 17.2287358289053)},
                               8: {'country': 'ES', 'location': (39.04397684446477, -3.584957140536688)},
                               9: {'country': 'GB', 'location': (50.722088190989055, -1.2868432968480525)}},
                    'Siemens Gamesa': {1: {'country': 'DK', 'location': (55.95591820576292, 9.122436361699371)},
                                       2: {'country': 'DK', 'location': (57.04323306831486, 10.033956589032089)},
                                       3: {'country': 'DE', 'location': (53.8432663362179, 8.751820362596261)},
                                       4: {'country': 'ES', 'location': (41.84642998231027, -1.9375829067207992)},
                                       5: {'country': 'ES', 'location': (43.288703816235945, -2.8593248806708322)},
                                       6: {'country': 'TR', 'location': (38.451039935268916, 27.17844981452182)},
                                       7: {'country': 'GB', 'location': (53.744468006219314, -0.30480462006378034)}},
                    'Enercon': {1: {'country': 'DE', 'location': (53.46900866545183, 7.461912729754879)},
                                2: {'country': 'DE', 'location': (52.198664285069384, 11.67724373576914)},
                                3: {'country': 'PT', 'location': (41.75127343145981, -8.688698916614396)}},
                    'Nordex': {1: {'country': 'DE', 'location': (53.67607989480183, 10.00207222593792)},
                               2: {'country': 'ES', 'location': (42.611427316908355, -1.6381422745701957)}},
                    'LM Wind': {1: {'country': 'ES', 'location': (42.56581346608258, -6.554743222531039)},
                                2: {'country': 'ES', 'location': (40.28628952931124, 0.08687809123315436)},
                                3: {'country': 'FR', 'location': (49.64941639036492, -1.5976205757000064)},
                                4: {'country': 'PL', 'location': (53.55423722170504, 14.789982179973517)},
                                5: {'country': 'TR', 'location': (39.08046596169773, 27.05453305967944)}}
                    }

# Mean share of steel production in Europe by country between 2017 and 2021 according to the
# European Steel in Figures Report 2022 (by Eurofer). Countries without gas inventory in Ecoinvent were assigned the
# inventory of a nearby country. Croatia and Slovenia were assigned Italy's inventory,
# Luxembourg was assigned Belgium's and Bulgaria was assigned Romania's.
# Europe is a net importer since 2017, but was a clear net exporter until then. Therefore, it is assumed
# that all the steel used in the turbines is European-made.
STEEL_DATA_EU27 = {
    'AT':
        {'share': 4.98,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'AT', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'AT', 'reference product': 'natural gas, high pressure'}},
    'BE':
        {'share': 4.91,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'BE', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'BE', 'reference product': 'natural gas, high pressure'}},
    'BG':
        {'share': 0.39,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'BG', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'RO', 'reference product': 'natural gas, high pressure'}},
    'CZ':
        {'share': 3.14,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'CZ', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'CZ', 'reference product': 'natural gas, high pressure'}},
    'DE':
        {'share': 26.97,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'DE', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'DE', 'reference product': 'natural gas, high pressure'}},
    'ES':
        {'share': 9.07,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'ES', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'ES', 'reference product': 'natural gas, high pressure'}},
    'FI':
        {'share': 2.61,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'FI', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'FI', 'reference product': 'natural gas, high pressure'}},
    'FR':
        {'share': 9.51,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'FR', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'FR', 'reference product': 'natural gas, high pressure'}},
    'GR':
        {'share': 0.94,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'GR', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'GR', 'reference product': 'natural gas, high pressure'}},
    'HR':
        {'share': 0.06,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'HR', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'IT', 'reference product': 'natural gas, high pressure'}},
    'HU':
        {'share': 1.11,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'HU', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'HU', 'reference product': 'natural gas, high pressure'}},
    'IT':
        {'share': 15.62,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'IT', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'IT', 'reference product': 'natural gas, high pressure'}},
    'LU':
        {'share': 1.41,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'LU', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'BE', 'reference product': 'natural gas, high pressure'}},
    'NL':
        {'share': 4.42,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'NL', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'NL', 'reference product': 'natural gas, high pressure'}},
    'PL':
        {'share': 6.14,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'PL', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'PL', 'reference product': 'natural gas, high pressure'}},
    'RO':
        {'share': 2.22,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'RO', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'RO', 'reference product': 'natural gas, high pressure'}},
    'SE':
        {'share': 3.09,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'SE', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'SE', 'reference product': 'natural gas, high pressure'}},
    'SI':
        {'share': 0.45,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'SI', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'IT', 'reference product': 'natural gas, high pressure'}},
    'SK':
        {'share': 2.97,
         'elect': {'name': 'market for electricity, medium voltage', 'location': 'SK', 'reference product': 'electricity, medium voltage'},
         'gas': {'name': 'market for natural gas, high pressure', 'location': 'SK', 'reference product': 'natural gas, high pressure'}}
}

OLD = {'AT': {'share': 4.98, 'elect_code': '6f61d8326ee98b75ab4136e87d0844c6',
                          'gas_code': '8fe71e00ac03aa41e073629a7bd4602e'},
                   'BE': {'share': 4.91, 'elect_code': '44032cfead0bec8f8f01c4303ee2aa2c',
                          'gas_code': '160194493d01c6ba3b3048c2147c9378'},
                   'BG': {'share': 0.39, 'elect_code': 'b6f1682a9b26383290e4b262b4d4f090',
                          'gas_code': '29d1c4a73720a7cdb66cb4d65900e445'},
                   'CZ': {'share': 3.14, 'elect_code': '9fe94e549ae2582efa46129959256f82',
                          'gas_code': 'a7555dee1e40b2d525daa09bc000a1cb'},
                   'DE': {'share': 26.97, 'elect_code': '6287db0c56830869ff2451fb1aa0f5f3',
                          'gas_code': 'c8de1013e1728007e0c0dfa8942b6dfb'},
                   'ES': {'share': 9.07, 'elect_code': 'f2324bfb8e5e1263effd6828b14a76b1',
                          'gas_code': 'af7929b977949d47873d533201e9c654'},
                   'FI': {'share': 2.61, 'elect_code': 'fb5e87faac73a0e194028e486c862ef1',
                          'gas_code': 'b9ed569ca351f4620cfe5b8fbd1e3bb7'},
                   'FR': {'share': 9.51, 'elect_code': '846e6df218567bf257f542dac9f3c6f7',
                          'gas_code': 'f1d5de00687347350cca9ae465fedb15'},
                   'GR': {'share': 0.94, 'elect_code': 'e5ccceae1f1eda69ad741cea2d13cc61',
                          'gas_code': '1a07bf6d0564b49db80ab54f5dc3f2c1'},
                   'HR': {'share': 0.06, 'elect_code': 'eb7379c7154fb29dcbeb6e98cfb6b6cd',
                          'gas_code': '59cf8916e90678fe85a12f5b5666972e'},
                   'HU': {'share': 1.11, 'elect_code': '9eeb0b0de8980074b490a694ee8c0b5b',
                          'gas_code': 'e7bb282744e6bbf0ab868a80ba3ce1bd'},
                   'IT': {'share': 15.62, 'elect_code': '5468040eda7cb574bb395c6e51431245',
                          'gas_code': '59cf8916e90678fe85a12f5b5666972e'},
                   'LU': {'share': 1.41, 'elect_code': '430debab821cc15c63f695cc837ff415',
                          'gas_code': '160194493d01c6ba3b3048c2147c9378'},
                   'NL': {'share': 4.42, 'elect_code': '03c0371b2d6143b9b1f74328caec3fc9',
                          'gas_code': '5fbddcb18d603dc3eb7572432db4c0ad'},
                   'PL': {'share': 6.14, 'elect_code': '57c178984cb1d22b4525fbd56906b0c6',
                          'gas_code': '84c52a94bf58be855ac22e6a538a16ba'},
                   'RO': {'share': 2.22, 'elect_code': '843cf854d71429d3a3058084ac5431d4',
                          'gas_code': '29d1c4a73720a7cdb66cb4d65900e445'},
                   'SE': {'share': 3.09, 'elect_code': '0e4b280caeeba40d5644b8d28328b0de',
                          'gas_code': '4493d6ebe7d2b80501f30fb24074e0d3'},
                   'SI': {'share': 0.45, 'elect_code': '11aaf6ce49e6061a64f7727526a2fa72',
                          'gas_code': '59cf8916e90678fe85a12f5b5666972e'},
                   'SK': {'share': 2.97, 'elect_code': 'bc95922472c85d01195a6bab4f30636e',
                          'gas_code': 'e88d630d212aae3e62d77689310ee8de'}}

# Share of secondary steel production (electric furnace)
SECONDARY_STEEL = {'other': 0.4162, '2012': 0.4304, '2013': 0.4172, '2014': 0.4080, '2015': 0.4094,
                   '2016': 0.4058, '2017': 0.4135, '2018': 0.4240, '2019': 0.4191, '2020': 0.4365, '2021': 0.4360}

PRINTED_WARNING_STEEL = False

# rare earth shares
RARE_EARTH_DICT = {'Praseodymium': {'dd_eesg': 9, 'dd_pmsg': 35, 'gb_pmsg': 4, 'gb_dfig': 0},
                   'Neodymium': {'dd_eesg': 28, 'dd_pmsg': 180, 'gb_pmsg': 51, 'gb_dfig': 12},
                   'Dysprosium': {'dd_eesg': 6, 'dd_pmsg': 17, 'gb_pmsg': 6, 'gb_dfig': 2},
                   'Terbium': {'dd_eesg': 1, 'dd_pmsg': 7, 'gb_pmsg': 1, 'gb_dfig': 0},
                   'Boron': {'dd_eesg': 0, 'dd_pmsg': 6, 'gb_pmsg': 1, 'gb_dfig': 0}
                   }

# scenarios lifetime extension, replacement, repowering
LONG_EXTENSION = {'steel': 0.08, 'c_steel': 0.08, 'iron': 1, 'aluminium': 0.65, 'copper': 0.81,
                  'plastics': 1, 'others': 1, 'foundations': 0, 'electronics_and_electrics': 0}
SHORT_EXTENSION = {'steel': 0.01, 'c_steel': 0.01, 'iron': 0.34, 'aluminium': 0, 'copper': 0,
                   'plastics': 0.91, 'others': 0, 'foundations': 0, 'electronics_and_electrics': 0}
REPLACEMENT_BASELINE = {'steel': 0.84, 'c_steel': 0.84, 'iron': 1, 'aluminium': 0.65, 'copper': 0.81,
                        'plastics': 1, 'others': 1, 'foundations': 0, 'electronics_and_electrics': 0}

# vestas_file path
cwd = os.getcwd()
VESTAS_FILE = os.path.join(cwd, 'clean_data.xlsx')

# variables to be set by the user
PROJECT_NAME = 'jupyter_notebook'
SPOLD_FILES = r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
NEW_DB_NAME = 'test'


