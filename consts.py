import os

# When possible, RER or Europe without Switzerland locations have been selected
MATERIALS_EI391_ACTIVITY_CODES = {
    'Low alloy steel': {'name': 'market for steel, low-alloyed', 'code': 'a81ce0e882f1b0ef617462fc8e7472e4'},
    'Low alloy steel_foundations': {'name': 'market for reinforcing steel', 'code': 'bd3a3818f60643fd78eccefa7b4390c1'},
    'Chromium steel': {'name': 'market for steel, chromium steel 18/8', 'code': '6af7fc101112f29fb43953d562902932'},
    'Chromium steel_foundations': {'name': 'market for steel, chromium steel 18/8',
                                   'code': '6af7fc101112f29fb43953d562902932'},
    'Cast iron': {'name': 'market for cast iron', 'code': 'e6ba5991b1ecab06c9e5ebc33af41364'},
    'Aluminium': {'name': 'market for aluminium, wrought alloy', 'code': 'd25c8e0755ee9899fb4e892990397a68'},
    'Copper': {'name': 'market for copper, cathode', 'code': '8b62f30ed586a5f23611ef196cc97b93'},
    'Copper_foundations': {'name': 'market for copper, cathode', 'code': '8b62f30ed586a5f23611ef196cc97b93'},
    'Epoxy resin': {'name': 'market for epoxy resin, liquid', 'code': '0ffc6bd2671c856cc5cc362bb1aba7b1'},
    'Rubber': {'name': 'market for synthetic rubber', 'code': 'c37f51d47f6f5d1e8d6c8d1eb61d335c'},
    'PUR': {'name': 'market for polyurethane, rigid foam', 'code': 'dfd8dafa15514464a61eb5b968e6ba86'},
    'PVC': {'name': 'market for polyvinylchloride, bulk polymerised', 'code': '342baff30ae57e0573e84257026d5e2e'},
    'PE': {'name': 'polyethylene, high density, granulate', 'code': '22db46f6ba1211c058e4da0ac386d3e7'},
    'Fiberglass': {'name': 'market for glass fibre reinforced plastic, polyamide, injection moulded',
                   'code': 'baf9fd46b7a5fc32fc070e2c1aa4674c'},
    'electronics': {'name': 'market for electronics, for control units', 'code': '23810b0bbe04fca71b0f53fc97d56360'},
    'Electrics': {'name': 'cable production, unspecified', 'code': 'e9c1cf3df7b69da66ba6c8859e33ae15'},
    'Lubricating oil': {'name': 'market for lubricating oil', 'code': '92391c8c6958ada25b22935e3fa6f06f'},
    'Ethyleneglycol': {'name': 'market for ethylene glycol', 'code': 'e229ace9c0c670aef7c1998446d5c3ca'},
    'Praseodymium': {'name': 'market for praseodymium oxide', 'code': '2569d108fa88c377eb339db47a45a03f'},
    'Neodymium': {'name': 'market for neodymium oxide', 'code': 'bfb5b92cc635472a79a03c702a79fd53'},
    'Dysprosium': {'name': 'market for dysprosium oxide', 'code': '975770f48967d9f9ea82f1ac4648f7c5'},
    'Terbium': {'name': 'market for terbium oxide', 'code': 'c44447ec9b7a72217943165e42a6e56d'},
    'Boron': {'name': 'market for boron carbide', 'code': 'db1891062e32e3ef010d13d2619c7ba3'},
    'Concrete_foundations': {'name': 'market group for concrete, normal',
                             'code': 'aa1ab624a71fdc35bdb2fabb8d02c8ec'},
    'hdpe': {'name': 'market for polyethylene, high density, granulate',
             'code': '22db46f6ba1211c058e4da0ac386d3e7'},
    'Gravel': {'name': 'market for gravel, round',
               'code': '306679b4bd7a801c5d60a2ff37fb3734'},
    'Cement': {'name': 'market for cement, Portland',
               'code': '172190d48ccdc5f567138ddafc2277f7'},
    'Lead': {'name': 'market for lead',
             'code': 'fbb0f24cb8a09b37b43d853dea138347'},
    'Paper': {'name': 'market for kraft paper',
              'code': '779418b7af49b2fdd8b2d8082ccf6a22'},
    'PP': {'name': 'market for polypropylene, granulate',
           'code': '61863071bd168004eb9af9df563661b5'},
    'Asphalt': {'name': 'market for bitumen seal',
                'code': '4b7b620b2d0bdf7250fa199dee87bc07'}
}

EOL_S1_EI391_ACTIVITY_CODES = {
    'Low alloy steel': {'landfill': {'name': 'treatment of scrap steel, inert material landfill',
                                     'code': 'f97add46fd2ba618668377d76a1b66bd'},
                        'incineration': {'name': 'treatment of scrap steel, municipal incineration',
                                         'code': '93367683cadf9f4ffdbce969a03f4c50'}},
    'Low alloy steel_foundations': {'landfill': {'name': 'treatment of scrap steel, inert material landfill',
                                                 'code': 'f97add46fd2ba618668377d76a1b66bd'},
                                    'incineration': {'name': 'treatment of scrap steel, municipal incineration',
                                                     'code': '93367683cadf9f4ffdbce969a03f4c50'}},
    'Chromium steel': {'landfill': {'name': 'treatment of scrap steel, inert material landfill',
                                    'code': 'f97add46fd2ba618668377d76a1b66bd'},
                       'incineration': {'name': 'treatment of scrap steel, municipal incineration',
                                        'code': '93367683cadf9f4ffdbce969a03f4c50'}},
    'Chromium steel_foundations': {'landfill': {'name': 'treatment of scrap steel, inert material landfill',
                                                'code': 'f97add46fd2ba618668377d76a1b66bd'},
                                   'incineration': {'name': 'treatment of scrap steel, municipal incineration',
                                                    'code': '93367683cadf9f4ffdbce969a03f4c50'}},
    'Cast iron': {'landfill': {'name': 'treatment of scrap steel, inert material landfill',  # proxy
                               'code': 'f97add46fd2ba618668377d76a1b66bd'},
                  'incineration': {'name': 'treatment of scrap steel, municipal incineration',  # proxy
                                   'code': '93367683cadf9f4ffdbce969a03f4c50'}},
    'Aluminium': {'landfill': {'name': 'treatment of waste aluminium, sanitary landfill',
                               'code': 'ee8bd09e74b8b4c4e1343b333aab7130'},
                  'incineration': {'name': 'treatment of scrap aluminium, municipal incineration',
                                   'code': 'e03c44cbb2aba9f10d107c135b7b8fad'}},
    'Copper': {'landfill': {'name': 'treatment of copper slag, residual material landfill',
                            'code': '0c9cd95424bf180649027c720b520da1'},
               'incineration': {'name': 'treatment of scrap copper, municipal incineration',
                                'code': '11373a42f15363b80251656bfcb1e55e'}},
    'Copper_foundations': {'landfill': {'name': 'treatment of copper slag, residual material landfill',
                                        'code': '0c9cd95424bf180649027c720b520da1'},
                           'incineration': {'name': 'treatment of scrap copper, municipal incineration',
                                            'code': '11373a42f15363b80251656bfcb1e55e'}},
    'Epoxy resin': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                                 'code': 'f04ff5258998b6a607333d69d377104d'},
                    'incineration': {'name': 'treatment of waste polyurethane, municipal incineration',  # proxy
                                     'code': 'b7d3a0fd6c740e74321b258671b8aa63'}},
    'Rubber': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                            'code': 'f04ff5258998b6a607333d69d377104d'},
               'incineration': {'name': 'treatment of waste rubber, unspecified, municipal incineration',
                                'code': '02468870e1d28d544ba2dfc774a7e0bf'}},
    'PUR': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                         'code': 'f04ff5258998b6a607333d69d377104d'},
            'incineration': {'name': 'treatment of waste polyurethane, municipal incineration',
                             'code': 'b7d3a0fd6c740e74321b258671b8aa63'}},
    'PVC': {'landfill': {'name': 'treatment of waste polyvinylchloride, sanitary landfill',
                         'code': 'd561874f1a0e2d39810805fdcddf6528'},
            'incineration': {'name': 'treatment of waste polyvinylchloride, municipal incineration',
                             'code': 'b98f7051abb38cf81cd4f402894e0759'}},
    'PE': {'landfill': {'name': 'treatment of waste polyethylene, sanitary landfill',
                        'code': 'f31b6ca086d5fd0ad2926e1f5be77b3f'},
           'incineration': {'name': 'treatment of waste polyethylene, municipal incineration',
                            'code': '3d0ff6a87049ddea69f6004f7f291d1e'}},
    'Fiberglass': {'landfill': {'name': 'treatment of waste glass, sanitary landfill',  # proxy
                                'code': '22f1877975b61ab2c622842dcabb8f57'},
                   'incineration': {'name': 'treatment of waste glass, municipal incineration',  # proxy
                                    'code': '4bc9d21071e8ba4901877ca4ab4ba175'}},
    'electronics': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                                 'code': 'f04ff5258998b6a607333d69d377104d'},
                    'incineration': {'name': None,
                                     'code': None}},
    'Electrics': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                               'code': 'f04ff5258998b6a607333d69d377104d'},
                  'incineration': {'name': None,
                                   'code': None}},
    'Lubricating oil': {'landfill': {'name': None,
                                     'code': None},
                        'incineration': {
                            'name': 'treatment of waste mineral oil, hazardous waste incineration, with energy recovery',
                            'code': 'ad6d0f2a8b45536da196238df879077b'}},
    'Praseodymium': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                                  'code': 'f04ff5258998b6a607333d69d377104d'},
                     'incineration': {'name': None,
                                      'code': None}},
    'Neodymium': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                               'code': 'f04ff5258998b6a607333d69d377104d'},
                  'incineration': {'name': None,
                                   'code': None}},
    'Dysprosium': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                                'code': 'f04ff5258998b6a607333d69d377104d'},
                   'incineration': {'name': None,
                                    'code': None}},
    'Terbium': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                             'code': 'f04ff5258998b6a607333d69d377104d'},
                'incineration': {'name': None,
                                 'code': None}},
    'Boron': {'landfill': {'name': 'treatment of inert waste, sanitary landfill',  # proxy
                           'code': 'f04ff5258998b6a607333d69d377104d'},
              'incineration': {'name': None,
                               'code': None}},
    'Concrete_foundations': {'landfill': {'name': 'treatment of waste concrete, inert material landfill',
                                          'code': '373d58f6430eb96257b7eef4a077f750'},
                             'incineration': {'name': None,
                                              'code': None}}
}

MATERIAL_PROCESSING_EI391_ACTIVITY_CODES = {
    'Copper': {'name': 'market for wire drawing, copper', 'code': 'd6369e6d8436b2dae14e1320d15428ef'},
    'Aluminium': {'name': 'market for sheet rolling, aluminium', 'code': 'acf665d13787149d4c1e71a8ae475b41'},
    'Chromium steel': {'name': 'market for sheet rolling, chromium steel', 'code': '43c86f918884ea7799596c841f5121fc'},
    'Steel_tower_rolling': {'name': 'market for section bar rolling, steel',
                            'code': 'd4a112fba0d7d6ef32c25cae044a5010'},
    'Steel_tower_welding': {'name': 'market for welding, arc, steel', 'code': 'f0f521d7de0e50b4a2014eec8800b2b7'},
    'Cast iron': {'name': 'market for section bar rolling, steel', 'code': 'd4a112fba0d7d6ef32c25cae044a5010'},
    'Zinc coating': {'name': 'zinc coating, pieces', 'code': '429ee8a600b97b759e5a0e9b45d4642f'},
    'PVC': {'name': 'extrusion, plastic film', 'code': 'd5adcf93135d0fc061c8bad0200f4cd1'},
    'PE': {'name': 'extrusion, plastic film', 'code': 'd5adcf93135d0fc061c8bad0200f4cd1'}
}

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
steel_data_EU27 = {'AT': {'share': 4.98, 'elect_code': '6f61d8326ee98b75ab4136e87d0844c6',
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
secondary_steel = {'other': 0.4162, '2012': 0.4304, '2013': 0.4172, '2014': 0.4080, '2015': 0.4094,
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
PROJECT_NAME = 'repowering'
SPOLD_FILES = r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
NEW_DB_NAME = 'new_db'


