#this file allows running different scenarios in WindTrace repowering
import pandas as pd
from geopy.distance import geodesic
import WindTrace_onshore
from typing import Optional, Literal, Tuple
import bw2data as bd
import bw2io as bi
import consts

# variables to be set by the user
PROJECT_NAME = 'repowering_Cabril_I' ###també  posat a consts.py (perque és d'on ho treu la funció)

#### LCA settings ####
# LCI data - Ecoinvent file path
SPOLD_FILES = r"C:\Users\1439891\OneDrive - UAB\Documentos\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets" #r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
method_name = 'EF v3.1'

### define path scenario data
path_scenario_data = r"C:\Users\1439891\UAB\LIVENlabTEAMS - JW4A - JW4A\WP4 - labs\Portugal\Repowering\WindTrace repowering\input_scenario_data.csv"

#######

#### Upload table with scenario data
table_scenarios = pd.read_csv(path_scenario_data,
                              index_col = 0,
                              sep = ";",
                              dtype = {"scenario_name": str,
                                       "scenario_name2": str,
                                       "extension_long": bool,
                                       "extension_short": bool,
                                       "substitution": bool,
                                       "repowering": bool,
                                       "park_name_i" : str,
                                       "park_power_i" : float,
                                       "number_of_turbines_i": int,
                                       "park_location_i" : str, #'PT'
                                       "park_coordinates_i" : str,
                                       "manufacturer_i" : str, # ['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind']
                                       "rotor_diameter_i" : float, #m
                                       "turbine_power_i": float, #MW
                                       "hub_height_i": float, #m
                                       "commissioning_year_i" : int,
                                       "recycled_share_steel_1" : float, #0.4
                                       "lifetime_i": int,
                                       "electricity_mix_steel_i": str, #'Norway', 'Europe', 'Poland'
                                       "generator_type_i" : str, #'dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'
                                       "lifetime_extension": int,
                                       "number_of_turbines_extension": int,
                                       "cf_extension": float,
                                       "attrition_rate_extension": float,
                                       "lifetime_substitution": int,
                                       "number_of_turbines_substitution": int,
                                       "cf_substitution": float,
                                       "attrition_rate_substitution": float,
                                       "recycled_share_steel_extension": float, #0.4
                                       "park_power_repowering": float, #MW
                                       "number_of_turbines_repowering": int,
                                       "manufacturer_repowering": str, #['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind']
                                       "rotor_diameter_repowering": float, #m
                                       "turbine_power_repowering": float, #MW
                                       "hub_height_repowering": float, #m
                                       "generator_type_repowering": str, #'dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'
                                       "electricity_mix_steel_repowering": str, #'Norway', 'Europe', 'Poland'
                                       "lifetime_repowering": int, #years
                                       "recycled_share_steel_repowering": float,
                                       "cf_repowering": float,
                                       "attrition_rate_repowering" : float,
                                       "land_use_permanent_intensity_repowering": float,
                                       "land_cover_type_repowering": str, #'industrial'
                                       "eol_scenario_repowering": int,
                                       "eol": bool,
                                       "transportation": bool,
                                       "use_and_maintenance": bool,
                                       "installation": bool}) #SEGURAMENT S'HA D'EDITAR AIXÒ


########
scenario_list = list(table_scenarios.columns)

for scenario in scenario_list :
    scenario_data = table_scenarios[scenario]
    lci_repowering(extension_long = scenario_data["extension_long"],
                    extension_short = scenario_data["extension_short"],
                    substitution = scenario_data["substitution"],
                    repowering = scenario_data["repowering"],
                    park_name_i = scenario_data["scenario_name2"], #temporary solution to multiple scenarios in one park: include scenario name (format: park_scenario)
                    park_power_i = scenario_data["park_power_1"],
                    number_of_turbines_i = scenario_data["number_of_turbines_i"],
                    park_location_i = scenario_data["park_location_i"],
                    park_coordinates_i = scenario_data["park_coordinates_i"],
                    manufacturer_i = scenario_data["manufacturer_i"],
                    rotor_diameter_i = scenario_data["rotor_diameter_i"],
                    turbine_power_i = scenario_data["turbine_power_i"],
                    hub_height_i = scenario_data["hub_height_i"],
                    commissioning_year_i = scenario_data["commissioning_year_i"],
                    recycled_share_steel_i = scenario_data["recycled_share_steel_i"],
                    lifetime_i = scenario_data["lifetime_i"],
                    electricity_mix_steel_i = scenario_data["electricity_mix_steel_i"],
                    generator_type_i = scenario_data["generator_type_i"],
                    lifetime_extension = scenario_data["lifetime_extension"],
                    number_of_turbines_extension = scenario_data["number_of_turbines_extension"],
                    cf_extension = scenario_data["cf_extension"],
                    attrition_rate_extension = scenario_data["attrition_rate_extension"],
                    lifetime_substitution = scenario_data["lifetime_substitution"],
                    number_of_turbines_substitution = scenario_data["number_of_turbines_substitution"],
                    cf_substitution = scenario_data["cf_substitution"],
                    attrition_rate_substitution = scenario_data["attrition_rate_substitution"],
                    recycled_share_steel_extension = scenario_data["recycled_share_steel_extension"],
                    park_power_repowering = scenario_data["park_power_repowering"],
                    number_of_turbines_repowering = scenario_data["number_of_turbines_repowering"],
                    manufacturer_repowering = scenario_data["manufacturer_repowering"],
                    rotor_diameter_repowering = scenario_data["rotor_diameter_repowering"],
                    turbine_power_repowering = scenario_data["turbine_power_repowering"],
                    hub_height_repowering = scenario_data["hub_height_repowering"],
                    generator_type_repowering = scenario_data["generator_type_repowering"],
                    electricity_mix_steel_repowering = scenario_data["electricity_mix_steel_repowering"],
                    lifetime_repowering = scenario_data["lifetime_repowering"],
                    recycled_share_steel_repowering = scenario_data["recycled_share_steel_repowering"],
                    cf_repowering = scenario_data["cf_repowering"],
                    attrition_rate_repowering = scenario_data["attrition_rate_repowering"],
                    land_use_permanent_intensity_repowering = scenario_data["land_use_permanent_intensity_repowering"],
                    land_cover_type_repowering = scenario_data["land_cover_type_repowering"],
                    eol_scenario_repowering = scenario_data["eol_scenario_repowering"],
                    eol = scenario_data["eol"],
                    transportation = scenario_data["transportation"],
                    use_and_maintenance = scenario_data["use_and_maintenance"],
                    installation = scenario_data["installation"])

    extension = extension_short or extension_long

    lci_excel_output(park_name = scenario_data["scenario_name2"], #temporary solution to multiple scenarios in one park: include scenario name (format: park_scenario)
                     extension = extension,
                     repowering = scenario_data["repowering"],
                     substitution = scenario_data["substitution"],
                     park_power_repowering = scenario_data["park_power_repowering"],
                     scenario_name = scenario_data["scenario_name"],
                     method_name = method_name)