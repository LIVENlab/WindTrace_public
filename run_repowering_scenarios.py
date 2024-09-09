#this file allows running different scenarios in WindTrace repowering
import pandas as pd
from geopy.distance import geodesic
import WindTrace_onshore
from typing import Optional, Literal, Tuple
import bw2data as bd
import bw2io as bi
import consts

# variables to be set by the user
PROJECT_NAME = 'repowering_Cabril_I'

#### Upload table with scenario data
path_scenario_data = r'C:\Users\34660\UAB\LIVENlabTEAMS - JW4A - JW4A\WP4 - labs\Portugal\Repowering\WindTrace repowering\input_scenario_data.csv'

table_scenarios = pd.read_csv(path_scenario_data,
                              index_col = 0) #SEGURAMENT S'HA D'EDITAR AIXÒ

#### LCA settings ####
# LCI data - Ecoinvent file path
SPOLD_FILES = r"C:\Users\1439891\OneDrive - UAB\Documentos\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets" #r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
method_name = 'EF v3.1'


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