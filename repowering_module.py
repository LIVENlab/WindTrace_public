import os
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


def lci_repowering(extension_long: bool, extension_short: bool, substitution: bool, repowering: bool,
                   park_name_i: str, park_power_i: float, number_of_turbines_i: int,
                   park_location_i: str,
                   park_coordinates_i: tuple,
                   manufacturer_i: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                   rotor_diameter_i: float,
                   turbine_power_i: float, hub_height_i: float, commissioning_year_i: int,
                   recycled_share_steel_i: float = None,
                   lifetime_i: int = 20,
                   electricity_mix_steel_i: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                   generator_type_i: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'] = 'gb_dfig',
                   lifetime_extension: int = None, number_of_turbines_extension: int = None,
                   cf_extension: float = None, attrition_rate_extension: float = 0.009,
                   lifetime_substitution: int = None, number_of_turbines_substitution: int = None,
                   cf_substitution: float = None, attrition_rate_substitution: float = 0.009,
                   recycled_share_steel_extension: float = None,
                   park_power_repowering: float = None,
                   number_of_turbines_repowering: int = None,
                   manufacturer_repowering: str = None,
                   rotor_diameter_repowering: float = None,
                   turbine_power_repowering: float = None,
                   hub_height_repowering: float = None,
                   generator_type_repowering: Optional[Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']] = 'gb_dfig',
                   electricity_mix_steel_repowering: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                   lifetime_repowering: int = 25,
                   recycled_share_steel_repowering: float = None,
                   cf_repowering: float = 0.24,
                   attrition_rate_repowering: float = 0.009,
                   land_use_permanent_intensity_repowering: int = 3000,
                   land_cover_type_repowering: str = 'industrial',
                   eol_scenario_repowering: int = 1,
                   eol: bool = True, transportation: bool = True, use_and_maintenance: bool = True,
                   installation: bool = True
                   ):
    # test input parameters
    if extension_short and extension_long:
        print('WARNING. You chose to have s short and a long extension and that is not possible. '
              'Choose either one, the other or none, by setting extension_long and extension_short '
              'variables to True or False')
        sys.exit()
    if (extension_short or extension_long) and (number_of_turbines_extension > number_of_turbines_i):
        print(f'WARNING. The initial number of turbines was {number_of_turbines_i}. The number of turbines extended '
              f'cannot be bigger than this, but you introduced {number_of_turbines_extension} turbines to be extended.')
        sys.exit()
    elif (extension_short or extension_long) and (number_of_turbines_extension < number_of_turbines_i):
        print(f'WARNING. You introduced less turbines to be extended ({number_of_turbines_extension}) '
              f'than the initial number of turbines ({number_of_turbines_i}. This means not all turbines in the park '
              f'are being extended')
    if substitution and repowering:
        print('Substitution and repowering cannot happen at the same time. Choose one or the other.')
        sys.exit()

    # calculate the year when the extension is happening
    year_of_extension = commissioning_year_i + lifetime_i
    year_of_repowering = commissioning_year_i + lifetime_i
    if (extension_short or extension_long) and (repowering or substitution):
        year_of_repowering = year_of_repowering + lifetime_extension

    # print the type of eol (e.g., short lifetime extension + repowering)
    if extension_short and repowering:
        print(f'You opted for a short lifetime extension ({lifetime_extension} years) and a repowering')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
        print(f'The repowering will start in {year_of_repowering}')
    elif extension_short and substitution:
        print(f'You opted for a short lifetime extension ({lifetime_extension} years) and a substitution')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
        print(f'The substitution will start in {year_of_repowering}')
    elif extension_long and repowering:
        print(f'You opted for a long lifetime extension ({lifetime_extension} years) and a repowering')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
        print(f'The repowering will start in {year_of_repowering}')
    elif extension_long and substitution:
        print(f'You opted for a long lifetime extension ({lifetime_extension} years) and a substitution')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
        print(f'The substitution will start in {year_of_repowering}')
    elif extension_short and not (repowering or substitution):
        print(f'You opted for a short lifetime extension ({lifetime_extension} years) with no substitution or '
              f'repowering afterwards')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
    elif extension_long and not (repowering or substitution):
        print(f'You opted for a long lifetime extension ({lifetime_extension} years) with no substitution or '
              f'repowering afterwards')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The lifetime extension will start in {year_of_extension}')
    elif repowering and not (extension_short or extension_long):
        print(f'You opted for a repowering, without a previous lifetime extension')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The repowering will start in {year_of_repowering}')
    elif substitution and not (extension_short or extension_long):
        print(f'You opted for a substitution, without a previous lifetime extension')
        print(f'The initial turbine was commissioned in {commissioning_year_i}')
        print(f'The substitution will start in {year_of_repowering}')

    # save variables in a comment
    comment = (f'extension_long: {extension_long}, extension_short: {extension_short}, substitution: {substitution}, '
               f'repowering: {repowering}, '
               f'park_name_i: {park_name_i}, park_power_i: {park_power_i}, number_of_turbines_i: {number_of_turbines_i},'
               f'park_location_i: {park_location_i},'
               f'park_coordinates_i: {park_coordinates_i},'
               f'manufacturer_i: {manufacturer_i},'
               f'rotor_diameter_i: {rotor_diameter_i},'
               f'turbine_power_i: {turbine_power_i}, hub_height_i: {hub_height_i}, commissioning_year_i: {commissioning_year_i},'
               f'recycled_share_steel_i: {recycled_share_steel_i},'
               f'lifetime_i: {lifetime_i},'
               f'electricity_mix_steel_i: {electricity_mix_steel_i},'
               f'generator_type_i: {generator_type_i},'
               f'lifetime_extension: {lifetime_extension}, number_of_turbines_extension: {number_of_turbines_extension},'
               f'cf_extension: {cf_extension}, attrition_rate_extension: {attrition_rate_extension},'
               f'lifetime_substitution: {lifetime_substitution}, number_of_turbines_substitution: {number_of_turbines_substitution},'
               f'cf_substitution: {cf_substitution}, attrition_rate_substitution: {attrition_rate_substitution},'
               f'recycled_share_steel_extension: {recycled_share_steel_extension},'
               f'park_power_repowering: {park_power_repowering},'
               f'number_of_turbines_repowering: {number_of_turbines_repowering},'
               f'manufacturer_repowering: {manufacturer_repowering},'
               f'rotor_diameter_repowering: {rotor_diameter_repowering},'
               f'turbine_power_repowering: {turbine_power_repowering},'
               f'hub_height_repowering: {hub_height_repowering},'
               f'generator_type_repowering: {generator_type_repowering},'
               f'electricity_mix_steel_repowering: {electricity_mix_steel_repowering},'
               f'lifetime_repowering: {lifetime_repowering},'
               f'recycled_share_steel_repowering: {recycled_share_steel_repowering}'
               f'cf_repowering: {cf_repowering},'
               f'attrition_rate_repowering: {attrition_rate_repowering},'
               f'land_use_permanent_intensity_repowering: {land_use_permanent_intensity_repowering},'
               f'land_cover_type_repowering:{land_cover_type_repowering},'
               f'eol_scenario_repowering: {eol_scenario_repowering},'
               f'eol: {eol}, transportation: {transportation}, use_and_maintenance: {use_and_maintenance},'
               f'installation: {installation}')

    #### initial turbine ####
    # It creates the activities 'park_name_single_turbine' (code: 'park_name_single_turbine'),
    # 'park_name_cables' (code: 'park_name_intra_cables') and park (park_name) (code: park_name) in the
    #  database 'new_db' in bw2.
    lci_materials = WindTrace_onshore.lci_materials(park_name=park_name_i, park_power=park_power_i,
                                                    number_of_turbines=number_of_turbines_i,
                                                    park_location=park_location_i,
                                                    park_coordinates=park_coordinates_i, manufacturer=manufacturer_i,
                                                    rotor_diameter=rotor_diameter_i, turbine_power=turbine_power_i,
                                                    hub_height=hub_height_i, commissioning_year=commissioning_year_i,
                                                    recycled_share_steel=recycled_share_steel_i, lifetime=lifetime_i,
                                                    electricity_mix_steel=electricity_mix_steel_i,
                                                    generator_type=generator_type_i, include_life_cycle_stages=True)
    for k, v in lci_materials.items():
        lci_materials[k] = v/number_of_turbines_i

    if extension_long and not substitution:
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.LONG_EXTENSION['steel'], c_steel=consts.LONG_EXTENSION['c_steel'],
                                     iron=consts.LONG_EXTENSION['iron'],
                                     aluminium=consts.LONG_EXTENSION['aluminium'],
                                     copper=consts.LONG_EXTENSION['copper'],
                                     plastics=consts.LONG_EXTENSION['plastics'],
                                     others=consts.LONG_EXTENSION['others'],
                                     foundations=consts.LONG_EXTENSION['foundations'],
                                     electronics_and_electrics=consts.LONG_EXTENSION['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_extension,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_extension)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_extension,
                                          lifetime_extended=lifetime_extension, cf_extended=cf_extension,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act)

    elif extension_short and not substitution:
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.SHORT_EXTENSION['steel'], c_steel=consts.SHORT_EXTENSION['c_steel'],
                                     iron=consts.SHORT_EXTENSION['iron'],
                                     aluminium=consts.SHORT_EXTENSION['aluminium'],
                                     copper=consts.SHORT_EXTENSION['copper'],
                                     plastics=consts.SHORT_EXTENSION['plastics'],
                                     others=consts.SHORT_EXTENSION['others'],
                                     foundations=consts.SHORT_EXTENSION['foundations'],
                                     electronics_and_electrics=consts.SHORT_EXTENSION['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_extension,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_extension)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_extension,
                                          lifetime_extended=lifetime_extension, cf_extended=cf_extension,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act)

    if extension_long and substitution:
        # if substitution is not set to False, the extension activity is given the name '_substitution'
        substitution = False
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.LONG_EXTENSION['steel'], c_steel=consts.LONG_EXTENSION['c_steel'],
                                     iron=consts.LONG_EXTENSION['iron'],
                                     aluminium=consts.LONG_EXTENSION['aluminium'],
                                     copper=consts.LONG_EXTENSION['copper'],
                                     plastics=consts.LONG_EXTENSION['plastics'],
                                     others=consts.LONG_EXTENSION['others'],
                                     foundations=consts.LONG_EXTENSION['foundations'],
                                     electronics_and_electrics=consts.LONG_EXTENSION['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_extension,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_extension)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_extension,
                                          lifetime_extended=lifetime_extension, cf_extended=cf_extension,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act)
        substitution = True

    # TODO: ara mateix quan faig extension + substitution, durant la substitution m'agafa el recycled steel share de
    #  l'extension. Hauria d'afegir una variable que permetés afegir recycled steel share de la substitució
    elif extension_short and substitution:
        substitution = False
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.SHORT_EXTENSION['steel'], c_steel=consts.SHORT_EXTENSION['c_steel'],
                                     iron=consts.SHORT_EXTENSION['iron'],
                                     aluminium=consts.SHORT_EXTENSION['aluminium'],
                                     copper=consts.SHORT_EXTENSION['copper'],
                                     plastics=consts.SHORT_EXTENSION['plastics'],
                                     others=consts.SHORT_EXTENSION['others'],
                                     foundations=consts.SHORT_EXTENSION['foundations'],
                                     electronics_and_electrics=consts.SHORT_EXTENSION['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_extension,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_extension)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_extension,
                                          lifetime_extended=lifetime_extension, cf_extended=cf_extension,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act)
        substitution = True

    if substitution and not (extension_short or extension_long):
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.REPLACEMENT_BASELINE['steel'],
                                     c_steel=consts.REPLACEMENT_BASELINE['c_steel'],
                                     iron=consts.REPLACEMENT_BASELINE['iron'],
                                     aluminium=consts.REPLACEMENT_BASELINE['aluminium'],
                                     copper=consts.REPLACEMENT_BASELINE['copper'],
                                     plastics=consts.REPLACEMENT_BASELINE['plastics'],
                                     others=consts.REPLACEMENT_BASELINE['others'],
                                     foundations=consts.REPLACEMENT_BASELINE['foundations'],
                                     electronics_and_electrics=consts.REPLACEMENT_BASELINE['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_substitution,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_substitution,
                                                substitution=substitution)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_substitution,
                                          lifetime_extended=lifetime_substitution, cf_extended=cf_substitution,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act,
                                          substitution=substitution)

    elif substitution and (extension_short or extension_long):
        # change the commissioning year to the year when the life extension ends
        turbine_act = life_extension(park_name=park_name_i, park_location=park_location_i,
                                     hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                                     manufacturer_i=manufacturer_i, park_coordinates_i=park_coordinates_i,
                                     steel=consts.REPLACEMENT_BASELINE['steel'],
                                     c_steel=consts.REPLACEMENT_BASELINE['c_steel'],
                                     iron=consts.REPLACEMENT_BASELINE['iron'],
                                     aluminium=consts.REPLACEMENT_BASELINE['aluminium'],
                                     copper=consts.REPLACEMENT_BASELINE['copper'],
                                     plastics=consts.REPLACEMENT_BASELINE['plastics'],
                                     others=consts.REPLACEMENT_BASELINE['others'],
                                     foundations=consts.REPLACEMENT_BASELINE['foundations'],
                                     electronics_and_electrics=consts.REPLACEMENT_BASELINE['electronics_and_electrics'],
                                     lci_materials_i=lci_materials, lifetime_extension=lifetime_extension,
                                     recycled_share_extension=recycled_share_steel_extension,
                                     substitution=substitution, year_of_extension=year_of_extension, comment=comment)
        park_extended_act = extension_wind_park(park_location=park_location_i, park_name=park_name_i,
                                                extension_turbine_act=turbine_act,
                                                number_of_turbines_extended=number_of_turbines_extension,
                                                substitution=substitution)
        electricity_production_activities(park_name=park_name_i, park_location=park_location_i, park_power=park_power_i,
                                          turbine_power=turbine_power_i,
                                          time_adjusted_cf_extended=attrition_rate_extension,
                                          lifetime_extended=lifetime_extension, cf_extended=cf_extension,
                                          park_extended_act=park_extended_act, turbine_extended_act=turbine_act,
                                          substitution=substitution)
    elif repowering:
        park_name_repowering = park_name_i + '_repowering'
        WindTrace_onshore.lci_wind_turbine(park_name=park_name_repowering, park_power=park_power_repowering,
                                           number_of_turbines=number_of_turbines_repowering,
                                           park_location=park_location_i, park_coordinates=park_coordinates_i,
                                           manufacturer=manufacturer_repowering,
                                           rotor_diameter=rotor_diameter_repowering,
                                           turbine_power=turbine_power_repowering, hub_height=hub_height_repowering,
                                           commissioning_year=year_of_repowering,
                                           generator_type=generator_type_repowering,
                                           recycled_share_steel=recycled_share_steel_repowering,
                                           electricity_mix_steel=electricity_mix_steel_repowering,
                                           lifetime=lifetime_repowering,
                                           land_use_permanent_intensity=land_use_permanent_intensity_repowering,
                                           land_cover_type=land_cover_type_repowering,
                                           eol_scenario=eol_scenario_repowering,
                                           cf=cf_repowering, time_adjusted_cf=attrition_rate_repowering,
                                           eol=eol, transportation=transportation,
                                           use_and_maintenance=use_and_maintenance, installation=installation,
                                           comment=comment
                                           )


def provisional_print(material, initial_amount, classification, share, final_amount):
    print(f'Material: {material}. Initial amount: {initial_amount}. '
          f'Classification: {classification}. Share: {share}. Final amount: {final_amount}')


def life_extension(park_name: str, park_location: str,
                   turbine_power_i: float, hub_height_i: float, park_coordinates_i: tuple,
                   manufacturer_i: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                   steel: float, c_steel: float, iron: float, aluminium: float, copper: float, plastics: float,
                   others: float, foundations: float, electronics_and_electrics: float,
                   year_of_extension,
                   lci_materials_i: dict,
                   lifetime_extension: int,
                   comment: str,
                   recycled_share_extension: float = None, eol_scenario_extension: Literal[1, 2, 3, 4] = 1,
                   substitution: bool = False
                   ):
    """"
    creates a new activity for each life cycle phase (except installation) [parkname_extension_lciphase]
    and links all of them to the extension inventory [parkname_extension]
    plastics include: epoxy resin, rubber, PUR, PVC, PE, fiberglass
    others include: lubricating oil, rare earth elements
    """
    # extension or substitution
    if substitution:
        name = 'substitution'
    else:
        name = 'extension'

    # calculate year of manufacturing
    year_of_manufacturing = year_of_extension - 1
    print(f'The manufacturing is assumed to be a year before ({year_of_manufacturing})')

    # create inventory for the lifetime extension
    extension_act = new_db.new_activity(name=park_name + f'_{name}', code=park_name + f'_{name}',
                                        location=park_location, unit='unit', comment=comment)
    extension_act.save()
    new_exc = extension_act.new_exchange(input=extension_act.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    extension_act.save()

    # 1. materials
    # create inventory for the materials
    materials_activity = new_db.new_activity(name=park_name + f'_{name}_materials',
                                             code=park_name + f'_{name}_materials',
                                             location=park_location, unit='unit')
    materials_activity.save()
    new_exc = materials_activity.new_exchange(input=materials_activity.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    materials_activity.save()

    # add materials from the turbine
    new_masses_extension = {}
    for material in lci_materials_i.keys():
        if any(element in material for element in ['Praseodymium', 'Neodymium', 'Dysprosium', 'Terbium', 'Boron',
                                                   'Lubricating oil']):
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * others)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='others',
                              share=others, final_amount=lci_materials_i[material] * others)
            new_masses_extension[material] = lci_materials_i[material] * others
        elif material == 'Low alloy steel':
            inp, ch = WindTrace_onshore.manipulate_steel_activities(commissioning_year=year_of_manufacturing,
                                                                    recycled_share=recycled_share_extension,
                                                                    electricity_mix=None,
                                                                    printed_warning=consts.PRINTED_WARNING_STEEL)
            consts.PRINTED_WARNING_STEEL = True
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * steel)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='steel',
                              share=steel, final_amount=lci_materials_i[material] * steel)
            new_masses_extension[material] = lci_materials_i[material] * steel
        elif material == 'Chromium steel':
            steel, ch = WindTrace_onshore.manipulate_steel_activities(commissioning_year=year_of_manufacturing,
                                                                      recycled_share=None,
                                                                      electricity_mix=None,
                                                                      printed_warning=consts.PRINTED_WARNING_STEEL)
            if ch:
                inp = ch[0]
            else:
                inp = cutoff391.get(consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * c_steel)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='c_steel',
                              share=c_steel, final_amount=lci_materials_i[material] * c_steel)
            new_masses_extension[material] = lci_materials_i[material] * c_steel
        elif material == 'Fiberglass':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * plastics)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='plastics',
                              share=plastics, final_amount=lci_materials_i[material] * plastics)
            new_masses_extension[material] = lci_materials_i[material] * plastics
        # foundations
        elif 'foundations' in material:
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * foundations)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='foundations',
                              share=foundations, final_amount=lci_materials_i[material] * foundations)
            new_masses_extension[material] = lci_materials_i[material] * foundations
        elif material == 'Copper':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * copper)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='copper',
                              share=copper, final_amount=lci_materials_i[material] * copper)
            new_masses_extension[material] = lci_materials_i[material] * copper
        elif material == 'Aluminium':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * aluminium)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='aluminium',
                              share=aluminium, final_amount=lci_materials_i[material] * aluminium)
            new_masses_extension[material] = lci_materials_i[material] * aluminium
        elif material == 'Cast iron':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * iron)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='iron',
                              share=iron, final_amount=lci_materials_i[material] * iron)
            new_masses_extension[material] = lci_materials_i[material] * iron
        elif any(element in material for element in ['Electrics', 'electronics']):
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * electronics_and_electrics)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material],
                              classification='electronics_and_electrics',
                              share=electronics_and_electrics,
                              final_amount=lci_materials_i[material] * electronics_and_electrics)
            new_masses_extension[material] = lci_materials_i[material] * electronics_and_electrics
        # plastics (except fiberglass, defined above)
        else:
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * plastics)
            ex.save()
            materials_activity.save()
            provisional_print(material=material, initial_amount=lci_materials_i[material], classification='plastics',
                              share=plastics, final_amount=lci_materials_i[material] * plastics)
            new_masses_extension[material] = lci_materials_i[material] * plastics

    # add materials to the extension inventory
    materials_ex = extension_act.new_exchange(input=materials_activity, type='technosphere', amount=1)
    materials_ex.save()
    extension_act.save()

    consts.PRINTED_WARNING_STEEL = False

    # 2. manufacturing
    # create inventory for the manufacturing
    manufacturing_activity = new_db.new_activity(name=park_name + f'_{name}_manufacturing',
                                                 code=park_name + f'_{name}_manufacturing',
                                                 location=park_location, unit='unit')
    manufacturing_activity.save()
    new_exc = manufacturing_activity.new_exchange(input=manufacturing_activity.key, amount=1.0, unit="unit",
                                                  type='production')
    new_exc.save()
    manufacturing_activity.save()
    # add turbine and foundations material processing activities
    processing_materials_list = ['Low alloy steel', 'Chromium steel', 'Cast iron', 'Aluminium', 'Copper',
                                 'Low alloy steel_foundations', 'Chromium steel_foundations', 'Zinc']
    for material in processing_materials_list:
        if material == 'Low alloy steel':
            # section bar rolling
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere',
                                                     amount=new_masses_extension[material])
            ex.save()
            manufacturing_activity.save()
            # welding
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_welding']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=hub_height_i * 2)
            ex.save()
            manufacturing_activity.save()
        elif material == 'Zinc':
            # We need the tower area to be coated. This is the perimeter of the tower multiplied by the hub height.
            # Perimeter of the tower: regression between the tower diameter and the power (data from Sacchi et al.)
            tower_diameter = [5, 5.5, 5.75, 6.75, 7.75]
            power = [3, 3.6, 4, 8, 10]  # in MW
            fit_diameter = np.polyfit(power, tower_diameter, 1)
            f_fit_diameter = np.poly1d(fit_diameter)
            outer_diameter = f_fit_diameter(turbine_power_i)  # in m
            perimeter = np.pi * outer_diameter
            tower_surface_area = perimeter * hub_height_i
            # create exchange
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Zinc coating']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=tower_surface_area)
            ex.save()
            manufacturing_activity.save()
        elif 'foundations' in material and 'alloy' not in material:
            material_name = material[:material.index('_')]
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES[material_name]['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere',
                                                     amount=new_masses_extension[material])
            ex.save()
            manufacturing_activity.save()
        elif 'foundations' in material and 'alloy' in material:
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere',
                                                     amount=new_masses_extension[material])
            ex.save()
            manufacturing_activity.save()
        else:
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES[material]['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere',
                                                     amount=new_masses_extension[material])
            ex.save()
            manufacturing_activity.save()

    # add manufacturing to the extension inventory
    manufacturing_ex = extension_act.new_exchange(input=manufacturing_activity, type='technosphere', amount=1)
    manufacturing_ex.save()
    extension_act.save()

    # 3. transport
    # create inventory for the transport
    transport_activity = new_db.new_activity(name=park_name + f'_{name}_transport',
                                             code=park_name + f'_{name}_transport',
                                             location=park_location, unit='unit')
    transport_activity.save()
    new_exc = transport_activity.new_exchange(input=transport_activity.key, amount=1.0, unit="unit",
                                              type='production')
    new_exc.save()
    transport_activity.save()

    for material in new_masses_extension.keys():
        if 'Concrete' in material:
            concrete_tkm = new_masses_extension[material] * 2.4 * 50 #50 km of road transport, 2.4 is the density
        elif material == 'Low alloy steel':
            steel_tower_tkm = new_masses_extension[material] / 1000 * 450 #450 km of road transport
        else:
            distance_dict = {}
            for location_id in consts.MANUFACTURER_LOC[manufacturer_i]:
                location = consts.MANUFACTURER_LOC[manufacturer_i][location_id]['location']
                distance = geodesic(park_coordinates_i, location).kilometers
                distance_dict[distance] = location_id
            others_tkm = (sum(new_masses_extension.values()) - new_masses_extension['Concrete_foundations']
                          - new_masses_extension['Low alloy steel']) / 1000 * min(distance_dict.keys())

    truck_trans = cutoff391.get(name='market for transport, freight, lorry >32 metric ton, EURO6',
                                code='508cc8b20d83e7b31af9848e1fb45815', location='RER')

    # add transport to the lci inventory
    # add steel transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=steel_tower_tkm)
    new_exc.save()
    # add foundations transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=concrete_tkm)
    new_exc.save()
    # add other materials transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=others_tkm)
    new_exc.save()

    # add transport to the extension inventory
    transport_ex = extension_act.new_exchange(input=transport_activity, type='technosphere', amount=1)
    transport_ex.save()
    extension_act.save()

    # 4. installation. There is NO installation in a lifetime extension

    # 5. maintenance. Includes inspection trips every 6 months and change oil every two years (oil waste included)
    # create inventory for the maintenance
    maintenance_activity = new_db.new_activity(name=park_name + f'_{name}_maintenance',
                                               code=park_name + f'_{name}_maintenance',
                                               location=park_location, unit='unit')
    maintenance_activity.save()
    new_exc = maintenance_activity.new_exchange(input=maintenance_activity.key, amount=1.0, unit="unit",
                                                type='production')
    new_exc.save()
    maintenance_activity.save()

    # Inspection trips
    inp = cutoff391.get(name='transport, passenger car, large size, diesel, EURO 4',
                        code='dceed1b2fd31e759a751c6dd912a45f3')
    ex = maintenance_activity.new_exchange(input=inp, type='technosphere', amount=200 * (lifetime_extension * 2))
    ex.save()
    maintenance_activity.save()

    # Change oil and lubrication
    inp = cutoff391.get(name='market for lubricating oil', code='92391c8c6958ada25b22935e3fa6f06f')
    ex = maintenance_activity.new_exchange(input=inp, type='technosphere',
                                           amount=lci_materials_i['Lubricating oil'] * (lifetime_extension / 2))
    ex.save()
    maintenance_activity.save()
    inp = cutoff391.get(name='treatment of waste mineral oil, hazardous waste incineration, with energy recovery',
                        code='ad6d0f2a8b45536da196238df879077b')
    ex = maintenance_activity.new_exchange(input=inp, type='technosphere',
                                           amount=lci_materials_i['Lubricating oil'] * -(lifetime_extension / 2))
    ex.save()
    maintenance_activity.save()

    # add maintenance to the extension inventory
    maintenance_ex = extension_act.new_exchange(input=maintenance_activity, type='technosphere', amount=1)
    maintenance_ex.save()
    extension_act.save()

    # 6. eol
    # create inventory for the eol
    eol_activity = new_db.new_activity(name=park_name + f'_{name}_eol',
                                       code=park_name + f'_{name}_eol',
                                       location=park_location, unit='unit')
    eol_activity.save()
    new_exc = eol_activity.new_exchange(input=eol_activity.key, amount=1.0, unit="unit",
                                        type='production')
    new_exc.save()
    eol_activity.save()

    # check that the eol is in the range 1-4.
    if 1 > eol_scenario_extension or eol_scenario_extension > 4:
        print('There are 4 eol scenarios in WindTrace. You chose a number that is not in the range 1-4. '
              'By default we applied the baseline scenario')
        eol_scenario_extension = 1

    # materials classification (according to EoL treatment groups)
    fe_alloys = ['Low alloy steel', 'Low alloy steel_foundations', 'Chromium steel',
                 'Chromium steel_foundations', 'Cast iron']
    copper = ['Copper', 'Copper_foundations']
    rare_earth_metals = ['Praseodymium', 'Neodymium', 'Dysprosium', 'Terbium', 'Boron']
    plastics = ['Rubber', 'PUR', 'PVC', 'PE']

    # add materials from the turbine
    for material in new_masses_extension.keys():
        # metals
        if any(element in material for element in fe_alloys):
            if eol_scenario_extension == 1 or eol_scenario_extension == 2 or eol_scenario_extension == 4:
                recycling_rate = 0.9
            # scenario == 3
            else:
                recycling_rate = 0.52
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=new_masses_extension[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in copper):
            if eol_scenario_extension == 1 or eol_scenario_extension == 4:
                recycling_rate = 0.9
            elif eol_scenario_extension == 2:
                recycling_rate = 0.53
            # scenario == 3
            else:
                recycling_rate = 0.42
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=new_masses_extension[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif 'Aluminium' in material:
            if eol_scenario_extension == 1 or eol_scenario_extension == 4:
                recycling_rate = 0.9
            elif eol_scenario_extension == 2:
                recycling_rate = 0.7
            # scenario == 3
            else:
                recycling_rate = 0.42
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=new_masses_extension[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in rare_earth_metals):
            if eol_scenario_extension == 1:
                recycling_rate = 0
            elif eol_scenario_extension == 2:
                recycling_rate = 0.21
            elif eol_scenario_extension == 3:
                recycling_rate = 0.01
            # scenario == 4
            else:
                recycling_rate = 0.7
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=new_masses_extension[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in plastics):
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=new_masses_extension[material] * (-1))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in ['Epoxy resin', 'Fiberglass']):
            # NOTE: 10% of glassfiber mass corresponds to the extra mass on the manufacturing process (i.e., waste)
            # and the waste is not accounted in here.
            if eol_scenario_extension == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=new_masses_extension[material] * (-0.3))
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=new_masses_extension[material] * (-1))
                ex.save()
                eol_activity.save()
        elif any(element in material for element in ['Lubricating oil', 'Ethyleneglycol']):
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=new_masses_extension[material] * (-1))
            ex.save()
            eol_activity.save()
        elif material == 'Concrete_foundations':
            # concrete modelled separately because the amount is in m3 and the landfill activity in kg
            # we use the density (2400 kg/m3)
            if eol_scenario_extension == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=new_masses_extension[material] * (-2400 * 0.5))
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=new_masses_extension[material] * (-2400))
                ex.save()
                eol_activity.save()
        else:
            if eol_scenario_extension == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=-new_masses_extension[material] * 0.5)
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=-new_masses_extension[material])
                ex.save()
                eol_activity.save()

    # add eol to the extension inventory
    eol_ex = extension_act.new_exchange(input=eol_activity, type='technosphere', amount=1)
    eol_ex.save()
    extension_act.save()

    return extension_act


def extension_wind_park(park_name: str, park_location: str, number_of_turbines_extended: int,
                        extension_turbine_act, substitution: bool = False):
    """
    It creates a wind park extension activity and adds the turbines as inputs. During a lifetime extension there
    is no need to update intra-array cables or transformer.
    """
    # extension or substitution?
    if substitution:
        name = 'substitution'
    else:
        name = 'extension'

    try:
        # create inventory for the wind park lifetime extension
        extension_park_act = new_db.new_activity(name=park_name + f'_park_{name}', code=park_name + f'_park_{name}',
                                                 location=park_location, unit='unit')
        extension_park_act.save()
        new_exc = extension_park_act.new_exchange(input=extension_park_act.key, amount=1.0, unit="unit",
                                                  type='production')
        new_exc.save()
        extension_park_act.save()
    except bd.errors.DuplicateNode:
        print(f'An inventory for a park with the name {park_name} was already created before in the database "new_db"')
        print('Give another name to the wind park. Otherwise, you may want to delete '
              'the content of "new_db" by running delete_new_db().')
        print(
            'WARNING: if you run delete_new_db() '
            'ALL WIND PARKS STORED IN THAT DATABASE WILL '
            'BE DELETED!')
        sys.exit()

    # add the turbines
    new_exc = extension_park_act.new_exchange(input=extension_turbine_act, amount=number_of_turbines_extended,
                                              unit="unit", type='technosphere')
    new_exc.save()
    extension_park_act.save()

    return extension_park_act


def electricity_production_activities(park_name: str, park_location: str, park_power: float, turbine_power: float,
                                      lifetime_extended: int,
                                      cf_extended: float, time_adjusted_cf_extended: float,
                                      turbine_extended_act, park_extended_act,
                                      substitution: bool = False):
    # substitution or extension?
    if substitution:
        name = 'substitution'
    else:
        name = 'extension'

    # Create electricity_production activity per turbine and per park (FU: kWh)
    try:
        # turbine
        if time_adjusted_cf_extended != 0:
            cf_comment = 'CF: ' + str(cf_extended) + '. Attrition rate: ' + str(time_adjusted_cf_extended)
        else:
            cf_comment = 'CF: ' + str(cf_extended) + '. Constant CF (no attrition rate)'
        elec_prod_turbine_act = new_db.new_activity(name=park_name + f'_turbine_{name}_kwh',
                                                    code=park_name + f'_turbine_{name}_kwh',
                                                    location=park_location, unit='kilowatt hour', comment=cf_comment)
        elec_prod_turbine_act.save()
        new_exc = elec_prod_turbine_act.new_exchange(input=elec_prod_turbine_act.key, amount=1.0, unit='kilowatt hour',
                                                     type='production')
        new_exc.save()
        # park
        elec_prod_park_act = new_db.new_activity(name=park_name + f'_park_{name}_kwh',
                                                 code=park_name + f'_park_{name}_kwh',
                                                 location=park_location, unit='kilowatt hour', comment=cf_comment)
        elec_prod_park_act.save()
        new_exc = elec_prod_park_act.new_exchange(input=elec_prod_park_act.key, amount=1.0, unit='kilowatt hour',
                                                  type='production')
        new_exc.save()
    except bd.errors.DuplicateNode:
        print(f'An inventory for a park with the name {park_name} was already created before in the database "new_db"')
        print('Give another name to the wind park. Otherwise, you may want to delete '
              'the content of "new_db" by running delete_new_db().')
        print(
            'WARNING: if you run delete_new_db() '
            'ALL WIND PARKS STORED IN THAT DATABASE WILL '
            'BE DELETED!')
        sys.exit()

    # add infrastructure
    elec_turbine, elec_park = electricity_production(park_power=park_power,
                                                     cf=cf_extended, time_adjusted_cf=time_adjusted_cf_extended,
                                                     lifetime=lifetime_extended, turbine_power=turbine_power)
    # to turbine activity
    turbine_amount = 1 / elec_turbine
    new_exc = elec_prod_turbine_act.new_exchange(input=turbine_extended_act, amount=turbine_amount, type='technosphere')
    new_exc.save()
    elec_prod_turbine_act.save()
    # to park activity
    park_amount = 1 / elec_park
    new_exc = elec_prod_park_act.new_exchange(input=park_extended_act, amount=park_amount, type='technosphere')
    new_exc.save()
    elec_prod_park_act.save()


def electricity_production(park_power: float,
                           cf: float, time_adjusted_cf: float,
                           lifetime: int, turbine_power: float):
    if time_adjusted_cf == 0:
        print('Constant cf (time-adjusted cf not applied)')
        elec_prod_turbine = cf * lifetime * 365 * 24 * turbine_power / 1000
        elec_prod_park = cf * lifetime * 365 * 24 * park_power / 1000
    else:
        # adjust a decay in yearly production according to CFage = CF2 * (1-time_adjusted_cf)^age [Xu et al. (2023)]
        print('Time-adjusted cf applied with an attrition coefficient of ' + str(time_adjusted_cf))
        year = 1
        adjusted_time = []
        while year <= lifetime:
            if year == 1:
                yearly_adjusted_time = 1
            else:
                yearly_adjusted_time = (1 - time_adjusted_cf) ** year
            adjusted_time.append(yearly_adjusted_time)
            year += 1
        adjusted_time = sum(adjusted_time)
        elec_prod_turbine = cf * 365 * 24 * turbine_power * adjusted_time * 1000
        elec_prod_park = cf * 365 * 24 * park_power * adjusted_time * 1000

    return elec_prod_turbine, elec_prod_park


def lca_wind_repowering(park_name: str, park_power: float, extension: bool, repowering: bool, substitution: bool,
                     method: str = 'ReCiPe 2016 v1.03, midpoint (H)',
                     indicators: Optional[Tuple[str, str, str]] = None):
    """
    Based on LCIs previously built with the function lci_wind_turbine, it creates a dictionary (total_park_results)
    with the lca score results of the total park. The dictionary follows this structure:
    total_park_results = {'method_1': park_score,
                          'method_2': park_score}
    Default turbine=True, it means that we calculate the impacts for the FU = 1 turbine.
    In case of wanting the impacts from the park: set turbine=False.
    """
    if indicators is None:
        indicators = []

    # setting the methods
    if indicators:
        methods = indicators
    else:
        methods = [m for m in bd.methods if method in m[0] and 'no LT' not in m[0]]

    results_acts = {}
    for lci_phase in ['materials', 'manufacturing', 'transport', 'installation', 'maintenance', 'eol', False]:
        turbine_act = False
        if extension and not lci_phase:
            # turbine
            name_turbine = park_name + '_extension'
            name_turbine_kwh = park_name + '_turbine_extension_kwh'
            turbine_act = new_db.get(name_turbine)
            turbine_kwh_act = new_db.get(name_turbine_kwh)

            # park
            name_park = park_name + '_park_extension'
            name_park_kwh = park_name + '_park_extension_kwh'
            park_act = new_db.get(name_park)
            park_kwh_act = new_db.get(name_park_kwh)

            name_surname = 'extension'

        elif extension and lci_phase:
            if lci_phase == 'installation':
                continue
            # turbine lci phases
            name_turbine_phase = park_name + '_extension_' + lci_phase
            lci_phase_act = new_db.get(name_turbine_phase)

            name_surname = 'extension'

        if substitution and not lci_phase:
            # turbine
            name_turbine = park_name + '_substitution'
            name_turbine_kwh = park_name + '_turbine_substitution_kwh'
            turbine_act = new_db.get(name_turbine)
            turbine_kwh_act = new_db.get(name_turbine_kwh)

            # park
            name_park = park_name + '_park_substitution'
            name_park_kwh = park_name + '_park_substitution_kwh'
            park_act = new_db.get(name_park)
            park_kwh_act = new_db.get(name_park_kwh)

            name_surname = 'substitution'

        elif substitution and lci_phase:
            if lci_phase == 'installation':
                continue
            # turbine lci phases
            name_turbine_phase = park_name + '_substitution_' + lci_phase
            lci_phase_act = new_db.get(name_turbine_phase)

            name_surname = 'substitution'

        elif repowering and not lci_phase:
            # turbine
            name_turbine = park_name + '_repowering_single_turbine'
            name_turbine_kwh = park_name + '_repowering_turbine_kwh'
            turbine_act = new_db.get(name_turbine)
            turbine_kwh_act = new_db.get(name_turbine_kwh)

            # park
            name_park = park_name + f'_repowering_{str(park_power)}'
            name_park_kwh = park_name + '_repowering_park_kwh'
            park_act = new_db.get(name_park)
            park_kwh_act = new_db.get(name_park_kwh)

            name_surname = 'repowering'

        elif repowering and lci_phase:
            # turbine lci phases
            name_turbine_phase = park_name + '_repowering_' + lci_phase
            lci_phase_act = new_db.get(name_turbine_phase)

            name_surname = 'repowering'

        name_activity = {0: 'turbine (FU unit)', 1: 'turbine (FU kWh)', 2: 'park (FU unit)', 3: 'park (FU kWh)'}
        if not turbine_act:
            print(str(lci_phase_act))
            lca_obj = lci_phase_act.lca(amount=1)
            results = {}
            for m in methods:
                lca_obj.switch_method(m)
                lca_obj.lcia()
                results[m[1]] = lca_obj.score
            results_acts[f'{lci_phase}_{name_surname}'] = results
        elif turbine_act:
            acts = [turbine_act, turbine_kwh_act, park_act, park_kwh_act]
            for i, act in enumerate(acts):
                print(str(act))
                lca_obj = act.lca(amount=1)
                results = {}
                for m in methods:
                    lca_obj.switch_method(m)
                    lca_obj.lcia()
                    results[m[1]] = lca_obj.score
                results_acts[name_activity[i]] = results
        df = pd.DataFrame(results_acts)

    return df


def test(park_name: str, extension: bool, repowering: bool, substitution: bool,
         lci_phase: Optional[Literal['materials', 'manufacturing', 'transport', 'installation', 'maintenance', 'eol']],
         park_power: float = None):
    turbine_ex = None
    turbine_kwh_ex = None
    park_ex = None
    park_kwh_ex = None
    lci_phase_ex = None
    if extension and not lci_phase:
        # turbine
        name_turbine = park_name + '_extension'
        name_turbine_kwh = park_name + '_turbine_extension_kwh'
        turbine_act = new_db.get(name_turbine)
        turbine_kwh_act = new_db.get(name_turbine_kwh)
        turbine_ex = [e for e in turbine_act.exchanges()]
        turbine_kwh_ex = [e for e in turbine_kwh_act.exchanges()]
        # park
        name_park = park_name + '_park_extension'
        name_park_kwh = park_name + '_park_extension_kwh'
        park_act = new_db.get(name_park)
        park_kwh_act = new_db.get(name_park_kwh)
        park_ex = [e for e in park_act.exchanges()]
        park_kwh_ex = [e for e in park_kwh_act.exchanges()]

    elif extension and lci_phase:
        # turbine lci phases
        name_turbine_phase = park_name + '_extension_' + lci_phase
        lci_phase_act = new_db.get(name_turbine_phase)
        lci_phase_ex = [e for e in lci_phase_act.exchanges()]
    if substitution and not lci_phase:
        # turbine
        name_turbine = park_name + '_substitution'
        name_turbine_kwh = park_name + '_turbine_substitution_kwh'
        turbine_act = new_db.get(name_turbine)
        turbine_kwh_act = new_db.get(name_turbine_kwh)
        turbine_ex = [e for e in turbine_act.exchanges()]
        turbine_kwh_ex = [e for e in turbine_kwh_act.exchanges()]
        # park
        name_park = park_name + '_park_substitution'
        name_park_kwh = park_name + '_park_substitution_kwh'
        park_act = new_db.get(name_park)
        park_kwh_act = new_db.get(name_park_kwh)
        park_ex = [e for e in park_act.exchanges()]
        park_kwh_ex = [e for e in park_kwh_act.exchanges()]

    elif substitution and lci_phase:
        # turbine lci phases
        name_turbine_phase = park_name + '_substitution_' + lci_phase
        lci_phase_act = new_db.get(name_turbine_phase)
        lci_phase_ex = [e for e in lci_phase_act.exchanges()]
    elif repowering and not lci_phase:
        # turbine
        name_turbine = park_name + '_repowering_single_turbine'
        name_turbine_kwh = park_name + '_repowering_turbine_kwh'
        turbine_act = new_db.get(name_turbine)
        turbine_kwh_act = new_db.get(name_turbine_kwh)
        turbine_ex = [e for e in turbine_act.exchanges()]
        turbine_kwh_ex = [e for e in turbine_kwh_act.exchanges()]
        # park
        name_park = park_name + f'_repowering_{str(park_power)}'
        name_park_kwh = park_name + '_repowering_park_kwh'
        park_act = new_db.get(name_park)
        park_kwh_act = new_db.get(name_park_kwh)
        park_ex = [e for e in park_act.exchanges()]
        park_kwh_ex = [e for e in park_kwh_act.exchanges()]
    elif repowering and lci_phase:
        # turbine lci phases
        name_turbine_phase = park_name + '_repowering_' + lci_phase
        lci_phase_act = new_db.get(name_turbine_phase)
        lci_phase_ex = [e for e in lci_phase_act.exchanges()]

    return turbine_ex, turbine_kwh_ex, park_ex, park_kwh_ex, lci_phase_ex


def lci_excel_output(park_name, extension, repowering, substitution, park_power_repowering, scenario_name, method_name):
    cwd = os.getcwd()
    file_name = f'results_{park_name}_{scenario_name}.xlsx'
    file_path = os.path.join(cwd, file_name)
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        if extension:
            turbine_act_code = park_name + '_extension'
        elif substitution:
            turbine_act_code = park_name + '_substitution'
        elif repowering:
            turbine_act_code = park_name + '_repowering_single_turbine'
        comment = new_db.get(turbine_act_code)._data['comment']
        new_comment = re.sub(r'(\d+), (\d+|-\d+)', r'\1; \2', comment)
        splitted = new_comment.split(',')
        input_variables = []
        input_values = []
        for a in splitted:
            input_variables.append(a.split(':')[0])
            input_values.append(a.split(':')[1][1:])
        data_dict = {'input variables': input_variables,
                     'input values': input_values}
        df = pd.DataFrame(data_dict)
        sheet_name = 'info'
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        for lci_phase in ['materials', 'manufacturing', 'transport', 'installation', 'maintenance', 'eol', False]:
            if extension and lci_phase:
                if lci_phase == 'materials':
                    output = test(park_name=park_name, extension=extension,
                                  repowering=False, substitution=False, park_power=park_power_repowering,
                                  lci_phase=lci_phase)
                    amount = []
                    unit = []
                    input_act = []
                    act_names = ['Steel', 'Chromium steel', 'Iron', 'Aluminium', 'Copper', 'Epoxy resin', 'Rubber',
                                 'PUR', 'PVC', 'Glass fiber', 'Electronics', 'Electrics', 'Lubricating oil',
                                 'Reinforcing steel', 'Chromium steel', 'Concrete', 'Praseodymium', 'Neodymium',
                                 'Dysprosium', 'Terbium', 'Boron', 'PE']
                    for ex in output[4]:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input': act_names,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{lci_phase}_extension'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif lci_phase != 'installation':
                    output = test(park_name=park_name, extension=extension,
                                  repowering=False, substitution=False, park_power=park_power_repowering,
                                  lci_phase=lci_phase)
                    amount = []
                    unit = []
                    input_act = []
                    for ex in output[4]:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{lci_phase}_extension'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif extension and not lci_phase:
                sheet_names = {0: 'turbine extension (FU unit)', 1: 'turbine extension (FU kWh)',
                               2: 'park extension (FU unit)', 3: 'park extension (FU kWh)'}
                output = test(park_name=park_name, extension=extension,
                              repowering=False, substitution=False, park_power=park_power_repowering,
                              lci_phase=None)

                for i, out in enumerate(output[:4]):
                    amount = []
                    unit = []
                    input_act = []
                    input_name = []
                    for ex in out:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                            start_position = str(ex.input).find(park_name) + len(park_name) + 1
                            name = str(ex.input)[start_position:].split("'")[0]
                            input_name.append(name)

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input': input_name,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{sheet_names[i]}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            if substitution and lci_phase:
                if lci_phase == 'materials':
                    output = test(park_name=park_name, extension=False,
                                  repowering=False, substitution=True, park_power=park_power_repowering,
                                  lci_phase=lci_phase)
                    amount = []
                    unit = []
                    input_act = []
                    act_names = ['Steel', 'Chromium steel', 'Iron', 'Aluminium', 'Copper', 'Epoxy resin', 'Rubber',
                                 'PUR', 'PVC', 'Glass fiber', 'Electronics', 'Electrics', 'Lubricating oil',
                                 'Reinforcing steel', 'Chromium steel', 'Concrete', 'Praseodymium', 'Neodymium',
                                 'Dysprosium', 'Terbium', 'Boron', 'PE']
                    for ex in output[4]:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input': act_names,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{lci_phase}_substitution'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif lci_phase != 'installation':
                    output = test(park_name=park_name, extension=False,
                                  repowering=False, substitution=True, park_power=park_power_repowering,
                                  lci_phase=lci_phase)
                    amount = []
                    unit = []
                    input_act = []
                    for ex in output[4]:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{lci_phase}_substitution'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif substitution and not lci_phase:
                sheet_names = {0: 'turbine substitution (FU unit)', 1: 'turbine substitution (FU kWh)',
                               2: 'park substitution (FU unit)', 3: 'park substitution (FU kWh)'}
                output = test(park_name=park_name, extension=False,
                              repowering=False, substitution=True, park_power=park_power_repowering,
                              lci_phase=None)

                for i, out in enumerate(output[:4]):
                    amount = []
                    unit = []
                    input_act = []
                    input_name = []
                    for ex in out:
                        if ex._data['type'] != 'production':
                            amount.append(ex.amount)
                            unit.append(ex.unit)
                            input_act.append(str(ex.input))

                            start_position = str(ex.input).find(park_name) + len(park_name) + 1
                            name = str(ex.input)[start_position:].split("'")[0]
                            input_name.append(name)

                    data_dict = {
                        'amount': amount,
                        'unit': unit,
                        'input': input_name,
                        'input (bw2 activity)': input_act
                    }
                    df = pd.DataFrame(data_dict)
                    sheet_name = f'lci_{sheet_names[i]}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

        df_lcia = lca_wind_repowering(park_name=park_name, park_power=park_power_repowering, extension=extension,
                                      repowering=repowering, substitution=substitution, method=method_name)
        df_lcia.to_excel(writer, sheet_name='lcia_results', index=True)


if __name__ == "__main__":
    bd.projects.set_current(consts.PROJECT_NAME)
    bi.bw2setup()
    spold_files = consts.SPOLD_FILES
    if "cutoff391" not in bd.databases:
        ei = bi.SingleOutputEcospold2Importer(spold_files, "cutoff391", use_mp=False)
        ei.apply_strategies()
        ei.write_database()
    cutoff391 = bd.Database("cutoff391")
    if 'new_db' not in bd.databases:
        new_db = bd.Database('new_db')
        new_db.register()
    new_db = bd.Database('new_db')
    pass

# example Laura:
    lci_repowering(extension_long=True, extension_short=False, substitution=False, repowering=True,
                   park_name_i='park_example_4', park_power_i=21.6, number_of_turbines_i=12,
                   park_location_i='PT',
                   park_coordinates_i=(40.7927, -8.5085),
                   manufacturer_i='Vestas',
                   rotor_diameter_i=90,
                   turbine_power_i=1.8, hub_height_i=100, commissioning_year_i=2008,
                   recycled_share_steel_i=None,
                   lifetime_i=25,
                   electricity_mix_steel_i=None,
                   generator_type_i='gb_dfig',
                   lifetime_extension=25, number_of_turbines_extension=12,
                   cf_extension=0.30, attrition_rate_extension=0.009,
                   park_power_repowering=21.6,
                   number_of_turbines_repowering=4,
                   manufacturer_repowering='Siemens Gamesa',
                   rotor_diameter_repowering=110,
                   turbine_power_repowering=5.4,
                   hub_height_repowering=120,
                   generator_type_repowering='gb_dfig',
                   electricity_mix_steel_repowering='Norway',
                   lifetime_repowering=25,
                   cf_repowering=0.42,
                   attrition_rate_repowering=0.009
                   )
lci_excel_output(park_name='park_example_4', extension=True, repowering=False, substitution=False, park_power_repowering=21.6, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='park_example_4', extension=False, repowering=True, substitution=False, park_power_repowering=21.6, scenario_name='repowering', method_name='EF v3.1')


# example of use lifetime extension long (no substitution or repowering):
lci_repowering(extension_long=True, extension_short=False, substitution=False, repowering=False,
               park_name_i='test_101_long', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               lifetime_extension=10, number_of_turbines_extension=5,
               cf_extension=0.30, attrition_rate_extension=0.009)

# example of use lifetime extension short (no substitution or repowering):
lci_repowering(extension_long=False, extension_short=True, substitution=False, repowering=False,
               park_name_i='test_101_short', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               lifetime_extension=5, number_of_turbines_extension=5,
               cf_extension=0.30, attrition_rate_extension=0.009)

# example of use substitution (no lifetime extension):
lci_repowering(extension_long=False, extension_short=False, substitution=True, repowering=False,
               park_name_i='test_103_substitution', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               lifetime_substitution=20, number_of_turbines_substitution=5,
               cf_substitution=0.30, attrition_rate_substitution=0.009)

# example of use repowering (no lifetime extension):
lci_repowering(extension_long=False, extension_short=False, substitution=False, repowering=True,
               park_name_i='test_101_repowering', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               park_power_repowering=10.0,
               number_of_turbines_repowering=2,
               manufacturer_repowering='Siemens Gamesa',
               rotor_diameter_repowering=130,
               turbine_power_repowering=5.0,
               hub_height_repowering=140,
               generator_type_repowering='dd_pmsg',
               electricity_mix_steel_repowering='Norway',
               lifetime_repowering=25,
               cf_repowering=0.4,
               attrition_rate_repowering=0.009)

# example of use long lifetime extension AND substitution:
lci_repowering(extension_long=True, extension_short=False, substitution=True, repowering=False,
               park_name_i='test_101_extension_substitution', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               lifetime_extension=5, number_of_turbines_extension=5,
               cf_extension=0.30, attrition_rate_extension=0.009,
               lifetime_substitution=20, number_of_turbines_substitution=5,
               cf_substitution=0.35, attrition_rate_substitution=0.009)

# example of use long lifetime extension AND repowering:
lci_repowering(extension_long=True, extension_short=False, substitution=False, repowering=True,
               park_name_i='test_101_extension_repowering', park_power_i=10.0, number_of_turbines_i=5,
               park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               lifetime_extension=5, number_of_turbines_extension=5,
               cf_extension=0.30, attrition_rate_extension=0.009,
               park_power_repowering=15.0,
               number_of_turbines_repowering=3,
               manufacturer_repowering='Siemens Gamesa',
               rotor_diameter_repowering=130,
               turbine_power_repowering=5.0,
               hub_height_repowering=140,
               generator_type_repowering='dd_pmsg',
               electricity_mix_steel_repowering='Norway',
               lifetime_repowering=25,
               cf_repowering=0.4,
               attrition_rate_repowering=0.009)

pass



# variables to be set by the user
PROJECT_NAME = 'repowering_Cabril_I'

#### LCA settings ####
# LCI data - Ecoinvent file path
SPOLD_FILES = r"C:\Users\1439891\OneDrive - UAB\Documentos\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets" #r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
method_name = 'EF v3.1'


### define path scenario data
path_scenario_data = r'C:\Users\34660\UAB\LIVENlabTEAMS - JW4A - JW4A\WP4 - labs\Portugal\Repowering\WindTrace repowering\input_scenario_data.csv'


#######

#### Upload table with scenario data
table_scenarios = pd.read_csv(path_scenario_data,
                              index_col = 0,
                              dtype = {"extension_long": bool,
                                       "extension_short": bool,
                                       "substitution": bool,
                                       "repowering": bool,
                                       "park_name_i" : str,
                                       "park_power_i" : float,
                                       "number_of_turbines_i": int,
                                       "park_location_i" : str, #'PT'
                                       "park_coordinates_i" : tuple,
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
                                       "land_use_permanent_intensity_repowering":float,
                                       "land_cover_type_repowering" : str, #'industrial'
                                       "eol_scenario_repowering" : int,
                                       "eol": int,
                                       "transportation": float,
                                       "use_and_maintenance": float,
                                       "installation": float,}) #SEGURAMENT S'HA D'EDITAR AIXÒ


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