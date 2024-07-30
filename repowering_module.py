import sys

import numpy as np
from geopy.distance import geodesic

import WindTrace_onshore
from typing import Optional, Literal
import bw2data as bd
import bw2io as bi
import consts

bd.projects.set_current("lci_model")
bi.bw2setup()
spold_files = r"C:\ecoinvent_data\3.9.1\cutoff\datasets"
if "cutoff391" not in bd.databases:
    ei = bi.SingleOutputEcospold2Importer(spold_files, "cutoff391", use_mp=False)
    ei.apply_strategies()
    ei.write_database()
cutoff391 = bd.Database("cutoff391")
if 'new_db' not in bd.databases:
    new_db = bd.Database('new_db')
    new_db.register()
new_db = bd.Database('new_db')


def lci_repowering(extension_long: bool, extension_short: bool, repowering: bool, substitution: bool,
                   park_name_i: str, park_power_i: float, number_of_turbines_i: int, park_location_i: str,
                   park_coordinates_i: tuple,
                   manufacturer_i: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                   rotor_diameter_i: float,
                   turbine_power_i: float, hub_height_i: float, commissioning_year_i: int,
                   recycled_share_steel_i: float = None,
                   lifetime_i: int = 20,
                   electricity_mix_steel_i: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                   generator_type_i: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'] = 'gb_dfig',
                   ):
    # test input parameters
    if extension_short and extension_long:
        print('WARNING. You chose to have s short and a long extension and that is not possible. '
              'Choose either one, the other or none, by setting extension_long and extension_short '
              'variables to True or False')
        sys.exit()

    #### initial turbine ####
    # It creates the activities 'park_name_single_turbine' (code: 'park_name_single_turbine'),
    # 'park_name_cables' (code: 'park_name_intra_cables') and park (park_name) (code: park_name) in the
    #  database 'new_db' in bw2.
    # TODO: Note -> it might be best to call lci_wind_turbine, so the wind park activity is created and it is just
    #  a matter of substituting the old turbine/s input for the new one!
    lci_materials = WindTrace_onshore.lci_materials(park_name=park_name_i, park_power=park_power_i,
                                                    number_of_turbines=number_of_turbines_i,
                                                    park_location=park_location_i,
                                                    park_coordinates=park_coordinates_i, manufacturer=manufacturer_i,
                                                    rotor_diameter=rotor_diameter_i, turbine_power=turbine_power_i,
                                                    hub_height=hub_height_i, commissioning_year=commissioning_year_i,
                                                    recycled_share_steel=recycled_share_steel_i, lifetime=lifetime_i,
                                                    electricity_mix_steel=electricity_mix_steel_i,
                                                    generator_type=generator_type_i, include_life_cycle_stages=True)

    if extension_long:
        life_extension(park_name=park_name_i, park_location=park_location_i,
                       commissioning_year_i=commissioning_year_i, lifetime_i=lifetime_i,
                       hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                       steel=LONG_EXTENSION['steel'], c_steel=LONG_EXTENSION['c_steel'], iron=LONG_EXTENSION['iron'],
                       aluminium=LONG_EXTENSION['aluminium'], copper=LONG_EXTENSION['copper'],
                       plastics=LONG_EXTENSION['plastics'],
                       others=LONG_EXTENSION['others'], foundations=LONG_EXTENSION['foundations'],
                       electronics_and_electrics=LONG_EXTENSION['electronics_and_electrics'],
                       lci_materials_i=lci_materials)
    elif extension_short:
        life_extension(park_name=park_name_i, park_location=park_location_i,
                       commissioning_year_i=commissioning_year_i, lifetime_i=lifetime_i,
                       hub_height_i=hub_height_i, turbine_power_i=turbine_power_i,
                       steel=SHORT_EXTENSION['steel'], c_steel=SHORT_EXTENSION['c_steel'], iron=SHORT_EXTENSION['iron'],
                       aluminium=SHORT_EXTENSION['aluminium'], copper=LONG_EXTENSION['copper'],
                       plastics=SHORT_EXTENSION['plastics'],
                       others=SHORT_EXTENSION['others'], foundations=SHORT_EXTENSION['foundations'],
                       electronics_and_electrics=SHORT_EXTENSION['electronics_and_electrics'],
                       lci_materials_i=lci_materials)


def provisional_print(material, initial_amount, classification, share, final_amount):
    print(f'Material: {material}. Initial amount: {initial_amount}. '
          f'Classification: {classification}. Share: {share}. Final amount: {final_amount}')


def life_extension(park_name: str, park_location: str, commissioning_year_i: int, lifetime_i: int,
                   turbine_power_i: float, hub_height_i: float, park_coordinates_i: tuple,
                   manufacturer_i: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                   steel: float, c_steel: float, iron: float, aluminium: float, copper: float, plastics: float,
                   others: float, foundations: float, electronics_and_electrics: float,
                   lci_materials_i: dict,
                   ):
    """"
    plastics include: epoxy resin, rubber, PUR, PVC, PE, fiberglass
    others include: lubricating oil, rare earth elements
    """
    # calculate the year when the extension is happening
    year_of_extension = commissioning_year_i + lifetime_i
    year_of_manufacturing = year_of_extension - 1
    print(f'The initial turbine was commissioned in {str(commissioning_year_i)}')
    print(f'The lifetime of the initial turbine was {str(lifetime_i)} years')
    print(f'Then, the extension started in {year_of_extension} and the manufacturing is assumed to be a year before '
          f'({year_of_manufacturing})')

    # create inventory for the lifetime extension
    extension_act = new_db.new_activity(name=park_name + '_extension', code=park_name + '_extension',
                                        location=park_location, unit='unit')
    extension_act.save()
    new_exc = extension_act.new_exchange(input=extension_act.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    extension_act.save()

    # 1. materials
    # create inventory for the materials
    materials_activity = new_db.new_activity(name=park_name + '_extension_materials',
                                             code=park_name + '_extension_materials',
                                             location=park_location, unit='unit')
    materials_activity.save()
    new_exc = materials_activity.new_exchange(input=materials_activity.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    materials_activity.save()

    # add materials from the turbine
    for material in lci_materials_i.keys():
        new_masses_extension = {}
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
        # TODO: include authomatic future recycling share according to projection
        elif material == 'Low alloy steel':
            inp, ch = WindTrace_onshore.manipulate_steel_activities(commissioning_year=year_of_manufacturing,
                                                                    recycled_share=None,
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
            # Mass includes 10% of waste produced in the manufacturing (Psomopoulos et al. 2019)
            ex = materials_activity.new_exchange(input=inp, type='technosphere',
                                                 amount=lci_materials_i[material] * 1.1 * plastics)
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

    consts.PRINTED_WARNING_STEEL = False

    # 2. manufacturing
    # create inventory for the manufacturing
    manufacturing_activity = new_db.new_activity(name=park_name + 'extension_manufacturing',
                                                 code=park_name + 'extension_manufacturing',
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

    # 3. transport
    # create inventory for the transport
    transport_activity = new_db.new_activity(name=park_name + 'extension_transport',
                                             code=park_name + 'extension_transport',
                                             location=park_location, unit='unit')
    transport_activity.save()
    new_exc = transport_activity.new_exchange(input=transport_activity.key, amount=1.0, unit="unit",
                                              type='production')
    new_exc.save()
    transport_activity.save()

    # TODO: continue here
    for material in new_masses_extension:
        if 'Concrete' in material:
            concrete_tkm = new_masses_extension[material] * 2.4 * 50
        elif material == 'Low alloy steel':
            steel_tower_tkm = new_masses_extension[material] / 1000 * 450
        else:
            distance_dict = {}
            for location_id in consts.MANUFACTURER_LOC[manufacturer_i]:
                location = consts.MANUFACTURER_LOC[manufacturer_i][location_id]['location']
                distance = geodesic(park_coordinates_i, location).kilometers
                distance_dict[distance] = location_id
            others_tkm = (sum(new_masses_extension.values()) - new_masses_extension['Concrete_foundations']
                          - new_masses_extension['Low alloy steel']) / 1000 * min(distance_dict.keys())


LONG_EXTENSION = {'steel': 0.08, 'c_steel': 0.08, 'iron': 1, 'aluminium': 0.65, 'copper': 0.81,
                  'plastics': 1, 'others': 1, 'foundations': 0, 'electronics_and_electrics': 0}
SHORT_EXTENSION = {'steel': 0.01, 'c_steel': 0.01, 'iron': 0.34, 'aluminium': 0, 'copper': 0,
                   'plastics': 0.91, 'others': 0, 'foundations': 0, 'electronics_and_electrics': 0}
REPLACEMENT_BASELINE = {'steel': 0.84, 'c_steel': 0.84}

pass

# example of use:
lci_repowering(extension_long=True, extension_short=False, repowering=False, substitution=False,
               park_name_i='repowering_test_23', park_power_i=2.0, number_of_turbines_i=1, park_location_i='ES',
               park_coordinates_i=(53.43568404210107, 11.214208299339631),
               manufacturer_i='Vestas',
               rotor_diameter_i=120,
               turbine_power_i=2.0, hub_height_i=120, commissioning_year_i=2009,
               recycled_share_steel_i=None,
               lifetime_i=20,
               electricity_mix_steel_i=None,
               generator_type_i='gb_dfig',
               )

pass
