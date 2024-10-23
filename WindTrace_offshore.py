import numpy as np
import bw2data as bd
import bw2io as bi
from typing import List, Literal
import sys
from stats_arrays import NormalUncertainty

import consts
from WindTrace_onshore import *
from consts import *

# TODO: write methods
# TODO:
#  1. Add a basic model of transport, installation, and maintenance (Tsai et al or
#  2. Add eol of the current materials
#  3. Function to build fleets

# create a bw25 project, import ecoinvent v.3.9.1 and create an empty database 'new_db'
bd.projects.set_current('premise')
bi.bw2setup()
spold_files = SPOLD_FILES
if "cutoff391" not in bd.databases:
    ei = bi.SingleOutputEcospold2Importer(spold_files, "cutoff391", use_mp=False)
    ei.apply_strategies()
    ei.write_database()
cutoff391 = bd.Database("cutoff391")
if NEW_DB_NAME not in bd.databases:
    new_db = bd.Database(NEW_DB_NAME)
    new_db.register()
new_db = bd.Database(NEW_DB_NAME)
biosphere3 = bd.Database('biosphere3')


# 1.1 turbine material inputs: the same (except foundations)

# 1.2 foundations

# depth: if it is in aarhus_wind_farm_market -> take it from there. Otherwise, use google API (Elevation)
# distance to shore: if it is in aarhus_wind_farm_market -> take it from there. Otherwise, ...?
# assumption (Bilgili et al., 2022). Confirmed with examples with Wu et al., 2019:
#   - gravity-based: <15 m
#   - monopile: 15-35 m
#   - tripod/jacket: 35-50 m
#   - floating: >50 m
# NOTE: accroding to Jiang (2021) [Installation of offshore wind turbines: A technical review]: gravity-based <10 m,
# monopile 20-40 m, jacket 50-70 m.

# monopile: modeled as Sacchi et al.
def monopile_parameters(sea_depth: float, power: float, park_name: str, commissioning_year: int,
                        recycled_share_steel: float = None,
                        electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None,
                        new_db=bd.Database('new_db')):
    """
        All credits to Sacchi et al.
    """
    #### NOTE: power in kW
    power = power * 1000
    penetration_depth = 19.4002294 * power / 1000 + 0.000944015847  # penetration in the seabed
    pile_length = 9 + penetration_depth + sea_depth  # 9m added to the

    # transition piece, grout and scour masses
    transition_lenght = 0.367 * pile_length + 1.147  # in meters
    transition_mass = (15.23 * transition_lenght - 73.86) * 1000  # in kg (steel)
    grout_volume = 2.93 * transition_lenght - 27.54  # in m3
    grout_mass = grout_volume * 2400  # in kg (cement: market for cement, Portland)
    scour_protection = 0.192 * power + 1643  # in m3
    scour_mass = scour_protection * 1680  # in kg (gravel: market for gravel, round)

    # monopile
    diameter = [5, 5.5, 5.75, 6.75, 7.75]  # in m
    p = [3000, 3600, 4000, 8000, 10000]  # in kW
    fit_diameter = np.polyfit(p, diameter, 1)
    f_fit_diameter = np.poly1d(fit_diameter)

    outer_diameter = f_fit_diameter(power)  # diameter for given power, in m
    outer_area = (np.pi / 4) * (outer_diameter ** 2)  # cross-section area of pile
    outer_volume = outer_area * pile_length  # pile volume, in m3

    inner_diameter = outer_diameter
    pile_thickness = np.interp(power, [2000, 3000, 3600, 4000, 8000, 10000], [0.07, 0.10, 0.13, 0.16, 0.19, 0.22])
    inner_diameter -= 2 * pile_thickness
    inner_area = (np.pi / 4) * (inner_diameter ** 2)
    inner_volume = inner_area * pile_length
    volume_steel = outer_volume - inner_volume
    steel_mass = 8000 * volume_steel  # in kg

    structural_mass = {'Low alloy steel': transition_mass + steel_mass,
                       'Cement': grout_mass, 'Gravel': scour_mass}
    monopile_name = str(park_name) + '_monopile'
    if not [act for act in new_db if monopile_name in act['name']]:
        new_act = new_db.new_activity(name=monopile_name, unit='unit', code=monopile_name)
        new_act['reference product'] = 'offshore turbine foundations, gravity-based'
        new_act.save()

        for material, mass in structural_mass.items():
            if material == 'Low alloy steel':
                inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                      recycled_share=recycled_share_steel,
                                                      electricity_mix=electricity_mix_steel)
                ex = new_act.new_exchange(input=inp, type='technosphere', amount=mass, unit="kilogram")
                ex.save()
                new_act.save()
            elif material == 'Concrete_foundations':
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass / 2400, unit="cubic meter",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
            else:
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass, unit="kilogram",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print('A monopile foundation activity was created.')
        return new_act


# gravity-based
def gravity_parameters(sea_depth: float, power: float, park_name: str, commissioning_year: int,
                       recycled_share_steel: float = None,
                       electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None,
                       new_db=bd.Database('new_db')):
    """
    Data for a 3 MW turbine from Tsai et al., 2016.
    Assumption:
    - Linear equation between power and foundations mass (in onshore turbines the relation is almost linear)
    Limitations:
    - The sea depth increase is always modeled as an increase in the transtion element and not the general structure
      (and this might not be the case)
    - No uncertainty "declared"
    """
    ### NOTA: power in kW
    power = power * 1000  # transform to kW from MW
    # data for a 3MW turbine
    t_reinforcing_steel_mass = 336000  # in kg
    t_concrete_mass = 1027 * 2.4 * 1000  # in kg
    t_gravel_mass = 12200000  # in kg
    # transition mass (steel) equivalent to the case if the foundation was a monopile (but without a penetration depth)
    pile_length = 9 + sea_depth
    transition_lenght = 0.367 * pile_length + 1.147  # in meters
    t_steel_mass = (15.23 * transition_lenght - 73.86) * 1000  # in kg (steel)

    # recalculated mass
    reinforcing_steel_mass = t_reinforcing_steel_mass * power / 3000
    concrete_mass = t_concrete_mass * power / 3000
    gravel_mass = t_gravel_mass * power / 3000
    steel_mass = t_steel_mass * power / 3000

    structural_mass = {'Low alloy steel': reinforcing_steel_mass + steel_mass, 'Concrete_foundations': concrete_mass,
                       'Gravel': gravel_mass}

    gravity_name = str(park_name) + '_gravity'
    if not [act for act in new_db if gravity_name in act['name']]:
        new_act = new_db.new_activity(name=gravity_name, unit='unit', code=gravity_name)
        new_act['reference product'] = 'offshore turbine foundations, gravity-based'
        new_act.save()

        for material, mass in structural_mass.items():
            if material == 'Low alloy steel':
                inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                      recycled_share=recycled_share_steel,
                                                      electricity_mix=electricity_mix_steel)
                ex = new_act.new_exchange(input=inp, type='technosphere', amount=mass, unit="kilogram")
                ex.save()
                new_act.save()
            elif material == 'Concrete_foundations':
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass / 2400, unit="cubic meter",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
            else:
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass, unit="kilogram",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print('A gravity-based foundation activity was created.')
        return new_act


# tripod
def tripod_parameters(sea_depth: float, power: float, park_name: str, commissioning_year: int,
                      recycled_share_steel: float = None,
                      electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None,
                      new_db=bd.Database('new_db')):
    """
        Data for a 3 MW turbine from Tsai et al., 2016.
        Assumption:
        - Linear equation between power and foundations mass (in onshore turbines the relation is almost linear)
        Limitations:
        - The sea depth increase is always modeled as an increase in the transition element and not the general
          structure (and this might not be the case)
        - No uncertainty "declared"
    """
    ### NOTA: power in kW
    power = power * 1000  # transform to kW from MW
    # data for a 3MW turbine
    t_steel_frame = 807000  # in kg
    t_steel_piles = 847000  # in kg
    t_concrete_mass = 63900  # in kg
    # transition mass (steel) equivalent to the case if the foundation was a monopile
    penetration_depth = 19.4002294 * power + 0.000944015847  # penetration in the seabed
    pile_length = 9 + penetration_depth + sea_depth  # 9m added to the
    transition_lenght = 0.367 * pile_length + 1.147
    t_steel_transition = (15.23 * transition_lenght - 73.86) * 1000  # in kg (steel)
    t_total_steel_mass = t_steel_piles + t_steel_frame + t_steel_transition

    # recalculated mass
    steel_mass = t_total_steel_mass * power / 3000
    concrete_mass = t_concrete_mass * power / 3000

    structural_mass = {'Low alloy steel': steel_mass, 'Concrete_foundations': concrete_mass}  # in kg

    tripod_name = str(park_name) + '_tripod'
    if not [act for act in new_db if tripod_name in act['name']]:
        new_act = new_db.new_activity(name=tripod_name, unit='unit', code=tripod_name)
        new_act['reference product'] = 'offshore turbine foundations, tripod'
        new_act.save()

        for material, mass in structural_mass.items():
            if material == 'Low alloy steel':
                inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                      recycled_share=recycled_share_steel,
                                                      electricity_mix=electricity_mix_steel)
                ex = new_act.new_exchange(input=inp, type='technosphere', amount=mass, unit="kilogram")
                ex.save()
                new_act.save()
            else:
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass / 2400, unit="cubic meter",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print('A tripod foundation activity was created.')
        return new_act


def floating_parameters(power: float, platform_type: List[Literal['semi_sub', 'spar_buoy_concrete', 'spar_buoy_iron',
'spar_buoy_steel', 'tension_leg', 'barge']],
                        park_name: str, commissioning_year: int, recycled_share_steel: float = None,
                        electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None,
                        new_db=bd.Database('new_db')):
    """
    Given the type of platform type among the following ('semi_sub', 'spar_buoy_concrete', 'spar_buoy_iron',
    'spar_buoy_steel', 'tension_leg', 'barge'), it creates the floating platform activity in new_db.
    """
    if platform_type == 'semi_sub':
        structure_mass_intensity = {'Low alloy steel': {'mass': 479892.33, 'std': 66723.18},
                                    'Concrete_foundations': {'mass': 7629.84, 'std': 13215.27},
                                    'Cast iron': {'mass': 40388.33, 'std': 69954.65},
                                    'hdpe': {'mass': 0, 'std': 0}
                                    }
    elif platform_type == 'spar_buoy_concrete':
        structure_mass_intensity = {'Low alloy steel': {'mass': 159477.81, 'std': 98377.13},
                                    'Concrete_foundations': {'mass': 835105.52, 'std': 98377.13},
                                    'Cast iron': {'mass': 0, 'std': 0},
                                    'hdpe': {'mass': 0, 'std': 0}
                                    }
    elif platform_type == 'spar_buoy_iron':
        structure_mass_intensity = {'Low alloy steel': {'mass': 585909.33, 'std': 0},
                                    'Concrete_foundations': {'mass': 0, 'std': 0},
                                    'Cast iron': {'mass': 832757.33, 'std': 0},
                                    'hdpe': {'mass': 0, 'std': 0}
                                    }
    elif platform_type == 'spar_buoy_steel':
        structure_mass_intensity = {'Low alloy steel': {'mass': 257585.67, 'std': 11326.45},
                                    'Concrete_foundations': {'mass': 6539.33, 'std': 11326.45},
                                    'Cast iron': {'mass': 0, 'std': 0},
                                    'hdpe': {'mass': 0, 'std': 0}
                                    }
    elif platform_type == 'tension_leg':
        structure_mass_intensity = {'Low alloy steel': {'mass': 180230.56, 'std': 11081.93},
                                    'Concrete_foundations': {'mass': 0, 'std': 0},
                                    'Cast iron': {'mass': 0, 'std': 0},
                                    'hdpe': {'mass': 13668.51, 'std': 11081.93}
                                    }
    elif platform_type == 'barge':
        structure_mass_intensity = {'Low alloy steel': {'mass': 456250.00, 'std': 0},
                                    'Concrete_foundations': {'mass': 2175000.00, 'std': 0},
                                    'Cast iron': {'mass': 30000, 'std': 0},
                                    'hdpe': {'mass': 0, 'std': 0}
                                    }
    else:
        print(str(platform_type) + " is not a valid platform type. Try one among this list: 'semi_sub', "
                                   "'spar_buoy_concrete', 'spar_buoy_iron', 'spar_buoy_steel', 'tension_leg', 'barge'")
        sys.exit()

    floating_platform_name = f"{str(park_name)}_{str(platform_type)}_floating_platform"
    if not [act for act in new_db if floating_platform_name in act['name']]:
        new_act = new_db.new_activity(name=floating_platform_name, unit='unit', code=floating_platform_name)
        new_act['reference product'] = f'offshore turbine foundations, floating, {floating_platform_name}'
        new_act.save()

        for material, data in structure_mass_intensity.items():
            if material == 'Low alloy steel':
                inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                      recycled_share=recycled_share_steel,
                                                      electricity_mix=electricity_mix_steel)
                ex = new_act.new_exchange(input=inp, type='technosphere', amount=data['mass'] * power, unit="kilogram")
                ex['uncertainty type'] = NormalUncertainty.id
                ex['loc'] = data['mass'] * power
                ex['scale'] = data['std'] * power
                ex['minimum'] = 0
                ex.save()
                new_act.save()
            elif material == 'Concrete_foundations':
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=data['mass'] * power / 2400, unit="kilogram",
                                               type='technosphere')
                # Uncertainty added as the standard deviation of the residuals
                new_exc['uncertainty type'] = NormalUncertainty.id
                new_exc['loc'] = data['mass'] * power
                new_exc['scale'] = data['std'] * power
                new_exc['minimum'] = 0
                new_exc.save()
                new_act.save()
            else:
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=data['mass'] * power, unit="kilogram",
                                               type='technosphere')
                # Uncertainty added as the standard deviation of the residuals
                new_exc['uncertainty type'] = NormalUncertainty.id
                new_exc['loc'] = data['mass'] * power
                new_exc['scale'] = data['std'] * power
                new_exc['minimum'] = 0
                new_exc.save()
                new_act.save()
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print(f'A floating platform activity was created of the type {str(platform_type)}')
        return new_act


# 1.3 intra-array cables
def submarine_cables(rotor_diameter: float, distance_to_shore: float, park_name: str, commissioning_year: int,
                     recycled_share_steel: float = None,
                     electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None):
    """
    It creates a dictionary mat_cable_output with the mass of the materials of the cables in kg.
    Data sources:
        - Material intensities (kg/m), from Brussa et al., 2023
        - Turbines spacing (7.7D), from Bosch et al., 2019
    Limitations:
        - 7.7D is the mean spacing of offshore wind parks in UK in 2019. It was 7.2D in Round 2 and 8.4D in Round 3, so
          the trend is that the spacing increases (probably to avoid the wake effect).
        - Data on material intensities from Brussa et al., 2023, where they model a huge OWF (190 turbines, 17MW,
          3 substations), so the cables are for a huge OWF, which might not be the case of current mostly test sites.
        - We are assuming that the shore transformer is exactly in the costal line, so no inland cables are required.
    """

    # MVAC 66kV mat list
    mat_list = ['Aluminium', 'hdpe', 'Lead', 'Low alloy steel']

    # MVAC 66kV 120mm2 intensities (kg/m)
    mvac_120 = {'Aluminium': 0.97, 'hdpe': 5, 'Lead': 6.4, 'Low alloy steel': 7.6}

    # MVAC 66kV 500mm2 intensities (kg/m)
    mvac_500 = {'Aluminium': 4, 'hdpe': 11, 'Lead': 16, 'Low alloy steel': 19}

    # HVDC 500kV export intensity (kg/m)
    hvdc = {'Copper': 17.8, 'Paper': 6.1, 'Lubricating oil': 4.7, 'Lead': 17.2, 'Low alloy steel': 19.2,
            'Asphalt': 0.73, 'PP': 2.2}

    # intra-array cabling (35% MVAC 120mm2, 65% MVAC 500mm2. Ref: Brussa et al., 2023)
    intra_array_mat_mean = {}  # in kg
    for mat in mat_list:
        intra_array_mat_mean[mat] = 7.7 * rotor_diameter * (
                0.35 * mvac_120[mat] + 0.65 * mvac_500[mat])  # 7.7D mean spacing offshore ref. Bosch et al., 2019

    # substation to shore cabling
    subs_to_shore_mat_mass = {}
    for mat in hvdc.keys():
        subs_to_shore_mat_mass[mat] = 7.7 * distance_to_shore * hvdc[mat]
        mat_list.append(mat)

    # total material
    mat_cable_output = {}
    total_mat_list = list(set(mat_list))
    for mat in total_mat_list:
        if mat in intra_array_mat_mean.keys() and mat in subs_to_shore_mat_mass.keys():
            mat_cable_output[mat] = intra_array_mat_mean[mat] + subs_to_shore_mat_mass[mat]
        elif mat in intra_array_mat_mean.keys():
            mat_cable_output[mat] = intra_array_mat_mean[mat]
        else:
            mat_cable_output[mat] = subs_to_shore_mat_mass[mat]

    cabling_offshore_name = str(park_name) + '_cabling_offshore'
    if not [act for act in new_db if cabling_offshore_name in act['name']]:
        new_act = new_db.new_activity(name=cabling_offshore_name, unit='unit', code=cabling_offshore_name)
        new_act['reference product'] = 'offshore turbine, cabling'
        new_act.save()

        for material, mass in mat_cable_output.items():
            if material == 'Low alloy steel':
                inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                      recycled_share=recycled_share_steel,
                                                      electricity_mix=electricity_mix_steel)
                ex = new_act.new_exchange(input=inp, type='technosphere', amount=mass, unit="kilogram")
                ex.save()
                new_act.save()
            else:
                material_act = cutoff391.get(code=MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
                new_exc = new_act.new_exchange(input=material_act, amount=mass, unit="kilogram",
                                               type='technosphere')
                new_exc.save()
                new_act.save()
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print('A submarine cables activity was created.')
        return new_act


# 1.4 substation
# Transformer substructure + HV transformer.
def substation_platform(park_name: str, commissioning_year: int,
                        recycled_share_steel: float = None,
                        electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None):
    """
    It creates an activity for the substation platform.
    Data from Brussa et al., 2023. (Supplementary SM6, Level 5)
    """
    semi_submersible_plat = 4014000  # steel, low alloyed in kg
    ballast_fixed = 2540000  # iron ore concentrate in kg
    ballast_fluid = 11300000  # water, unspecified natural origin in kg

    substation_name = f"{str(park_name)}_floating_substation"
    if not [act for act in new_db if substation_name in act['name']]:
        new_act = new_db.new_activity(name=substation_name, unit='unit', code=substation_name)
        new_act['reference product'] = 'offshore turbine, substation'
        new_act.save()

        # iron
        iron = cutoff391.get(code='b3d48f2f5446c645c128b06b5de93f21')  # cast iron
        new_exc = new_act.new_exchange(input=iron.key, amount=ballast_fixed, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # water
        water = biosphere3.get('478e8437-1c21-4032-8438-872a6b5ddcdf')  # water
        new_exc = new_act.new_exchange(input=water.key, amount=ballast_fluid / 1000, unit="cubic meter",
                                       type='biosphere')
        new_exc.save()
        new_act.save()

        # steel
        inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                              recycled_share=recycled_share_steel,
                                              electricity_mix=electricity_mix_steel)
        ex = new_act.new_exchange(input=inp, type='technosphere', amount=semi_submersible_plat, unit="kilogram")
        ex.save()
        new_act.save()

        # output
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

        print('A substation platform for the transformer was created.')
        return new_act


def materials_foundations_offshore(offshore_type: List[Literal['monopile', 'gravity', 'tripod', 'floating']],
                                   power: float, sea_depth: float, park_name: str, commissioning_year: int,
                                   floating_platform: List[Literal['semi_sub', 'spar_buoy_concrete', 'spar_buoy_iron',
                                   'spar_buoy_steel', 'tension_leg', 'barge']] = None,
                                   recycled_share_steel: float = None,
                                   electricity_mix_steel: List[Literal['Europe', 'Poland', 'Norway']] = None
                                   ):
    """
    :return:
    """
    # structural materials
    if offshore_type == 'monopile':
        foundations_act = monopile_parameters(sea_depth=sea_depth, power=power, park_name=park_name,
                                              commissioning_year=commissioning_year,
                                              recycled_share_steel=recycled_share_steel,
                                              electricity_mix_steel=electricity_mix_steel)
    elif offshore_type == 'gravity':
        foundations_act = gravity_parameters(sea_depth=sea_depth, power=power, park_name=park_name,
                                             commissioning_year=commissioning_year,
                                             recycled_share_steel=recycled_share_steel,
                                             electricity_mix_steel=electricity_mix_steel)
    elif offshore_type == 'tripod':
        foundations_act = tripod_parameters(sea_depth=sea_depth, power=power, park_name=park_name,
                                            commissioning_year=commissioning_year,
                                            recycled_share_steel=recycled_share_steel,
                                            electricity_mix_steel=electricity_mix_steel)
    elif offshore_type == 'floating':
        foundations_act = floating_parameters(power=power, platform_type=floating_platform, park_name=park_name,
                                              commissioning_year=commissioning_year,
                                              recycled_share_steel=recycled_share_steel,
                                              electricity_mix_steel=electricity_mix_steel)
    else:
        print(str(offshore_type) + ' is not an allowed parameter for the offshore type. Try "monopile", "tripod", '
                                   '"gravel" or "floating" instead.')
        sys.exit()
    return foundations_act


def offshore_turbine_materials(
        park_name: str, park_power: float, number_of_turbines: int, park_location: str,
        park_coordinates: tuple,
        manufacturer: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
        rotor_diameter: float,
        turbine_power: float, hub_height: float, commissioning_year: int,
        offshore_type: List[Literal['monopile', 'gravity', 'tripod', 'floating']],
        sea_depth: float, distance_to_shore: float,
        floating_platform: List[Literal['semi_sub', 'spar_buoy_concrete', 'spar_buoy_iron',
        'spar_buoy_steel', 'tension_leg', 'barge']] = None,
        recycled_share_steel: float = None,
        lifetime: int = 20,
        electricity_mix_steel: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
        generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'] = 'gb_dfig',
        new_db=bd.Database('new_db')
):
    """
    It creates an activity 'park_name_turbine_materials' that contains:
    - Materials of the turbine
    - Materials of the foundations
    - Materials of the cabling
    - Materials of the substation
    It returns this activity AND the manufacturing and eol activities
    """
    # 1. create turbine without foundations
    lci_materials(park_name=park_name, park_power=park_power, number_of_turbines=number_of_turbines,
                  park_location=park_location, park_coordinates=park_coordinates, manufacturer=manufacturer,
                  rotor_diameter=rotor_diameter, turbine_power=turbine_power, hub_height=hub_height,
                  commissioning_year=commissioning_year, recycled_share_steel=recycled_share_steel, lifetime=lifetime,
                  electricity_mix_steel=electricity_mix_steel, generator_type=generator_type
                  )
    materials_act = new_db.get(f'{park_name}_materials')
    materials_act['name'] = f'{park_name}_turbine_materials'
    materials_act.save()
    deletable_ex = [e for e in materials_act.technosphere()][13:16]
    for e in deletable_ex:
        e.delete()
    manufacturing_act = new_db.get(f'{park_name}_manufacturing')
    deletable_ex = [e for e in manufacturing_act.technosphere()][6:8]
    for e in deletable_ex:
        e.delete()
    transport_act = new_db.get(f'{park_name}_transport')
    transport_act.delete()
    installation_act = new_db.get(f'{park_name}_installation')
    installation_act.delete()
    om_act = new_db.get(f'{park_name}_maintenance')
    om_act.delete()
    eol_act = new_db.get(f'{park_name}_eol')

    # 2. create offshore foundations
    foundations_act = materials_foundations_offshore(offshore_type=offshore_type, power=turbine_power,
                                                     sea_depth=sea_depth,
                                                     park_name=park_name, commissioning_year=commissioning_year,
                                                     floating_platform=floating_platform,
                                                     recycled_share_steel=recycled_share_steel,
                                                     electricity_mix_steel=electricity_mix_steel)
    # 3. create offshore cables
    cables_act = submarine_cables(rotor_diameter=rotor_diameter, distance_to_shore=distance_to_shore,
                                  park_name=park_name,
                                  commissioning_year=commissioning_year, recycled_share_steel=recycled_share_steel,
                                  electricity_mix_steel=electricity_mix_steel)
    # 4. create offshore substation
    substation_act = substation_platform(park_name=park_name, commissioning_year=commissioning_year,
                                         recycled_share_steel=recycled_share_steel,
                                         electricity_mix_steel=electricity_mix_steel)
    # 5. put together all materials in one activity
    offshore_materials_act = new_db.new_activity(name=f'{park_name}_offshore_materials', unit='unit',
                                                 code=f'{park_name}_offshore_materials')
    offshore_materials_act['reference product'] = 'offshore turbine, materials'
    offshore_materials_act.save()
    new_ex = offshore_materials_act.new_exchange(input=offshore_materials_act.key, type='production', amount=1)
    new_ex.save()
    for act in [materials_act, foundations_act, cables_act, substation_act]:
        new_ex = offshore_materials_act.new_exchange(input=act, type='technosphere', amount=1)
        new_ex.save()

    return offshore_materials_act, manufacturing_act, eol_act, foundations_act, cables_act, substation_act


def offshore_manufacturing(park_name: str, park_power: float, number_of_turbines: int, park_location: str,
                           park_coordinates: tuple,
                           manufacturer: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                           rotor_diameter: float,
                           turbine_power: float, hub_height: float, commissioning_year: int,
                           offshore_type: List[Literal['monopile', 'gravity', 'tripod', 'floating']],
                           sea_depth: float, distance_to_shore: float,
                           floating_platform: List[Literal['semi_sub', 'spar_buoy_concrete', 'spar_buoy_iron',
                           'spar_buoy_steel', 'tension_leg', 'barge']] = None,
                           recycled_share_steel: float = None,
                           lifetime: int = 20,
                           electricity_mix_steel: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                           generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'] = 'gb_dfig',
                           new_db=bd.Database('new_db')):
    """
    It creates an activity for the manufacturing of substation, foundations and cabling, respectively,
    and add its inputs using the activities in consts.MANUFACTURING. It later creates an activity for the manufacturing
    of everything, where all the substation, cabling, foundation activities AND TOWER MANUFACTURING are added.
    This activity is called {park_name}_offshore_manufacturing
    """

    (offshore_materials_act, manufacturing_act, eol_act,
     foundations_act, cables_act, substation_act) = offshore_turbine_materials(
        park_name=park_name, park_power=park_power, number_of_turbines=number_of_turbines, park_location=park_location,
        park_coordinates=park_coordinates, manufacturer=manufacturer, rotor_diameter=rotor_diameter,
        turbine_power=turbine_power, hub_height=hub_height, commissioning_year=commissioning_year,
        offshore_type=offshore_type, sea_depth=sea_depth, floating_platform=floating_platform,
        recycled_share_steel=recycled_share_steel, lifetime=lifetime,
        electricity_mix_steel=electricity_mix_steel, generator_type=generator_type, new_db=new_db
        )

    # create foundations manufacturing
    foundations_man_act = new_db.new_activity(name=f'{park_name}_foundations_manufacturing', unit='unit', location=park_location)
    foundations_man_act['reference product'] = 'offshore turbine foundations, manufacturing'
    foundations_man_act.save()
    new_ex = foundations_man_act.new_exchange(input=foundations_man_act.key, type='production', amount=1)
    new_ex.save()
    ex_foundations = [e for e in foundations_act.technosphere()]
    for e in ex_foundations:
        # steel, concrete, iron, hdpe, gravel
        if 'steel' in e.input['name'] or 'iron' in e.input['name']:
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code']
            new_ex = foundations_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        elif 'hdpe' in e.input['name']:
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['PVC']['code']
            new_ex = foundations_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        else:
            pass
    # create manufacturing process for cabling
    cabling_man_act = new_db.new_activity(name=f'{park_name}_cabling_manufacturing', unit='unit', location=park_location)
    cabling_man_act['reference product'] = 'offshore turbine cabling, manufacturing'
    cabling_man_act.save()
    new_ex = cabling_man_act.new_exchange(input=cabling_man_act.key, type='production', amount=1)
    new_ex.save()
    ex_cables = [e for e in cables_act.technosphere()]
    for e in ex_cables:
        if 'steel' in e.input['name'] or 'iron' in e.input['name']:
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code']
            new_ex = cabling_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        elif 'hdpe' in e.input['name'] or 'polypropylene':
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['PVC']['code']
            new_ex = cabling_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        elif 'copper' in e.input['name'] or 'aluminium' in e.input['name']:
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Copper']['code']
            new_ex = cabling_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        else:
            pass
    # create substation manufacturing process
    substation_man_act = new_db.new_activity(name=f'{park_name}_substation_manufacturing', unit='unit', location=park_location)
    substation_man_act['reference product'] = 'offshore turbine substation, manufacturing'
    substation_man_act.save()
    new_ex = substation_man_act.new_exchange(input=substation_man_act.key, type='production', amount=1)
    new_ex.save()
    ex_substation = [e for e in substation_act.technosphere()]
    for e in ex_substation:
        # steel, iron, water
        if 'steel' in e.input['name'] or 'iron' in e.input['name']:
            input_act = consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code']
            new_ex = substation_man_act.new_exchange(input=input_act, type='technosphere', amount=e.amount)
            new_ex.save()
        else:
            pass

    # create manufacturing process
    new_act = new_db.new_activity(name=f'{park_name}_offshore_manufacturing', unit='unit', location=park_location)
    new_act['reference product'] = 'offshore turbine, manufacturing'
    new_act.save()
    new_ex = new_act.new_exchange(input=new_act.key, type='production', amount=1)
    new_ex.save()
    for act in [substation_man_act, foundations_man_act, cabling_man_act, manufacturing_act]:
        new_ex = new_act.new_exchange(input=act, type='technosphere', amount=1)
        new_ex.save()

    return new_act, eol_act
