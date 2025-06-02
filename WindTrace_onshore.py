import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bw2data as bd
from geopy.distance import geodesic
import random
from typing import Optional, List, Literal, Tuple
from stats_arrays import NormalUncertainty
from statistics import linear_regression
import sys
import consts

# TODO: update documentation
# TODO: do a few tests on the new material_mass functions

def steel_turbine(plot_mat: bool = False, regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    It returns a dictionary 'materials_polyfits' that contains the fitting curve of steel (mass vs hub height). The
    dictionary has the keys 'polyfit' and 'confidence_95%' where the values are stored. The uncertainty (to be added
    to the lci) is stored as 'std_dev' also in the same dictionary and corresponds to the standard deviation of the
    residuals. If plot_mat is set to True, all the materials fitting plots will be shown.
    """
    vestas_data = pd.read_excel(consts.VESTAS_FILE, sheet_name="1_MATERIALS_TURBINE", dtype=None, decimal=";", header=0)

    if regression_adjustment == 'Hub height':
        short_vestas_data = vestas_data[vestas_data['Hub height'] <= 84]
        # Extracting columns
        x = vestas_data['Hub height']
    else:
        short_vestas_data = vestas_data[vestas_data['D2h'] <= 1053696]
        x = vestas_data['D2h']
    y = vestas_data['Low alloy steel']
    # Remove NaN values
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    new_x = x[valid_indices]
    new_y = y[valid_indices]
    # Dictionary to save the polyfits and confidence intervals
    materials_polyfits = {}
    materials_polyfits_short = {}
    # Linear regression (steel mass vs height) and statistics
    fit_steel = np.polyfit(new_x, new_y, 1)
    predict_steel = np.poly1d(fit_steel)
    # Calculate residuals
    residuals = new_y - predict_steel(new_x)
    # Calculate standard error of the estimate
    std_error = np.sqrt(np.mean(residuals ** 2))
    # Calculate confidence intervals (95%) for interpolated x values
    confidence = 1.96 * std_error  # 95% confidence interval multiplier
    residual_variance = np.mean(residuals ** 2)
    residual_std_dev = np.sqrt(residual_variance)
    # long_short = {}
    polyfit_and_confidence = {'polyfit': predict_steel, 'confidence_95%': confidence, 'std_dev': residual_std_dev}
    materials_polyfits['Low alloy steel'] = polyfit_and_confidence
    # Extract short data
    if regression_adjustment == 'Hub height':
        short_x = short_vestas_data['Hub height']
    else:
        short_x = short_vestas_data['D2h']
    short_y = short_vestas_data['Low alloy steel']
    slope, intercept = linear_regression(short_x, short_y, proportional=True)
    # slope, intercept = linear_regression(short_x, short_y)
    short_predict_steel = np.poly1d([slope, intercept])
    # Calculate residuals
    residuals = short_y - short_predict_steel(short_x)
    # Calculate standard error of the estimate
    std_error = np.sqrt(np.mean(residuals ** 2))
    # Calculate confidence intervals (95%) for interpolated x values
    confidence = 1.96 * std_error  # 95% confidence interval multiplier
    # residual_variance = np.mean(residuals ** 2)
    residual_std_dev = np.sqrt(residual_variance)
    # We mantain the same confidence and std_dev as the main function.
    polyfit_and_confidence_short = {'polyfit': short_predict_steel, 'confidence_95%': confidence,
                                    'std_dev': residual_std_dev}
    if plot_mat:
        plot_materials(x=short_x, y=short_y, residuals=residuals, interpolation_eq=short_predict_steel,
                       confidence=confidence,
                       xlabel='Hub height (m)', ylabel='Steel mass (t)', title='Steel')
    materials_polyfits_short['Low alloy steel'] = polyfit_and_confidence_short

    # where do the linear equations intersect?
    intersection_poly = np.poly1d(short_predict_steel - predict_steel)
    intersection_x = np.roots(intersection_poly)
    intersection = {'Low alloy steel': intersection_x}

    return vestas_data, materials_polyfits, materials_polyfits_short, intersection


def other_turbine_materials(plot_mat=False, regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h') -> (tuple, dict, dict):
    """
    It returns a dictionary 'materials_polyfits' that contains the fitting curves of steel and turbine materials.
    The dictionary has the keys 'polyfit' and 'confidence_95%' where the values are stored.
    If plot_mat is set to True, all the materials fitting plots will be shown.
    """
    (vestas_data, materials_polyfits,
     materials_polyfits_short, intersection) = steel_turbine(regression_adjustment=regression_adjustment)
    columns = list(vestas_data)
    last_index = columns.index('Lubricating oil')
    initial_index = columns.index('Low alloy steel') + 1

    while initial_index <= last_index:
        short = False
        materials_to_adjust_3mw = ['PUR', 'PVC']
        materials_to_adjust_1mw = ['Low alloy steel', 'Chromium steel', 'Epoxy resin', 'Fiberglass', 'Rubber',
                                   'Aluminium']
        if columns[initial_index] in materials_to_adjust_3mw:
            short_vestas_data = vestas_data[vestas_data['Power (MW)'] <= 3.0]
            short = True
        elif columns[initial_index] in materials_to_adjust_1mw:
            short_vestas_data = vestas_data[vestas_data['Power (MW)'] <= 1.0]
            short = True
        x = vestas_data[columns[columns.index('Power (MW)')]]  # power (MW)
        y = vestas_data[columns[initial_index]]  # material mass (t)
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        new_x = x[valid_indices]
        new_y = y[valid_indices]
        fit = np.polyfit(new_x, new_y, 1)
        predict_mat = np.poly1d(fit)
        residuals = new_y - predict_mat(new_x)
        std_error = np.sqrt(np.mean(residuals ** 2))
        confidence = 1.96 * std_error
        residual_variance = np.mean(residuals ** 2)
        residual_std_dev = np.sqrt(residual_variance)
        polyfit_and_confidence = {'polyfit': predict_mat, 'confidence_95%': confidence, 'std_dev': residual_std_dev}
        materials_polyfits[columns[initial_index]] = polyfit_and_confidence
        if short:
            short_x = short_vestas_data[columns[columns.index('Power (MW)')]]
            short_y = short_vestas_data[columns[initial_index]]
            valid_indices = ~np.isnan(x) & ~np.isnan(y)
            short_x = short_x[valid_indices]
            short_y = short_y[valid_indices]
            slope, intercept = linear_regression(short_x, short_y, proportional=True)
            # slope, intercept = linear_regression(short_x, short_y)
            short_predict_mat = np.poly1d([slope, intercept])
            residuals = short_y - short_predict_mat(short_x)
            # Calculate standard error of the estimate
            std_error = np.sqrt(np.mean(residuals ** 2))
            # Calculate confidence intervals (95%) for interpolated x values
            confidence = 1.96 * std_error  # 95% confidence interval multiplier
            # residual_variance = np.mean(residuals ** 2)
            residual_std_dev = np.sqrt(residual_variance)
            polyfit_and_confidence_short = {'polyfit': short_predict_mat, 'confidence_95%': confidence,
                                            'std_dev': residual_std_dev}
            materials_polyfits_short[columns[initial_index]] = polyfit_and_confidence_short

            intersection_poly = np.poly1d(short_predict_mat - predict_mat)
            intersection_x = np.roots(intersection_poly)
            intersection[columns[initial_index]] = intersection_x

        if plot_mat:
            plot_materials(x=new_x, y=new_y, residuals=residuals, interpolation_eq=predict_mat, confidence=confidence,
                           xlabel='Power (MW)', ylabel=columns[initial_index] + ' (t)', title=columns[initial_index])
        initial_index += 1
    return materials_polyfits, materials_polyfits_short, intersection


def rare_earth(generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']):
    """
    It returns a dictionary 'rare_earth_int' that contains the intensities of the rare earth materials according to the
    generator type that the turbine uses.
    Material intensity data according to Ferrara et al. (2020). Units: t/GW
    generator_type: accepted arguments 'dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'.
    """

    rare_earth_int = {'Praseodymium': consts.RARE_EARTH_DICT['Praseodymium'][generator_type],
                      'Neodymium': consts.RARE_EARTH_DICT['Neodymium'][generator_type],
                      'Dysprosium': consts.RARE_EARTH_DICT['Dysprosium'][generator_type],
                      'Terbium': consts.RARE_EARTH_DICT['Terbium'][generator_type],
                      'Boron': consts.RARE_EARTH_DICT['Boron'][generator_type]}
    return rare_earth_int


def foundations_mat(mat_file: str, plot_mat=False, regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    It returns a dictionary 'materials_polyfits' that contains the fitting curves of all the materials (steel, turbine
    and foundations). The dictionary has the keys 'polyfit' and 'confidence_95%' where the values are stored.
    If plot_mat is set to True, all the materials fitting plots will be shown.
    """
    (materials_polyfits, mat_polyfits_short,
     intersection) = other_turbine_materials(regression_adjustment=regression_adjustment)
    vestas_data = pd.read_excel(mat_file, sheet_name="1_MATERIALS_FOUNDATIONS", dtype=None, decimal=";", header=0)
    columns = list(vestas_data)
    last_index = columns.index('Concrete')
    initial_index = columns.index('Low alloy steel')
    while initial_index <= last_index:
        x = vestas_data[
            columns[columns.index('Power (MW)')]]  # power (MW). Maybe in the future I use tip momentum (D^2*h)
        y = vestas_data[columns[initial_index]]  # material mass (t)
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        new_x = x[valid_indices]
        new_y = y[valid_indices]
        fit = np.polyfit(new_x, new_y, 1)
        predict_mat = np.poly1d(fit)
        residuals = new_y - predict_mat(new_x)
        residual_variance = np.mean(residuals ** 2)
        residual_std_dev = np.sqrt(residual_variance)
        std_error = np.sqrt(np.mean(residuals ** 2))
        confidence = 1.96 * std_error
        polyfit_and_confidence = {}
        polyfit_and_confidence['polyfit'] = predict_mat
        polyfit_and_confidence['confidence_95%'] = confidence
        polyfit_and_confidence['std_dev'] = residual_std_dev
        materials_polyfits[columns[initial_index] + '_foundations'] = polyfit_and_confidence
        if plot_mat:
            plot_materials(x=new_x, y=new_y, residuals=residuals, interpolation_eq=predict_mat, confidence=confidence,
                           xlabel='Power (MW)', ylabel=columns[initial_index] + ' (t)', title=columns[initial_index])
        initial_index += 1
    return materials_polyfits, mat_polyfits_short, intersection


def plot_materials(x, y, residuals, interpolation_eq, confidence, xlabel: str, ylabel: str, title: str, grid=True,
                   adjusted_plot=True):
    """
    for the scatter points of x and y, given the residuals, fitting curve (interpolation_eq), and confidence 95% (value
    that stablishes the minimim and maximum deviation from the mean that guarantees that 95% of the values will fall in
    that range), it shows the corresponding plot. It's not saving it, just showing.
    Note:
    The variable adjusted_plot allows to extend the plot from 0 to 15 MW.
    """
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_residual = np.sum(residuals ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    if adjusted_plot:
        x_interpolate = np.linspace(min(x), max(x), 100)
        y_interpolated = interpolation_eq(x_interpolate)
    else:
        x_interpolate = np.linspace(0, 15, 100)
        y_interpolated = interpolation_eq(x_interpolate)
    # Plot the data, fitted curve, and confidence interval
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue')
    plt.plot(x_interpolate, y_interpolated, color='red')
    plt.fill_between(x_interpolate, y_interpolated - confidence, y_interpolated + confidence,
                     color='red', alpha=0.2)
    plt.annotate(f"R-squared = {r_squared:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    plt.show()


def materials_mass(generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'],
                   turbine_power: float, hub_height: float, rotor_diameter: float,
                   regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    returns a dictionary 'mass_materials' with material names as keys and their masses in kg as values.
    generator_type: it only accepts the models (strings) 'dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'.
    """
    mass_materials = {}
    (materials_polyfits, mat_polyfits_short,
     intersection) = foundations_mat(mat_file=consts.VESTAS_FILE, regression_adjustment=regression_adjustment)

    uncertainty = {}
    if regression_adjustment == 'Hub height':
        turbine_power_is_larger = hub_height > intersection['Low alloy steel'].item()
    else:
        turbine_power_is_larger = hub_height * rotor_diameter * rotor_diameter > intersection['Low alloy steel'].item()
    if not turbine_power_is_larger:
        materials_polyf = mat_polyfits_short
    else:
        materials_polyf = materials_polyfits
    uncertainty['Low alloy steel'] = materials_polyf['Low alloy steel']['std_dev']
    if regression_adjustment == 'Hub height':
        steel_mass_turbine = materials_polyf['Low alloy steel']['polyfit'](hub_height) * 1000
    else:
        steel_mass_turbine = (materials_polyf['Low alloy steel']['polyfit']
                              (hub_height * rotor_diameter * rotor_diameter) * 1000)
    mass_materials['Low alloy steel'] = steel_mass_turbine

    is_larger_true_list = []
    for k in materials_polyfits.keys():
        if k in intersection.keys():
            turbine_power_is_larger = turbine_power > intersection[k].item()
            is_larger_true_list.append(turbine_power_is_larger)
        else:
            is_larger_true_list.append(True)

    counter = 0
    for k in materials_polyfits.keys():
        is_larger = is_larger_true_list[counter]
        if not is_larger:
            materials_polyf = mat_polyfits_short
        else:
            materials_polyf = materials_polyfits
        uncertainty[k] = materials_polyf[k]['std_dev']
        if k != 'Low alloy steel':
            # in kg instead of tonnes
            mass = materials_polyf[k]['polyfit'](turbine_power) * 1000
            if mass < 0:
                mass = 0.0
            mass_materials[k] = mass
        if k == 'Concrete_foundations':
            # transform concrete mass (t) to volume in m3
            volume = materials_polyf[k]['polyfit'](turbine_power) / 2.4

            if volume < 0:
                volume = 0.0
            mass_materials[k] = volume
        counter += 1

    rare_earth_int = rare_earth(generator_type)
    for k in rare_earth_int.keys():
        # in kg
        mass = rare_earth_int[k] * turbine_power
        mass_materials[k] = mass
    return mass_materials, uncertainty


def cabling_materials(turbine_power: float, rotor_diameter: float, number_of_turbines: int,
                      cu_density=8960, al_density=2700, pe_density=930, pvc_density=1400):
    """
    It returns a dictionary 'cable_mat_mass' that contains the mass in kg of copper, aluminium, polyethylene (PE) and
    polyvinyl chloride (PVC) of the cables.
    :param: park_size, power of the park in MW.
    Material densities data in kg/m3
    Cable section data from Nexans. Nexans cables 33kV 3-core are assumed. Data from Nexans'
    technical datasheets. Assumptions:
        1. buried cables
        2. 50% Cu - 50% Al
    Data from the technical characteristics (thewindpower.net)
    """
    distance_between_turbines = 8 * rotor_diameter  # 8-12D is recommended by McKenna et al., (2022) and also the
    # technical guidelines from ABB 'cuaderno técnico nº12 Planta eólicas' pp.16. We will consider that all the
    # turbines are always in line, so we don't apply the crosswind recommended distance of 4-6D.
    # To compensate we choose 8D, as it is the smallest distance in the range.
    total_distance = distance_between_turbines * (number_of_turbines - 1)
    section_y = np.array([50, 70, 95, 120, 150, 185, 240, 300, 400, 500])
    al_power_x = np.array([5.247, 6.402, 7.656, 8.712, 9.735, 11.022, 12.804, 14.454, 16.566, 18.81])
    cu_power_x = np.array([6.765, 8.25, 9.867, 11.235, 12.573, 14.223, 16.467, 18.579, 21.120, 23.694])
    cu_fit = np.polyfit(cu_power_x, section_y, 2)
    al_fit = np.polyfit(al_power_x, section_y, 2)
    predict_cu = np.poly1d(cu_fit)
    predict_al = np.poly1d(al_fit)
    cu_section = predict_cu(turbine_power) * 3  # Nexans' cross-section is per individual core and cables are three-core
    al_section = predict_al(turbine_power) * 3  # Nexans' cross-section is per individual core and cables are three-core
    cu_mass = cu_section / 1000000 * (total_distance / 2) * cu_density
    al_mass = al_section / 1000000 * (total_distance / 2) * al_density
    pe_mass_total = (cu_section + al_section) * (total_distance / 2) / 1000000 / 0.61 * 0.3 * pe_density
    pvc_mass_total = (cu_section + al_section) * (total_distance / 2) / 1000000 / 0.61 * 0.09 * pvc_density
    cable_mat_mass = {'Copper': cu_mass, 'Aluminium': al_mass, 'PE': pe_mass_total, 'PVC': pvc_mass_total}
    return cable_mat_mass


def mva500_transformer(new_db: bd.Database, cutoff391: bd.Database):
    """
    It creates the activity "Power transformer TrafoStar 500 MVA" in the database 'new_db' in brightway2. It returns
    the recently created transformer activity as a variable.
    Data from ABB. Code credits to Romain Sacchi et al. (with small changes)
    """
    if not [act for act in new_db if 'Power transformer TrafoStar 500 MVA' in act['name']]:
        new_act = new_db.new_activity(name="Power transformer TrafoStar 500 MVA", unit='unit', code='TrafoStar_500')
        new_act.save()

        # electric steel
        steel = cutoff391.get(code='b3d48f2f5446c645c128b06b5de93f21')
        new_exc = new_act.new_exchange(input=steel.key, amount=99640.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # transformer oil
        oil = cutoff391.get(code='92391c8c6958ada25b22935e3fa6f06f')
        new_exc = new_act.new_exchange(input=oil.key, amount=63000.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # copper
        copper = cutoff391.get(code='8b62f30ed586a5f23611ef196cc97b93')
        new_exc = new_act.new_exchange(input=copper.key, amount=39960.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # insulation
        insulation = cutoff391.get(code='1548660cbdd613eab4b00ddbd388c490')
        new_exc = new_act.new_exchange(input=insulation.key, amount=6500.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # wood
        wood = cutoff391.get(code='31d3bc7c09fc6efcd9c626cca48f6e47')
        new_exc = new_act.new_exchange(input=wood.key, amount=15000.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # porcelain
        porcelain = cutoff391.get(code='245eaef2fb637e428e0425deb295ec37')
        new_exc = new_act.new_exchange(input=porcelain.key, amount=2650.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # construction steel
        c_steel = cutoff391.get(code='d872e0d78319cb13e12b96de83e19dd7')
        new_exc = new_act.new_exchange(input=c_steel.key, amount=53618.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # paint
        paint = cutoff391.get(code='9291eac91d350e0a56be6f433a25ad3a')
        new_exc = new_act.new_exchange(input=paint.key, amount=2200.0, unit="kilogram", type='technosphere')
        new_exc.save()
        new_act.save()

        # electricity, medium
        elec = cutoff391.get(code='0e4b280caeeba40d5644b8d28328b0de')
        new_exc = new_act.new_exchange(input=elec.key, amount=750000.0, unit="kilowatt hour", type='technosphere')
        new_exc.save()
        new_act.save()

        # heat
        heat = cutoff391.get(code='e73087e282f26de5d3a9fec2edc19e61')
        new_exc = new_act.new_exchange(input=heat.key, amount=1080000.0, unit="megajoule", type='technosphere')
        new_exc.save()
        new_act.save()

        # output
        new_exc = new_act.new_exchange(input=new_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        new_act.save()

    transformer = new_db.get('TrafoStar_500')
    return transformer


def manipulate_steel_activities(new_db: bd.Database, cutoff391: bd.Database, commissioning_year: int, recycled_share: float = None,
                                electricity_mix: Optional[Literal['Europe', 'Poland', 'Norway']] = None,
                                printed_warning: bool = False):
    """
    This function creates a copy of the secondary and primary steel production activities in Ecoinvent
    and adapts the location of its electricity and gas inputs. The adaptation is made according to the share of steel
    production in Europe by country (between 2017 and 2021) reported in the European Steel in Figures Report 2022.
    Then, a market activity with the recycled and primary steel inputs depending on the year of manufacturing is
    created. If the turbine was manufactured before 2012, a mean value between data from 2012 and 2021 is taken.
    Assumptions:
    - Data refers to EU countries. Therefore, big producers outside the EU like Great Britain are excluded.
      As a consequence, we assume that no steel is produced in GB for the turbine making.
    - All countries produce the same share of recycled and non-recycled steel.
    - The consumption amount of gas and electricity in the furnaces does not change between countries.
    - The manufacture of the turbine takes place 1 year before the commissioning
    :param: recycled_share -> needs to be inputted with a value from 0 to 1 (or None)
    :param: electricity_mix -> ony accepts 'Europe', 'Poland' or 'Norway' (or None). For other values, 'Europe'
    is applied by default.
    """
    # test if the input variables are correct or not
    if recycled_share is not None and not printed_warning:
        if recycled_share > 1 or recycled_share < 0:
            print('WARNING. The recycling share must be inputed with values from 0 to 1.')
            sys.exit()
        if electricity_mix is None:
            print('WARNING. You did not select any electricity_mix. '
                  'The mean shares by country applied in the steel industry between 2017 and 2021 will be used')
        elif electricity_mix not in ['Europe', 'Poland', 'Norway']:
            print('WARNING. ' + str(electricity_mix) + ' is not an accepted input for the electricity_mix '
                                                       'variable. You can only use "Poland", '
                                                       '"Norway" or "Europe". A European electricity mix will '
                                                       'be applied by default!!')
    if str(commissioning_year) not in list(
            consts.secondary_steel.keys()) and recycled_share is None and not printed_warning:
        if commissioning_year > 2021:
            print('WARNING. This wind turbine was commissioned after 2021 for which WindTrace '
                  'does not have data from the steel industry. '
                  'We recommend you to specify an expected recycling rate if you have an estimation. '
                  'Otherwise, 41.6% of recycled steel will be considered by default.')
            print('Steel recycling share: 41.6%')
        elif commissioning_year < 2012:
            print('WARNING. This wind turbine was commissioned before 2012, for which WindTrace '
                  'does not have data from the steel industry. We recommend you to specify an expected recycling '
                  'rate if you have an estimation. Otherwise, 41.6% of recycled steel will be considered by '
                  'default.')
            print('Steel recycling share: 41.6%')

    if recycled_share is None and electricity_mix is None:
        if not printed_warning:
            print('WARNING. You did not select any electricity_mix. '
                  'The mean shares by country applied in the steel industry between 2017 and 2021 will be used')
            print('Electricity mix: European mix provided by Eurofer')
        act_name = "market for steel, low-alloyed, " + str(commissioning_year - 1)
        code_name = "steel, " + str(commissioning_year - 1)
        try:
            steel_act = new_db.get(code=code_name)
        except bd.errors.UnknownObject:
            steel_act = None
    else:
        act_name = "market for steel, low-alloyed, defined " + str(recycled_share) + str(electricity_mix)
        code_name = "steel, defined " + str(recycled_share) + str(electricity_mix)
        try:
            steel_act = new_db.get(code=code_name)
        except bd.errors.UnknownObject:
            steel_act = None

    # check if we already created a steel market for that year and skip if we did
    steel_act_check = 0
    if steel_act is not None:
        steel_act_check = len(steel_act.exchanges())

    if steel_act_check == 0:
        # find recycled steel production activity in Ecoinvent
        recycled_ei = cutoff391.get(code='b3d48f2f5446c645c128b06b5de93f21',
                                    name='steel production, electric, low-alloyed',
                                    location='Europe without Switzerland and Austria')
        # find primary steel production activity in Ecoinvent
        primary_ei = cutoff391.get(code='89cb4e1a47b707fe43b99135b81fcaba',
                                   name='steel production, converter, low-alloyed',
                                   location='RER')
        # Create a copy to manipulate them in the new_db database
        recycled_act = recycled_ei.copy(database=consts.NEW_DB_NAME)
        primary_act = primary_ei.copy(database=consts.NEW_DB_NAME)
        acts = [recycled_act, primary_act]

        # Manipulate both the primary and secondary activities in the same way
        for act in acts:
            # Calculate the total amount of gas and electricity inputs in the activity
            elect_ex = [e for e in act.technosphere() if
                        e.input._data['name'] == 'market for electricity, medium voltage'
                        or e.input._data['name'] == 'market group for electricity, medium voltage']
            gas_ex = [e for e in act.technosphere() if
                      e.input._data['name'] == 'market for natural gas, high pressure' or
                      e.input._data['name'] == 'market group for natural gas, high pressure']
            total_elect_amount = sum([a['amount'] for a in elect_ex])
            total_gas_amount = sum([a['amount'] for a in gas_ex])
            # Delete current gas and electricity exchanges
            for ex in elect_ex:
                ex.delete()
            for ex in gas_ex:
                ex.delete()

            # Add new exchanges with adjusted location. The total amount of gas and electricity inputs are maintained
            # from the original Ecoinvent activity. The only main change is the share of each country.
            if electricity_mix is None:
                for country in consts.steel_data_EU27.keys():
                    elect_act = cutoff391.get(code=consts.steel_data_EU27[country]['elect_code'])
                    gas_act = cutoff391.get(code=consts.steel_data_EU27[country]['gas_code'])
                    elect_amount = total_elect_amount * consts.steel_data_EU27[country]['share'] / 100
                    gas_amount = total_gas_amount * consts.steel_data_EU27[country]['share'] / 100
                    new_elect_ex = act.new_exchange(input=elect_act, amount=elect_amount, unit='kilowatt hour',
                                                    type='technosphere')
                    new_gas_ex = act.new_exchange(input=gas_act, amount=gas_amount, unit='cubic meter',
                                                  type='technosphere')
                    new_elect_ex.save()
                    new_gas_ex.save()
            else:
                # gas is always changed independently of the electricity mix chosen
                for country in consts.steel_data_EU27.keys():
                    gas_act = cutoff391.get(code=consts.steel_data_EU27[country]['gas_code'])
                    gas_amount = total_gas_amount * consts.steel_data_EU27[country]['share'] / 100
                    new_gas_ex = act.new_exchange(input=gas_act, amount=gas_amount, unit='cubic meter',
                                                  type='technosphere')
                    new_gas_ex.save()
                if electricity_mix == 'Norway':
                    electricity_norway = [a for a in cutoff391 if
                                          a._data['name'] == 'market for electricity, medium voltage' and a._data[
                                              'location'] == 'NO'][0]
                    new_elect_ex = act.new_exchange(input=electricity_norway, amount=total_elect_amount,
                                                    unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                elif electricity_mix == 'Poland':
                    electricity_poland = [a for a in cutoff391 if
                                          a._data['name'] == 'market for electricity, medium voltage' and a._data[
                                              'location'] == 'PL'][0]
                    new_elect_ex = act.new_exchange(input=electricity_poland, amount=total_elect_amount,
                                                    unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                elif electricity_mix == 'Europe':
                    electricity_europe = [a for a in cutoff391 if
                                          a._data['name'] == 'market group for electricity, medium voltage' and a._data[
                                              'location'] == 'RER'][0]
                    new_elect_ex = act.new_exchange(input=electricity_europe, amount=total_elect_amount,
                                                    unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                # if the electricity_mix variable inputed is not in the list ('Europe', 'Norway' or 'Poland'),
                # Europe is chosen by default
                else:
                    electricity_europe = [a for a in cutoff391 if
                                          a._data['name'] == 'market group for electricity, medium voltage' and a._data[
                                              'location'] == 'RER'][0]
                    new_elect_ex = act.new_exchange(input=electricity_europe, amount=total_elect_amount,
                                                    unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
        if electricity_mix is not None:
            ch_act_name = 'market for steel, chromium steel 18/8' + str(electricity_mix)
            ch_act = [a for a in new_db if a._data['name'] == ch_act_name]
            if len(ch_act) == 0:
                ch_steel_act_cutoff = [a for a in cutoff391 if a._data['name'] ==
                                       'market for steel, chromium steel 18/8'][0]
                ch_steel_act_newdb = ch_steel_act_cutoff.copy(database=consts.NEW_DB_NAME)
                ch_steel_act_newdb._data['name'] = 'market for steel, chromium steel 18/8' + str(electricity_mix)
                ch_steel_act_newdb.save()
                ch_steel_electric_input = [e.input for e in ch_steel_act_newdb.technosphere() if
                                           'transport' not in e.input._data['name'] and e.input._data[
                                               'location'] == 'RER'][0]
                ch_steel_act = ch_steel_electric_input.copy(database=consts.NEW_DB_NAME)
                ch_steel_act._data['name'] = 'steel production, electric, chromium steel 18/8' + str(electricity_mix)
                ch_steel_act.save()
                # Calculate the total amount of electricity inputs in the activity
                elect_ex = [e for e in ch_steel_act.technosphere() if
                            e.input._data['name'] == 'market for electricity, medium voltage'
                            or e.input._data['name'] == 'market group for electricity, medium voltage']
                total_elect_amount = sum([a['amount'] for a in elect_ex])
                for ex in elect_ex:
                    ex.delete()
                if electricity_mix == 'Norway':
                    electricity_norway = [a for a in cutoff391 if
                                          a._data['name'] == 'market for electricity, medium voltage' and a._data[
                                              'location'] == 'NO'][0]
                    new_elect_ex = ch_steel_act.new_exchange(input=electricity_norway, amount=total_elect_amount,
                                                             unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                elif electricity_mix == 'Poland':
                    electricity_poland = [a for a in cutoff391 if
                                          a._data['name'] == 'market for electricity, medium voltage' and a._data[
                                              'location'] == 'PL'][0]
                    new_elect_ex = ch_steel_act.new_exchange(input=electricity_poland, amount=total_elect_amount,
                                                             unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                elif electricity_mix == 'Europe':
                    electricity_europe = [a for a in cutoff391 if
                                          a._data['name'] == 'market group for electricity, medium voltage' and a._data[
                                              'location'] == 'RER'][0]
                    new_elect_ex = ch_steel_act.new_exchange(input=electricity_europe, amount=total_elect_amount,
                                                             unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                # if the electricity_mix variable inputted is not in the list ('Europe', 'Norway' or 'Poland'),
                # Europe is chosen by default
                else:
                    electricity_europe = [a for a in cutoff391 if
                                          a._data['name'] == 'market group for electricity, medium voltage' and a._data[
                                              'location'] == 'RER'][0]
                    new_elect_ex = ch_steel_act.new_exchange(input=electricity_europe, amount=total_elect_amount,
                                                             unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()

                original_inputs = [e for e in ch_steel_act_newdb.technosphere() if
                                   'transport' not in e.input._data['name']]
                for e in original_inputs:
                    name = e.input._data['name'] + str(electricity_mix)
                    new_db_act = [a for a in new_db if a._data['name'] == name][0]
                    amount = e.amount
                    new_elect_ex = ch_steel_act_newdb.new_exchange(input=new_db_act, amount=amount,
                                                                   unit='kilowatt hour', type='technosphere')
                    new_elect_ex.save()
                    e.delete()

        # Create an empty market activity in new_db
        try:
            steel_market = new_db.new_activity(name=act_name, code=code_name, unit='kilogram', location='RER')
            steel_market.save()
        except bd.errors.DuplicateNode:
            steel_market = new_db.get(code=code_name)

        # Add exchanges with the annual share of primary a secondary steel to the recently created activity
        # Historic primary and secondary shares according to Eurofer data.
        if recycled_share is None:
            if str(commissioning_year - 1) in consts.secondary_steel.keys():
                # We assume that the turbine was manufactured a year before the commissioning date
                secondary_amount = consts.secondary_steel[str(commissioning_year - 1)]
                primary_amount = 1 - secondary_amount
            else:
                secondary_amount = consts.secondary_steel['other']
                primary_amount = 1 - secondary_amount
            # Add primary steel
            primary_ex = steel_market.new_exchange(input=primary_act, amount=primary_amount, unit='kilogram',
                                                   type='technosphere')
            primary_ex.save()
            # Add secondary steel
            secondary_ex = steel_market.new_exchange(input=recycled_act, amount=secondary_amount, unit='kilogram',
                                                     type='technosphere')
            secondary_ex.save()
        # Manually selected primary and secondary shares
        else:
            # Add primary steel
            primary_share = 1 - recycled_share
            primary_ex = steel_market.new_exchange(input=primary_act, amount=primary_share, unit='kilogram',
                                                   type='technosphere')
            primary_ex.save()
            # Add secondary steel
            secondary_ex = steel_market.new_exchange(input=recycled_act, amount=recycled_share, unit='kilogram',
                                                     type='technosphere')
            secondary_ex.save()
        # Add production and save
        production_exc = steel_market.new_exchange(input=steel_market.key, amount=1.0, unit="kilogram",
                                                   type='production')
        production_exc.save()
        steel_market.save()

        ch_name = 'market for steel, chromium steel 18/8' + str(electricity_mix)
        ch_steel_market = [a for a in new_db if a._data['name'] == ch_name]
        return steel_market, ch_steel_market

    elif steel_act_check == 3:
        ch_name = 'market for steel, chromium steel 18/8' + str(electricity_mix)
        ch_steel_market = [a for a in new_db if a._data['name'] == ch_name]
        return steel_act, ch_steel_market
    else:
        print('Something went wrong during the creation of the steel market')


def lci_materials(new_db: bd.Database, cutoff391: bd.Database, park_name: str, park_power: float, number_of_turbines: int, park_location: str,
                  park_coordinates: tuple,
                  manufacturer: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'],
                  rotor_diameter: float,
                  turbine_power: float, hub_height: float, commissioning_year: int,
                  recycled_share_steel: float = None,
                  lifetime: int = 20,
                  electricity_mix_steel: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                  generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'] = 'gb_dfig',
                  include_life_cycle_stages: bool = True,
                  regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h',
                  comment: str = ''):
    """
    It creates the activities 'park_name_single_turbine' (code: 'park_name_single_turbine'),
    'park_name_cables' (code: 'park_name_intra_cables') and park (park_name) (code: park_name) in the
    database 'new_db' in bw2. The park activity, contains as many turbines as there are in the park, the cable inputs
    and the transformer activity scaled to the size of the park.
    generator_type: it only accepts the models (strings) 'dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'.
    manufacturer: it only accepts 'Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind'.
    """
    # create the turbine activity and cables activity in bw2 including the production exchange
    try:
        turbine_act = new_db.new_activity(name=park_name + '_single_turbine', code=park_name + '_single_turbine',
                                          location=park_location, unit='unit', comment=comment)
        turbine_act.save()
        new_exc = turbine_act.new_exchange(input=turbine_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        turbine_act.save()
    except bd.errors.DuplicateNode:
        print(
            'An inventory for a park with the name ' + '"' + park_name + '"' + 'was already created before in the '
                                                                               'database ')
        print('"new_db". You may want to think about giving '
              'another name to the wind park you are trying to '
              'analyse. Otherwise, you may want to delete '
              'the content of "new_db" by runing delete_new_db().')
        print(
            'WARNING: if you run delete_new_db() '
            'ALL WIND PARKS STORED IN THAT DATABASE WILL '
            'BE DELETED!')
        sys.exit()

    cables_act = new_db.new_activity(name=park_name + '_cables', code=park_name + '_intra_cables', unit='unit')
    cables_act.save()
    new_exc = cables_act.new_exchange(input=cables_act.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    cables_act.save()

    # create an activity for each life cycle stage
    if include_life_cycle_stages:
        # 1. materials
        materials_act = new_db.new_activity(name=park_name + '_materials', code=park_name + '_materials',
                                            location=park_location, unit='unit')
        materials_act.save()
        new_exc = materials_act.new_exchange(input=materials_act.key, amount=1.0, unit="unit", type='production')
        new_exc.save()
        materials_act.save()
        # 2. manufacturing
        manufacturing_act = new_db.new_activity(name=park_name + '_manufacturing', code=park_name + '_manufacturing',
                                                location=park_location, unit='unit')
        manufacturing_act.save()
        new_exc = manufacturing_act.new_exchange(input=manufacturing_act.key, amount=1.0, unit="unit",
                                                 type='production')
        new_exc.save()
        manufacturing_act.save()
        # 3. transport
        transport_act = new_db.new_activity(name=park_name + '_transport', code=park_name + '_transport',
                                            location=park_location, unit='unit')
        transport_act.save()
        new_exc = transport_act.new_exchange(input=transport_act.key, amount=1.0, unit="unit",
                                             type='production')
        new_exc.save()
        transport_act.save()
        # 4. installation
        installation_act = new_db.new_activity(name=park_name + '_installation', code=park_name + '_installation',
                                               location=park_location, unit='unit')
        installation_act.save()
        new_exc = installation_act.new_exchange(input=installation_act.key, amount=1.0, unit="unit",
                                                type='production')
        new_exc.save()
        installation_act.save()
        # 5. operation & maintenance
        om_act = new_db.new_activity(name=park_name + '_maintenance', code=park_name + '_maintenance',
                                     location=park_location, unit='unit')
        om_act.save()
        new_exc = om_act.new_exchange(input=om_act.key, amount=1.0, unit="unit",
                                      type='production')
        new_exc.save()
        om_act.save()
        # 6. eol
        eol_act = new_db.new_activity(name=park_name + '_eol', code=park_name + '_eol',
                                      location=park_location, unit='unit')
        eol_act.save()
        new_exc = eol_act.new_exchange(input=eol_act.key, amount=1.0, unit="unit",
                                       type='production')
        new_exc.save()
        eol_act.save()

        materials_activity = materials_act
        manufacturing_activity = manufacturing_act
    else:
        materials_activity = turbine_act
        manufacturing_activity = turbine_act

    # add materials from the turbine
    if generator_type not in ['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']:
        print(generator_type, 'is not an allowed value. We selected the default gb_dfig instead.')
        generator_type = 'gb_dfig'
    mass_materials, material_polyfits = materials_mass(generator_type, turbine_power,
                                                       hub_height, regression_adjustment=regression_adjustment,
                                                       rotor_diameter=rotor_diameter)
    for material in mass_materials.keys():
        if any(element in material for element in ['Praseodymium', 'Neodymium', 'Dysprosium', 'Terbium', 'Boron']):
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            ex.save()
            materials_activity.save()
        elif any(element in material for element in ['Low alloy steel', 'Low alloy steel_foundations']):
            inp, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                  recycled_share=recycled_share_steel,
                                                  electricity_mix=electricity_mix_steel,
                                                  printed_warning=consts.PRINTED_WARNING_STEEL,
                                                  cutoff391=cutoff391, new_db=new_db)
            consts.PRINTED_WARNING_STEEL = True
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            # Uncertainty added as the standard deviation of the residuals
            ex['uncertainty type'] = NormalUncertainty.id
            ex['loc'] = mass_materials[material]
            ex['scale'] = material_polyfits[material] * 1000
            ex['minimum'] = 0
            ex.save()
            materials_activity.save()
        elif (any(element in material for element in ['Chromium steel', 'Chromium steel_foundations'])
              and electricity_mix_steel is not None):
            steel, ch = manipulate_steel_activities(commissioning_year=commissioning_year,
                                                    recycled_share=recycled_share_steel,
                                                    electricity_mix=electricity_mix_steel,
                                                    printed_warning=consts.PRINTED_WARNING_STEEL,
                                                    new_db=new_db, cutoff391=cutoff391)
            if ch:
                inp = ch[0]
            else:
                inp = cutoff391.get(consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            # Uncertainty added as the standard deviation of the residuals
            ex['uncertainty type'] = NormalUncertainty.id
            ex['loc'] = mass_materials[material]
            ex['scale'] = material_polyfits[material] * 1000
            ex['minimum'] = 0
            ex.save()
            materials_activity.save()
        elif material == 'Fiberglass':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            # Mass includes 10% of waste produced in the manufacturing (Psomopoulos et al. 2019)
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material] * 1.1)
            # Uncertainty added as the standard deviation of the residuals
            ex['uncertainty type'] = NormalUncertainty.id
            ex['loc'] = mass_materials[material]
            ex['scale'] = material_polyfits[material] * 1000
            ex['minimum'] = 0
            ex.save()
            materials_activity.save()
        elif material == 'Concrete_foundations':
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            # Uncertainty added as the standard deviation of the residuals
            ex['uncertainty type'] = NormalUncertainty.id
            ex['loc'] = mass_materials[material]
            ex['scale'] = material_polyfits[material] / 2.4
            ex['minimum'] = 0
            ex.save()
            materials_activity.save()
        else:
            inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
            ex = materials_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            # Uncertainty added as the standard deviation of the residuals
            ex['uncertainty type'] = NormalUncertainty.id
            ex['loc'] = mass_materials[material]
            ex['scale'] = material_polyfits[material] * 1000
            ex['minimum'] = 0
            ex.save()
            materials_activity.save()

    # add turbine and foundations material processing activities
    processing_materials_list = ['Low alloy steel', 'Chromium steel', 'Cast iron', 'Aluminium', 'Copper',
                                 'Low alloy steel_foundations', 'Chromium steel_foundations', 'Zinc']
    for material in processing_materials_list:
        if material == 'Low alloy steel':
            # section bar rolling
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            ex.save()
            manufacturing_activity.save()
            # welding
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_welding']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=hub_height * 2)
            ex.save()
            manufacturing_activity.save()
        elif material == 'Zinc':
            # We need the tower area to be coated. This is the perimeter of the tower multiplied by the hub height.
            # Perimeter of the tower: regression between the tower diameter and the power (data from Sacchi et al.)
            tower_diameter = [5, 5.5, 5.75, 6.75, 7.75]
            power = [3, 3.6, 4, 8, 10]  # in MW
            fit_diameter = np.polyfit(power, tower_diameter, 1)
            f_fit_diameter = np.poly1d(fit_diameter)
            outer_diameter = f_fit_diameter(turbine_power)  # in m
            perimeter = np.pi * outer_diameter
            tower_surface_area = perimeter * hub_height
            # create exchange
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Zinc coating']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=tower_surface_area)
            ex.save()
            manufacturing_activity.save()
        elif 'foundations' in material and 'alloy' not in material:
            material_name = material[:material.index('_')]
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES[material_name]['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            ex.save()
            manufacturing_activity.save()
        elif 'foundations' in material and 'alloy' in material:
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Steel_tower_rolling']['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            ex.save()
            manufacturing_activity.save()
        else:
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES[material]['code'])
            ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material])
            ex.save()
            manufacturing_activity.save()

    # add electricity
    power = [0.03, 0.15, 0.6, 0.8, 2]  # in MW
    electricity = [575, 3987, 17510, 17510, 67500]  # in kWh
    fit_electricity = np.polyfit(power, electricity, 1)
    f_fit_diameter = np.poly1d(fit_electricity)
    electricity_input = f_fit_diameter(turbine_power)
    # find the closest manufacturer location
    distance_dict = {}
    if manufacturer is None or manufacturer not in ['Vestas', 'Siemens Gamesa', 'Nordex', 'Enercon', 'LM Wind']:
        print(manufacturer, 'is not an allowed value. We chose LM Wind by default instead')
        manufacturer = 'LM Wind'
    for location_id in consts.MANUFACTURER_LOC[manufacturer]:
        location = consts.MANUFACTURER_LOC[manufacturer][location_id]['location']
        distance = geodesic(park_coordinates, location).kilometers
        distance_dict[distance] = location_id
    loc_id_min_distance = distance_dict[min(distance_dict.keys())]
    closest_country = consts.MANUFACTURER_LOC[manufacturer][loc_id_min_distance]['country']
    inp = [a for a in cutoff391 if
           'market for electricity, low voltage' in a._data['name'] and closest_country in a._data['location']][0]
    ex = manufacturing_activity.new_exchange(input=inp, type='technosphere', amount=electricity_input)
    ex.save()
    manufacturing_activity.save()

    # add materials from the cables
    cable_mass = cabling_materials(turbine_power, rotor_diameter, number_of_turbines)
    for material in cable_mass.keys():
        inp = cutoff391.get(code=consts.MATERIALS_EI391_ACTIVITY_CODES[material]['code'])
        ex = cables_act.new_exchange(input=inp, type='technosphere', amount=cable_mass[material])
        ex.save()

    # add cable material processing activities
    processing_materials_list = ['Aluminium_cables', 'Copper', 'PE', 'PVC']
    for material in processing_materials_list:
        if material == 'Aluminium_cables':
            # copper wire drawing
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES['Copper']['code'])
            ex = cables_act.new_exchange(input=inp, type='technosphere', amount=cable_mass['Aluminium'])
            ex.save()
            cables_act.save()
        else:
            inp = cutoff391.get(code=consts.MATERIAL_PROCESSING_EI391_ACTIVITY_CODES[material]['code'])
            ex = cables_act.new_exchange(input=inp, type='technosphere', amount=cable_mass[material])
            ex.save()
            cables_act.save()

    # add exchanges (material and manufacturing stages) to the wind turbine activity, if needed
    if include_life_cycle_stages:
        materials_ex = turbine_act.new_exchange(input=materials_activity, type='technosphere', amount=1)
        materials_ex.save()
        manufacturing_ex = turbine_act.new_exchange(input=manufacturing_activity, type='technosphere', amount=1)
        manufacturing_ex.save()

    # create a new activity for the whole wind park
    comment = 'Lifetime=' + str(lifetime) + ', Turbine power=' + str(turbine_power)
    park_act = new_db.new_activity(name=park_name,
                                   code=park_name + '_' + str(park_power), location=park_location, unit='unit',
                                   comment=comment)
    park_act.save()

    # add turbines
    turbine_ex = park_act.new_exchange(input=turbine_act, type='technosphere', amount=number_of_turbines)
    turbine_ex.save()

    # add cables
    cables_ex = park_act.new_exchange(input=cables_act, type='technosphere', amount=1.0)
    cables_ex.save()

    # add materials from the transformer (downscaled from 500MVA to the park power)
    transformer = mva500_transformer(cutoff391=cutoff391, new_db=new_db)
    transformer_ex = park_act.new_exchange(input=transformer, type='technosphere', amount=park_power / 500)
    transformer_ex.save()

    # output
    new_exc = park_act.new_exchange(input=park_act.key, amount=1.0, unit="unit", type='production')
    new_exc.save()
    park_act.save()

    # convert single turbine mass_materials to total park mass and add cable_mass too
    total_mass_turbines = {key: value * number_of_turbines for key, value in mass_materials.items()}
    for key, value in cable_mass.items():
        if key in total_mass_turbines:
            total_mass_turbines[key] += value
        else:
            total_mass_turbines[key] = value
    return total_mass_turbines


def end_of_life(new_db: bd.Database, cutoff391: bd.Database, scenario: int, park_name: str,
                generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'],
                turbine_power: float, hub_height: float, rotor_diameter: float, include_life_cycle_stages: bool = True,
                regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    This function adds the end-of-life of materials to the turbine activity.
    The variable 'scenario' is used to retrieve the dictionary with the activity codes that should be taken
    (defined in consts.py).
    Recycling activities are not included in the system boundaries (EPD approach for EoL).
    Note: Avoided impacts of 'virgin material substitution' are also not included.
    """
    if 1 > scenario or scenario > 4:
        print('There are 4 eol scenarios in WindTrace. You chose a number that is not in the range 1-4. '
              'By default we applied the baseline scenario')
        scenario = 1

    # materials classification (according to EoL treatment groups)
    fe_alloys = ['Low alloy steel', 'Low alloy steel_foundations', 'Chromium steel',
                 'Chromium steel_foundations', 'Cast iron']
    copper = ['Copper', 'Copper_foundations']
    rare_earth_metals = ['Praseodymium', 'Neodymium', 'Dysprosium', 'Terbium', 'Boron']
    plastics = ['Rubber', 'PUR', 'PVC', 'PE']

    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    if include_life_cycle_stages:
        eol_act = new_db.search(park_name + '_eol')[0]
        eol_activity = eol_act
    else:
        eol_activity = turbine_act

    # add materials from the turbine
    mass_materials, material_polyfits = materials_mass(generator_type, turbine_power, hub_height,
                                                       regression_adjustment=regression_adjustment,
                                                       rotor_diameter=rotor_diameter)
    for material in mass_materials.keys():
        # metals
        if any(element in material for element in fe_alloys):
            if scenario == 1 or scenario == 2 or scenario == 4:
                recycling_rate = 0.9
            # scenario == 3
            else:
                recycling_rate = 0.52
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=mass_materials[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in copper):
            if scenario == 1 or scenario == 4:
                recycling_rate = 0.9
            elif scenario == 2:
                recycling_rate = 0.53
            # scenario == 3
            else:
                recycling_rate = 0.42
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=mass_materials[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif 'Aluminium' in material:
            if scenario == 1 or scenario == 4:
                recycling_rate = 0.9
            elif scenario == 2:
                recycling_rate = 0.7
            # secenario == 3
            else:
                recycling_rate = 0.42
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=mass_materials[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in rare_earth_metals):
            if scenario == 1:
                recycling_rate = 0
            elif scenario == 2:
                recycling_rate = 0.21
            elif scenario == 3:
                recycling_rate = 0.01
            # scenario == 4
            else:
                recycling_rate = 0.7
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                           amount=mass_materials[material] * (-(1 - recycling_rate)))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in plastics):
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material] * (-1))
            ex.save()
            eol_activity.save()
        elif any(element in material for element in ['Epoxy resin', 'Fiberglass']):
            # NOTE: 10% of glassfiber mass corresponds to the extra mass on the manufacturing process (i.e., waste)
            # and the waste is not accounted in here.
            if scenario == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material] * (-0.3))
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material] * (-1))
                ex.save()
                eol_activity.save()
        elif any(element in material for element in ['Lubricating oil', 'Ethyleneglycol']):
            inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['incineration']['code'])
            ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=mass_materials[material] * (-1))
            ex.save()
            eol_activity.save()
        elif material == 'Concrete_foundations':
            # concrete modelled separatelly because the amount is in m3 and the landfill activity in kg
            # we use the density (2400 kg/m3)
            if scenario == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=mass_materials[material] * (-2400 * 0.5))
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere',
                                               amount=mass_materials[material] * (-2400))
                ex.save()
                eol_activity.save()
        else:
            if scenario == 4:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=-mass_materials[material] * 0.5)
                ex.save()
                eol_activity.save()
            else:
                inp = cutoff391.get(code=consts.EOL_S1_EI391_ACTIVITY_CODES[material]['landfill']['code'])
                ex = eol_activity.new_exchange(input=inp, type='technosphere', amount=-mass_materials[material])
                ex.save()
                eol_activity.save()

    if include_life_cycle_stages:
        eol_ex = turbine_act.new_exchange(input=eol_activity, type='technosphere', amount=1)
        eol_ex.save()


def maintenance(new_db: bd.Database, cutoff391: bd.Database, park_name: str,
                generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'],
                turbine_power: float, hub_height: float, lifetime: int, rotor_diameter: float,
                include_life_cycle_stages: bool = True,
                regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    This function adds the maintenance activities (inspection trips and oil changes) to the turbine activity.
    Inspection trips: 200 km every 6 months (Elsan Engineering, 2004)
    Change oil and lubrication: change all the oil every two years (adaptation from Abeliotis and Pactiti, 2014)
    Replacement of parts: NOT INCLUDED
    """
    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    if include_life_cycle_stages:
        om_act = new_db.search(park_name + '_maintenance')[0]
        om_activity = om_act
    else:
        om_activity = turbine_act
    # Inspection trips
    inp = cutoff391.get(name='transport, passenger car, large size, diesel, EURO 4',
                        code='dceed1b2fd31e759a751c6dd912a45f3')
    ex = om_activity.new_exchange(input=inp, type='technosphere', amount=200 * (lifetime * 2))
    ex.save()
    om_activity.save()

    # Change oil and lubrication
    mass_materials, m_poly = materials_mass(generator_type=generator_type,
                                            turbine_power=turbine_power, hub_height=hub_height,
                                            regression_adjustment=regression_adjustment, rotor_diameter=rotor_diameter)
    inp = cutoff391.get(name='market for lubricating oil', code='92391c8c6958ada25b22935e3fa6f06f')
    ex = om_activity.new_exchange(input=inp, type='technosphere',
                                  amount=mass_materials['Lubricating oil'] * (lifetime / 2))
    ex.save()
    om_activity.save()
    inp = cutoff391.get(name='treatment of waste mineral oil, hazardous waste incineration, with energy recovery',
                        code='ad6d0f2a8b45536da196238df879077b')
    ex = om_activity.new_exchange(input=inp, type='technosphere',
                                  amount=mass_materials['Lubricating oil'] * -(lifetime / 2))
    ex.save()
    om_activity.save()

    if include_life_cycle_stages:
        om_ex = turbine_act.new_exchange(input=om_activity, type='technosphere', amount=1)
        om_ex.save()


def transport(new_db: bd.Database, cutoff391: bd.Database,
              manufacturer: Literal['Vestas', 'Siemens Gamesa', 'Nordex', 'ENERCON', 'LM Wind'],
              park_coordinates: tuple, park_name: str, rotor_diameter: float,
              generator_type: Optional[Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']], turbine_power: float,
              hub_height: float, include_life_cycle_stages: bool = True,
              regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    **Tower transport. Distance data (in km) according to Vestas report on Vestas V162-6.2 MW.**
    Limitations:
    1) Vestas report refers to a location in Germany and to its suppliers. Other companies will have different suppliers
       (some outside Europe).
    2) Not all the steel in the turbine is used in the tower. However, most of it does, so it is an approximation.
    **Foundation transport. Distance data (in km) according to Vestas report on Vestas V162-6.2 MW.**
    Limitations:
    1) Vestas report refers to a location in Germany and to its suppliers. Other companies will have different
       suppliers.
    **Other materials transport. We assume that the transport is from the closest manufacturing site from the
    manufacturer to the installation location. We only have data for Enercon, Nordex, Vestas and Siemens Gamesa (~65%
    of European turbines). In case it is not from one of these manufacturers, we assume it is from LM Wind, another big
    firm in the turbine parts manufacturing business in Europe.

    **General Limitations**
    1) Cable transport not included
    """
    mat_mass, material_polyfits = materials_mass(generator_type, turbine_power, hub_height,
                                                 regression_adjustment=regression_adjustment,
                                                 rotor_diameter=rotor_diameter)

    for material in mat_mass.keys():
        if 'Concrete' in material:
            foundations_amount = mat_mass[material] * 2.4 * 50
        elif material == 'Low alloy steel':
            steel_amount = mat_mass[material] / 1000 * 450
        else:
            distance_dict = {}
            for location_id in consts.MANUFACTURER_LOC[manufacturer]:
                location = consts.MANUFACTURER_LOC[manufacturer][location_id]['location']
                distance = geodesic(park_coordinates, location).kilometers
                distance_dict[distance] = location_id
            others_amount = (sum(mat_mass.values()) - mat_mass['Concrete_foundations']
                             - mat_mass['Low alloy steel']) / 1000 * min(distance_dict.keys())

    truck_trans = cutoff391.get(name='market for transport, freight, lorry >32 metric ton, EURO6',
                                code='508cc8b20d83e7b31af9848e1fb45815', location='RER')

    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    if include_life_cycle_stages:
        transport_act = new_db.search(park_name + '_transport')[0]
        transport_activity = transport_act
    else:
        transport_activity = turbine_act
    # add transport to the lci inventory
    # add steel transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=steel_amount)
    new_exc.save()
    # add foundations transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=foundations_amount)
    new_exc.save()
    # add other materials transport
    new_exc = transport_activity.new_exchange(input=truck_trans, type='technosphere', amount=others_amount)
    new_exc.save()

    if include_life_cycle_stages:
        transport_ex = turbine_act.new_exchange(input=transport_activity, type='technosphere', amount=1)
        transport_ex.save()


def generate_events_with_probability():
    """
    This function returns the land cover type of each park according to its probability. Data from
    'Electricity from a European onshore wind farm using SG 6.2-170 / SG 6.6-170 wind turbines' (2022).
    """
    # Probabilities for each event type
    land_cover_prob = {
        'industrial, from': 0.044,
        'crops, non-irrigated': 0.21,
        'row crops': 0.115,
        'shrubland': 0.174,
        'pasture': 0.079,
        'forest': 0.306,
        'unspecified': 0.072
    }

    cumulative_prob = 0

    while True:
        rand_val = random.random()  # Generate a random number between 0 and 1

        for event_type, probability in land_cover_prob.items():
            cumulative_prob += probability
            if rand_val <= cumulative_prob:
                return event_type


def land_use(new_db: bd.Database, biosphere3: bd.Database, turbine_power: float, park_name: str, lifetime: int, manual_land_cover: str = None,
             include_life_cycle_stages: bool = True, land_use_permanent_intensity: int = 3000):
    """
    Function to define the flows with the biosphere regarding the land use.

    manual_land_cover allowed values: 'industrial_from', 'crops, non-irrigated', 'row crops', 'shrubland', 'grassland',
    'pasture', 'forest', 'unspecified'.

    All data comes from the report
    'Land-Use Requirements of Modern Wind Power Plants in the United States' Denholm et al., 2009.
    Permanent impacts intensity = 3000 m2/MW (79% roads, 10% turbine, 6% substation, 2% transformer, 2% others)
    Temporal impacts intensity = 7000 m2/MW (62% temporal roads, 30% staging area, 6% substation construction, 3% other)
    **Limitations:**
    1) Old data (installations from 2009 and before)
    2) Each park has a randomly assigned land cover (adjusted to the probability). So the lci of the individual parks
    won't represent properly the land covers, but overall it will be okay, according to US data.
    """
    # Equivalence of land covers with biosphere3 codes
    equivalence = {
        'industrial, from': {'occupation_code': None,
                             'transformation_code': 'b6dcefd8-3848-4338-9c3e-fe6e91f20937'},
        'crops, non-irrigated': {'occupation_code': 'a6889a22-e99e-42ea-85cd-4a68d7975dcd',
                                 'transformation_code': '4b420f19-0421-461e-a0b6-7efbf580089b'},
        'unspecified': {'occupation_code': 'c7cb5880-4219-4051-9357-10fdd08c6f2b',
                        'transformation_code': '29630a65-f38c-48a5-9744-c0121f586640'},
        'row crops': {'occupation_code': 'c5aafa60-495c-461c-a1d4-b262a34c45b9',
                      'transformation_code': 'f05cca02-ec18-4acc-9939-59658ff9a554'},
        'shrubland': {'occupation_code': 'c199261c-8234-43c5-b906-5b67707e4395',
                      'transformation_code': '17a5a406-333f-4b9e-8852-c2de50bc9585'},
        'grassland': {'occupation_code': '2b8a0f87-bd2a-4b10-8dd9-714487f59fc9',
                      'transformation_code': 'b905c2e0-a0db-4e66-80d2-8bdfc93c6218'},
        'pasture': {'occupation_code': '59ded913-17fe-4b3e-80cb-79b97cdbef9a',
                    'transformation_code': '2c126bcc-bb63-4d63-bd72-f02a1e616809'},
        'forest': {'occupation_code': 'b91d0527-9a01-4a86-b420-c62b70629ba4',
                   'transformation_code': '0930b6b8-d9c6-4462-966f-ac7495b63bed'},
        'industrial': {'occupation_code': 'fe9c3a98-a6d2-452d-a9a4-a13e64f1b95b',
                       'transformation_code': '4624deff-2016-41d4-b2bf-3db8dab88779'},
        'road': {'occupation_code': '26efe47c-92a5-4dea-b4d0-eac13e418a58',
                 'transformation_code': 'a42347d2-09f1-405e-95dd-bf6ac03765d8'}
    }

    if manual_land_cover:
        land_cover_type = manual_land_cover
    else:
        land_cover_type = generate_events_with_probability()

    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    if include_life_cycle_stages:
        installation_act = new_db.search(park_name + '_installation')[0]
        installation_activity = installation_act
    else:
        installation_activity = turbine_act

    ### transformation flows ###
    # transformation, from
    try:
        land_cover_type_activity = biosphere3.get(equivalence[land_cover_type]['transformation_code'])
    except bd.errors.UnknownObject:
        print('You introduced a wrong land cover type: ' + str(manual_land_cover))
        print('Only the following land cover types are allowed: industrial_from, crops, non-irrigated, row crops, '
              'shrubland, grassland, pasture, forest, unspecified.')
        sys.exit()
    amount_land_cover = (
                                land_use_permanent_intensity + 7000) * turbine_power  # 3000 m2/MW permanent, 7000 m2/MW temporal.

    # transformation, to
    industrial_type_activity = biosphere3.get(equivalence['industrial']['transformation_code'])
    amount_industrial = (
                                0.21 * land_use_permanent_intensity + 2660) * turbine_power  # 630 m2/MW permanent, 2660 m2/MW temporal.
    road_type_activity = biosphere3.get(equivalence['road']['transformation_code'])
    amount_road = (
                          0.79 * land_use_permanent_intensity + 4340) * turbine_power  # 2370 m2/MW permanent, 4340 m2/MW temporal.

    # occupation flows
    industrial_occ_activity = biosphere3.get(equivalence['industrial']['occupation_code'])
    amount_industrial_occ = (0.21 * land_use_permanent_intensity * turbine_power * lifetime) + (
            2660 * turbine_power * 2.5)  # 630 m2/MW permanent, 2660 m2/MW temporal.
    # 2.5 in years, is the mean recovery time from a temporal land cover change (data also from the report)
    road_occ_activity = biosphere3.get(equivalence['road']['occupation_code'])
    amount_road_occ = (0.79 * land_use_permanent_intensity * turbine_power * lifetime) + (
            4340 * turbine_power * 2.5)  # 2370 m2/MW permanent, 4340 m2/MW temporal.

    # add flows to the lci inventory
    # add transformation, from (land cover)
    new_exc = installation_activity.new_exchange(input=land_cover_type_activity, type='biosphere',
                                                 amount=amount_land_cover)
    new_exc.save()

    # add transformation, to (industrial area)
    new_exc = installation_activity.new_exchange(input=industrial_type_activity, type='biosphere',
                                                 amount=amount_industrial)
    new_exc.save()

    # add transformation, to (road network)
    new_exc = installation_activity.new_exchange(input=road_type_activity, type='biosphere',
                                                 amount=amount_road)
    new_exc.save()

    # add occupation (industrial area)
    new_exc = installation_activity.new_exchange(input=industrial_occ_activity, type='biosphere',
                                                 amount=amount_industrial_occ)
    new_exc.save()

    # add occupation (road network)
    new_exc = installation_activity.new_exchange(input=road_occ_activity, type='biosphere',
                                                 amount=amount_road_occ)
    new_exc.save()

    # to keep track of the transformation and occupation flows amounts
    transformation_flows = amount_land_cover
    occupation_flows = amount_industrial_occ + amount_road_occ
    return transformation_flows, occupation_flows


def auxiliary_road_materials(new_db: bd.Database, cutoff391: bd.Database,
                             turbine_power: float, park_name: str, include_life_cycle_stages: bool = True):
    """
    Adds the auxiliary road materials (exchange with: market for road) to the lci.
    *Data sources*
    -Width of auxiliary roads 5.0 m. Data from 'Environmental Impact Assessment Report. Proposed
    Drumnahough Wind Farm Co. Donegal. Volume 2 (MAIN REPORT)', Malachy Walsh and Partners (2020).
     (https://www.pleanala.ie/publicaccess/EIAR-NIS/308806/EIAR/Vol%202/Chapter%203%20Civil%20Engineering%20(Sept%202020).pdf)
    -road network intensity: 2370 m2/MW (permanents). Land use requirements... Dunholm et al., 2009.
    -We assume that all roads are rural roads. We modify the road activity to only include diesel,
    excavation electricity as transforming activities and gravel as a material.
    """
    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    if include_life_cycle_stages:
        installation_act = new_db.search(park_name + '_installation')[0]
        installation_activity = installation_act
    else:
        installation_activity = turbine_act

    amount_road = 2370 * turbine_power / 5  # we don't multiply per liftime, because the functional unit is 'meters and year' (so, the intensity of meters constructed every year)

    # remove the land use from the road activity
    # remove bitumen, concrete, reinforcing steel and steel. We will assume that all roads are rural roads (gravel)
    # (copying the activity in the database new_db and making the changes there)
    road_act_in_new_db = new_db.search('road construction')

    if not road_act_in_new_db:
        road_act = cutoff391.get(code='3d1d98819862a4057c75095315820d52')
        road_new = road_act.copy(database=consts.NEW_DB_NAME)
        technosphere_activities_to_remove = ['bitumen', 'concrete', 'steel']
        for ex in road_new.biosphere():
            if 'Transformation' in ex.input._data['name'] or 'Occupation' in ex.input._data['name']:
                ex.delete()
        for ex in road_new.technosphere():
            if any(element in ex.input._data['name'] for element in technosphere_activities_to_remove):
                ex.delete()
            # adjust amount of waste
            elif 'inert waste' in ex.input._data['name']:
                ex['amount'] = -106
                ex.save()
    else:
        road_new = road_act_in_new_db[0]
    # add road
    new_exc = installation_activity.new_exchange(input=road_new, type='technosphere', amount=amount_road)
    new_exc.save()


def excavation_activities(new_db: bd.Database, cutoff391: bd.Database,
                          generator_type: Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig'],
                          turbine_power: float, hub_height: float, rotor_diameter: float,
                          number_of_turbines: int, park_name: str, include_life_cycle_stages: bool = True,
                          regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h'):
    """
    Adds to the lci the hydraulic digger activities necessary build the foundations and bury the cables.
    *Data*
    Depth (1.1 m) and with (0.6-1.1 m) of the cabling trenches. Data from Ardente et al., 2008.
    Limitations:
    Distance between turbines: 8*D (could be too much)
    """
    # foundations volume
    mat_mass, material_polyfits = materials_mass(generator_type, turbine_power, hub_height,
                                                 regression_adjustment=regression_adjustment,
                                                 rotor_diameter=rotor_diameter)
    volume_steel = mat_mass['Low alloy steel_foundations'] / 7800  # 7.8 t/m3 is the density of steel
    volume_ch_steel = mat_mass['Chromium steel_foundations'] / 7190  # 7.19 t/m3 is the density of chromium steel
    # volume_copper = mat_mass['Copper_foundations'] / 8960  # 8.96 t/m3 is the density of copper
    volume_concrete = mat_mass['Concrete_foundations']
    foundations_volume = volume_concrete + volume_steel + volume_ch_steel  # + volume_copper

    # cabling volume
    distance_between_turbines = 8 * rotor_diameter
    cable_length = distance_between_turbines * (number_of_turbines - 1)
    cabling_volume = 1.1 * 0.85 * cable_length

    # define brightway activities
    digger_act = cutoff391.get(name='excavation, hydraulic digger',
                               code='dc208e3cd1b01954185c03259c97a36a', location='RER')
    turbine_act = new_db.search(park_name + '_single_turbine')[0]
    cables_act = new_db.search(park_name + '_cables')[0]
    if include_life_cycle_stages:
        installation_act = new_db.search(park_name + '_installation')[0]
        installation_activity = installation_act
    else:
        installation_activity = turbine_act

    # add digger foundations activities
    new_exc = installation_activity.new_exchange(input=digger_act, type='technosphere', amount=foundations_volume)
    new_exc.save()

    # add digger cabling activities
    new_exc = cables_act.new_exchange(input=digger_act, type='technosphere', amount=cabling_volume)
    new_exc.save()

    if include_life_cycle_stages:
        installation_ex = turbine_act.new_exchange(input=installation_activity, type='technosphere', amount=1)
        installation_ex.save()


def electricity_production(new_db: bd.Database, park_name: str, park_power: float,
                           cf: float, time_adjusted_cf: float):
    # Finds park activity to extract lifetime and turbine power (saved in comments).
    try:
        real_park_activity = new_db.get(park_name + '_' + str(park_power))
        lifetime = int(real_park_activity._data['comment'].split('=')[1].split(',')[0])
        turbine_power = float(real_park_activity._data['comment'].split('=')[-1])
    except bd.errors.UnknownObject:
        print('There is no inventory created with the park name you specified. '
              'Please, make sure you entered the right park name, or use the function lci_wind_turbine to create a '
              'wind park inventory with this name.')
        sys.exit()

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


def lci_wind_turbine(new_db: bd.Database, cutoff391: bd.Database, biosphere3: bd.Database,
                     park_name: str, park_power: float, number_of_turbines: int, park_location: str,
                     park_coordinates: tuple, manufacturer: str, rotor_diameter: int,
                     turbine_power: float, hub_height: float, commissioning_year: int,
                     generator_type: Optional[Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']] = 'gb_dfig',
                     recycled_share_steel: float = None,
                     electricity_mix_steel: Optional[Literal['Norway', 'Europe', 'Poland']] = None,
                     lifetime: int = 20, land_use_permanent_intensity: int = 3000, land_cover_type: str = None,
                     eol_scenario: int = 1,
                     cf: float = 0.24, time_adjusted_cf: float = 0.009,
                     regression_adjustment: Literal['D2h', 'Hub height'] = 'D2h',
                     include_life_cycle_stages: bool = True,
                     eol: bool = True, transportation: bool = True,
                     use_and_maintenance: bool = True, installation: bool = True,
                     comment: str = ''):
    """
    It creates the life-cycle inventories per unit (turbine and wind park) and per kwh (also turbine and wind park)
    and store them as activities in the database new_db.
    It is possible to run only specific stages of the life-cycle, by setting them to True or False in the parameters.
    By default, lifetime set to 20 years.
    By default, the land cover is chosen randomly according to probability. It can be manually selected
    by changing manual_land_cover. The allowed values are: 'industrial_from', 'crops, non-irrigated', 'row crops',
    'shrubland', 'grassland', 'pasture', 'forest', 'unspecified'.
    The variables recycled_share_steel and electricity_mix_steel allow to manually select the amount of recycled steel
    in the turbine and the type of electricity mix used in the steel production. By default, the values are set to None,
    and the recycling share and electricity_mix is set automatically according to historic data from the steel industry
    (Eurofer).
    By default, cf is 0.24 (European mean in 2023) and time_adjusted_cf is 0.009 (Xu et al. 2023)
    By default, all the stages are included (set to True).
    By default, include_life_cycle_stages = True, that means that a new activity is created for each stage. In case
    this is set to False, all the inputs would be added to the single_turbine activity. Important note: if this is set
    to False, the lca_wind_turbine_extended() function WON'T WORK AS EXPECTED (because it relies on inventories that
    contain an activity for each stage).
    """
    if park_power != number_of_turbines * turbine_power:
        print("WARNING. The power of the park does not match the power sum of the unitary turbines. "
              "The inventory of the park will use the specified number of turbines and their unitary powers. "
              "However, the transformer will be scaled according to the park power.")
        print("Park power: " + str(park_power), 'Number of turbines: ' + str(number_of_turbines) +
              'Unitary power: ' + str(turbine_power))
        print('Consider redoing the analysis if you think there was a mistake with any of the numbers')
    if comment == '':
        comment = (f'park_name: {park_name}, park_power: {park_power} MW, number_of_turbines: {number_of_turbines}, '
                   f'park_location: {park_location}, park_coordinates: {park_coordinates}, manufacturer: {manufacturer}, '
                   f'rotor_diameter: {rotor_diameter} m, turbine_power: {turbine_power} MW, hub_height: {hub_height} m, '
                   f'commissioning_year: {commissioning_year}, generator_type: {generator_type}, '
                   f'recycled_share_steel: {recycled_share_steel}, electricity_mix_steel: {electricity_mix_steel} '
                   f'lifetime: {lifetime} years, land_use_permanent_intensity: {land_use_permanent_intensity} m2/MW,'
                   f'land_cover_type: {land_cover_type}, eol_scenario: {eol_scenario}, cf: {cf*100} %, '
                   f'annual attrition rate: {time_adjusted_cf}'
                   )
    mass_materials_park = lci_materials(park_name=park_name, park_power=park_power,
                                        number_of_turbines=number_of_turbines,
                                        park_location=park_location, park_coordinates=park_coordinates,
                                        manufacturer=manufacturer,
                                        rotor_diameter=rotor_diameter, generator_type=generator_type,
                                        recycled_share_steel=recycled_share_steel,
                                        electricity_mix_steel=electricity_mix_steel,
                                        lifetime=lifetime,
                                        turbine_power=turbine_power,
                                        hub_height=hub_height, commissioning_year=commissioning_year,
                                        include_life_cycle_stages=include_life_cycle_stages, comment=comment,
                                        regression_adjustment=regression_adjustment,
                                        new_db=new_db, cutoff391=cutoff391)
    if transportation:
        transport(manufacturer=manufacturer, park_coordinates=park_coordinates, park_name=park_name,
                  generator_type=generator_type, turbine_power=turbine_power, hub_height=hub_height,
                  include_life_cycle_stages=include_life_cycle_stages, regression_adjustment=regression_adjustment,
                  rotor_diameter=rotor_diameter,
                  new_db=new_db, cutoff391=cutoff391)
    if installation and not land_cover_type:
        trans, occ = land_use(turbine_power=turbine_power, park_name=park_name,
                              include_life_cycle_stages=include_life_cycle_stages, lifetime=lifetime,
                              land_use_permanent_intensity=land_use_permanent_intensity,
                              biosphere3=biosphere3, new_db=new_db)
        auxiliary_road_materials(turbine_power=turbine_power, park_name=park_name,
                                 include_life_cycle_stages=include_life_cycle_stages,
                                 new_db=new_db, cutoff391=cutoff391)
        excavation_activities(generator_type=generator_type, turbine_power=turbine_power, hub_height=hub_height,
                              rotor_diameter=rotor_diameter, number_of_turbines=number_of_turbines, park_name=park_name,
                              include_life_cycle_stages=include_life_cycle_stages,
                              regression_adjustment=regression_adjustment,
                              new_db=new_db, cutoff391=cutoff391)
    elif installation and land_cover_type:
        trans, occ = land_use(turbine_power=turbine_power, park_name=park_name, manual_land_cover=land_cover_type,
                              include_life_cycle_stages=include_life_cycle_stages, lifetime=lifetime,
                              land_use_permanent_intensity=land_use_permanent_intensity,
                              biosphere3=biosphere3, new_db=new_db)
        auxiliary_road_materials(turbine_power=turbine_power, park_name=park_name,
                                 include_life_cycle_stages=include_life_cycle_stages,
                                 new_db=new_db, cutoff391=cutoff391)
        excavation_activities(generator_type=generator_type, turbine_power=turbine_power, hub_height=hub_height,
                              rotor_diameter=rotor_diameter, number_of_turbines=number_of_turbines, park_name=park_name,
                              include_life_cycle_stages=include_life_cycle_stages,
                              regression_adjustment=regression_adjustment,
                              new_db=new_db, cutoff391=cutoff391)
    if use_and_maintenance:
        maintenance(park_name=park_name, generator_type=generator_type, turbine_power=turbine_power,
                    hub_height=hub_height, include_life_cycle_stages=include_life_cycle_stages, lifetime=lifetime,
                    regression_adjustment=regression_adjustment, rotor_diameter=rotor_diameter,
                    new_db=new_db, cutoff391=cutoff391)
    if eol:
        end_of_life(scenario=eol_scenario, park_name=park_name, generator_type=generator_type,
                    turbine_power=turbine_power,
                    hub_height=hub_height, include_life_cycle_stages=include_life_cycle_stages,
                    regression_adjustment=regression_adjustment, rotor_diameter=rotor_diameter,
                    new_db=new_db, cutoff391=cutoff391)

    # Create electricity_production activity per turbine and per park (per kWh)
    try:
        # turbine
        if time_adjusted_cf != 0:
            cf_comment = 'CF: ' + str(cf) + '. Attrition rate: ' + str(time_adjusted_cf)
        else:
            cf_comment = 'CF: ' + str(cf) + '. Constant CF (no attrition rate)'
        elec_prod_turbine_act = new_db.new_activity(name=park_name + '_turbine_kwh', code=park_name + '_turbine_kwh',
                                                    location=park_location, unit='kilowatt hour', comment=cf_comment)
        elec_prod_turbine_act.save()
        new_exc = elec_prod_turbine_act.new_exchange(input=elec_prod_turbine_act.key, amount=1.0, unit='kilowatt hour',
                                                     type='production')
        new_exc.save()
        # park
        elec_prod_park_act = new_db.new_activity(name=park_name + '_park_kwh', code=park_name + '_park_kwh',
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
    elec_turbine, elec_park = electricity_production(park_name=park_name, park_power=park_power,
                                                     cf=cf, time_adjusted_cf=time_adjusted_cf,
                                                     new_db=new_db)
    # to turbine activity
    turbine_amount = 1 / elec_turbine
    turbine_act = new_db.get(park_name + '_single_turbine')
    new_exc = elec_prod_turbine_act.new_exchange(input=turbine_act, amount=turbine_amount, type='technosphere')
    new_exc.save()
    elec_prod_turbine_act.save()
    # to park activity
    park_amount = 1 / elec_park
    park_act = new_db.get(park_name + '_' + str(park_power))
    new_exc = elec_prod_park_act.new_exchange(input=park_act, amount=park_amount, type='technosphere')
    new_exc.save()
    elec_prod_park_act.save()

    consts.PRINTED_WARNING_STEEL = False

    if installation:
        return mass_materials_park, trans, occ
    else:
        trans, occ = 0, 0
        return mass_materials_park, trans, occ


def lca_wind_turbine(new_db: bd.Database,
                     park_name: str, park_power: float,
                     method: str = 'ReCiPe 2016 v1.03, midpoint (H)',
                     indicators: List[Tuple[str, str, str]] = None,
                     turbine: bool = True):
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
    results = {}
    results_kwh = {}

    # setting the methods
    if indicators:
        methods = indicators
    else:
        methods = [m for m in bd.methods if method in m[0] and 'no LT' not in m[0]]

    if turbine:
        print('Functional unit: 1 turbine')
        if indicators:
            print('LCIA methods: ' + str(indicators)[1:-1])
        else:
            print('LCIA methods: ' + str(method))
        try:
            act = new_db.get(park_name + '_single_turbine')
            act_kwh = new_db.get(park_name + '_turbine_kwh')
            print(act_kwh._data['comment'])
        except bd.errors.UnknownObject:
            print('There is no inventory created with the park name you specified. '
                  'Please, make sure you entered the right park name, or use the function lci_wind_turbine to create a '
                  'wind park inventory with this name.')
            sys.exit()
    else:
        print('Functional unit: 1 wind park')
        if indicators:
            print('LCIA methods: ' + str(indicators)[1:-1])
        else:
            print('LCIA methods: ' + str(method))
        try:
            act = new_db.get(park_name + '_' + str(park_power))
            act_kwh = new_db.get(park_name + '_park_kwh')
            print(act_kwh._data['comment'])
        except bd.errors.UnknownObject:
            print('There is no inventory created with the park name you specified. '
                  'Please, make sure you entered the right park name, or use the function lci_wind_turbine to create a '
                  'wind park inventory with this name.')
            sys.exit()

    # per unit
    lca_obj = act.lca(amount=1)
    for m in methods:
        lca_obj.switch_method(m)
        lca_obj.lcia()
        results[m[1]] = lca_obj.score
    # per kwh
    lca_obj = act_kwh.lca(amount=1)
    for m in methods:
        lca_obj.switch_method(m)
        lca_obj.lcia()
        results_kwh[m[1]] = lca_obj.score

    return results, results_kwh


def delete_new_db(new_db: bd.Database):
    for a in new_db:
        a.delete()
    if len(new_db) == 0:
        print('The database new_db was cleared and it is now empty')


pass

