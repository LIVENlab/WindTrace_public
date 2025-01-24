

park_name ='Cabril_II_R'
name_turbine = park_name + '_repowering_single_turbine'
name_turbine_kwh = park_name + '_repowering_turbine_kwh'
turbine_act = new_db.get(name_turbine)
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



lci_repowering(extension_long=True, extension_short=False, substitution=False, repowering=True,
                   park_name_i='Cabril_I_LER', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=25, number_of_turbines_extension=9,
                   cf_extension=0.208, attrition_rate_extension=0.009,
                   park_power_repowering=18,
                   number_of_turbines_repowering=4,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009
                   )
lci_excel_output(park_name='Cabril_I_LER', extension=True, repowering=False, substitution=False,
                     park_power_repowering=18, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_I_LER', extension=False, repowering=True, substitution=False,
                     park_power_repowering=18, scenario_name='repowering', method_name='EF v3.1')

# Long extension + repowering
lci_repowering(extension_long=False, extension_short=True, substitution=False, repowering=True,
                   park_name_i='Cabril_I_SER', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=10, number_of_turbines_extension=9,
                   cf_extension=0.208, attrition_rate_extension=0.009,
                   park_power_repowering=18,
                   number_of_turbines_repowering=4,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009
                   )
lci_excel_output(park_name='Cabril_I_SER', extension=True, repowering=False, substitution=False,
                     park_power_repowering=18, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_I_SER', extension=False, repowering=True, substitution=False,
                     park_power_repowering=18, scenario_name='repowering', method_name='EF v3.1')

# repowering
lci_repowering(extension_long=False, extension_short=False, substitution=False, repowering=True,
                   park_name_i='Cabril_I_R', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   park_power_repowering=18,
                   number_of_turbines_repowering=4,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009
                   )

lci_excel_output(park_name='Cabril_I_R', extension=False, repowering=True, substitution=False,
                     park_power_repowering=18, scenario_name='repowering', method_name='EF v3.1')

# Long extension + substitution
lci_repowering(extension_long=True, extension_short=False, substitution=True, repowering=False,
                   park_name_i='Cabril_I_LES', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=25, number_of_turbines_extension=9,
                   cf_extension=0.208, attrition_rate_extension=0.009,
                   lifetime_substitution=25, number_of_turbines_substitution=9,
                   cf_substitution=0.208, attrition_rate_substitution=0.009)

lci_excel_output(park_name='Cabril_I_LES', extension=True, repowering=False, substitution=False,
                     park_power_repowering=0,scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_I_LES', extension=False, repowering=False, substitution=True,
                     park_power_repowering=0,scenario_name='substitution', method_name='EF v3.1')

# Short extension + substitution
lci_repowering(extension_long=False, extension_short=True, substitution=True, repowering=False,
                   park_name_i='Cabril_I_SES', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=10, number_of_turbines_extension=9,
                   cf_extension=0.208, attrition_rate_extension=0.009,
                   lifetime_substitution=25, number_of_turbines_substitution=9,
                   cf_substitution=0.208, attrition_rate_substitution=0.009
                   )
lci_excel_output(park_name='Cabril_I_SES', extension=True, repowering=False, substitution=False,
                     park_power_repowering=0,scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_I_SES', extension=False, repowering=False, substitution=True,
                     park_power_repowering=0,scenario_name='substitution', method_name='EF v3.1')

# Substitution
lci_repowering(extension_long=False, extension_short=False, substitution=True, repowering=False,
                   park_name_i='Cabril_I_S', park_power_i=16.2, number_of_turbines_i=9,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=66,
                   turbine_power_i=1.8, hub_height_i=84.35, commissioning_year_i=2002,
                   recycled_share_steel_i=None,
                   lifetime_i=22,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_substitution=25, number_of_turbines_substitution=9,
                   cf_substitution=0.208, attrition_rate_substitution=0.009
                   )

lci_excel_output(park_name='Cabril_I_S', extension=False, repowering=False, substitution=True,
                     park_power_repowering=0,scenario_name='substitution', method_name='EF v3.1')

###CABRIL II
lci_repowering(extension_long=True, extension_short=False, substitution=False, repowering=True,
                   park_name_i='Cabril_II_LER', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   recycled_share_steel_i=None,
                   lifetime_i=19,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=25, number_of_turbines_extension=2,
                   cf_extension=0.223, attrition_rate_extension=0.009,
                   park_power_repowering=4.5,
                   number_of_turbines_repowering=1,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009
                   )
lci_excel_output(park_name='Cabril_II_LER', extension=True, repowering=False, substitution=False,
                     park_power_repowering=4.5, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_II_LER', extension=False, repowering=True, substitution=False,
                     park_power_repowering=4.5, scenario_name='repowering', method_name='EF v3.1')

# Short extension + repowering
lci_repowering(extension_long=False, extension_short=True, substitution=False, repowering=True,
                   park_name_i='Cabril_II_SER', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   recycled_share_steel_i=None,
                   lifetime_i=19,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=10, number_of_turbines_extension=2,
                   cf_extension=0.223, attrition_rate_extension=0.009,
                   park_power_repowering=4.5,
                   number_of_turbines_repowering=1,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009)
lci_excel_output(park_name='Cabril_II_SER', extension=True, repowering=False, substitution=False,
                     park_power_repowering=4.5, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_II_SER', extension=False, repowering=True, substitution=False,
                     park_power_repowering=4.5, scenario_name='repowering', method_name='EF v3.1')

# repowering
lci_repowering(extension_long=False, extension_short=False, substitution=False, repowering=True,
                   park_name_i='Cabril_II_R', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   recycled_share_steel_i=None,
                   lifetime_i=19,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   park_power_repowering=4.5,
                   number_of_turbines_repowering=1,
                   manufacturer_repowering='Enercon',
                   rotor_diameter_repowering=114,
                   turbine_power_repowering=4.5,
                   hub_height_repowering=120,
                   generator_type_repowering='dd_eesg',
                   electricity_mix_steel_repowering='Europe',
                   lifetime_repowering=25,
                   cf_repowering=0.293,
                   attrition_rate_repowering=0.009
                   )

lci_excel_output(park_name='Cabril_II_R', extension=False, repowering=True, substitution=False,
                     park_power_repowering=4.5, scenario_name='repowering', method_name='EF v3.1')

# Long extension + substitution
lci_repowering(extension_long=True, extension_short=False, substitution=True, repowering=False,
                   park_name_i='Cabril_II_LES', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   lifetime_i=19,
                   recycled_share_steel_i=None,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=25, number_of_turbines_extension=2,
                   cf_extension=0.223, attrition_rate_extension=0.009,
                   lifetime_substitution=25, number_of_turbines_substitution=2,
                   cf_substitution=0.223, attrition_rate_substitution=0.009)

lci_excel_output(park_name='Cabril_II_LES', extension=True, repowering=False, substitution=False,
                     park_power_repowering=0, scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_II_LES', extension=False, repowering=False, substitution=True,
                    park_power_repowering=0, scenario_name='substitution', method_name='EF v3.1')

# Short extension + substitution
lci_repowering(extension_long=False, extension_short=True, substitution=True, repowering=False,
                   park_name_i='Cabril_II_SES', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   recycled_share_steel_i=None,
                   lifetime_i=19,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_extension=10, number_of_turbines_extension=2,
                   cf_extension=0.223, attrition_rate_extension=0.009,
                   lifetime_substitution=25, number_of_turbines_substitution=2,
                   cf_substitution=0.208, attrition_rate_substitution=0.009
                   )
lci_excel_output(park_name='Cabril_II_SES', extension=True, repowering=False, substitution=False,
                     park_power_repowering=0,scenario_name='extension', method_name='EF v3.1')
lci_excel_output(park_name='Cabril_II_SES', extension=False, repowering=False, substitution=True,
                     park_power_repowering=0,scenario_name='substitution', method_name='EF v3.1')

    # Substitution
lci_repowering(extension_long=False, extension_short=False, substitution=True, repowering=False,
                   park_name_i='Cabril_II_S', park_power_i=4, number_of_turbines_i=2,
                   park_location_i='PT',
                   park_coordinates_i=(40.98, -8.045),
                   manufacturer_i='Enercon',
                   rotor_diameter_i=70,
                   turbine_power_i=2, hub_height_i=87.6, commissioning_year_i=2005,
                   recycled_share_steel_i=None,
                   lifetime_i=19,
                   electricity_mix_steel_i=None,
                   generator_type_i='dd_eesg',
                   lifetime_substitution=25, number_of_turbines_substitution=2,
                   cf_substitution=0.223, attrition_rate_substitution=0.009
                   )

lci_excel_output(park_name='Cabril_II_S', extension=False, repowering=False, substitution=True,
                     park_power_repowering=0,scenario_name='substitution', method_name='EF v3.1')


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
