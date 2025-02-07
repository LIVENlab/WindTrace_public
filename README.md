# WindTrace 

## About
WindTrace is a python open-source parametric life-cycle inventory model to create tailor-made inventories of onshore wind turbines and park fleets and assess their environmental impacts.

## Getting started
To get ready to use WindTrace, you must first create a virtual environment using the YAML file provided.
WindTrace creates a [Brightway2.5](https://github.com/brightway-lca/brightway25) project and a database within this project where all the inventories you create will be stored. Before you start, you must change the variables `PROJECT_NAME`, and `NEW_DB_NAME` in `consts.py` and choose the names that you wish. 
**IMPORTANT**: at the moment WindTrace only works with **Ecoinvent v3.9.1**. In consts.py, you must change the **SPOLD_FILES** variable and choose the route where you have the spold files of this Ecoinvent version stored in your computer.

In case you don't want to create wind turbine inventories yourself, you can import into Brightway the database in the folder [turbine_examples](https://github.com/LIVENlab/WindTrace_public/tree/main/turbine_examples), where **2.0 MW, 4.5 MW, 6.0 MW and 8.0 MW turbines with different characteristics are sampled**. As explained later, the individual turbines inventories are stored per unit of turbine under the code `'Park_name'` + `'_single_turbine'`. The same inventories per kWh have the code `'Park_name'` + `'_turbine_kwh'`. To know how the parameters where defined to create these inventories (the technical parameters of the turbine) you can check the comment in the `'Park_name'` + `'_single_turbine'` inventory.


## Parameters
The function allowing to create customized inventories is **lci_wind_turbine()**, which admits the following parameters:
| Parameter                   | Description                                                  | Default value                                   | Type                                                           |
|-----------------------------|--------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------|
| Park name                   | Name of the wind park                                        | No                                              | str                                                            |
| Park power                  | Total power of the wind park [MW]                            | No                                              | float                                                          |
| Number of turbines          | Number of turbines in the wind park                          | No                                              | int                                                            |
| Park location               | Abbreviation of the country (ISO 3166-1 alpha-2 codes)       | No                                              | str                                                            |
| Park coordinates            | WGS84 coordinates of the wind park (latitude, longitude)     | No                                              | tuple                                                          |
| Manufacturer                | Name of the turbine manufacturer     | LM Wind                                         | str [Literal['LM Wind','Vestas','Siemens Gamesa','Enercon','Nordex']                   |
| Rotor diameter              | Diameter of the rotor (in meters)                            | No                                              | int                                                            |
| Turbine rated power         | Nominal power of the individual turbines [MW]                | No                                              | float                                                          |
| Hub height                  | Height of the turbine (in meters)                           | No                                              | float                                                          |
| Commissioning year          | Year when the turbine started operation                      | No                                              | int                                                            |
| Recycled steel share        | Recycled content share of steel                       | Data from Eurofer (2012-2021)                   | float                                                          |
| Land use permanent intensity | Permanent direct land transformation per MW                 | 3000 m²/MW                                      | float                                                          |
| Electricity mix steel       | Country of the electricity mix for steel production                         | Mix per Eurofer data by country                 | Optional[Literal['Norway', 'Europe', 'Poland']]                |
| Generator type              | Gearbox and type of generator                                | Doubly fed induction generator (gearbox)        | Optional[Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']]  |                                                           |
| Land cover type             | Land cover type prior to turbine installation                | No                                              | str[Literal['industrial, from','crops, non-irrigated','unspecified', 'row crops', 'shrubland', 'grassland', 'pasture', 'forest', 'industrial','road']                                              |
| End-of-life scenario        | Scenario based on material recycling rates (1: baseline, 2: optimistic, 3: pessimistic)                   | Baseline                                        | str                                                            |
| Lifetime                    | Expected lifetime of the turbine (years)                            | 20 years                                        | int                                                            |
| Capacity factor             | Ratio of average delivered electricity to maximum theoretical production | 0.24                                            | float                                                          |
| Attrition rate              | Annual efficiency reduction due to wear                      | 0.009                                           | float                                                          |

Parameters without a default value must be filled. Parameters with a default value are not compulsory but can be adapted to the user needs.

## Example of application:

### Generation of Life Cycle Inventories: lci_wind_turbine()

```ruby
lci_wind_turbine(park_name='Garriguella', park_power=10.0, number_of_turbines=2, park_location='ES', park_coordinates=(41.502, -1.126), manufacturer='Vestas', rotor_diameter=97, turbine_power=5.0, hub_height=110, commissioning_year=2015, generator_type='gb_dfig', recycled_share_steel=0.43, electricity_mix_steel='Europe', lifetime=20, eol_scenario=1, cf=0.24, time_adjusted_cf=0.009)
```

The inventories are given by unit of turbine, unit of park, and kWh (for both turbine and park). They are stored in a new database ('new_db') in your brightway2.5 project. In this example, the inventory codes would be:
- Turbine (FU=unit): Garriguella_single_turbine
- Turbine (FU=kWh): Garriguella_turbine_kwh
- Park (FU=unit): Garriguella_10.0
- Park (FU=kWh): Garriguella_park_kwh
  
For generating activities and calculating impacts for a given wind park with different input parameters, users must introduce different park_name parameters. For example, for two different electricity mixes for steel production in Garriguella wind park:

```ruby
lci_wind_turbine(park_name='Garriguella_EMEurope', park_power=10.0, number_of_turbines=2, park_location='ES', park_coordinates=(41.502, -1.126), manufacturer='Vestas', rotor_diameter=97, turbine_power=5.0, hub_height=110, commissioning_year=2015, generator_type='gb_dfig', recycled_share_steel=0.43, electricity_mix_steel='Europe', lifetime=20, eol_scenario=1, cf=0.24, time_adjusted_cf=0.009)
```

```ruby
lci_wind_turbine(park_name='Garriguella_EMPoland', park_power=10.0, number_of_turbines=2, park_location='ES', park_coordinates=(41.502, -1.126), manufacturer='Vestas', rotor_diameter=97, turbine_power=5.0, hub_height=110, commissioning_year=2015, generator_type='gb_dfig', recycled_share_steel=0.43, electricity_mix_steel='Poland', lifetime=20, eol_scenario=1, cf=0.24, time_adjusted_cf=0.009)
```

### Calculation of Life Cycle Impact Assessment: lca_wind_turbine() 

LCA impacts of a park or an individual turbine of the park can be calculated with **lca_wind_turbine()**. It returns two dictionaries: the first one being results per unit and the second one being results per kWh.
Default results are given per turbine. For the whole park, you must set the parameter 'turbine' to False. 
_ReCiPe 2016 v1.03, midpoint (H)_ is the default LCIA method. It can be manually changed by giving the parameter 'method' another name of a method in Brightway.

Here an example of application for the whole park:

```ruby
lca_wind_turbine(park_name='Garriguella', park_power=10.0, method='EF v3.1', turbine=False)
```

## Citing WindTrace

If you use WindTrace for academic work please cite:

Sierra-Montoya, M., Muñoz-Liesa, J., Pérez-Sánchez, L. À., de Tomás-Pascual, A., & Madrid-López, C. (2024). _WindTrace: a parametric life-cycle inventory model for wind turbines. Unveiling the influence of technical parameters on environmental impacts_ [manuscript submitted for publication].

