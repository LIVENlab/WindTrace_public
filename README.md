### WindTrace ###

WindTrace, an open-source parametric life-cycle inventory model to create tailor-made inventories of onshore wind turbines and park fleets and assess their environmental impacts.

The function allowing to create customized inventories is lci_wind_turbine(), which admits the following parameters:
| Parameter                   | Description                                                  | Default value                                   | Type                                                           |
|-----------------------------|--------------------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------|
| Park name                   | Name of the wind park                                        | No                                              | str                                                            |
| Park power                  | Total power of the wind park                                 | No                                              | float                                                          |
| Number of turbines          | Number of turbines in the wind park                          | No                                              | int                                                            |
| Park location               | Abbreviation of the country (ISO 3166-1 alpha-2 codes)       | No                                              | str                                                            |
| Park coordinates            | WGS84 coordinates of the wind park (latitude, longitude)     | No                                              | tuple                                                          |
| Manufacturer                | Name of the turbine manufacturer (Vestas, Siemens, etc.)     | LM Wind                                         | str                                                            |
| Rotor diameter              | Diameter of the rotor (in meters)                            | No                                              | int                                                            |
| Turbine rated power         | Nominal power of the individual turbines                     | No                                              | float                                                          |
| Hub height                  | Height of the turbine                                        | No                                              | float                                                          |
| Commissioning year          | Year when the turbine started operation                      | No                                              | int                                                            |
| Recycled steel share        | Share of recycled steel in the turbine                       | Data from Eurofer (2012-2021)                   | Optional[Literal['dd_eesg', 'dd_pmsg', 'gb_pmsg', 'gb_dfig']]  |
| Land use permanent intensity | Permanent direct land transformation per MW                 | 3000 m²/MW                                      | float                                                          |
| Electricity mix steel       | Electricity mix for steel production                         | Mix per Eurofer data by country                 | Optional[Literal['Norway', 'Europe', 'Poland']]                |
| Generator type              | Gearbox and type of generator                                | Doubly fed induction generator (gearbox)        | int                                                            |
| Land cover type             | Land cover type prior to turbine installation                | No                                              | int                                                            |
| End-of-life scenario        | Scenario based on material recycling rates                   | Baseline                                        | str                                                            |
| Lifetime                    | Expected lifetime of the turbine                             | 20 years                                        | int                                                            |
| Capacity factor             | Ratio of average delivered electricity to maximum production | 0.24                                            | float                                                          |
| Attrition rate              | Annual efficiency reduction due to wear                      | 0.009                                           | float                                                          |

Parameters without a default value must be filled. Parameters with a default value are not compulsory but can be adapted to the user needs.

Examples of application:
`lci_wind_turbine(park_name='example', park_power=10.0, number_of_turbines=2, park_location='ES', park_coordinates='41.502, -1.126', manufacturer='Vestas', rotor_diameter=97, turbine_power=5.0, hub_height=110, commissioning_year=2015, generator_type='gb_dfig', recycled_share_steel=0.43, electricity_mix_steel='Europe', lifetime=20, eol_scenario=1, cf=0.24, time_adjusted_cf=0.009)`

The inventories are given by unit of turbine, unit of park, and kWh (for both turbine and park). They are stored in a new database ('new_db') in your brightway2.5 project. In this example, the inventory codes would be:
- Turbine (FU=unit): example_single_turbine
- Turbine (FU=kWh): example_turbine_kwh
- Park (FU=unit): example_10.0
- Park (FU=kWh): example_park_kwh

You can calculate the impacts of the park or an individual turbine of the park by calling lca_wind_turbine(). It returns two dictionaries: the first one being results per unit and the second one being results per kWh.
If you want to analyse the whole park, you must set the parameter 'turbine' to False. Otherwise, a single turbine of the park will be analysed. By default, ReCiPe 2016 v1.03, midpoint (H) is used, but it can be manually changed by giving the variable 'method' another name of a method in Brightway. An example of another method below.
Here an example of application for the whole park:
`lca_wind_turbine(park_name='example', park_power=10.0, method='EF v3.1', turbine=False)`
