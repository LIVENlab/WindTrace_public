# WindTrace 

## About
WindTrace is a python open-source parametric life-cycle inventory model to create tailor-made inventories of onshore wind turbines and park fleets and assess their environmental impacts.
It has been funded by the European Commission though the Horizon Europe project "JUSTWIND4ALL" (GA 101083936). 


## Getting started
To get ready to use WindTrace, you must first create a virtual environment using the YAML file provided.
WindTrace creates a [Brightway2.5](https://github.com/brightway-lca/brightway25) project and a database within this project where all the inventories you create will be stored. Before you start, you must change the variables `PROJECT_NAME`, and `NEW_DB_NAME` in `consts.py` and choose the names that you wish. 
**IMPORTANT**: at the moment WindTrace is compatible with **Ecoinvent v3.9.1, v3.10 and v3.10.1**. You must have an Ecoinvent license to run WindTrace. \
In consts.py, you must change the **SPOLD_FILES** variable and choose the route where you have the spold files of this Ecoinvent version stored in your computer.

In case you don't want to create wind turbine inventories yourself, you can import into Brightway the database in the folder [turbine_examples](https://github.com/LIVENlab/WindTrace_public/tree/main/turbine_examples), where **2.0 MW, 4.5 MW, 6.0 MW and 8.0 MW turbines with different characteristics are sampled**. As explained later, the individual turbines inventories are stored per unit of turbine under the code `'Park_name'` + `'_single_turbine'`. The same inventories per kWh have the code `'Park_name'` + `'_turbine_kwh'`. To know how the parameters where defined to create these inventories (the technical parameters of the turbine) you can check the comment in the `'Park_name'` + `'_single_turbine'` inventory.


## Parameters
The function allowing to create customized inventories is **lci_wind_turbine()**, which admits the following parameters:
| Parameter                     | Description                                                                  | Default                             | Type |
|------------------------------|------------------------------------------------------------------------------|-------------------------------------|------|
| Park name                    | Name of the wind park                                                        | *Required*                          | `str` |
| Park power                   | Total power of the wind park (MW)                                            | *Required*                          | `float` |
| Number of turbines           | Number of turbines in the wind park                                          | *Required*                          | `int` |
| Park location                | Country code (ISO 3166-1 alpha-2)                                            | *Required*                          | `str` |
| Park coordinates             | Latitude and longitude (WGS84)                                               | *Required*                          | `tuple` |
| Manufacturer                 | Turbine manufacturer                                                         | `LM Wind`                           | `'LM Wind'`, `'Vestas'`, `'Siemens Gamesa'`, `'Enercon'`, `'Nordex'` |
| Rotor diameter               | Rotor diameter (m)                                                           | *Required*                          | `int` |
| Turbine rated power          | Nominal power per turbine (MW)                                               | *Required*                          | `float` |
| Hub height                   | Turbine hub height (m)                                                        | *Required*                          | `float` |
| Regression adjustment        | Steel mass as function of `D2h` or `Hub height`                              | `D2h`                               | `'D2h'`, `'Hub height'` |
| Commissioning year           | Year turbine started operation                                               | *Required*                          | `int` |
| Recycled steel share         | Recycled content share of steel                                              | Eurofer data (2012–2021)            | `float` |
| Land use permanent intensity | Permanent land transformation per MW (m²/MW)                                 | `3000`                              | `float` |
| Electricity mix steel        | Electricity mix for steel production                                         | Eurofer country mix                 | `'Norway'`, `'Europe'`, `'Poland'` *(optional)* |
| Generator type               | Gearbox and generator type                                                   | `Doubly fed induction generator`    | `'dd_eesg'`, `'dd_pmsg'`, `'gb_pmsg'`, `'gb_dfig'` *(optional)* |
| Land cover type              | Land cover before installation                                               | *Required*                          | `'crops, non-irrigated'`, `'grassland'`, `'road'`, etc. |
| End-of-life scenario         | Material recycling scenario: <br> `1`: baseline<br> `2`: optimistic<br> `3`: pessimistic | `Baseline` | `str` |
| Lifetime                     | Expected lifetime of the turbine (years)                                     | `20`                                | `int` |
| Capacity factor              | Avg. delivered electricity / theoretical max                                 | `0.24`                              | `float` |
| Attrition rate               | Annual performance degradation                                               | `0.009`                             | `float` |

Parameters without a default value must be filled. Parameters with a default value are not compulsory but can be adapted to the user needs.

## Example of application:

The repository has a Jupyter Notebook with a step-by-step guide on how to use WindTrace.

## Citing WindTrace

If you use WindTrace for academic work please cite:

Sierra‐Montoya, Miquel, et al. "WindTrace: Assessing the environmental impacts of wind energy designs with a parametric life cycle inventory model." Journal of Industrial Ecology (2025). https://doi.org/10.1111/jiec.70114

