from WindTrace_onshore import steel_turbine, other_turbine_materials, foundations_mat, materials_mass
import consts


def test_steel_turbine_outputs():
    _, full, short, intersection = steel_turbine(regression_adjustment='Hub height')
    print("Intersection at hub height:", intersection['Low alloy steel'].item())
    assert isinstance(full, dict)
    assert 'Low alloy steel' in full
    polyfit = full['Low alloy steel']['polyfit']
    assert callable(polyfit)
    assert isinstance(full['Low alloy steel']['std_dev'], float)


def test_other_materials_structure():
    full, short, intersection = other_turbine_materials()
    assert isinstance(full, dict)
    assert 'Aluminium' in full or 'Epoxy resin' in full
    for k, v in full.items():
        assert 'polyfit' in v and callable(v['polyfit'])


def test_foundation_materials_fit():
    full, short, intersection = foundations_mat(mat_file=consts.VESTAS_FILE)
    assert 'Concrete_foundations' in full
    assert 'polyfit' in full['Concrete_foundations']
    pred = full['Concrete_foundations']['polyfit'](2.0)
    assert pred > 0


def test_materials_mass_outputs():
    materials, uncertainty = materials_mass(
        generator_type='dd_eesg',
        turbine_power=2.0,
        hub_height=80,
        rotor_diameter=120,
        regression_adjustment='D2h'
    )
    assert isinstance(materials, dict)
    assert 'Low alloy steel' in materials
    assert materials['Low alloy steel'] > 0
    assert 'Neodymium' in materials
    assert materials['Neodymium'] > 0


def test_chromium_steel_mass_matches_data_v120():
    # V120-2.0: Power = 2.0 MW, Chromium steel = 27.14 t
    expected_mass_tonnes = 27.14

    full, _, _ = other_turbine_materials(regression_adjustment="Hub height")
    chromium_poly = full['Chromium steel']['polyfit']
    predicted_mass_tonnes = chromium_poly(2.0)

    assert abs(predicted_mass_tonnes - expected_mass_tonnes) < 2.0, (
        f"Expected ~{expected_mass_tonnes} t, got {predicted_mass_tonnes:.2f} t"
    )


def test_steel_mass_matches_data_v136():
    # V136-3.45: Power = 3.45 MW, Hub height = 132.0 m, steel = 424.88 t
    expected_mass_tonnes = 424.88

    _, full, _, _ = steel_turbine(regression_adjustment="Hub height")
    predicted_mass_tonnes = full['Low alloy steel']['polyfit'](132.0)

    assert abs(predicted_mass_tonnes - expected_mass_tonnes) < 15.0, (
        f"Expected ~{expected_mass_tonnes} t, got {predicted_mass_tonnes:.2f} t"
    )


def test_steel_mass_matches_data_v117_d2h():
    # V117-3.3: D2h = 1,252,543.5, steel = 243.09 t
    expected_mass_tonnes = 243.09
    d2h = 117 ** 2 * 92.5  # Rotor diameter squared × hub height

    _, full, _, _ = steel_turbine(regression_adjustment="D2h")
    predicted_mass_tonnes = full['Low alloy steel']['polyfit'](d2h)

    assert abs(predicted_mass_tonnes - expected_mass_tonnes) < 10.0, (
        f"Expected ~{expected_mass_tonnes} t, got {predicted_mass_tonnes:.2f} t"
    )
