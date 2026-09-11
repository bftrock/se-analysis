"""Reading ProtoNode Modbus exports, and the standard expectations for their signals.

SPEC is what a healthy ProtoNode file looks like. It is deliberately expressed in
the named physical limits from se_analysis.validation, so a change to what counts
as a plausible irradiance lands everywhere at once. Site-specific deviations
belong in the notebook, via SPEC.override(...).
"""

import pandas as pd

from se_analysis.validation import (
    ALBEDO_IRRADIANCE,
    ARRAY_TILT,
    BAROMETRIC_PRESSURE,
    DNI_IRRADIANCE,
    GHI_TILT,
    GHI_IRRADIANCE,
    ISC_SOILING_RATIO,
    LATITUDE,
    LOGR_VOLTAGE,
    LONGITUDE,
    RATIO,
    RELATIVE_HUMIDITY,
    SOLAR_AZIMUTH,
    SOLAR_ELEVATION,
    SOLAR_ZENITH,
    STATE,
    SolarAngles,
    Spec,
    TEMPERATURE,
    WIND_DIRECTION,
    WIND_SPEED,
)

TIMESTAMP = "ProtoNode.Modbus_Poll_Timestamp"
PROTONODE_COUNTER = (0, 100)

# Columns that are plumbing rather than measurements
NON_SIGNAL_PREFIXES = ("reserved", "ProtoNode", "Unnamed", "__")

RANGES = {
    "Logger_Voltage": LOGR_VOLTAGE,
    "Ambient_Temp": TEMPERATURE,
    "RH": RELATIVE_HUMIDITY,
    "Counter": PROTONODE_COUNTER,
    "Wind_Speed": WIND_SPEED,
    "Wind_Speed_Max_": WIND_SPEED,
    "Wind_Dir_Corrected_": WIND_DIRECTION,
    "BP": BAROMETRIC_PRESSURE,
    "GHI": GHI_IRRADIANCE,
    "DHI": GHI_IRRADIANCE,
    "DNI": DNI_IRRADIANCE,
    "GHI_Albedo": GHI_IRRADIANCE,
    "RHI_Albedo": ALBEDO_IRRADIANCE,
    "Albedo": RATIO,
    "RefCell_Down": ALBEDO_IRRADIANCE,
    "RefCell_Down_Temp": TEMPERATURE,
    "GHI_ClassA_Pressure": BAROMETRIC_PRESSURE,
    "POA_ClassA_Pressure": BAROMETRIC_PRESSURE,
    "GHI_ClassA_Temp": TEMPERATURE,
    "GHI_ClassA": GHI_IRRADIANCE,
    "Site_Latitude": LATITUDE,
    "Site_Longitude": LONGITUDE,
    "Solar_Zenith": SOLAR_ZENITH,
    "Solar_Elevation": SOLAR_ELEVATION,
    "Solar_Azimuth": SOLAR_AZIMUTH,
    "GHI_Tilt_ClassA": GHI_TILT,
    "VentGHI_FanStatus_1": STATE,
    "Heat_State": STATE,
    "POA_ClassA": GHI_IRRADIANCE,
    "POA_ClassA_Temp": TEMPERATURE,
    "POA_Tilt_ClassA": ARRAY_TILT,
    "GHI_ClassA_RH": RELATIVE_HUMIDITY,
    "POA_ClassA_RH": RELATIVE_HUMIDITY,
    "RefCell_1": GHI_IRRADIANCE,
    "RefCell_Temp_1": TEMPERATURE,
    "Soiling_Tilt": ARRAY_TILT,
    "G_Clean_uncorrected": GHI_IRRADIANCE,
    "G_Soiled_uncorrected": GHI_IRRADIANCE,
    "BOM_Temp_Clean": TEMPERATURE,
    "BOM_Temp_Soiled": TEMPERATURE,
    "Soiling_Isc_Ratio": ISC_SOILING_RATIO,
    "G_Clean": GHI_IRRADIANCE,
    "G_Soiled": GHI_IRRADIANCE,
}

# Both signals should carry the same value
EQUIVALENT = (("stationname", "Logger_SN"),)

# Left signal should always stay below the right
LESSER = (
    ("DHI", "GHI"),
    ("RHI_Albedo", "GHI_Albedo"),
)

# Signals that should never sit still
DYNAMIC = (
    "Counter",
    "Ambient_Temp",
    "RH",
    "Wind_Speed",
    "BP",
    "GHI",
    "DHI",
    "DNI",
    "GHI_Albedo",
    "RHI_Albedo",
    "Albedo",
    "RefCell_Down",
    "RefCell_Down_Temp",
    "GHI_ClassA_Pressure",
    "POA_ClassA_Pressure",
    "GHI_ClassA_Temp",
    "GHI_ClassA",
    "GHI_Tilt_ClassA",
    "POA_ClassA",
    "POA_ClassA_Temp",
    "POA_Tilt_ClassA",
    "RefCell_1",
    "RefCell_Temp_1",
    "G_Clean_uncorrected",
    "G_Soiled_uncorrected",
    "BOM_Temp_Clean",
    "BOM_Temp_Soiled",
    "Soiling_Isc_Ratio",
    "G_Clean",
    "G_Soiled",
)

# Configuration that should be present, non-zero and unchanging
CONSTANT = (
    "stationname",
    "Logger_SN",
    "GHI_SN",
    "SPN1_Mult",
    "GHI_Mult_1",
    "Isc_STC_Clean",
    "Isc_STC_Soiled",
    "POA_SN_1",
    "POA_Mult_1",
    "Site_Latitude",
    "Site_Longitude",
)

# Signals that should sit at one particular value all file long. A ventilator
# fan reads 1 while it is running, so anything else is the fan having stopped.
EXPECTED = {
    "VentGHI_FanStatus_1": 1,
    "VentPOA_FanStatus_2": 1,
}

# Signals expected to read the invalid-data sentinel (-99) throughout
SENTINEL = (
    "MetSENS500_SN",
    "Wind_Quality",
    "Shunt_Res_c",
)

# The angles the logger works out for itself, checked against a position computed
# from the site coordinates. Stamps in these files are written by whatever machine
# polled the ProtoNode, in whatever timezone it was set to, so the offset is left
# to be inferred rather than assumed; once the files carry UTC, pinning
# utc_offset=pd.Timedelta(0) turns that inference into an assertion.
SOLAR = SolarAngles(
    latitude="Site_Latitude",
    longitude="Site_Longitude",
    zenith="Solar_Zenith",
    elevation="Solar_Elevation",
    azimuth="Solar_Azimuth",
)

SPEC = Spec(
    ranges=RANGES,
    equivalent=EQUIVALENT,
    lesser=LESSER,
    dynamic=DYNAMIC,
    constant=CONSTANT,
    expected=EXPECTED,
    sentinel=SENTINEL,
    solar=SOLAR,
)


def read(path) -> pd.DataFrame:
    """A ProtoNode tab-delimited export, indexed by its Modbus poll timestamp."""
    data = pd.read_csv(path, delimiter="\t")
    data[TIMESTAMP] = pd.to_datetime(data[TIMESTAMP])
    return data.set_index(TIMESTAMP)


def signal_columns(data: pd.DataFrame) -> list[str]:
    """Measurement columns, with the reserved/ProtoNode/unnamed plumbing dropped."""
    return [col for col in data.columns if not col.startswith(NON_SIGNAL_PREFIXES)]
