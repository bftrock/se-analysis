"""Fixtures shared by the suite: synthetic LOGR headers and frames.

Everything here is built by hand rather than read off a sample file. A committed
.dat would make the tests depend on one site's configuration, and the point of
most of these is to pin behaviour on shapes no real file we have happens to
carry -- a channel reconfigured mid-record, a statistic outside the vocabulary,
a row that came up short.
"""

import numpy as np
import pandas as pd
import pytest

# Burlington, VT -- the northern-hemisphere site the geometry tests are written
# around. Nothing depends on it being this site, only on it being well north of
# the tropics so the sun stays south at noon.
LATITUDE = 44.5
LONGITUDE = -73.2


def site_info(rows) -> pd.DataFrame:
    """A Site_info frame in the two-column key/value shape nrgpy hands back."""
    return pd.DataFrame(rows)


@pytest.fixture
def header() -> pd.DataFrame:
    """A LOGR header: top-level fields, then one Sensor History block per channel.

    Channel 1 is an analog Pt1000 carrying only a Description; channel 108 is a
    Modbus pyranometer carrying a Sensor Type, a Description and a Measurand,
    which is the case channel_names() has to prefer the Description for.
    """
    return site_info(
        [
            ("Site Number:", "002682"),
            ("Latitude:", "44.5"),
            ("Longitude:", "-73.2"),
            ("Serial Number:", "820900123"),
            ("Channel:", "1"),
            ("Description:", "Pt1000"),
            ("Units:", "deg_C"),
            ("Serial Number:", "SN-PT-1"),
            ("Channel:", "108"),
            ("Sensor Type:", "Hukseflux SR30"),
            ("Description:", "Hukseflux SR30-POA"),
            ("Measurand:", "Irradiance"),
            ("Units:", "W/m^2"),
            ("Serial Number:", "SN-SR30-9"),
        ]
    )


@pytest.fixture
def statistical() -> pd.DataFrame:
    """A statistical frame: several columns per channel, plus the NotUsed slot."""
    index = pd.date_range("2024-05-01", periods=6, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "Ch1_Avg_deg_C": np.linspace(10.0, 11.0, 6),
            "Ch1_Min_deg_C": np.linspace(9.5, 10.5, 6),
            "Ch1_Max_deg_C": np.linspace(10.5, 11.5, 6),
            "Ch1_SD_deg_C": np.full(6, 0.2),
            "Ch1_NotUsed": np.zeros(6),
            "Ch108_Avg_W/m^2": np.linspace(300.0, 800.0, 6),
        },
        index=index,
    )


@pytest.fixture
def daily() -> pd.DataFrame:
    """A whole day at ten-minute cadence, complete and gap-free."""
    index = pd.date_range("2024-05-01", "2024-05-01 23:50", freq="10min", tz="UTC")
    return pd.DataFrame({"signal": np.arange(len(index), dtype=float)}, index=index)
