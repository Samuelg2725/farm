import numpy as np

from vision.plateinfo import describe, estimate_colour, issue_period


def test_uk_current_plate():
    d = describe("YX19 ABC")
    assert d == {"country": "United Kingdom", "format": "UK current (2001+)", "region": "Yorkshire",
                 "registered": "Mar–Aug 2019"}
    assert describe("KL70XYZ")["registered"] == "Sep 2020 – Feb 2021"
    assert describe("KL70XYZ")["region"] == "Milton Keynes & Luton"


def test_issue_period_edges():
    assert issue_period(2) == "Mar–Aug 2002" and issue_period(51) == "Sep 2001 – Feb 2002"
    assert issue_period(50) == "" and issue_period(1) == ""


def test_older_ni_foreign_private():
    assert describe("A123BCD")["format"] == "UK prefix (1983–2001)"
    assert describe("ABC123A")["format"] == "UK suffix (1963–1983)"
    assert describe("AIZ1234")["region"] == "Northern Ireland"
    assert describe("B12345", "Germany", 0.9)["country"] == "Germany"
    assert describe("SAM1", "Unknown", 0.9)["format"] == "private or unrecognised"


def test_colour():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:] = (200, 40, 30)  # BGR blue
    assert estimate_colour(img) == "blue"
    img[:] = (30, 30, 220)
    assert estimate_colour(img) == "red"
    img[:] = (240, 240, 240)
    assert estimate_colour(img) == "white"
    img[:] = (20, 20, 20)
    assert estimate_colour(img) == "black"
