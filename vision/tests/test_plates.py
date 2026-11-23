from pathlib import Path

import pytest

from vision.matching import Vehicle, find_match, normalise
from vision.plates import PlateBook, Whitelist

WL = [Vehicle("AB12 CDE", "Owner"), Vehicle("YX19ABC", "Haulier")]


def test_matching():
    assert normalise(" ab12 cde ") == "AB12CDE"
    assert find_match("ab12cde", WL).owner == "Owner"
    assert find_match("A812CDE", WL).owner == "Owner"      # B misread as 8
    assert find_match("AB12CDF", WL) is not None           # one wrong char ok
    assert find_match("AB12CXY", WL) is None               # two wrong = different car
    assert find_match("KL70XYZ", WL) is None


def test_book_needs_two_reads_or_timeout():
    b = PlateBook(min_score=0.7, delay=8)
    assert b.want_grab(1, 0) and not b.want_grab(1, 1) and b.want_grab(1, 2.5)
    b.add_read(1, "YX19ABC", 0.9, 1)
    assert b.decide(1, 2, WL) is None                      # only one read yet
    b.add_read(1, "YX19A8C", 0.8, 3)
    info = b.decide(1, 3, WL)
    assert info and info.known and info.owner == "Haulier" and info.plate == "YX19ABC"
    assert b.decide(1, 4, WL) is None                      # only decided once
    assert not b.want_grab(1, 100)                         # no more grabs once decided


def test_book_unknown_on_leave_and_low_score_ignored():
    b = PlateBook(min_score=0.7)
    b.add_read(2, "KL70XYZ", 0.95, 0)
    info = b.decide(2, 1, WL, lost=True)
    assert info and not info.known and info.status == "unknown"
    b.add_read(3, "ZZ99ZZZ", 0.4, 0)
    assert b.decide(3, 100, WL, lost=True) is None         # too unsure to accuse anyone


def test_whitelist_file(tmp_path: Path):
    f = tmp_path / "w.yaml"
    f.write_text("vehicles:\n  - plate: AB12 CDE\n    owner: Owner\n")
    assert Whitelist(str(f)).get()[0].owner == "Owner"


def test_reads_plate_from_fixture():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("fast_alpr")
    from vision.plates import PlateReader, crop_scaled
    img = cv2.imread(str(Path(__file__).parent / "fixtures" / "plate_test.jpg"))
    reads = PlateReader().read(crop_scaled(img, [300, 200, 980, 600], (1280, 720), pad=0))
    assert reads[0][0] == "YX19ABC" and reads[0][1] > 0.7
