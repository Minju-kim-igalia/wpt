# mypy: allow-untyped-defs

from types import SimpleNamespace
from unittest import mock

import pytest

from wptrunner.browsers import chrome


def _test_case(color_space=None, testdriver_features=None, path="/reftest.html"):
    return SimpleNamespace(
        environment={},
        testdriver_features=testdriver_features,
        path=path,
        color_space=color_space,
    )


@pytest.mark.parametrize("color_space, expected", [
    (None, False),
    ("srgb", True),
    ("display-p3", True),
    ("rec2020", True),
    ("rec2100-pq", True),
    ("rec2100-hlg", True),
])
def test_settings_require_bidi_for_color_space_reftest(
    color_space, expected
):
    browser = chrome.ChromeBrowser(
        mock.Mock(), manager_number=0, webdriver_binary="chromedriver"
    )
    assert (browser.settings(_test_case(color_space))["require_webdriver_bidi"]
            is expected)


def test_settings_return_to_classic_after_color_space_reftest():
    browser = chrome.ChromeBrowser(
        mock.Mock(), manager_number=0, webdriver_binary="chromedriver"
    )

    assert browser.settings(
        _test_case("display-p3"))["require_webdriver_bidi"] is True
    assert browser.settings(
        _test_case())["require_webdriver_bidi"] is False


def test_settings_accept_test_without_color_space_attribute():
    browser = chrome.ChromeBrowser(
        mock.Mock(), manager_number=0, webdriver_binary="chromedriver"
    )
    test = _test_case()
    del test.color_space

    assert browser.settings(test)["require_webdriver_bidi"] is False


@pytest.mark.parametrize("testdriver_features, path", [
    (["bidi"], "/test.html"),
    (["extensions"], "/test.html"),
    (None, "/web-extensions/test.html"),
])
def test_settings_preserve_existing_bidi_requirements(
    testdriver_features, path
):
    browser = chrome.ChromeBrowser(
        mock.Mock(), manager_number=0, webdriver_binary="chromedriver"
    )

    settings = browser.settings(_test_case(
        testdriver_features=testdriver_features,
        path=path,
    ))
    assert settings["require_webdriver_bidi"] is True
