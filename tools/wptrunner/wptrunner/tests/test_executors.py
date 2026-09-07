# mypy: allow-untyped-defs

from types import SimpleNamespace
from unittest import mock

import pytest

from ..executors import base
from ..executors import executorchrome
from ..executors import executorwebdriver


@pytest.mark.parametrize("ranges_value, total_pages, expected", [
    ([], 3, {1, 2, 3}),
    ([[1, 2]], 3, {1, 2}),
    ([[1], [3, 4]], 5, {1, 3, 4}),
    ([[1],[3]], 5, {1, 3}),
    ([[2, None]], 5, {2, 3, 4, 5}),
    ([[None, 2]], 5, {1, 2}),
    ([[None, 2], [2, None]], 5, {1, 2, 3, 4, 5}),
    ([[1], [6, 7], [8]], 5, {1})])
def test_get_pages_valid(ranges_value, total_pages, expected):
    assert base.get_pages(ranges_value, total_pages) == expected


@pytest.mark.parametrize("test_window, expected_window", [
    (None, "current-window"),
    ("explicit-window", "explicit-window"),
])
def test_webdriver_testdriver_run_uses_selected_window(
    test_window, expected_window
):
    current_window = "current-window"
    message = [0, "complete", {}]
    webdriver = SimpleNamespace(url=None)
    loop = mock.Mock()
    loop.run_until_complete.return_value = mock.sentinel.serialized_message
    bidi_script = SimpleNamespace(
        call_function=mock.Mock(return_value=mock.sentinel.awaitable)
    )
    parent = SimpleNamespace(
        base=SimpleNamespace(current_window=current_window),
        webdriver=webdriver,
        logger=mock.Mock(),
        loop=loop,
        bidi_script=bidi_script,
    )
    part = executorwebdriver.WebDriverTestDriverProtocolPart(parent)
    part.setup()
    callback = mock.Mock(return_value=(True, mock.sentinel.result))

    with mock.patch.object(
        executorwebdriver, "bidi_deserialize", return_value=message
    ), mock.patch.object(
        executorwebdriver,
        "WebDriverAsyncCallbackHandler",
        return_value=callback,
    ) as callback_cls:
        result = part.run(
            "https://example.test/test.html", "resume script", test_window
        )

    assert result is mock.sentinel.result
    assert part._test_window == expected_window
    callback_cls.assert_called_once_with(
        parent.logger, parent, expected_window, loop
    )
    assert bidi_script.call_function.call_args.kwargs["target"] == {
        "context": expected_window
    }


@pytest.mark.parametrize("require_bidi, expected", [
    (False, executorchrome.ChromeDriverProtocol),
    (True, executorchrome.ChromeDriverBidiProtocol),
])
def test_chromedriver_reftest_selects_protocol(require_bidi, expected):
    browser = SimpleNamespace(
        pac=None,
        webdriver_url="http://127.0.0.1:4444/",
        leak_check=False,
        is_extension_test=False,
    )
    assert (executorchrome.ChromeDriverRefTestExecutor.protocol_cls is
            executorchrome.ChromeDriverProtocol)

    executor = None
    try:
        executor = executorchrome.ChromeDriverRefTestExecutor(
            logger=mock.Mock(),
            browser=browser,
            server_config={},
            screenshot_cache={},
            capabilities=None,
            browser_settings={"require_webdriver_bidi": require_bidi}
        )
        assert executor.protocol_cls is expected
        assert type(executor.protocol) is expected
        assert (executorchrome.ChromeDriverRefTestExecutor.protocol_cls is
                executorchrome.ChromeDriverProtocol)
    finally:
        if require_bidi and executor is not None:
            executor.protocol.loop.close()
