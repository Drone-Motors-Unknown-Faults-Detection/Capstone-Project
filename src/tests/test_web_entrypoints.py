"""`python -m web.server`／`python -m web.guide` 要跑正式模組的 main，模組狀態只有一份。"""
import runpy
import unittest
import warnings
from unittest import mock

import web.guide
import web.server


class EntrypointTests(unittest.TestCase):
    def test_module_entrypoints_call_canonical_main(self):
        for module in (web.server, web.guide):
            with self.subTest(module=module.__name__), mock.patch.object(module, "main") as main:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # runpy 提示模組已在 sys.modules
                    runpy.run_module(module.__name__, run_name="__main__")
                main.assert_called_once_with()

    def test_allow_host_reaches_guide_socket_module(self):
        """guide 掛載的 /guide/ws 讀 web.server.ALLOWED_HOSTS；--allow-host 必須寫進這一份。"""
        saved = set(web.server.ALLOWED_HOSTS)
        try:
            web.server.allow_hosts("0.0.0.0", ["140.134.138.33"])
            self.assertIn("140.134.138.33", web.server.ALLOWED_HOSTS)
            self.assertIs(web.guide.GuideSocket.prepare, web.server.LocalSocketHandler.prepare)
        finally:
            web.server.ALLOWED_HOSTS.clear()
            web.server.ALLOWED_HOSTS.update(saved)


if __name__ == "__main__":
    unittest.main()
