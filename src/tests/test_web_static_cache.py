"""靜態檔版本網址：Cloudflare／瀏覽器不能在部署後拿舊 JS 配新伺服器。"""
import re
import tempfile
from pathlib import Path

import tornado.testing
import tornado.web

import web.server as server

STATIC_REFS = re.compile(r'(?:src|href)="(/static/[^"]+)"')


class VersionedStaticTests(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        return server.make_app()

    def refs(self):
        resp = self.fetch("/")
        self.assertEqual(resp.code, 200)
        self.assertEqual(resp.headers["Cache-Control"], "no-store")
        return STATIC_REFS.findall(resp.body.decode())

    def test_every_static_reference_is_versioned(self):
        refs = self.refs()
        names = {r.split("?")[0] for r in refs}
        for name in ("/static/experiments.js", "/static/ai_chat.js", "/static/markdown.js", "/static/icon.png"):
            self.assertIn(name, names)
        for ref in refs:
            self.assertRegex(ref, r"^/static/[^?]+\?v=[0-9a-f]{32,}$")

    def test_versioned_url_cached_long_and_served(self):
        ref = next(r for r in self.refs() if r.startswith("/static/experiments.js?"))
        resp = self.fetch(ref)
        self.assertEqual(resp.code, 200)
        max_age = int(re.search(r"max-age=(\d+)", resp.headers["Cache-Control"]).group(1))
        self.assertGreaterEqual(max_age, 365 * 24 * 3600)   # 網址含內容雜湊，可放心長快取
        on_disk = (server.STATIC_DIR / "experiments.js").read_bytes()
        self.assertEqual(resp.body, on_disk)

    def test_unversioned_url_must_revalidate(self):
        resp = self.fetch("/static/experiments.js")
        self.assertEqual(resp.code, 200)
        self.assertEqual(resp.headers["Cache-Control"], "no-cache")


class VersionChangesWithContentTests(tornado.testing.AsyncHTTPTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.static = Path(self.temp.name)
        (self.static / "index.html").write_text('<script src="/static/app.js"></script>', encoding="utf-8")
        (self.static / "app.js").write_text("console.log(1);", encoding="utf-8")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        tornado.web.StaticFileHandler.reset()
        self.temp.cleanup()

    def get_app(self):
        return server.make_app(self.static)

    def version(self):
        return re.search(r"app\.js\?v=([0-9a-f]+)", self.fetch("/").body.decode()).group(1)

    def test_new_content_gets_new_url(self):
        before = self.version()
        (self.static / "app.js").write_text("console.log(2);", encoding="utf-8")
        tornado.web.StaticFileHandler.reset()   # 正式環境：部署重啟行程即清空雜湊快取
        self.assertNotEqual(self.version(), before)
