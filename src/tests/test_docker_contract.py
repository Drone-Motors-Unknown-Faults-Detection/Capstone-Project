"""docker/ 的 compose 與 Dockerfile 契約：版本訂死、模型名稱與 tool calling 對得上 web 端、資料唯讀。"""
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"


def load(name):
    return yaml.safe_load((DOCKER / name).read_text(encoding="utf-8"))


def default(value):
    """展開 ${VAR:-default} 成預設值。"""
    return re.sub(r"\$\{[A-Z0-9_]+:-([^}]*)\}", r"\1", str(value))


class DockerContractTests(unittest.TestCase):
    def setUp(self):
        self.vllm = load("vllm.compose.yaml")["services"]["vllm"]
        self.web = load("app.compose.yaml")["services"]["web"]

    def test_compose_entry_includes_both_and_shares_project(self):
        main = load("compose.yaml")
        self.assertEqual(main["include"], ["vllm.compose.yaml", "app.compose.yaml"])
        for name in ("compose.yaml", "vllm.compose.yaml", "app.compose.yaml"):
            self.assertEqual(load(name)["name"], "lineage", name)

    def test_vllm_serves_qwen3_with_tool_calling(self):
        cmd = [default(c) for c in self.vllm["command"]]
        self.assertEqual(cmd[:2], ["vllm", "serve"])
        self.assertTrue(cmd[2].startswith("Qwen/Qwen3-"), cmd[2])
        self.assertIn("--enable-auto-tool-choice", cmd)
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "hermes")
        served = cmd[cmd.index("--served-model-name") + 1]
        web_cmd = [default(c) for c in self.web["command"]]
        self.assertEqual(web_cmd[web_cmd.index("--llm-model") + 1], served)
        self.assertEqual(web_cmd[web_cmd.index("--llm-url") + 1], "http://vllm:8000/v1")

    def test_images_pinned(self):
        self.assertRegex(default(self.vllm["image"]), r"^nvcr\.io/nvidia/vllm:\d+\.\d+-py3$")
        froms = re.findall(r"^FROM (\S+)", (DOCKER / "Dockerfile").read_text(encoding="utf-8"), re.M)
        self.assertTrue(froms)
        for image in froms:
            self.assertRegex(image, r":\d+\.\d+\.\d+", image)
            self.assertNotIn("latest", image)

    def test_dockerfile_installs_from_lock_without_test_extras(self):
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("uv sync --locked", text)
        self.assertNotIn("--extra", text)
        self.assertIn('ENTRYPOINT ["python", "-m", "web.server"]', text)

    def test_dockerignore_keeps_data_out_of_image(self):
        lines = (DOCKER / "Dockerfile.dockerignore").read_text(encoding="utf-8").splitlines()
        rules = [x for x in lines if x and not x.startswith("#")]
        self.assertEqual(rules[0], "*")
        self.assertNotIn("!data/", rules)
        self.assertIn("src/tests/", rules)

    def test_web_mounts_data_read_only(self):
        volumes = self.web["volumes"]
        data = next(v for v in volumes if isinstance(v, dict) and v["target"] == "/app/data")
        self.assertEqual(data["source"], "../data")
        self.assertTrue(data["read_only"])
        self.assertFalse(data["bind"]["create_host_path"])
        self.assertIn("../logs:/app/logs", volumes)
        self.assertIn("../output:/app/output", volumes)

    def test_env_example_covers_compose_variables(self):
        env = (DOCKER / ".env.example").read_text(encoding="utf-8")
        keys = set(re.findall(r"^([A-Z0-9_]+)=", env, re.M))
        used = set()
        for name in ("vllm.compose.yaml", "app.compose.yaml"):
            used |= set(re.findall(r"\$\{([A-Z0-9_]+)", (DOCKER / name).read_text(encoding="utf-8")))
        self.assertEqual(used - keys, set())


if __name__ == "__main__":
    unittest.main()
