"""docker/ 的 compose 與 Dockerfile 契約：版本訂死、模型名稱與 tool calling 對得上 web 端、資料唯讀。"""
import os
import re
import shutil
import subprocess
import tempfile
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
        # AI 問答讀實驗手冊：映像要帶 docs/experiments，Dockerfile 也要 COPY
        self.assertIn("!docs/experiments/", rules)
        self.assertIn("COPY docs/experiments ./docs/experiments",
                      (DOCKER / "Dockerfile").read_text(encoding="utf-8"))

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


class RestartScriptTests(unittest.TestCase):
    """以假 docker 記錄 scripts/opt/restart_docker.sh 的呼叫順序；不需要 Docker daemon。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "scripts/opt").mkdir(parents=True)
        shutil.copy(ROOT / "scripts/opt/restart_docker.sh", self.root / "scripts/opt/")
        (self.root / "docker").mkdir()
        shutil.copy(DOCKER / ".env.example", self.root / "docker/.env.example")
        (self.root / "data").mkdir()
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        self.calls = self.root / "calls.txt"
        fakes = {
            "docker": f'echo "$*" >> "{self.calls}"\n'
                      '[ "$1 $2" = "info --format" ] && echo "$FAKE_RUNTIMES"\nexit 0\n',
            "nvidia-smi": '[ -n "$FAKE_GPU_MEM" ] || exit 1\necho "$FAKE_GPU_MEM"\n',
            "fake-hook": "exit 0\n",
        }
        for name, body in fakes.items():
            (bin_dir / name).write_text("#!/bin/bash\n" + body, encoding="utf-8")
            (bin_dir / name).chmod(0o755)
        # 預設模擬裝好 toolkit 的 L40S（48 GB 獨立顯存）
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "NVIDIA_HOOK": "fake-hook", "FAKE_GPU_MEM": "46068", "FAKE_RUNTIMES": '{"runc":{}}'}

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, *args, **env):
        return subprocess.run(["bash", str(self.root / "scripts/opt/restart_docker.sh"), *args],
                              env={**self.env, **env}, capture_output=True, text=True)

    def env_value(self, key):
        text = (self.root / "docker/.env").read_text(encoding="utf-8")
        return re.search(rf"^{key}=(.*)$", text, re.M).group(1)

    def lines(self):
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []

    def test_all_downs_then_rebuilds(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.lines(), [
            "info",
            "compose -f compose.yaml down --remove-orphans",
            "compose -f compose.yaml up -d --build",
            "compose -f compose.yaml ps",
        ])
        env = (self.root / "docker/.env").read_text(encoding="utf-8")
        self.assertIn(f"APP_UID={os.getuid()}", env)
        self.assertIn(f"APP_GID={os.getgid()}", env)

    def test_partial_restart_keeps_other_service(self):
        for target, expected in (("app", "compose -f app.compose.yaml up -d --build"),
                                 ("vllm", "compose -f vllm.compose.yaml up -d")):
            with self.subTest(target=target):
                self.calls.unlink(missing_ok=True)
                self.assertEqual(self.run_script(target).returncode, 0)
                lines = self.lines()
                self.assertEqual(lines[1], f"compose -f {target}.compose.yaml down")
                self.assertEqual(lines[2], expected)
                self.assertFalse(any("-v" in line.split() or "--rmi" in line for line in lines))

    def test_existing_env_untouched_and_no_build(self):
        (self.root / "docker/.env").write_text("WEB_PORT=9000\n", encoding="utf-8")
        result = self.run_script("all", "--no-build")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compose -f compose.yaml up -d", self.lines())
        self.assertEqual((self.root / "docker/.env").read_text(encoding="utf-8"), "WEB_PORT=9000\n")
        self.assertIn("http://localhost:9000", result.stdout)

    def test_missing_data_stops_before_down(self):
        (self.root / "data").rmdir()
        result = self.run_script("app")
        self.assertEqual(result.returncode, 2)
        self.assertIn("data/", result.stderr)
        self.assertEqual(self.lines(), ["info"])

    def test_no_container_toolkit_stops_before_down(self):
        for target in ("all", "vllm"):
            with self.subTest(target=target):
                self.calls.unlink(missing_ok=True)
                result = self.run_script(target, NVIDIA_HOOK="missing-hook")
                self.assertEqual(result.returncode, 2)
                self.assertIn("nvidia-ctk runtime configure", result.stderr)
                self.assertEqual(self.lines(), ["info", "info --format {{json .Runtimes}}"])

    def test_app_only_does_not_need_gpu(self):
        result = self.run_script("app", NVIDIA_HOOK="missing-hook")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_nvidia_runtime_registered_is_enough(self):
        result = self.run_script("vllm", NVIDIA_HOOK="missing-hook",
                                 FAKE_RUNTIMES='{"nvidia":{"path":"nvidia-container-runtime"}}')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_memory_utilization_follows_gpu_kind(self):
        for gpu_mem, expected in (("46068", "0.9"), ("[N/A]", "0.5"), ("", "0.9")):
            with self.subTest(gpu_mem=gpu_mem):
                (self.root / "docker/.env").unlink(missing_ok=True)
                self.assertEqual(self.run_script(FAKE_GPU_MEM=gpu_mem).returncode, 0)
                self.assertEqual(self.env_value("VLLM_GPU_MEMORY_UTILIZATION"), expected)

    def test_existing_env_mismatch_warns_without_rewriting(self):
        (self.root / "docker/.env").write_text("VLLM_GPU_MEMORY_UTILIZATION=0.5\n", encoding="utf-8")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("建議 0.9", result.stderr)
        self.assertEqual(self.env_value("VLLM_GPU_MEMORY_UTILIZATION"), "0.5")

    def test_bad_target(self):
        self.assertEqual(self.run_script("nope").returncode, 2)
        self.assertEqual(self.lines(), [])


if __name__ == "__main__":
    unittest.main()
