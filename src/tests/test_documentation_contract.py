"""操作文件的本機目標與版本界線；不以文字測試代替模型驗證。"""
from pathlib import Path
import unittest
import json
import subprocess
import re
from urllib.parse import quote, unquote

from tornado.testing import AsyncHTTPTestCase

from tests.issue_delivery_evidence import local_links


ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ("docs/README.md", "docs/ci_contract.md", "docs/runtime_policy.md",
             "docs/formal_materialization_contract.md", "docs/experiments/exp8_health.md",
             "docs/navigation/exp13_README.md")
GUIDE_DOCUMENTS = ("exp13_README.md", "exp13_實驗總覽.md", "exp13_來源索引.md",
                   "exp13_結果白話解讀.md", "exp13_示範腳本.md")


def heading_anchors(text):
    """回傳 Markdown ATX 標題的 GitHub 錨點集合；忽略程式區塊，同名標題加 -N 後綴。"""
    result, counts, fence = set(), {}, None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            character = marker[1][0]
            if fence is None:
                fence = character
            elif fence == character:
                fence = None
            continue
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if match:
            title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", match[1]).lower()
            slug = re.sub(r"[^\w\s-]", "", title).replace(" ", "-")
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            result.add(slug if count == 0 else f"{slug}-{count}")
    return result


class DocumentationContractTests(unittest.TestCase):
    def test_report_new_chapter_uses_global_references_and_actual_book(self):
        text = (ROOT / "docs/FinalReport/畢業專題報告.md").read_text(encoding="utf-8")
        body, references = text.split("## 參考文獻\n", 1)
        self.assertNotRegex(body, r"\[(?:Jardine|Lee|Taguchi|Schölkopf|Pimentel|Bendale|Scheirer|Tax|沈品志|Ledoit|Campello|Mahalanobis) \d{4}")
        books = references.split("### 書籍\n", 1)[1].split("### 期刊論文", 1)[0]
        self.assertIn("Taguchi", books)
        self.assertNotIn("未引用書籍", books)
        self.assertIn("版次仍待原始版權頁核實", books)

    def test_publication_fields_keep_sources_and_unverified_edition(self):
        text = (ROOT / "docs/FinalReport/畢業專題報告.md").read_text(encoding="utf-8")
        references = text.split("## 參考文獻\n", 1)[1]
        entries = dict(re.findall(r'<a id="ref-(\d+)"></a>\[\d+\] ([^\n]+)', references))
        support = next(value for value in entries.values()
                       if "Estimating the Support" in value)
        self.assertIn("July 2001", support)
        self.assertIn("direct.mit.edu/neco/article/13/7/1443/6529", support)
        lee = next(value for value in entries.values()
                   if "A Simple Unified Framework" in value)
        self.assertIn("pp. 7167–7177", lee)
        self.assertIn("4–6 December 2018（主會議）", lee)
        self.assertIn("Montréal, Canada", lee)
        self.assertIn("-Metadata.json", lee)
        self.assertIn("Conferences/2018/Dates", lee)
        book = next(value for value in entries.values()
                    if "The Mahalanobis–Taguchi Strategy" in value)
        self.assertIn("New York: John Wiley & Sons", book)
        self.assertIn("kansalliskirjasto.finna.fi/Record/vaari.1825773", book)
        self.assertIn("版次尚未取得原始版權頁核實", book)
        web_index = references.split("### 網頁\n", 1)[1].split("### 學位論文", 1)[0]
        for title in ("OAS", "MLPRegressor"):
            number = next(number for number, value in entries.items()
                          if f'“{title},”' in value)
            self.assertIn(f"{title} [{number}](#ref-{number})", web_index)

    def test_final_report_citations_follow_first_occurrence(self):
        text = (ROOT / "docs/FinalReport/畢業專題報告.md").read_text(encoding="utf-8")
        body, references = text.split("## 參考文獻\n", 1)
        definitions = re.findall(r'<a id="ref-(\d+)"></a>\[(\d+)\]', references)
        self.assertTrue(definitions)
        expected = [str(i) for i in range(1, len(definitions) + 1)]
        self.assertEqual(definitions, [(n, n) for n in expected])
        citations = re.findall(r"\[(\d+)\]\(#ref-(\d+)\)", body)
        self.assertTrue(all(label == target for label, target in citations))
        first = list(dict.fromkeys(target for _, target in citations))
        self.assertEqual(first, expected)
        for label, target in re.findall(r"\[(\d+)\]\(#ref-(\d+)\)", references):
            self.assertEqual(label, target)
            self.assertIn(target, expected)

    def test_final_report_local_links_and_section_status(self):
        path = ROOT / "docs/FinalReport/畢業專題報告.md"
        text = path.read_text(encoding="utf-8")
        for row in local_links(path, ROOT):
            self.assertTrue(row["exists"], row)
            if row["anchor"]:
                target = (ROOT / row["target"]).read_text(encoding="utf-8")
                anchors = heading_anchors(target) | set(re.findall(r'<a id="([^"]+)">', target))
                self.assertIn(unquote(row["anchor"]), anchors, row)
        for section in ("2.2", "2.3", "2.4"):
            content = re.search(rf"^### {re.escape(section)} .*?(?=^### |^## |\Z)",
                                text, re.M | re.S)
            self.assertIsNotNone(content)
            self.assertNotIn("（待撰）", content[0])
        self.assertIn("Word", text)

    def test_scoped_local_anchors_exist(self):
        documents = {ROOT / name for name in DOCUMENTS}
        documents.update(ROOT.glob("docs/navigation/*.md"))
        documents.update(ROOT / "docs/experiments" / name for name in (
            "README.md", "exp1_cold_start.md", "exp3_trend.md", "exp4_polar_map.md",
            "exp8_health.md", "exp6_formal_benchmark.md", "exp13_experiment_navigation.md"))
        documents.add(ROOT / "docs/formal_feature_schema_contract.md")
        documents.add(ROOT / "docs/content_provenance_contract.md")
        documents.update(ROOT.glob("reports/Andy_20261008_*/*.md"))
        documents.update(ROOT.glob("reports/Andy_20261009_*/*.md"))
        for document in sorted(documents):
            for row in local_links(document, ROOT):
                self.assertTrue(row["exists"], row)
                if row["anchor"]:
                    target = ROOT / row["target"]
                    self.assertEqual(target.suffix, ".md", row)
                    self.assertIn(unquote(row["anchor"]),
                                  heading_anchors(target.read_text(encoding="utf-8")), row)

    def test_anchor_fixture_rejects_missing_and_handles_duplicates(self):
        anchors = heading_anchors("# 共用模型生命週期\n# 共用模型生命週期\n"
                                  "```python\n# 假標題\n```\n## Web 串流生命週期\n")
        self.assertEqual(anchors, {"共用模型生命週期", "共用模型生命週期-1", "web-串流生命週期"})
        self.assertNotIn("假標題", anchors)
        self.assertNotIn("不存在", anchors)

    def test_overview_keeps_frontend_eight_column_contract(self):
        text = (ROOT / "docs/navigation/exp13_實驗總覽.md").read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("|")]
        self.assertEqual(len(rows), 26)
        self.assertTrue(all(len(line.strip("|").split("|")) == 8 for line in rows))
        self.assertIn("| 實驗名稱 |", rows[0])
        frontend = (ROOT / "src/web/static/guide.js").read_text(encoding="utf-8")
        self.assertIn('fetch(guideUrl("/docs/exp13_實驗總覽.md"))', frontend)
        self.assertNotIn('fetch("/docs/exp13_實驗總覽.md")', frontend)

    def test_report_attachments_and_readers_resolve(self):
        from tests.ci_evidence import DOCUMENTS as ci_documents
        reports = [*ROOT.glob("reports/Andy_20261008_*/*.md"),
                   *ROOT.glob("reports/Andy_20261009_*/*.md")]
        self.assertTrue(reports)
        for path in reports:
            self.assertFalse([row for row in local_links(path, ROOT) if not row["exists"]], path)
        self.assertTrue(all((ROOT / name).is_file() for name in ci_documents))

    def test_each_original_delivery_has_one_summary_and_no_deleted_draft_reader(self):
        expected = {
            "reports/Andy_20261008_分支整合": "integration_report.md",
            "reports/Andy_20261008_議題交付": "delivery.md",
            "reports/Andy_20261009_主線交付": "README.md",
        }
        summary_names = {"README.md", "integration_report.md", "delivery.md", "branch_inventory.md",
                         "audit_followup.md", "audit_followups.md", "runtime_delivery.md", "pr_description.md"}
        for directory, name in expected.items():
            actual = {path.name for path in (ROOT / directory).glob("*.md")} & summary_names
            self.assertEqual(actual, {name})
        for name in ("ci_evidence.py", "project_closeout_evidence.py", "audit_followup_evidence.py"):
            text = (ROOT / "src/tests" / name).read_text(encoding="utf-8")
            self.assertNotIn("/runtime_delivery.md", text)
            self.assertNotIn("/audit_followups.md", text)
            self.assertNotIn('主線交付/issues', text)

    def test_backup_manifest_preserves_bytes_and_paths(self):
        path = "reports/Andy_20261009_主線交付/backup_manifest.json"
        original = subprocess.check_output(
            ["git", "show", "42ee1b7abee92f85088b6150d64e113931104203:docs/project_closeout_20261009/backup_manifest.json"],
            cwd=ROOT)
        current = (ROOT / path).read_bytes()
        self.assertEqual(current.replace(b"\r\n", b"\n"), original.replace(b"\r\n", b"\n"))
        record = json.loads(current)
        self.assertEqual(record["verified_archive"]["status"], "VERIFIED")
        self.assertEqual(record["preserved_failed_archive"]["status"], "FAILED_PARTIAL")

    def test_local_targets_exist(self):
        for name in DOCUMENTS:
            with self.subTest(document=name):
                rows = local_links(ROOT / name, ROOT)
                self.assertFalse([row for row in rows if not row["exists"]])

    def test_operations_and_history_have_separate_contracts(self):
        runtime = (ROOT / "docs/runtime_policy.md").read_text(encoding="utf-8")
        self.assertIn("uv sync", runtime)
        self.assertIn("uv.lock", runtime)
        self.assertIn("-m pytest src/tests", runtime)
        ci = (ROOT / "docs/ci_contract.md").read_text(encoding="utf-8")
        self.assertIn("./scripts/quantity/run_ruff.sh", ci)
        self.assertIn("./scripts/quantity/run_pytest.sh", ci)
        self.assertIn("不在其中重跑 pytest／ruff", ci)
        self.assertIn("只有 Ubuntu", ci)


class GuideDocumentHTTPTests(AsyncHTTPTestCase):
    def get_app(self):
        from web.guide import GuideHub, application
        self.hub = GuideHub([])
        return application(self.hub)

    def tearDown(self):
        self.hub.executor.shutdown(wait=True)
        super().tearDown()

    def test_five_document_routes_load_without_private_paths_or_fit(self):
        for name in GUIDE_DOCUMENTS:
            with self.subTest(document=name):
                response = self.fetch("/docs/" + quote(name))
                self.assertEqual(response.code, 200)
                text = response.body.decode("utf-8")
                self.assertTrue(text.startswith("# "))
                self.assertNotIn("D:/schoolshit", text)
                self.assertNotIn("C:/Users/", text)
                self.assertNotIn("404: Not Found", text)
        self.assertIsNone(self.hub.demo)
        self.assertEqual(self.hub.full_state()["step"], "缺少資料")
