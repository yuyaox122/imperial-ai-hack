"""agent.installer: the vetting rules, the plan, and the approval-gated install into the toolbox.

No network and no real uv: _http, _run, _find_uv, _target and _now are patched. The files in
tests/fixtures/installer/ are trimmed real answers recorded on 2026-11-15: PyPI's JSON for pyproj 3.8.0 and for
docopt 0.6.2 (a source distribution only), OSV's answer for requests 2.19.0, and uv pip compile's output for
pyproj on macOS arm64 (pyproj's hashes trimmed to the files kept in its PyPI fixture). Other packages are built
in memory, each failing exactly one rule, so a test fails if its rule is removed.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_installer.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import installer

FIXTURES = Path(__file__).parent / "fixtures" / "installer"
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
MAC_WHEEL = "cp312-cp312-macosx_14_0_arm64"
TARGET = {
    "version": "3.12.12",
    "platform": "aarch64-apple-darwin",
    "macos": "27.0",
    "tags": ["cp312-cp312-macosx_27_0_arm64", MAC_WHEEL, "cp312-abi3-macosx_14_0_arm64", "cp312-none-any",
             "py3-none-macosx_14_0_arm64", "py3-none-any"],
    "installed": {"numpy": "2.2.6", "certifi": "2026.10.23"},
}
PYPROJ_SHA = "8606ccbc110ce2e8cbe4c9ba6c3f24cd73f603795ac0fce1ceb8fc27a9bb786e"  # its macosx_14_0_arm64 wheel


def fixture(name: str):
    text = (FIXTURES / name).read_text()
    return json.loads(text) if name.endswith(".json") else text


def stamp(when: datetime) -> str:
    return when.isoformat().replace("+00:00", "Z")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeIndex:
    """PyPI, OSV and uv answered from memory. Records every request and command."""

    def __init__(self):
        self.projects: dict[str, dict] = {}
        self.releases: dict[tuple[str, str], dict] = {}
        self.vulns: dict[tuple[str, str], list] = {}
        self.resolution: list[tuple[str, str, tuple[str, ...]]] = []
        self.compile_stdout: str | None = None
        self.compile_rc, self.compile_stderr = 0, ""
        self.install_rc, self.install_writes = 0, True
        self.http_calls: list[tuple[str, str, dict | None]] = []
        self.commands: list[dict] = []

    # --- building packages

    def add(self, name, version, *, first_days_ago=4000, n_releases=20, licence="MIT", urls=True,
            wheels=(MAC_WHEEL,), sdist=True, vulns=(), via=("requested",), resolve=True):
        first = NOW - timedelta(days=first_days_ago)
        files = []
        for tag in wheels:
            fn = f"{name.replace('-', '_')}-{version}-{tag}.whl"
            files.append({"filename": fn, "packagetype": "bdist_wheel", "digests": {"sha256": sha(fn)},
                          "url": f"https://files.pythonhosted.org/packages/ab/{fn}",
                          "upload_time_iso_8601": stamp(NOW - timedelta(days=1)), "yanked": False})
        if sdist:
            fn = f"{name}-{version}.tar.gz"
            files.append({"filename": fn, "packagetype": "sdist", "digests": {"sha256": sha(fn)},
                          "url": f"https://files.pythonhosted.org/packages/cd/{fn}",
                          "upload_time_iso_8601": stamp(NOW - timedelta(days=1)), "yanked": False})
        self.releases[(name, version)] = {
            "info": {"summary": f"The {name} test package", "license_expression": licence, "license": None,
                     "classifiers": [], "home_page": None, "version": version, "yanked": False,
                     "project_urls": {"Source": f"https://github.com/example/{name}"} if urls else None},
            "urls": files,
            "vulnerabilities": [],
        }
        step = first_days_ago / max(n_releases, 1)
        history = {f"0.{i}": [{"upload_time_iso_8601": stamp(first + timedelta(days=i * step))}]
                   for i in range(n_releases - 1)}
        history[version] = [{"upload_time_iso_8601": stamp(first + timedelta(days=(n_releases - 1) * step))}]
        self.projects[name] = {"info": {"version": version}, "releases": history}
        self.vulns[(name, version)] = list(vulns)
        if resolve:
            self.resolution.append((name, version, tuple(via)))
        return self

    def add_real(self, name, version):
        self.projects[name] = fixture(f"pypi_{name}.json")
        self.releases[(name, version)] = fixture(f"pypi_{name}_{version}.json")
        self.vulns.setdefault((name, version), [])
        return self

    # --- the patched helpers

    def http(self, method, url, *, body=None, timeout=None):
        self.http_calls.append((method, url, body))
        if url == installer.OSV_QUERY:
            assert method == "POST" and body["package"]["ecosystem"] == "PyPI"
            found = self.vulns.get((body["package"]["name"], body["version"]), [])
            return 200, ({"vulns": found} if found else {})
        parts = urllib.parse.urlsplit(url)
        assert parts.scheme == "https" and parts.hostname == "pypi.org", url
        path = parts.path.strip("/").split("/")
        if len(path) == 3:
            data = self.projects.get(path[1])
        else:
            data = self.releases.get((path[1], path[2]))
        return (200, data) if data is not None else (404, None)

    def stdout(self) -> str:
        if self.compile_stdout is not None:
            return self.compile_stdout
        lines = []
        for name, version, via in self.resolution:
            hashes = [f["digests"]["sha256"] for f in self.releases[(name, version)]["urls"]] if \
                (name, version) in self.releases else [sha(f"{name}-{version}")]
            lines.append(f"{name}=={version} \\")
            lines += [f"    --hash=sha256:{h} \\" for h in hashes[:-1]] + [f"    --hash=sha256:{hashes[-1]}"]
            lines.append("    # via " + ", ".join("-r requirements.in" if v == "requested" else v for v in via))
        return "\n".join(lines) + "\n"

    def run(self, cmd, *, timeout, env, cwd=None):
        record = {"cmd": list(cmd), "env": dict(env), "cwd": cwd, "timeout": timeout}
        self.commands.append(record)
        if cmd[1:3] == ["pip", "compile"]:
            record["requirements.in"] = (Path(cwd) / "requirements.in").read_text()
            record["constraints"] = (Path(cwd) / "constraints.txt").read_text()
            return subprocess.CompletedProcess(cmd, self.compile_rc, self.stdout(), self.compile_stderr)
        if cmd[1:3] == ["pip", "install"]:
            text = Path(cmd[cmd.index("-r") + 1]).read_text()
            record["requirements"] = text
            target = Path(cmd[cmd.index("--target") + 1])
            if self.install_writes and self.install_rc == 0:
                for name, version in re.findall(r"^(\S+)==(\S+) \\$", text, re.MULTILINE):
                    info = target / f"{name.replace('-', '_')}-{version}.dist-info"
                    info.mkdir(parents=True, exist_ok=True)
                    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
            return subprocess.CompletedProcess(cmd, self.install_rc, "Installed\n", "" if self.install_rc == 0
                                               else "error: hash mismatch\n")
        raise AssertionError(f"unexpected command {cmd}")

    def compiles(self):
        return [c for c in self.commands if c["cmd"][1:3] == ["pip", "compile"]]

    def installs(self):
        return [c for c in self.commands if c["cmd"][1:3] == ["pip", "install"]]

    def urls_mentioning(self, name: str):
        return [u for _, u, b in self.http_calls if f"/{name}/" in u or (b and b["package"]["name"] == name)]


@pytest.fixture
def index(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    fake = FakeIndex()
    monkeypatch.setattr(installer, "_http", fake.http)
    monkeypatch.setattr(installer, "_run", fake.run)
    monkeypatch.setattr(installer, "_find_uv", lambda: "/fake/bin/uv")
    monkeypatch.setattr(installer, "_target", lambda python: {**TARGET, "python": python,
                                                               "installed": dict(TARGET["installed"])})
    monkeypatch.setattr(installer, "_now", lambda: NOW)
    return fake


def pyproj_plan(index):
    index.add_real("pyproj", "3.8.0")
    index.compile_stdout = fixture("uv_compile_pyproj.txt")
    return installer.plan([{"name": "pyproj", "why": "geodesy"}])


def check(plan, name):
    return next(p for p in plan.packages if p.name == name)


# ---------------------------------------------------------------- names and versions


@pytest.mark.parametrize("name", [
    "https://evil.example/pkg-1.0-py3-none-any.whl", "./local/pkg", "../pkg.tar.gz", "/tmp/pkg",
    "git+https://github.com/x/y.git", "--index-url x", "-r other.txt", "a;b", "numpy @ https://x/numpy.whl",
    "numpy\n--index-url=https://evil.example/simple", "numpy[ex tra]", "numpy[x;y]", "numpy==1.0", "", 42,
])
def test_invalid_names_are_refused_before_reaching_uv(index, name):
    plan = installer.plan([{"name": name, "why": "x"}])
    assert plan.verdict == "refuse"
    [bad] = plan.packages
    assert bad.verdict == "refuse" and "not a valid package name" in bad.reasons[0]
    assert index.commands == [] and index.http_calls == []
    json.dumps(plan.to_dict())


def test_an_invalid_name_never_reaches_the_resolver_beside_valid_ones(index):
    index.add("tinylib", "1.0")
    plan = installer.plan([{"name": "tinylib"}, {"name": "--index-url https://evil.example/simple"}])
    assert index.compiles()[0]["requirements.in"] == "tinylib\n"
    assert check(plan, "tinylib").verdict == "ok"
    assert plan.verdict == "refuse"


@pytest.mark.parametrize("version,line", [("3.8.0", "pyproj==3.8.0"), (">=3.6, <4", "pyproj>=3.6,<4"),
                                          ("~=3.8", "pyproj~=3.8")])
def test_versions_become_requirement_lines(index, version, line):
    index.add_real("pyproj", "3.8.0")
    index.compile_stdout = fixture("uv_compile_pyproj.txt")
    installer.plan([{"name": "pyproj", "version": version}])
    assert index.compiles()[0]["requirements.in"] == line + "\n"


def test_no_requests_or_too_many_are_refused(index):
    assert installer.plan([]).problems == ["no packages were requested"]
    plan = installer.plan([{"name": f"lib{i}"} for i in range(installer.MAX_REQUESTS + 1)])
    assert plan.verdict == "refuse" and plan.problems[0] == "at most 10 packages can be requested at once"
    assert index.compiles()[0]["requirements.in"].count("\n") == installer.MAX_REQUESTS


@pytest.mark.parametrize("version", ["1.0; os_name=='nt'", "1.0 --hash=sha256:00", "latest", "==1.0 || >2"])
def test_invalid_versions_are_refused(index, version):
    plan = installer.plan([{"name": "pyproj", "version": version}])
    assert plan.verdict == "refuse" and "not a valid version" in plan.packages[0].reasons[0]
    assert index.commands == []


# ---------------------------------------------------------------- resolution


def test_mature_package_is_ok_with_its_facts_and_a_single_pinned_hash(index):
    plan = pyproj_plan(index)
    pyproj = check(plan, "pyproj")
    assert (pyproj.verdict, pyproj.reasons, plan.verdict, plan.problems) == ("ok", [], "ok", [])
    facts = pyproj.facts
    assert facts["licence"] == "MIT" and facts["requested"] and facts["via"] == ["requested"]
    assert facts["wheel_filename"] == "pyproj-3.8.0-cp312-cp312-macosx_14_0_arm64.whl"
    assert facts["wheel_sha256"] == PYPROJ_SHA and facts["wheel_url"].startswith("https://files.pythonhosted.org/")
    assert facts["first_release"].startswith("2006-") and facts["n_releases"] >= 50
    assert facts["vulnerabilities"] == [] and facts["similar_to"] == [] and facts["project_urls"]
    assert plan.requirements == f"pyproj==3.8.0 \\\n    --hash=sha256:{PYPROJ_SHA}\n"
    assert plan.summary().startswith("OK to install pyproj (1 new package for the toolbox, 1 already installed)")


def test_compile_uses_pinned_platform_wheels_only_and_holds_installed_versions(index, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    pyproj_plan(index)
    [compile_] = index.compiles()
    cmd = compile_["cmd"]
    for flag, value in [("--only-binary", ":all:"), ("--python-version", "3.12.12"),
                        ("--python-platform", "aarch64-apple-darwin"), ("--index-url", "https://pypi.org/simple"),
                        ("--python", sys.executable), ("--constraint", "constraints.txt")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert {"--generate-hashes", "--no-config", "--no-python-downloads"} <= set(cmd)
    assert compile_["constraints"].splitlines() == ["certifi==2026.10.23", "numpy==2.2.6"]
    assert compile_["env"]["MACOSX_DEPLOYMENT_TARGET"] == "27.0"
    assert "OPENAI_API_KEY" not in compile_["env"] and "UV_INDEX_URL" not in compile_["env"]
    assert compile_["timeout"] == installer.COMPILE_TIMEOUT_S


def test_already_installed_packages_are_not_vetted_or_installed(index):
    index.add("meshkit", "2.0", via=("requested",))
    index.resolution.append(("numpy", "2.2.6", ("meshkit",)))
    plan = installer.plan([{"name": "meshkit"}])
    numpy = check(plan, "numpy")
    assert numpy.facts["already_installed"] and numpy.verdict == "ok" and numpy.facts["via"] == ["meshkit"]
    assert index.urls_mentioning("numpy") == [] and index.urls_mentioning("meshkit")
    assert "numpy" not in plan.requirements and "meshkit==2.0" in plan.requirements
    assert [p.name for p in plan.to_install()] == ["meshkit"]


def test_a_package_already_in_the_toolbox_is_not_vetted_again(index):
    site = installer.toolbox_dir() / "meshkit-2.0.dist-info"
    site.mkdir(parents=True)
    (site / "METADATA").write_text("Metadata-Version: 2.1\nName: meshkit\nVersion: 2.0\n")
    index.add("meshkit", "2.0")
    plan = installer.plan([{"name": "meshkit"}])
    assert check(plan, "meshkit").facts["already_installed"] and index.urls_mentioning("meshkit") == []
    assert "meshkit==2.0" in index.compiles()[0]["constraints"]


def test_resolver_failure_refuses_and_still_vets_the_request(index):
    index.add_real("docopt", "0.6.2")
    index.compile_rc = 1
    index.compile_stderr = ("  × No solution found when resolving dependencies:\n  ╰─▶ Because all "
                            "versions of docopt have no usable wheels and you require docopt, we can conclude that "
                            "your requirements are unsatisfiable.\n")
    plan = installer.plan([{"name": "docopt", "why": "parse args"}])
    assert plan.verdict == "refuse"
    assert plan.problems == ["uv could not resolve the request: No solution found when resolving dependencies: "
                             "Because all versions of docopt have no usable wheels and you require docopt, we can "
                             "conclude that your requirements are unsatisfiable."]
    docopt = check(plan, "docopt")
    assert docopt.version == "0.6.2" and docopt.verdict == "refuse"
    assert any("no wheel for this platform" in r and "only a source distribution" in r for r in docopt.reasons)
    assert plan.requirements == ""


def test_a_conflict_with_an_installed_package_is_explained(index):
    index.compile_rc = 1
    index.compile_stderr = ("  x No solution found when resolving dependencies:\n  Because you require numpy==1.26.4 "
                            "and numpy==2.2.6, we can conclude that your requirements are unsatisfiable.\n")
    index.add("numpy", "1.26.4", resolve=False)
    plan = installer.plan([{"name": "numpy", "version": "1.26.4"}])
    assert "numpy==2.2.6 is already installed and held at that version" in plan.problems[0]


def test_resolver_output_that_is_not_pinned_and_hashed_is_refused(index):
    index.compile_stdout = "-e git+https://github.com/x/y.git#egg=y\n"
    index.add("y", "1.0", resolve=False)
    plan = installer.plan([{"name": "y"}])
    assert plan.verdict == "refuse" and "not pinned, hashed requirements" in plan.problems[0]


def test_missing_uv_refuses(index, monkeypatch):
    monkeypatch.setattr(installer, "_find_uv", lambda: None)
    index.add("tinylib", "1.0")
    plan = installer.plan([{"name": "tinylib"}])
    assert plan.verdict == "refuse" and plan.problems[0].startswith("uv was not found")


# ---------------------------------------------------------------- vetting rules


def test_a_package_without_a_wheel_for_this_platform_is_refused(index):
    index.add("sdistonly", "1.0", wheels=())
    plan = installer.plan([{"name": "sdistonly"}])
    pkg = check(plan, "sdistonly")
    assert pkg.verdict == "refuse" and plan.verdict == "refuse"
    assert pkg.reasons == ["no wheel for this platform (aarch64-apple-darwin); PyPI has only a source distribution, "
                           "and building from source would run the package's own build code"]
    assert pkg.facts["wheel_for_this_platform"] is False and plan.requirements == ""


def test_wheels_for_other_platforms_only_are_refused(index):
    index.add("linuxonly", "1.0", wheels=("cp312-cp312-manylinux_2_28_x86_64", "cp312-cp312-win_amd64"))
    pkg = check(installer.plan([{"name": "linuxonly"}]), "linuxonly")
    assert pkg.verdict == "refuse" and "wheels for other platforms only" in pkg.reasons[0]


def test_the_best_wheel_for_this_interpreter_is_chosen(index):
    index.add("multi", "1.0", wheels=("py3-none-any", MAC_WHEEL, "cp312-cp312-win_amd64"))
    pkg = check(installer.plan([{"name": "multi"}]), "multi")
    assert pkg.facts["wheel_filename"] == f"multi-1.0-{MAC_WHEEL}.whl"


def test_a_wheel_the_resolver_did_not_see_is_refused(index):
    index.add("swapped", "1.0")
    index.compile_stdout = f"swapped==1.0 \\\n    --hash=sha256:{'0' * 64}\n    # via -r requirements.in\n"
    pkg = check(installer.plan([{"name": "swapped"}]), "swapped")
    assert pkg.verdict == "refuse" and "not among the files the resolver saw" in pkg.reasons[0]


def test_a_vulnerable_exact_version_is_refused_and_names_the_ids(index):
    index.add("requests", "2.19.0", vulns=fixture("osv_requests_2.19.0.json")["vulns"])
    plan = installer.plan([{"name": "requests", "version": "2.19.0"}])
    pkg = check(plan, "requests")
    assert pkg.verdict == "refuse" and plan.verdict == "refuse"
    assert pkg.reasons[0] == ("known high vulnerability in requests 2.19.0: GHSA-x84v-xcm2-53pg "
                              "(CVE-2018-18074, PYSEC-2018-28)")
    assert [r for r in pkg.reasons if r.startswith("known high") or "no severity" in r] == pkg.reasons[:1]
    assert sum(r.startswith("known medium vulnerability") for r in pkg.reasons) == 4
    vulns = pkg.facts["vulnerabilities"]
    assert len(vulns) == 5 and vulns[0]["id"] == "GHSA-x84v-xcm2-53pg" and vulns[0]["severity"] == "high"
    assert vulns[0]["summary"] == "Insufficiently Protected Credentials in Requests"
    assert ("POST", installer.OSV_QUERY, {"package": {"name": "requests", "ecosystem": "PyPI"}, "version": "2.19.0"}) \
        in index.http_calls


def test_a_vulnerability_without_any_severity_is_refused(index):
    index.add("quietbug", "1.0", vulns=[{"id": "PYSEC-2099-1", "summary": "Something bad", "aliases": []}])
    pkg = check(installer.plan([{"name": "quietbug"}]), "quietbug")
    assert pkg.verdict == "refuse"
    assert pkg.reasons == ["known vulnerability with no severity score in quietbug 1.0: PYSEC-2099-1"]


def test_a_critical_score_from_a_cvss_vector_alone_is_refused(index):
    vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    index.add("badcve", "1.0", vulns=[{"id": "GHSA-aaaa-bbbb-cccc", "severity": [{"type": "CVSS_V3", "score": vector}]}])
    pkg = check(installer.plan([{"name": "badcve"}]), "badcve")
    assert pkg.verdict == "refuse" and pkg.reasons[0].startswith("known critical vulnerability")


def test_a_medium_vulnerability_is_only_a_caution(index):
    index.add("meh", "1.0", vulns=[{"id": "GHSA-mmmm-mmmm-mmmm", "database_specific": {"severity": "MODERATE"}}])
    pkg = check(installer.plan([{"name": "meh"}]), "meh")
    assert pkg.verdict == "caution"
    assert pkg.reasons == ["known medium vulnerability in meh 1.0: GHSA-mmmm-mmmm-mmmm"]


def test_alias_records_share_the_best_known_severity(index):
    vulns = [{"id": "PYSEC-2099-7", "aliases": ["CVE-2099-7"]},
             {"id": "GHSA-llll-llll-llll", "aliases": ["CVE-2099-7"], "database_specific": {"severity": "LOW"}}]
    index.add("lowbug", "1.0", vulns=vulns)
    pkg = check(installer.plan([{"name": "lowbug"}]), "lowbug")
    assert pkg.verdict == "caution"
    assert pkg.reasons == ["known low vulnerability in lowbug 1.0: GHSA-llll-llll-llll (CVE-2099-7, PYSEC-2099-7)"]


def test_withdrawn_vulnerabilities_are_ignored(index):
    index.add("fixedup", "1.0", vulns=[{"id": "PYSEC-2099-9", "withdrawn": "2099-01-01T00:00:00Z"}])
    assert check(installer.plan([{"name": "fixedup"}]), "fixedup").verdict == "ok"


@pytest.mark.parametrize("vector,score", [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N", 7.5),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1),
    ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:H/A:N", 5.5),
    ("CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:H/I:N/A:N", 5.3),
    ("CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N", None),
])
def test_cvss3_base_scores(vector, score):
    assert installer._cvss3_score(vector) == score


def test_a_new_single_release_package_gets_caution(index):
    index.add("freshpkg", "0.1.0", first_days_ago=12, n_releases=1)
    pkg = check(installer.plan([{"name": "freshpkg"}]), "freshpkg")
    assert pkg.verdict == "caution"
    assert pkg.reasons == ["first release on PyPI was 12 days ago (under 90)",
                           "only 1 release on PyPI (fewer than 3)"]


def test_a_new_project_with_many_releases_gets_caution(index):
    index.add("busynew", "0.9", first_days_ago=30, n_releases=9)
    pkg = check(installer.plan([{"name": "busynew"}]), "busynew")
    assert pkg.verdict == "caution" and pkg.reasons == ["first release on PyPI was 30 days ago (under 90)"]


def test_an_old_project_with_two_releases_gets_caution(index):
    index.add("sleepy", "2.0", first_days_ago=3000, n_releases=2)
    pkg = check(installer.plan([{"name": "sleepy"}]), "sleepy")
    assert pkg.verdict == "caution" and pkg.reasons == ["only 2 releases on PyPI (fewer than 3)"]


def test_no_licence_gets_caution(index):
    index.add("nolicence", "1.0", licence=None)
    pkg = check(installer.plan([{"name": "nolicence"}]), "nolicence")
    assert pkg.verdict == "caution" and pkg.reasons == ["no licence is declared"]


def test_a_yanked_release_gets_caution(index):
    index.add("pulled", "1.0")
    index.releases[("pulled", "1.0")]["info"].update(yanked=True, yanked_reason="broken build")
    pkg = check(installer.plan([{"name": "pulled", "version": "1.0"}]), "pulled")
    assert pkg.verdict == "caution" and pkg.reasons == ["this release was yanked from PyPI: broken build"]


def test_no_project_urls_gets_caution(index):
    index.add("nowhere", "1.0", urls=False)
    pkg = check(installer.plan([{"name": "nowhere"}]), "nowhere")
    assert pkg.verdict == "caution" and pkg.reasons == ["no project URLs (no homepage or source repository)"]


@pytest.mark.parametrize("name,popular", [("numpyy", "numpy"), ("reqeusts", "requests"), ("scikit-lern", "scikit-learn")])
def test_a_typosquat_name_gets_caution(index, name, popular):
    index.add(name, "1.0")
    pkg = check(installer.plan([{"name": name}]), name)
    assert pkg.verdict == "caution" and popular in pkg.facts["similar_to"]
    assert pkg.reasons == [f"name is within two edits of the popular {', '.join(pkg.facts['similar_to'])} "
                           "(possible typosquat)"]


def test_a_typosquat_that_is_not_on_pypi_is_refused_and_flagged(index):
    index.compile_rc, index.compile_stderr = 1, "Because reqeusts was not found in the package registry ..."
    plan = installer.plan([{"name": "reqeusts"}])
    pkg = check(plan, "reqeusts")
    assert pkg.verdict == "refuse" and pkg.reasons[0] == "PyPI has no project named reqeusts"
    assert pkg.facts["similar_to"] == ["requests"]


def test_popular_names_and_their_listed_neighbours_are_not_flagged():
    assert installer._similar_to("numpy") == [] and installer._similar_to("dash") == []
    assert installer._similar_to("ipython") == [] and installer._similar_to("Scikit_Learn") == []
    assert installer._similar_to("pyprojx") == ["pyproj"]


def test_the_plan_verdict_is_the_worst_package_verdict(index):
    index.add("goodlib", "1.0").add("nolicence", "1.0", licence=None, via=("goodlib",))
    plan = installer.plan([{"name": "goodlib"}])
    assert [check(plan, n).verdict for n in ("goodlib", "nolicence")] == ["ok", "caution"]
    assert plan.verdict == "caution"
    index.add("sdistonly", "1.0", wheels=(), via=("goodlib",))
    plan = installer.plan([{"name": "goodlib"}])
    assert check(plan, "goodlib").verdict == "ok" and plan.verdict == "refuse"
    assert plan.summary().startswith("Refused, goodlib cannot be installed: sdistonly: no wheel for this platform")


def test_unreachable_osv_falls_back_to_pypi_and_says_so(index, monkeypatch):
    index.add("offline", "1.0")
    index.releases[("offline", "1.0")]["vulnerabilities"] = [{"id": "PYSEC-2099-3", "aliases": []}]
    real = index.http

    def no_osv(method, url, *, body=None, timeout=None):
        if url == installer.OSV_QUERY:
            raise ConnectionError("down")
        return real(method, url, body=body, timeout=timeout)

    monkeypatch.setattr(installer, "_http", no_osv)
    pkg = check(installer.plan([{"name": "offline"}]), "offline")
    assert pkg.verdict == "refuse"
    assert pkg.reasons == ["known vulnerability with no severity score in offline 1.0: PYSEC-2099-3",
                           "OSV could not be reached; only PyPI's own vulnerability list was checked"]


# ---------------------------------------------------------------- install


def test_install_refuses_without_approval(index):
    plan = pyproj_plan(index)
    for approved in (False, None, "yes", 1):
        with pytest.raises(installer.InstallRefused):
            installer.install(plan, approved=approved)
    assert index.installs() == [] and not installer.toolbox_dir().exists()


def test_install_refuses_a_refused_plan_even_if_its_verdict_was_edited(index):
    index.add("sdistonly", "1.0", wheels=())
    plan = installer.plan([{"name": "sdistonly"}])
    with pytest.raises(installer.InstallRefused, match="refused"):
        installer.install(plan, approved=True)
    plan.verdict = "ok"
    with pytest.raises(installer.InstallRefused, match="refused"):
        installer.install(plan, approved=True)
    assert index.installs() == []


def test_install_runs_the_exact_uv_command_and_records_the_ledger(index, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    index.add("meshkit", "2.0").add("tinydep", "0.5", via=("meshkit",))
    plan = installer.plan([{"name": "meshkit", "why": "read meshes"}])
    result = installer.install(plan, approved=True)
    [run] = index.installs()
    cmd, target = run["cmd"], str(installer.toolbox_dir())
    assert cmd[:3] == ["/fake/bin/uv", "pip", "install"]
    assert cmd[cmd.index("--target") + 1] == target
    assert cmd[cmd.index("--only-binary") + 1] == ":all:" and cmd[cmd.index("--python") + 1] == sys.executable
    assert {"--require-hashes", "--no-deps", "--no-config"} <= set(cmd)
    assert cmd[cmd.index("--index-url") + 1] == "https://pypi.org/simple"
    assert run["requirements"] == plan.requirements == (
        f"meshkit==2.0 \\\n    --hash=sha256:{sha(f'meshkit-2.0-{MAC_WHEEL}.whl')}\n"
        f"tinydep==0.5 \\\n    --hash=sha256:{sha(f'tinydep-0.5-{MAC_WHEEL}.whl')}\n")
    assert "OPENAI_API_KEY" not in run["env"] and run["timeout"] == installer.INSTALL_TIMEOUT_S
    assert result["target"] == target and isinstance(result["seconds"], float) and "Installed" in result["log_tail"]
    assert [i["name"] for i in result["installed"]] == ["meshkit", "tinydep"]
    ledger = json.loads((installer.toolbox_dir().parent / "installed.json").read_text())
    assert [(r["name"], r["version"], r["requested_by"], r["why"]) for r in ledger] == [
        ("meshkit", "2.0", "agent", "read meshes"), ("tinydep", "0.5", "agent", "dependency of meshkit")]
    assert ledger[0]["sha256"] == sha(f"meshkit-2.0-{MAC_WHEEL}.whl") and ledger[0]["approved_at"] == NOW.isoformat()
    assert installer.sys_path_entry() == target

    index.add("otherlib", "1.0")
    installer.install(installer.plan([{"name": "otherlib"}]), approved=True)
    ledger = json.loads((installer.toolbox_dir().parent / "installed.json").read_text())
    assert [r["name"] for r in ledger] == ["meshkit", "tinydep", "otherlib"]


@pytest.mark.parametrize("tamper", [
    lambda text: text.replace(PYPROJ_SHA, "0" * 64),
    lambda text: text + "--index-url https://evil.example/simple\n",
    lambda text: "-e git+https://github.com/x/y.git#egg=y\n" + text,
])
def test_install_refuses_requirements_that_do_not_match_the_vetted_packages(index, tamper):
    plan = pyproj_plan(index)
    plan.requirements = tamper(plan.requirements)
    with pytest.raises(installer.InstallRefused, match="requirements"):
        installer.install(plan, approved=True)
    assert index.installs() == []


def test_install_refuses_a_plan_made_for_another_interpreter(index):
    plan = pyproj_plan(index)
    with pytest.raises(installer.InstallRefused, match="made for"):
        installer.install(plan, approved=True, python="/somewhere/else/python")


def test_install_failure_raises_with_the_log_tail(index):
    plan = pyproj_plan(index)
    index.install_rc = 1
    with pytest.raises(installer.InstallFailed) as caught:
        installer.install(plan, approved=True)
    assert "hash mismatch" in caught.value.log_tail
    assert not (installer.toolbox_dir().parent / "installed.json").exists()


def test_install_checks_the_toolbox_holds_every_planned_package(index):
    plan = pyproj_plan(index)
    index.install_writes = False
    with pytest.raises(installer.InstallFailed, match="lacks pyproj==3.8.0"):
        installer.install(plan, approved=True)


def test_nothing_to_install_runs_nothing(index):
    index.compile_stdout = f"numpy==2.2.6 \\\n    --hash=sha256:{'a' * 64}\n    # via -r requirements.in\n"
    plan = installer.plan([{"name": "numpy"}])
    assert plan.verdict == "ok" and plan.requirements == ""
    assert installer.install(plan, approved=True)["installed"] == [] and index.installs() == []


# ---------------------------------------------------------------- serialisation and helpers


def test_plan_to_dict_is_json_and_round_trips_into_install(index):
    index.add("meshkit", "2.0", vulns=[{"id": "GHSA-mmmm-mmmm-mmmm", "database_specific": {"severity": "LOW"}}])
    plan = installer.plan([{"name": "meshkit", "why": "read meshes"}])
    data = json.loads(json.dumps(plan.to_dict()))
    assert data["verdict"] == "caution" and data["summary"] == plan.summary()
    assert set(data["packages"][0]["facts"]) == set(installer._empty_facts())
    again = installer.InstallPlan.from_dict(data)
    assert again.to_dict() == plan.to_dict()
    assert installer.install(again, approved=True)["installed"][0]["name"] == "meshkit"


def test_toolbox_paths(index):
    assert installer.toolbox_dir() == Path(installer.config.home()) / "toolbox" / "site"
    assert installer.sys_path_entry() is None
    installer.toolbox_dir().mkdir(parents=True)
    assert installer.sys_path_entry() == str(installer.toolbox_dir())


def test_subprocess_env_carries_no_secrets_or_index_overrides(monkeypatch):
    for key in ("OPENAI_API_KEY", "LINKUP_API_KEY", "HF_TOKEN", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL",
                "PIP_INDEX_URL", "PYTHONPATH"):
        monkeypatch.setenv(key, "x")
    env = installer._subprocess_env("27.0")
    assert env["MACOSX_DEPLOYMENT_TARGET"] == "27.0" and env["PATH"] == installer.os.environ["PATH"]
    assert not {"OPENAI_API_KEY", "LINKUP_API_KEY", "HF_TOKEN", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL",
                "PIP_INDEX_URL", "PYTHONPATH"} & set(env)


def test_http_helper_sends_no_credentials_and_only_reaches_pypi_and_osv(monkeypatch):
    monkeypatch.delenv("SENSE_FAKE", raising=False)
    sent = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b'{"vulns": []}'

    class Opener:
        def open(self, request, timeout):
            sent.append((request, timeout))
            return Response()

    monkeypatch.setattr(installer.urllib.request, "build_opener", lambda *handlers: Opener())
    status, data = installer._http("POST", installer.OSV_QUERY, body={"package": {"name": "x"}})
    request, timeout = sent[0]
    assert (status, data, timeout) == (200, {"vulns": []}, installer.HTTP_TIMEOUT_S)
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers["user-agent"] == "sense-harness" and "authorization" not in headers
    assert json.loads(request.data) == {"package": {"name": "x"}}
    for url in ("https://evil.example/pypi/x/json", "http://pypi.org/pypi/x/json",
                "https://user:pw@pypi.org/pypi/x/json", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            installer._http("GET", url)
    assert len(sent) == 1


def test_redirects_may_not_leave_pypi_or_osv():
    handler = installer._SameHostRedirects()
    request = installer.urllib.request.Request("https://pypi.org/pypi/PyYAML/json")
    moved = handler.redirect_request(request, None, 301, "Moved", {}, "https://pypi.org/pypi/pyyaml/json")
    assert moved.full_url == "https://pypi.org/pypi/pyyaml/json"
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.example/pypi/pyyaml/json")


def test_fake_mode_fetches_and_runs_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path))

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("forbidden")

    monkeypatch.setattr(installer.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(installer.subprocess, "run", forbidden)
    monkeypatch.setattr(installer, "_find_uv", lambda: "/fake/bin/uv")
    plan = installer.plan([{"name": "pyproj"}])
    assert calls == []
    assert plan.verdict == "refuse" and plan.problems == ["uv could not be run: fake mode runs no installer subprocess"]
    assert check(plan, "pyproj").reasons[0] == "PyPI could not be reached to find its latest version"
    assert installer.ask_llm_review(plan) is None


def test_the_running_interpreter_is_described_without_a_subprocess(monkeypatch):
    import platform
    from importlib import metadata

    def forbidden(*args, **kwargs):
        raise AssertionError("the running interpreter needs no subprocess")

    monkeypatch.setattr(installer.subprocess, "run", forbidden)
    target = installer._target(sys.executable)
    assert target["python"] == sys.executable and target["version"] == platform.python_version()
    assert "py3-none-any" in target["tags"]
    assert target["installed"]["pytest"] == metadata.version("pytest")


def test_llm_review_is_advisory_and_sees_the_vetted_facts(index, monkeypatch):
    from agent import llm

    index.add("meshkit", "2.0")
    plan = installer.plan([{"name": "meshkit", "why": "read meshes"}])
    seen = []
    monkeypatch.setattr(llm, "ask_json", lambda instruction, data, schema, **kw: seen.append(data) or
                        {"opinion": "A small, established mesh reader.\x00"})
    assert installer.ask_llm_review(plan) is None and seen == []  # fake mode asks nothing
    monkeypatch.delenv("SENSE_FAKE")
    assert installer.ask_llm_review(plan) == "A small, established mesh reader."
    assert seen[0]["packages"][0]["name"] == "meshkit" and seen[0]["verdict"] == plan.verdict == "ok"
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: None)
    assert installer.ask_llm_review(plan) is None
