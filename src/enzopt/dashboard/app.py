"""enzopt dashboard (Streamlit). Start with ``enzopt dashboard`` (or ``streamlit run src/enzopt/dashboard/app.py``).

Left: projects, [새 실행], runs (newest first, with state). Top: stage tabs ①–⑤. The page only reads run folders;
runs are started as separate engine processes, so closing this page or the server does not stop a computation.
"""

from __future__ import annotations

import streamlit as st

from enzopt.dashboard import data as D
from enzopt.dashboard.launch import cancel_run, resume_run, start_run
from enzopt.figures import sig3
from enzopt.pipeline.runner import STAGE_TITLES, STAGES

st.set_page_config(page_title="enzopt", layout="wide")


# ------------------------------------------------------------------------------------------------ sidebar
def sidebar():
    st.sidebar.title("enzopt")
    st.sidebar.caption("속도 상수를 몰라도, 효소 반응 조건은 정할 수 있습니다.")
    projects = D.list_projects()
    if not projects:
        st.sidebar.info(f"프로젝트 없음: {D.projects_root()}에 project.yaml이 있는 폴더를 두세요")
        return None, None
    names = [p.name for p in projects]
    sel = st.sidebar.selectbox("프로젝트", names, key="project")
    proj = projects[names.index(sel)]
    quick = st.sidebar.checkbox("짧은 모드(시험용)", key="quick")
    no_hopsan = st.sidebar.checkbox("Hopsan 끄기", key="no_hopsan")
    if st.sidebar.button("새 실행", type="primary", key="new_run"):
        rd, pid = start_run(proj, D.runs_root(), quick, hopsan=not no_hopsan)
        st.session_state["run"] = rd.name
        st.session_state["flash"] = f"새 실행 시작: {rd.name} (엔진 프로세스 {pid})"
    runs = D.project_runs(proj)
    if not runs:
        st.sidebar.caption("실행 없음")
        return proj, None
    rnames = [r.name for r in runs]
    labels = {r.name: r.label for r in runs}
    cur = st.session_state.get("run")
    idx = rnames.index(cur) if cur in rnames else 0
    choice = st.sidebar.radio("실행 (최신순)", rnames, index=idx, format_func=lambda n: labels[n])
    st.session_state["run"] = choice
    return proj, runs[rnames.index(choice)]


# ------------------------------------------------------------------------------------------------ status
def status_panel(run: D.RunView) -> None:
    if run.state == "running":
        progress_box(run)
    elif run.state == "failed":
        st.error(run.failure_line())
        stg = run.failed_stage
        if stg and st.button(f"{STAGE_TITLES.get(stg, stg)}부터 다시", key="redo"):
            resume_run(run.path, stg)
            st.session_state["flash"] = f"{STAGE_TITLES.get(stg, stg)}부터 다시 시작"
            st.rerun()
    elif run.state == "interrupted":
        st.warning("중단된 실행: 끝난 단계는 저장되어 있음")
        if st.button("이어서 실행", key="resume"):
            resume_run(run.path)
            st.rerun()
    elif run.state == "cancelled":
        st.warning("취소된 실행")
    warns = run.manifest.get("warnings") or []
    if warns:
        with st.expander(f"경고 {len(warns)}건"):
            for w in warns:
                st.markdown(f"- {w}")


@st.fragment(run_every=2)
def progress_box(run: D.RunView) -> None:
    r = D.load_run(run.path)
    if r.state != "running":
        st.rerun(scope="app")
    p = r.progress or {}
    frac = float(p.get("overall_fraction") or 0.0)
    stage = p.get("stage_title") or "시작 준비"
    eta = p.get("eta_s")
    eta_txt = f" · 남은 시간 약 {eta / 60:.1f}분" if eta else ""
    st.progress(min(max(frac, 0.0), 1.0), text=f"진행 중: {stage} {p.get('message', '')}{eta_txt}")
    if st.button("취소", key="cancel"):
        cancel_run(r.path)
        st.toast("취소 요청")


def banner(run: D.RunView) -> None:
    b = run.banner
    if b:
        st.error(f"⚠ {b}")


def figures(run: D.RunView, stage: str) -> None:
    figs = D.stage_figures(run, stage)
    if not figs:
        return
    cols = st.columns(2)
    for i, f in enumerate(figs):
        with cols[i % 2]:
            st.image(str(run.path / stage / f["png"]), caption=f["name"])


def _short(v):
    """Display value: numbers to 4 significant digits (the CSV files keep full precision)."""
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            return v
        return int(f) if f.is_integer() and abs(f) < 1e9 else float(f"{f:.4g}")
    if isinstance(v, float):
        return float(f"{v:.4g}")
    return v


def table(rows: list[dict] | None, key: str | None = None) -> None:
    if rows:
        st.dataframe([{k: _short(v) for k, v in r.items()} for r in rows], hide_index=True, key=key)


# ------------------------------------------------------------------------------------------------ tabs
def tab_verdict(run):
    d, why = D.stage_json(run, "1_verdict", "verdict.json")
    if d is None:
        st.caption(why)
        return
    main = d["networks"]["main"]
    st.subheader(main["headline"])
    st.info("보장하는 것 / 전제 — " + main["guarantee_line"])
    rows = []
    names = {"main": "식별", "control": "비약가역 대조" if main["weakly_reversible"] else "대조", "design": "설계"}
    names.update(d.get("labels") or {})
    for k, v in d["networks"].items():
        rows.append(
            {
                "반응망": f"{names.get(k, k)} ({v['name']})",
                "복합체": v["n_complexes"],
                "연결 성분": v["n_linkage_classes"],
                "랭크": v["rank"],
                "결핍": v["deficiency"],
                "약가역": "예" if v["weakly_reversible"] else "아니오",
                "보존 법칙": len(v["conservation_laws"]),
                "등급": v["grade"],
            }
        )
    table(rows, "verdict_table")
    hs = (run.manifest.get("flags") or {}).get("hopsan_status")
    if hs:
        st.caption(f"해석 백엔드: {hs}")
    figures(run, "1_verdict")
    p, _ = D.stage_file(run, "1_verdict", "report.txt")
    if p:
        with st.expander("판정 보고 전문"):
            st.code(p.read_text(encoding="utf-8"), language=None)


def tab_data(run):
    d, why = D.stage_json(run, "2_data", "data.json")
    if d is None:
        st.caption(why)
        return
    obs = (d.get("observable") or {}).get("name") or "A340"  # what the data files hold (A340, NADH_mM, …)
    rows = []
    for k, v in d["experiments"].items():
        t = v.get("temperature") or {}
        rows.append(
            {
                "실험": k,
                "역할": v["role"],
                "점 수": v["n"],
                "시간(s)": f"{v['t_first_s']:g}–{v['t_last_s']:g}",
                f"{obs} 처음": round(v["A_first"], 4),
                f"{obs} 끝": round(v["A_last"], 4),
                "Et (nM)": v["Et_nM"],
                "온도 평균(℃)": round(t.get("mean_C", float("nan")), 2) if t else None,
                "온도 범위(℃)": round(t.get("range_C", float("nan")), 3) if t else None,
            }
        )
    table(rows, "data_table")
    figures(run, "2_data")


def tab_fit(run):
    d, why = D.stage_json(run, "3_fit", "fit_summary.json")
    if d is None:
        st.caption(why)
        return
    dev = d.get("device") or {}
    c = st.columns(4)
    c[0].metric("표본(남음/시드)", f"{len(d['kept'])}/{len(d['seeds'])}")
    c[1].metric("장치", f"{dev.get('used', '?')} {dev.get('precision', '')}")
    c[2].metric("학습 시간", f"{d['train_seconds']:.0f} s")
    hs = {k: v["mean"] for k, v in d["holdout_rmse"]["summary"].items() if v["mean"] is not None}
    if len(hs) > 4:  # many held-out sets: one line, not a list (per set in the comparison table below)
        vals = list(hs.values())
        line = f"평균 {sig3(sum(vals) / len(vals))} · 최대 {sig3(max(vals))}"
        c[3].metric(f"홀드아웃 RMSE(표본 평균, {len(vals)}개)", line)
    else:
        c[3].metric("홀드아웃 RMSE(평균)", ", ".join(f"{k} {sig3(v)}" for k, v in hs.items()))
    if dev.get("fallback"):
        st.warning(dev["fallback"])
    for s, reason in (d.get("excluded") or {}).items():
        st.caption(f"시드 {s} 제외: {reason}")
    if d.get("design_blocked"):
        st.error("남은 표본 < 2: 설계 단계에서 멈춤")
    p, _ = D.stage_file(run, "3_fit", "holdout_comparison.csv")
    if p:
        with st.expander("홀드아웃 외삽 비교(PINN · 고전 피팅 · 대조)"):
            table(D.read_table(p), "holdout_cmp")
    figures(run, "3_fit")


def tab_sim(run):
    d, why = D.stage_json(run, "4_sim", "sim_summary.json")
    if d is None:
        st.caption(why)
        return
    h = d.get("hopsan") or {}
    c = st.columns(3)
    c[0].metric("CRNT 검증", f"{d['n_pass']}/{d['n_runs']} 통과")
    c[1].metric("수치 오류 의심 재실행", d["n_retried"])
    full = h.get("full_representative") or {}
    c[2].metric(
        "Hopsan vs scipy 최대 차이(관측값)",
        sig3(full["max_abs_diff_A340"]) if full.get("max_abs_diff_A340") is not None else "Hopsan 미사용",
    )
    if not h.get("used"):
        st.warning(h.get("note") or "Hopsan 미사용")
    elif full.get("note"):
        st.caption(full["note"])
    figures(run, "4_sim")
    p, _ = D.stage_file(run, "4_sim", "validation.csv")
    if p:
        with st.expander("검증 표(실행별)"):
            table(D.read_table(p), "validation")


def tab_design(run):
    d, why = D.stage_json(run, "5_design", "design_summary.json")
    if d is None:
        st.caption(why)
        return
    if d.get("skipped"):
        st.caption(d["skipped"])
        return
    c = st.columns(4)
    if d.get("goal") == "peak":
        lo, hi = d["stop_window_s"]
        c[0].metric("강건 ρ*", sig3(d["rho_star"]))
        c[1].metric("효소량", f"{sig3(d['Et_nM'])} nM")
        c[2].metric("멈출 시각", f"{d['robust_stop_s']:.0f} s")
        c[3].metric("최악 표본 수율", f"{d['robust_yield'] * 100:.1f} %")
        st.caption(f"{d['window_fraction'] * 100:.0f} % 이상 유지 구간 {lo:.0f}~{hi:.0f} s")
    elif d.get("goal") == "time":
        c[0].metric("강건 ρ*", sig3(d["rho_star"]))
        c[1].metric("효소량", f"{sig3(d['Et_nM'])} nM")
        c[2].metric("최악 표본 도달 시간", f"{d['robust_time_s']:.0f} s")
    if d.get("goal") in ("peak", "time"):
        p, _ = D.stage_file(run, "5_design", "pipette.csv")
        if p:
            st.markdown("**피펫 표**")
            table(D.read_table(p), "pipette")
            st.download_button(
                "피펫 표 내려받기(CSV)", p.read_bytes(), file_name=f"pipette_{run.name}.csv", key="dl_pip"
            )
        figures(run, "5_design")
        return
    c[0].metric("강건 ρ*", sig3(d["rho_star"]))
    c[1].metric("강건 Et*", f"{sig3(d['robust_Et_star_nM'])} nM")
    c[2].metric("E₀ 배수", sig3(d["robust_Et_over_E0"]) if d.get("robust_Et_over_E0") else "—")
    c[3].metric("공칭 Et(미달 표본)", f"{sig3(d['nominal_Et_at_rho_star_nM'])} nM ({d['nominal_fail']})")
    if d.get("rho_min"):
        st.caption(
            f"CRNT 평형 한계 ρ_min = {sig3(d['rho_min'])} (K' = {sig3(d['Kp'])}): "
            "이보다 작으면 효소를 아무리 넣어도 불가능"
        )
    elif not d.get("use_equilibrium_limit", True):
        st.caption("평형 한계 대수식 미사용(약가역·결핍 0 아님): 정방향 해석만으로 탐색")
    p, _ = D.stage_file(run, "5_design", "pipette.csv")
    if p:
        st.markdown("**피펫 표**")
        table(D.read_table(p), "pipette")
        st.download_button("피펫 표 내려받기(CSV)", p.read_bytes(), file_name=f"pipette_{run.name}.csv", key="dl_pip")
    figures(run, "5_design")


RENDER = {"1_verdict": tab_verdict, "2_data": tab_data, "3_fit": tab_fit, "4_sim": tab_sim, "5_design": tab_design}


# ------------------------------------------------------------------------------------------------ page
def main() -> None:
    proj, run = sidebar()
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)
    if proj is None:
        return
    if run is None:
        st.info("프로젝트를 고르고 [새 실행]을 누르세요 (같은 엔진: enzopt run projects/<이름>)")
        return
    m = run.manifest
    st.header(f"{proj.name} · {run.name}")
    git = (m.get("git") or {}).get("commit") or "-"
    st.caption(
        f"상태 {D.STATE_LABEL.get(run.state, run.state)} · {'짧은 모드' if m.get('mode') == 'quick' else '전체 모드'}"
        f" · 시작 {m.get('created', '-')} · enzopt {m.get('enzopt_version', '-')} · 커밋 {git[:8]}"
    )
    if run.old_format:
        st.warning("옛 형식의 실행 폴더: 일부 결과를 읽지 못할 수 있음")
    status_panel(run)
    tabs = st.tabs([STAGE_TITLES[s] for s in STAGES])
    for tab, s in zip(tabs, STAGES, strict=True):
        with tab:
            banner(run)
            stt = run.stage_status(s)
            if stt in ("cached",):
                st.caption(f"이전 실행 결과 재사용({run.stages.get(s, {}).get('cached_from')})")
            try:
                RENDER[s](run)
            except Exception as e:  # noqa: BLE001 — one broken file must not break the whole page
                st.caption(f"결과를 표시하지 못함: {type(e).__name__}: {e}")


main()
