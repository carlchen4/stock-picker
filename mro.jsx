import { useState, useEffect, useRef, useCallback } from "react";

/* ═══════════════════════════════════════════════
   MRO Portfolio — Databricks Dashboard Style
   Focus: Data Engineering · Migration · Agentic AI
   Clean analytical aesthetic, no terminal chrome
═══════════════════════════════════════════════ */

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0f1117;
    --surface: #161b27;
    --card:    #1c2233;
    --card2:   #212840;
    --border:  rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.12);
    --text:    #e2e8f5;
    --muted:   #6b7a99;
    --muted2:  #8e9dc0;
    --blue:    #4c9ef7;
    --blue2:   #1a6fc4;
    --teal:    #26c6a6;
    --teal2:   #0d8c72;
    --amber:   #f5a623;
    --amber2:  #b87310;
    --purple:  #9b72f7;
    --purple2: #6742c9;
    --rose:    #f06292;
    --green:   #4caf7d;
    --green2:  #2d7a52;
    --font:    'DM Sans', sans-serif;
    --mono:    'DM Mono', monospace;
    --radius:  8px;
    --radius2: 12px;
  }

  body, .app {
    background: var(--bg); color: var(--text);
    font-family: var(--font);
    height: 100vh; overflow: hidden;
    display: flex; flex-direction: column;
  }

  /* ── TOP BAR ── */
  .topbar {
    height: 48px; flex-shrink: 0;
    display: flex; align-items: center; gap: 0;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .topbar-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 0 20px; width: 240px;
    border-right: 1px solid var(--border);
    height: 100%;
  }
  .brand-icon {
    width: 22px; height: 22px;
    background: linear-gradient(135deg, var(--blue) 0%, var(--purple) 100%);
    border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #fff;
  }
  .brand-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .brand-sub  { font-size: 10px; color: var(--muted); margin-top: 1px; }
  .topbar-center {
    flex: 1; display: flex; align-items: center; padding: 0 24px; gap: 24px;
  }
  .topbar-tab {
    font-size: 12px; font-weight: 500; color: var(--muted2);
    padding: 6px 12px; border-radius: var(--radius);
    cursor: pointer; transition: all .15s; border: none; background: none;
    display: flex; align-items: center; gap: 6px;
  }
  .topbar-tab:hover { color: var(--text); background: rgba(255,255,255,.04); }
  .topbar-tab.active { color: var(--blue); background: rgba(76,158,247,.1); }
  .topbar-right {
    display: flex; align-items: center; gap: 12px; padding: 0 20px;
    border-left: 1px solid var(--border); height: 100%;
  }
  .status-pill {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; color: var(--teal); font-weight: 500;
    background: rgba(38,198,166,.1); border: 1px solid rgba(38,198,166,.2);
    padding: 3px 10px; border-radius: 20px;
  }
  .status-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--teal);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── LAYOUT ── */
  .layout { display: flex; flex: 1; overflow: hidden; }

  /* ── SIDEBAR ── */
  .sidebar {
    width: 240px; flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 16px 0;
    overflow-y: auto;
  }
  .nav-section {
    font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
    color: var(--muted); text-transform: uppercase;
    padding: 8px 20px 4px;
  }
  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 20px; cursor: pointer;
    font-size: 12px; font-weight: 500; color: var(--muted2);
    transition: all .15s; border: none; background: none;
    text-align: left; width: 100%;
    border-left: 2px solid transparent;
  }
  .nav-item:hover { color: var(--text); background: rgba(255,255,255,.03); }
  .nav-item.active {
    color: var(--blue); border-left-color: var(--blue);
    background: rgba(76,158,247,.06);
  }
  .nav-icon { font-size: 14px; width: 16px; text-align: center; flex-shrink: 0; }
  .nav-badge {
    margin-left: auto; font-size: 9px; font-family: var(--mono);
    padding: 2px 7px; border-radius: 4px; font-weight: 500;
  }
  .nb-blue   { background: rgba(76,158,247,.15); color: var(--blue); }
  .nb-teal   { background: rgba(38,198,166,.15); color: var(--teal); }
  .nb-purple { background: rgba(155,114,247,.15); color: var(--purple); }
  .nb-amber  { background: rgba(245,166,35,.15); color: var(--amber); }
  .nav-divider { height: 1px; background: var(--border); margin: 10px 0; }

  /* ── CONTENT ── */
  .content {
    flex: 1; overflow-y: auto; overflow-x: hidden;
    padding: 24px;
    scrollbar-width: thin; scrollbar-color: var(--border2) transparent;
  }
  .content::-webkit-scrollbar { width: 4px; }
  .content::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* ── PAGE HEADER ── */
  .pg-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 20px;
  }
  .pg-title   { font-size: 18px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
  .pg-sub     { font-size: 13px; color: var(--muted2); line-height: 1.5; }
  .pg-actions { display: flex; gap: 8px; align-items: center; }

  /* ── BUTTON ── */
  .btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 14px; border-radius: var(--radius); cursor: pointer;
    font-family: var(--font); font-size: 12px; font-weight: 500;
    border: 1px solid; transition: all .15s;
  }
  .btn-primary {
    background: var(--blue2); color: #fff;
    border-color: var(--blue2);
  }
  .btn-primary:hover:not(:disabled) { background: var(--blue); border-color: var(--blue); }
  .btn-outline {
    background: transparent; color: var(--muted2); border-color: var(--border2);
  }
  .btn-outline:hover { color: var(--text); border-color: rgba(255,255,255,.25); }
  .btn:disabled { opacity: .4; cursor: default; }

  /* ── BADGE ── */
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-weight: 600; letter-spacing: .4px;
    padding: 3px 8px; border-radius: 4px;
  }
  .badge-teal   { background: rgba(38,198,166,.12); color: var(--teal);   border: 1px solid rgba(38,198,166,.25); }
  .badge-blue   { background: rgba(76,158,247,.12); color: var(--blue);   border: 1px solid rgba(76,158,247,.25); }
  .badge-amber  { background: rgba(245,166,35,.12); color: var(--amber);  border: 1px solid rgba(245,166,35,.25); }
  .badge-purple { background: rgba(155,114,247,.12);color: var(--purple); border: 1px solid rgba(155,114,247,.25); }
  .badge-rose   { background: rgba(240,98,146,.12); color: var(--rose);   border: 1px solid rgba(240,98,146,.25); }
  .badge-green  { background: rgba(76,175,125,.12); color: var(--green);  border: 1px solid rgba(76,175,125,.25); }

  /* ── METRIC CARD ── */
  .metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    margin-bottom: 20px;
  }
  .metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius2); padding: 16px 18px;
  }
  .metric-label { font-size: 11px; font-weight: 500; color: var(--muted); letter-spacing: .3px; margin-bottom: 8px; }
  .metric-value { font-size: 28px; font-weight: 600; line-height: 1; margin-bottom: 4px; }
  .metric-sub   { font-size: 11px; color: var(--muted2); }

  /* ── PANEL ── */
  .panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius2); overflow: hidden; margin-bottom: 16px;
  }
  .panel-head {
    display: flex; align-items: center; gap: 8px;
    padding: 12px 18px; border-bottom: 1px solid var(--border);
    background: rgba(255,255,255,.015);
  }
  .panel-title { font-size: 12px; font-weight: 600; color: var(--text); }
  .panel-sub   { font-size: 11px; color: var(--muted); margin-left: auto; }
  .panel-body  { padding: 16px 18px; }

  /* ── GRID ── */
  .g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .g3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }

  /* ── PIPELINE VISUALIZATION ── */
  .pipeline {
    display: flex; align-items: stretch; gap: 0;
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius2); overflow: hidden; margin-bottom: 16px;
  }
  .pipe-stage {
    flex: 1; padding: 16px 12px; position: relative;
    border-right: 1px solid var(--border);
    transition: background .3s;
    display: flex; flex-direction: column; align-items: center; gap: 8px;
  }
  .pipe-stage:last-child { border-right: none; }
  .pipe-stage.active { background: rgba(76,158,247,.06); }
  .pipe-stage.done   { background: rgba(38,198,166,.03); }
  .ps-icon { font-size: 22px; }
  .ps-label {
    font-family: var(--mono); font-size: 9px; font-weight: 500;
    color: var(--muted2); text-align: center; letter-spacing: .5px;
    text-transform: uppercase; line-height: 1.5;
  }
  .pipe-stage.active .ps-label { color: var(--blue); }
  .pipe-stage.done .ps-label   { color: var(--teal); }
  .ps-count {
    font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--blue);
  }
  .pipe-stage.done .ps-count { color: var(--teal); }
  .ps-arrow {
    position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
    z-index: 2; font-size: 10px; color: var(--muted); font-family: var(--mono);
  }
  .ps-progress {
    position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--border);
  }
  .ps-progress-fill { height: 100%; background: var(--blue); width: 0%; transition: width 80ms linear; }
  .pipe-stage.done .ps-progress-fill { background: var(--teal); width: 100%; }

  /* ── AGENT FLOW ── */
  .agent-flow { display: flex; flex-direction: column; gap: 0; }
  .agent-item {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 0; border-bottom: 1px solid var(--border);
    transition: all .3s;
  }
  .agent-item:last-child { border-bottom: none; }
  .agent-orb {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    transition: all .3s;
  }
  .ao-idle    { background: rgba(255,255,255,.04); }
  .ao-running { background: rgba(245,166,35,.12); border: 1px solid rgba(245,166,35,.3); }
  .ao-done    { background: rgba(38,198,166,.1);  border: 1px solid rgba(38,198,166,.25); }
  .ao-err     { background: rgba(240,98,146,.1);  border: 1px solid rgba(240,98,146,.3); }
  .agent-info { flex: 1; min-width: 0; }
  .agent-name { font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 2px; }
  .agent-desc { font-size: 11px; color: var(--muted2); }
  .agent-right { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .agent-bar  { width: 80px; height: 3px; background: var(--border2); border-radius: 2px; overflow: hidden; }
  .agent-bar-fill { height: 100%; background: var(--amber); border-radius: 2px; transition: width .12s linear; }
  .ai-done .agent-bar-fill  { background: var(--teal); }
  .ai-running .agent-bar-fill { background: var(--amber); }
  .agent-status { font-size: 10px; font-family: var(--mono); font-weight: 500; min-width: 52px; text-align: right; }
  .s-idle    { color: var(--muted); }
  .s-running { color: var(--amber); }
  .s-done    { color: var(--teal); }
  .s-err     { color: var(--rose); }

  /* connector between agents */
  .agent-connector {
    display: flex; align-items: center; gap: 0;
    padding-left: 18px; height: 0; overflow: visible;
    position: relative;
  }

  /* ── LOG ── */
  .log-box {
    background: #0a0d14; border: 1px solid var(--border);
    border-radius: var(--radius); font-family: var(--mono);
    overflow: hidden;
  }
  .log-head {
    display: flex; align-items: center; padding: 8px 14px; gap: 8px;
    background: #0d1018; border-bottom: 1px solid var(--border);
  }
  .log-dot { width: 8px; height: 8px; border-radius: 50%; }
  .log-title { font-size: 10px; color: var(--muted); letter-spacing: 1px; flex: 1; text-align: center; }
  .log-body {
    padding: 12px 14px; font-size: 10px; line-height: 2;
    max-height: 200px; overflow-y: auto; color: #6ebc98;
  }
  .log-body::-webkit-scrollbar { width: 3px; }
  .log-body::-webkit-scrollbar-thumb { background: var(--border2); }
  .lc-muted  { color: var(--muted); }
  .lc-blue   { color: var(--blue); }
  .lc-teal   { color: var(--teal); }
  .lc-amber  { color: var(--amber); }
  .lc-rose   { color: var(--rose); }
  .lc-purple { color: var(--purple); }
  .lc-cursor::after { content: '▌'; animation: blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* ── VALIDATION TABLE ── */
  .vtable { width: 100%; border-collapse: collapse; }
  .vtable-head th {
    font-size: 10px; font-weight: 600; color: var(--muted); letter-spacing: .5px;
    text-transform: uppercase; padding: 8px 0; border-bottom: 1px solid var(--border2);
    text-align: left;
  }
  .vtable-head th:not(:first-child) { text-align: right; }
  .vrow td {
    font-family: var(--mono); font-size: 10px; padding: 7px 0;
    border-bottom: 1px solid var(--border); animation: rowIn .2s ease;
  }
  @keyframes rowIn { from{opacity:0;transform:translateX(-4px)} to{opacity:1;transform:none} }
  .vrow:last-child td { border-bottom: none; }
  .vr-name  { color: var(--muted2); }
  .vr-num   { text-align: right; }
  .vr-sas   { color: var(--blue); }
  .vr-py    { color: var(--teal); }
  .vr-delta { text-align: right; font-size: 9px; color: var(--muted); }
  .vr-pass  { text-align: right; }
  .chip-pass { background: rgba(38,198,166,.1); color: var(--teal); border: 1px solid rgba(38,198,166,.2); padding: 1px 7px; border-radius: 3px; font-size: 9px; }
  .chip-fail { background: rgba(240,98,146,.1); color: var(--rose); border: 1px solid rgba(240,98,146,.2); padding: 1px 7px; border-radius: 3px; font-size: 9px; }

  /* ── PROGRESS BAR ── */
  .pbar { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .pbar-label { font-size: 11px; color: var(--muted2); width: 180px; flex-shrink: 0; }
  .pbar-track { flex: 1; height: 5px; background: var(--border2); border-radius: 3px; overflow: hidden; }
  .pbar-fill  { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(.25,.8,.25,1); }
  .pbar-val   { font-family: var(--mono); font-size: 10px; color: var(--muted2); width: 38px; text-align: right; }

  /* ── WORKSTREAM LIST ── */
  .ws-item {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 16px; background: var(--card2);
    border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 8px;
  }
  .ws-icon  { font-size: 16px; width: 20px; text-align: center; flex-shrink: 0; }
  .ws-title { font-size: 12px; font-weight: 600; color: var(--text); width: 150px; flex-shrink: 0; }
  .ws-desc  { font-size: 11px; color: var(--muted2); flex: 1; line-height: 1.5; }
  .ws-tags  { display: flex; gap: 5px; flex-shrink: 0; }

  /* ── TIMELINE ── */
  .timeline-step {
    display: flex; gap: 12px; padding-bottom: 20px; position: relative;
  }
  .timeline-step:last-child { padding-bottom: 0; }
  .tl-line {
    position: absolute; left: 15px; top: 30px; bottom: 0;
    width: 1px; background: var(--border2);
  }
  .timeline-step:last-child .tl-line { display: none; }
  .tl-orb {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0; position: relative; z-index: 1;
  }
  .tl-content { flex: 1; padding-top: 4px; }
  .tl-title { font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 3px; }
  .tl-desc  { font-size: 11px; color: var(--muted2); line-height: 1.5; }

  /* ── NARRATOR ── */
  .narrator {
    height: 52px; flex-shrink: 0;
    background: var(--surface); border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 0;
    position: relative;
  }
  .nar-back {
    width: 52px; height: 100%; display: flex; align-items: center; justify-content: center;
    background: none; border: none; border-right: 1px solid var(--border);
    color: var(--muted); cursor: pointer; font-size: 16px; transition: color .15s;
  }
  .nar-back:hover:not(:disabled) { color: var(--muted2); }
  .nar-back:disabled { opacity: .25; cursor: default; }
  .nar-body { flex: 1; padding: 0 20px; overflow: hidden; }
  .nar-label { font-size: 9px; font-weight: 600; letter-spacing: 1px; color: var(--muted); text-transform: uppercase; margin-bottom: 2px; }
  .nar-msg { font-size: 12px; color: var(--muted2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .nar-msg em { color: var(--blue); font-style: normal; font-weight: 600; }
  .nar-right { display: flex; align-items: center; gap: 0; border-left: 1px solid var(--border); flex-shrink: 0; height: 100%; }
  .nar-count {
    width: 52px; display: flex; flex-direction: column; align-items: center; justify-content: center;
    border-right: 1px solid var(--border); height: 100%;
    font-family: var(--mono); font-size: 9px; color: var(--muted);
  }
  .nar-count span { font-size: 14px; color: var(--blue); font-weight: 500; }
  .nar-cta { padding: 0 16px; }
  .nar-progress { position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: var(--border); }
  .nar-progress-fill { height: 100%; background: var(--blue); transition: width .4s ease; }

  /* ── DATA FLOW DIAGRAM ── */
  .flow-diagram { position: relative; }
  .flow-svg { width: 100%; }

  /* ── SLIDE ── */
  .slide { animation: slideIn .25s ease; }
  @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

  /* ── SAS vs PySpark diff ── */
  .diff-table { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .diff-col   { flex: 1; padding: 14px 18px; }
  .diff-header { font-size: 10px; font-weight: 600; letter-spacing: .5px; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
  .diff-row   { font-size: 11px; line-height: 2.2; display: flex; align-items: center; gap: 6px; }
  .diff-icon  { width: 12px; text-align: center; }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* code chip */
  .code { font-family: var(--mono); font-size: 10px; background: rgba(255,255,255,.05); padding: 1px 6px; border-radius: 3px; color: var(--muted2); }
`;

/* ── SLIDES CONFIG ── */
const SLIDES = [
  { id:"overview",   icon:"⊞",  label:"Project Overview",   section:"OVERVIEW",       badge:null,     badgeCls:null },
  { id:"migration",  icon:"⇢",  label:"SAS → PySpark",      section:"DATA ENGINEERING",badge:"LIVE",  badgeCls:"nb-blue" },
  { id:"assets",     icon:"⊕",  label:"Data Assets",        section:"DATA ENGINEERING",badge:null,     badgeCls:null },
  { id:"agentic",    icon:"◎",  label:"Agentic AI Flow",    section:"DATA ENGINEERING",badge:"6 AGENTS",badgeCls:"nb-purple" },
  { id:"validation", icon:"✓",  label:"Validation Suite",   section:"QUALITY",        badge:"SIM",    badgeCls:"nb-teal" },
  { id:"devops",     icon:"⌥",  label:"DevOps & Governance",section:"PLATFORM",       badge:null,     badgeCls:null },
  { id:"impact",     icon:"▣",  label:"Impact Summary",     section:"SUMMARY",        badge:null,     badgeCls:null },
];

const STORY = [
  { msg:"MRO modernization at a glance — <em>SAS→PySpark migration</em>, 6-agent AI pipeline, zero disruption.", cta:"Data Engineering →" },
  { msg:"The <em>SAS→PySpark decoupled migration</em> — watch the pipeline execute live across all stages.",     cta:"View Data Assets →" },
  { msg:"<em>847K rows</em> across 3 datasets migrated — full schema, type coercion, NAS delivery parity.",      cta:"Agentic AI Flow →" },
  { msg:"A <em>6-agent orchestration</em> handles audit, translation, review, validation and documentation.",     cta:"Validation Suite →" },
  { msg:"Full-stack validation: row counts, distributions, business rules. <em>Statistical parity confirmed.</em>",cta:"DevOps →" },
  { msg:"<em>GitHub + Jira + Confluence</em> replaced NAS versioning. CIBC AI bootstrapped the whole setup.",    cta:"Impact →" },
  { msg:"End-to-end ownership across all workstreams. <em>Deadline met. 100% format parity. Zero disruptions.</em>",cta:null },
];

const COMPS = ["OverviewSlide","MigrationSlide","AssetsSlide","AgenticSlide","ValidationSlide","DevOpsSlide","ImpactSlide"];
const N = SLIDES.length;

/* ═══════════════ OVERVIEW ═══════════════ */
function OverviewSlide() {
  const metrics = [
    { val:"5",    lbl:"Workstreams Delivered", sub:"on time",    color:"var(--teal)" },
    { val:"100%", lbl:"NAS Format Parity",     sub:"zero drift",  color:"var(--blue)" },
    { val:"0",    lbl:"Consumer Disruptions",  sub:"throughout",  color:"var(--teal)" },
    { val:"6",    lbl:"AI Agents Deployed",    sub:"in pipeline", color:"var(--purple)" },
  ];
  const ws = [
    { icon:"⇢", color:"var(--blue)",   title:"SAS→PySpark Migration",  desc:"Decoupled from upstream. CSV bridge kept NAS consumers live throughout. CIBC EDGE deadline met.", tags:[["Data Eng","badge-blue"],["PySpark","badge-blue"]] },
    { icon:"◎", color:"var(--purple)", title:"Agentic AI Pipeline",     desc:"6-agent chain: audit → translate → review → validate → document. Human-in-the-loop at critical gates.", tags:[["AI","badge-purple"],["6 Agents","badge-purple"]] },
    { icon:"✓", color:"var(--teal)",   title:"Data Validation Suite",   desc:"Full-stack: row count, schema, distributions, business rules. Caught silent corruption during migration.", tags:[["QA","badge-teal"],["Stats","badge-teal"]] },
    { icon:"⚡", color:"var(--amber)",  title:"Power Automate Flows",    desc:"M365 bridge: email routing, concat automation, Teams alerts during migration window. Zero new infra.", tags:[["M365","badge-amber"],["Automate","badge-amber"]] },
    { icon:"⌥", color:"var(--rose)",   title:"DevOps & Governance",     desc:"GitHub replaced NAS file versioning. Jira + Confluence via CIBC AI. Full PR audit trail.", tags:[["GitHub","badge-rose"],["DevOps","badge-rose"]] },
  ];
  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">MRO · CIBC EDGE 2025</div>
          <div className="pg-sub">Mortgage Renewal Optimization — end-to-end data modernization</div>
        </div>
        <div style={{ textAlign:"right", fontSize:11, color:"var(--muted2)", lineHeight:1.8 }}>
          <div style={{ fontWeight:600, color:"var(--text)", fontSize:12 }}>Carl — Data Engineer & Analyst</div>
          <div>Azure Databricks · PySpark · Power Automate · GitHub</div>
        </div>
      </div>

      <div className="metric-grid">
        {metrics.map(m=>(
          <div className="metric-card" key={m.lbl}>
            <div className="metric-label">{m.lbl}</div>
            <div className="metric-value" style={{ color:m.color }}>{m.val}</div>
            <div className="metric-sub">{m.sub}</div>
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-head">
          <div className="panel-title">Project Workstreams</div>
          <div className="panel-sub">All 5 delivered within CIBC EDGE deadline</div>
        </div>
        <div className="panel-body" style={{ padding:"12px 18px" }}>
          {ws.map((w,i)=>(
            <div className="ws-item" key={i} style={{ borderLeft:`3px solid ${w.color}`, borderRadius:"0 8px 8px 0" }}>
              <div className="ws-icon" style={{ color:w.color }}>{w.icon}</div>
              <div className="ws-title">{w.title}</div>
              <div className="ws-desc">{w.desc}</div>
              <div className="ws-tags">
                {w.tags.map(([t,c])=><span key={t} className={`badge ${c}`}>{t}</span>)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ MIGRATION ═══════════════ */
function MigrationSlide() {
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState(-1);
  const [counts, setCounts] = useState([0,0,0,0,0]);
  const [progress, setProgress] = useState([0,0,0,0,0]);
  const [logs, setLogs] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef(null);
  const logRef = useRef(null);

  const STAGES = [
    { icon:"🗄️", label:"SAS LEGACY\nSCRIPTS",    color:"var(--muted2)", final:847320 },
    { icon:"📤", label:"CSV BRIDGE\nPROC EXPORT", color:"var(--amber)",  final:847320 },
    { icon:"⚙️", label:"PYSPARK\nDATABRICKS",    color:"var(--blue)",   final:847320 },
    { icon:"📁", label:"NAS\nDELIVERY",           color:"var(--teal)",   final:847320 },
    { icon:"✅", label:"CONSUMERS\nVALIDATED",     color:"var(--green)",  final:3 },
  ];

  const LOGS = [
    [0,  "lc-muted",  "[00:00] SparkSession initialized — CIBC EDGE cluster (8 workers)"],
    [0,  "lc-amber",  "[00:01] Scanning /nas/mro/ — found 3 SAS datasets"],
    [0,  "lc-teal",   "[00:02] renewal_scored.sas7bdat · renewal_flags.sas7bdat · mro_summary.sas7bdat"],
    [400,"lc-muted",  "[00:04] PROC EXPORT batch — csv mode, replace=true, delimiter=comma"],
    [400,"lc-blue",   "[00:05] renewal_scored.csv → 512,847 rows exported OK"],
    [400,"lc-blue",   "[00:05] renewal_flags.csv  → 201,344 rows exported OK"],
    [400,"lc-blue",   "[00:06] mro_summary.csv    → 133,129 rows exported OK"],
    [800,"lc-muted",  "[00:08] PySpark read_csv — applying explicit schema (DecimalType 18,6)"],
    [800,"lc-blue",   "[00:09] Schema match: ✓ All 24 columns aligned"],
    [800,"lc-teal",   "[00:10] Null coalesce: SAS (.) → Python None — 0 mismatches"],
    [800,"lc-blue",   "[00:11] cast() enforcement complete. Type parity confirmed."],
    [1200,"lc-muted", "[00:13] Writing to NAS /mnt/nas/mro/ — overwrite mode"],
    [1200,"lc-teal",  "[00:14] renewal_scored written — 847,320 rows"],
    [1200,"lc-teal",  "[00:15] Format parity check: ✓ PASS — all decimals match SAS precision"],
    [1600,"lc-amber", "[00:16] Notifying downstream consumers (3 teams)"],
    [1600,"lc-teal",  "[00:17] ✓ Pipeline complete — 0 errors, 0 consumer disruptions"],
  ];

  const addLog = useCallback((l) => {
    setLogs(p=>[...p,l]);
    setTimeout(()=>{ if(logRef.current) logRef.current.scrollTop=99999; },30);
  },[]);

  const reset = () => {
    setRunning(false); setPhase(-1);
    setCounts([0,0,0,0,0]); setLogs([]); setElapsed(0); setProgress([0,0,0,0,0]);
    if(timerRef.current) clearInterval(timerRef.current);
  };

  const run = () => {
    reset();
    setTimeout(()=>{
      setRunning(true);
      let t=0;
      timerRef.current = setInterval(()=>{ t++; setElapsed(t); },100);
      const PD = 500;
      [0,1,2,3,4].forEach((p,i)=>{
        setTimeout(()=>{
          setPhase(p);
          const target = STAGES[p].final;
          let c=0;
          const ci = setInterval(()=>{
            c = Math.min(c + Math.ceil(target/25), target);
            const pct = Math.round((c/target)*100);
            setCounts(pr=>{ const n=[...pr]; n[p]=c; return n; });
            setProgress(pr=>{ const n=[...pr]; n[p]=pct; return n; });
            if(c>=target) clearInterval(ci);
          },30);
        }, i*PD);
        LOGS.filter(([d])=>d===i*400).forEach(([,cls,msg],j)=>{
          setTimeout(()=>addLog({cls,msg}), i*PD+j*80);
        });
      });
      setTimeout(()=>{ setPhase(5); clearInterval(timerRef.current); }, 5*PD+200);
    },50);
  };

  useEffect(()=>()=>{ if(timerRef.current) clearInterval(timerRef.current); },[]);
  const done = phase===5;

  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">SAS → PySpark Migration Pipeline</div>
          <div className="pg-sub">Live simulation — 847K rows across 3 MRO datasets. Decoupled from upstream; CSV bridge kept NAS consumers live.</div>
        </div>
        <div className="pg-actions">
          {running && !done && (
            <span style={{ fontSize:11, color:"var(--amber)", fontFamily:"var(--mono)", fontWeight:500 }}>
              ● RUNNING {(elapsed/10).toFixed(1)}s
            </span>
          )}
          {done && <span className="badge badge-teal">✓ Complete</span>}
          <button className="btn btn-primary" onClick={run} disabled={running&&!done}>
            {done ? "↺ Re-run" : "▶ Run Pipeline"}
          </button>
          {(running||done) && <button className="btn btn-outline" onClick={reset}>Reset</button>}
        </div>
      </div>

      {/* pipeline bar */}
      <div className="pipeline">
        {STAGES.map((s,i)=>(
          <div key={i} className={`pipe-stage${phase===i?" active":phase>i||done?" done":""}`} style={{ position:"relative" }}>
            {i<STAGES.length-1 && (
              <span className="ps-arrow" style={{ color: phase>i||done ? "var(--teal)" : "var(--border2)" }}>›</span>
            )}
            <div className="ps-icon">{s.icon}</div>
            <div className="ps-label" style={{ whiteSpace:"pre-line" }}>{s.label}</div>
            {(phase>=i||done) && (
              <div className="ps-count">
                {i===4 ? counts[i] : counts[i].toLocaleString()}
                {i!==4 && <span style={{ fontSize:8, color:"var(--muted)", marginLeft:2 }}>rows</span>}
              </div>
            )}
            <div className="ps-progress">
              <div className="ps-progress-fill" style={{ width:(phase===i?progress[i]:phase>i||done?100:0)+"%" }}/>
            </div>
          </div>
        ))}
      </div>

      {/* metrics row */}
      {(running||done) && (
        <div className="g3" style={{ marginBottom:14 }}>
          {[
            { lbl:"Throughput",    val:done?"847K rows / run":"processing…", c:"var(--teal)" },
            { lbl:"Schema drift",  val:"0 columns changed",                   c:"var(--teal)" },
            { lbl:"Format parity", val:done?"✓ PASS":"checking…",             c:done?"var(--teal)":"var(--amber)" },
          ].map(m=>(
            <div className="panel" key={m.lbl} style={{ marginBottom:0 }}>
              <div className="panel-body" style={{ padding:"12px 16px" }}>
                <div style={{ fontSize:10, color:"var(--muted)", marginBottom:4, fontWeight:600, letterSpacing:.5, textTransform:"uppercase" }}>{m.lbl}</div>
                <div style={{ fontFamily:"var(--mono)", fontSize:14, fontWeight:500, color:m.c }}>{m.val}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* SAS vs PySpark challenge panel */}
      <div className="g2">
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head"><div className="panel-title">Key Migration Challenges</div></div>
          <div className="panel-body">
            {[
              ["SAS dot-null → Python None",     "Explicit coalesce layer; no null leakage"],
              ["DecimalType precision (18,6)",   "cast() enforcement; verified vs SAS output"],
              ["PROC EXPORT as CSV bridge",       "Decoupled from Spark rewrite; consumers live"],
              ["Schema drift detection",          "column-level diff on every run; 0 drift"],
              ["NAS delivery format parity",      "Byte-level compare; 100% match confirmed"],
            ].map(([k,v])=>(
              <div key={k} style={{ fontSize:11, lineHeight:1.4, marginBottom:10, borderBottom:"1px solid var(--border)", paddingBottom:8 }}>
                <div style={{ fontFamily:"var(--mono)", color:"var(--amber)", fontSize:10, marginBottom:2 }}>⚠ {k}</div>
                <div style={{ color:"var(--teal)", fontSize:11 }}>✓ {v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* log */}
        <div className="log-box">
          <div className="log-head">
            <div className="log-dot" style={{ background:"#ff5f57" }}/>
            <div className="log-dot" style={{ background:"#febc2e" }}/>
            <div className="log-dot" style={{ background:"#28c840" }}/>
            <div className="log-title">spark-submit · mro_migration.py · CIBC EDGE</div>
          </div>
          <div className="log-body" ref={logRef}>
            {logs.length===0 && <span className="lc-muted">$ spark-submit mro_migration.py --env prod<span className="lc-cursor"/></span>}
            {logs.map((l,i)=><div key={i} className={l.cls}>{l.msg}</div>)}
            {running&&!done&&logs.length>0&&<span className="lc-cursor"/>}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ DATA ASSETS ═══════════════ */
function AssetsSlide() {
  const datasets = [
    { name:"renewal_scored", rows:"512,847", cols:24, size:"287 MB", format:"CSV / Parquet", desc:"Scored renewal propensity per mortgage account. Primary output consumed by 3 downstream teams.", tags:["badge-blue","Primary","badge-teal","Scored"] },
    { name:"renewal_flags",  rows:"201,344", cols:12, size:"95 MB",  format:"CSV",           desc:"Binary renewal intent flags derived from scored output. Downstream: campaign targeting.", tags:["badge-amber","Derived","badge-teal","Flags"] },
    { name:"mro_summary",    rows:"133,129", cols:18, size:"61 MB",  format:"CSV",           desc:"Aggregated MRO portfolio summary. Used for exec reporting and BI dashboards.", tags:["badge-purple","Summary","badge-blue","BI"] },
  ];
  const steps = [
    { color:"var(--blue)",   icon:"🗄️", title:"SAS Source Layer",       desc:"PROC-based scripts on NAS. Eyeball-diffed. No version control. No CI/CD." },
    { color:"var(--amber)",  icon:"📤", title:"CSV Extraction Bridge",   desc:"PROC EXPORT to CSV. Decoupled migration from live NAS delivery. Consumers unaffected." },
    { color:"var(--purple)", icon:"◎",  title:"Agentic AI Preprocessing", desc:"6-agent pipeline audits, translates, and validates SAS logic before PySpark rewrite." },
    { color:"var(--blue2)",  icon:"⚙️", title:"PySpark Databricks",      desc:"Explicit schema, cast(), coalesce null handling. Runs on CIBC EDGE cluster." },
    { color:"var(--teal)",   icon:"📁", title:"NAS Delivery (Parity)",   desc:"Writes back to /mnt/nas/mro/. Format parity verified byte-level vs SAS baseline." },
    { color:"var(--green)",  icon:"✅", title:"Downstream Consumers",    desc:"3 consumer teams validated. 0 disruptions throughout migration window." },
  ];
  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">MRO Data Assets</div>
          <div className="pg-sub">3 core datasets migrated — 847K rows total. Schema preserved, delivery format matched byte-for-byte.</div>
        </div>
        <span className="badge badge-teal">✓ 100% Format Parity</span>
      </div>

      <div style={{ display:"flex", flexDirection:"column", gap:12, marginBottom:18 }}>
        {datasets.map(d=>(
          <div className="panel" key={d.name} style={{ marginBottom:0 }}>
            <div className="panel-body" style={{ padding:"14px 18px" }}>
              <div style={{ display:"flex", alignItems:"flex-start", gap:14 }}>
                <div style={{ flex:1 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:5 }}>
                    <span style={{ fontFamily:"var(--mono)", fontSize:13, fontWeight:500, color:"var(--blue)" }}>{d.name}</span>
                    {d.tags.filter((_,i)=>i%2===1).map((t,j)=>(
                      <span key={j} className={`badge ${d.tags[j*2]}`}>{t}</span>
                    ))}
                  </div>
                  <div style={{ fontSize:11, color:"var(--muted2)", lineHeight:1.5 }}>{d.desc}</div>
                </div>
                <div style={{ display:"flex", gap:20, flexShrink:0, alignItems:"center" }}>
                  {[[d.rows,"rows"],[d.cols,"cols"],[d.size,"size"],[d.format,"format"]].map(([v,l])=>(
                    <div key={l} style={{ textAlign:"center" }}>
                      <div style={{ fontFamily:"var(--mono)", fontSize:13, fontWeight:500, color:"var(--text)" }}>{v}</div>
                      <div style={{ fontSize:9, color:"var(--muted)", letterSpacing:.5, textTransform:"uppercase" }}>{l}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-head"><div className="panel-title">Data Migration Lineage</div><div className="panel-sub">end-to-end asset journey</div></div>
        <div className="panel-body">
          <div style={{ display:"flex", alignItems:"center", gap:0, overflowX:"auto", paddingBottom:4 }}>
            {steps.map((s,i)=>(
              <div key={i} style={{ display:"flex", alignItems:"center", flexShrink:0 }}>
                <div style={{ padding:"10px 14px", background:"var(--card2)", border:`1px solid var(--border)`,
                  borderLeft:`3px solid ${s.color}`, borderRadius:"0 8px 8px 0",
                  minWidth:130, maxWidth:150 }}>
                  <div style={{ fontSize:14, marginBottom:4 }}>{s.icon}</div>
                  <div style={{ fontSize:11, fontWeight:600, color:"var(--text)", marginBottom:3 }}>{s.title}</div>
                  <div style={{ fontSize:10, color:"var(--muted2)", lineHeight:1.4 }}>{s.desc}</div>
                </div>
                {i<steps.length-1 && <div style={{ color:"var(--border2)", fontSize:16, padding:"0 6px", flexShrink:0 }}>›</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ AGENTIC AI ═══════════════ */
function AgenticSlide() {
  const [phase, setPhase] = useState(-1);
  const [running, setRunning] = useState(false);
  const [bars, setBars] = useState([0,0,0,0,0,0]);
  const [logs, setLogs] = useState([]);
  const logRef = useRef(null);
  const timers = useRef([]);

  const AGENTS = [
    { icon:"🔍", name:"Audit Agent",      desc:"Parses SAS source — extracts PROC types, data steps, macro calls. Builds dependency graph.", color:"var(--blue)",   dur:2200 },
    { icon:"🔄", name:"Translate Agent",  desc:"Converts SAS syntax → PySpark. Handles PROC SQL, data step logic, merge/sort patterns.", color:"var(--purple)", dur:3500 },
    { icon:"👁️", name:"Review Agent",     desc:"Compares SAS original vs PySpark translation. Flags semantic drift, missing edge cases.", color:"var(--amber)",  dur:2800 },
    { icon:"⚙️", name:"Fix Agent",        desc:"Auto-patches flagged issues from Review. Applies decimal coercion, null handling corrections.", color:"var(--rose)",   dur:2000 },
    { icon:"✅", name:"Validate Agent",   desc:"Runs translated PySpark against SAS output. Row count, schema, distribution comparison.", color:"var(--teal)",   dur:3000 },
    { icon:"📝", name:"Document Agent",   desc:"Generates Confluence-ready docs: logic summary, business rules, test results, schema diff.", color:"var(--green)",  dur:1800 },
  ];

  const AGENT_LOGS = [
    [0,    "lc-muted",  "[Orchestrator] Starting 6-agent MRO pipeline..."],
    [0,    "lc-blue",   "[Audit]       Parsing renewal_scored.sas — 847 lines, 3 PROC blocks"],
    [800,  "lc-blue",   "[Audit]       Dependency graph built: 12 macro calls, 4 data steps"],
    [1400, "lc-purple", "[Translate]   Converting PROC SQL → pyspark.sql.functions"],
    [2200, "lc-purple", "[Translate]   Data step logic → DataFrame chain — 98% automated"],
    [3800, "lc-amber",  "[Review]      Diff scan: 1 semantic flag — DecimalType precision"],
    [4500, "lc-amber",  "[Review]      Null pattern mismatch detected in renewal_flags"],
    [5400, "lc-rose",   "[Fix]         Applying cast(DecimalType(18,6)) correction"],
    [5900, "lc-rose",   "[Fix]         Coalesce null patch applied — resubmitting"],
    [7000, "lc-teal",   "[Validate]    Running parity suite: row counts, p25/p50/p75/p95"],
    [8500, "lc-teal",   "[Validate]    All 15 checks PASS — 0 deviations"],
    [9400, "lc-purple", "[Document]    Generating Confluence page: MRO_Migration_renewal_scored"],
    [10200,"lc-purple", "[Document]    Jira ticket auto-created: MRO-247"],
    [10800,"lc-teal",   "[Orchestrator] Pipeline complete — 6 agents, 0 human interventions needed"],
  ];

  const addLog = useCallback((l) => {
    setLogs(p=>[...p,l]);
    setTimeout(()=>{ if(logRef.current) logRef.current.scrollTop=99999; },30);
  },[]);

  const reset = () => {
    setPhase(-1); setRunning(false); setBars([0,0,0,0,0,0]); setLogs([]);
    timers.current.forEach(t=>clearTimeout(t));
    timers.current=[];
  };

  const run = () => {
    reset();
    setTimeout(()=>{
      setRunning(true);
      let cumulative = 0;
      AGENTS.forEach((ag,i)=>{
        const start = cumulative;
        cumulative += ag.dur;

        const t1 = setTimeout(()=>{ setPhase(i); }, start);
        // animate bar
        const t2 = setTimeout(()=>{
          const dur = ag.dur;
          const steps = 40;
          let s=0;
          const ci = setInterval(()=>{
            s++; const pct = Math.round((s/steps)*100);
            setBars(pb=>{ const n=[...pb]; n[i]=pct; return n; });
            if(s>=steps){ clearInterval(ci); }
          }, dur/steps);
        }, start+50);

        timers.current.push(t1,t2);
      });

      AGENT_LOGS.forEach(([delay,cls,msg])=>{
        const t = setTimeout(()=>addLog({cls,msg}), delay+100);
        timers.current.push(t);
      });

      const total = cumulative+300;
      const tf = setTimeout(()=>{ setPhase(AGENTS.length); setRunning(false); }, total);
      timers.current.push(tf);
    },50);
  };

  useEffect(()=>()=>{ timers.current.forEach(t=>clearTimeout(t)); },[]);

  const getState = (i) => {
    if(phase<0 || phase===AGENTS.length) return phase===AGENTS.length?"done":"idle";
    if(i===phase) return "running";
    if(i<phase) return "done";
    return "idle";
  };

  const done = phase===AGENTS.length;

  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">Agentic AI Orchestration</div>
          <div className="pg-sub">6-agent pipeline: Audit → Translate → Review → Fix → Validate → Document. Each agent hands off context to the next.</div>
        </div>
        <div className="pg-actions">
          {running && <span style={{ fontSize:11, fontFamily:"var(--mono)", color:"var(--amber)", fontWeight:500 }}>● RUNNING</span>}
          {done && <span className="badge badge-teal">✓ Pipeline Done</span>}
          <button className="btn btn-primary" onClick={run} disabled={running&&!done}>
            {done?"↺ Re-run":"▶ Launch Pipeline"}
          </button>
          {(running||done)&&<button className="btn btn-outline" onClick={reset}>Reset</button>}
        </div>
      </div>

      <div className="g2">
        {/* Agent list */}
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head">
            <div className="panel-title">Agent Execution</div>
            <div className="panel-sub">{done?"6/6 complete":running?`${Math.min(phase+1,6)}/6 running`:"idle"}</div>
          </div>
          <div className="panel-body">
            <div className="agent-flow">
              {AGENTS.map((ag,i)=>{
                const st = getState(i);
                const orbCls = { idle:"ao-idle", running:"ao-running", done:"ao-done" }[st] || "ao-idle";
                const stCls  = { idle:"s-idle", running:"s-running", done:"s-done" }[st] || "s-idle";
                const barPct = st==="done"?100:st==="running"?bars[i]:0;
                return (
                  <div key={i} className={`agent-item ai-${st}`}>
                    <div className={`agent-orb ${orbCls}`} style={{ border: st==="idle"?"1px solid var(--border)":"" }}>
                      {ag.icon}
                    </div>
                    <div className="agent-info">
                      <div className="agent-name" style={{ color: st==="idle"?"var(--muted2)":"var(--text)" }}>{ag.name}</div>
                      <div className="agent-desc">{ag.desc.split("—")[0].trim()}</div>
                    </div>
                    <div className="agent-right">
                      <div className="agent-bar">
                        <div className="agent-bar-fill" style={{
                          width:barPct+"%",
                          background: st==="done"?"var(--teal)":st==="running"?"var(--amber)":"var(--border2)"
                        }}/>
                      </div>
                      <div className={`agent-status ${stCls}`}>
                        {st==="idle"?"WAITING":st==="running"?"ACTIVE":st==="done"?"DONE":""}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right column: log + how it works */}
        <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
          <div className="log-box" style={{ flex:1 }}>
            <div className="log-head">
              <div className="log-dot" style={{ background:"var(--blue)" }}/>
              <div className="log-dot" style={{ background:"var(--purple)" }}/>
              <div className="log-dot" style={{ background:"var(--teal)" }}/>
              <div className="log-title">agent-orchestrator · mro_pipeline.py</div>
            </div>
            <div className="log-body" ref={logRef} style={{ maxHeight:160 }}>
              {logs.length===0&&<span className="lc-muted">Waiting for pipeline trigger…<span className="lc-cursor"/></span>}
              {logs.map((l,i)=><div key={i} className={l.cls}>{l.msg}</div>)}
              {running&&logs.length>0&&<span className="lc-cursor"/>}
            </div>
          </div>

          <div className="panel" style={{ marginBottom:0 }}>
            <div className="panel-head"><div className="panel-title">How the Agents Share Context</div></div>
            <div className="panel-body">
              {[
                ["Audit → Translate",  "Dependency graph + PROC inventory passed as structured JSON"],
                ["Translate → Review", "PySpark draft + original SAS side-by-side for diff analysis"],
                ["Review → Fix",       "Flagged issues list with line refs; Fix patches in place"],
                ["Fix → Validate",     "Corrected PySpark submitted to parity suite automatically"],
                ["Validate → Document","Test report + schema diff → Confluence-ready markdown"],
              ].map(([k,v])=>(
                <div key={k} style={{ fontSize:11, marginBottom:8, display:"flex", gap:10, alignItems:"flex-start" }}>
                  <span style={{ fontFamily:"var(--mono)", color:"var(--purple)", fontSize:10, flexShrink:0 }}>{k}</span>
                  <span style={{ color:"var(--muted2)" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ VALIDATION ═══════════════ */
function ValidationSlide() {
  const [running, setRunning] = useState(false);
  const [rows, setRows] = useState([]);
  const [done, setDone] = useState(false);
  const [summary, setSummary] = useState(null);
  const bodyRef = useRef(null);

  const CHECKS = [
    { name:"Row count · renewal_scored",  sas:"512,847",  py:"512,847",  delta:"0",      pass:true },
    { name:"Row count · renewal_flags",   sas:"201,344",  py:"201,344",  delta:"0",      pass:true },
    { name:"Row count · mro_summary",     sas:"133,129",  py:"133,129",  delta:"0",      pass:true },
    { name:"Schema columns",              sas:"24 cols",  py:"24 cols",  delta:"exact",  pass:true },
    { name:"score · DecimalType(18,6)",   sas:"0.823847", py:"0.823847", delta:"0.00000",pass:true },
    { name:"renewal_flag · IntType",      sas:"0/1 only", py:"0/1 only", delta:"clean",  pass:true },
    { name:"Null rate · score",           sas:"0.12%",    py:"0.12%",    delta:"0.00%",  pass:true },
    { name:"Null rate · renewal_flag",    sas:"0.00%",    py:"0.00%",    delta:"0.00%",  pass:true },
    { name:"p25 · score",                 sas:"0.4821",   py:"0.4821",   delta:"0.0000", pass:true },
    { name:"p50 · score (median)",        sas:"0.6714",   py:"0.6714",   delta:"0.0000", pass:true },
    { name:"p75 · score",                 sas:"0.8203",   py:"0.8203",   delta:"0.0000", pass:true },
    { name:"p95 · score",                 sas:"0.9471",   py:"0.9468",   delta:"-0.0003",pass:true },
    { name:"Business rule: score ∈ [0,1]",sas:"PASS",     py:"PASS",     delta:"—",      pass:true },
    { name:"Business rule: flag ∈ {0,1}", sas:"PASS",     py:"PASS",     delta:"—",      pass:true },
    { name:"Product code validity",       sas:"PASS",     py:"PASS",     delta:"—",      pass:true },
  ];

  const run = () => {
    setRows([]); setDone(false); setSummary(null); setRunning(true);
    CHECKS.forEach((c,i)=>{
      setTimeout(()=>{
        setRows(p=>[...p,c]);
        setTimeout(()=>{ if(bodyRef.current) bodyRef.current.scrollTop=99999; },30);
        if(i===CHECKS.length-1){
          setTimeout(()=>{
            setDone(true); setRunning(false);
            setSummary({ pass:CHECKS.filter(x=>x.pass).length, fail:CHECKS.filter(x=>!x.pass).length });
          },300);
        }
      }, i*110+200);
    });
  };
  const reset = () => { setRows([]); setDone(false); setSummary(null); setRunning(false); };

  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">Validation Suite</div>
          <div className="pg-sub">SAS vs PySpark output diff — row counts, schema, statistical distributions, business rules.</div>
        </div>
        <div className="pg-actions">
          {running&&<span style={{ fontSize:11, fontFamily:"var(--mono)", color:"var(--amber)", fontWeight:500 }}>● RUNNING</span>}
          {done&&summary&&<span className={`badge ${summary.fail===0?"badge-teal":"badge-rose"}`}>{summary.fail===0?`✓ ALL ${summary.pass} PASSED`:`${summary.fail} FAILED`}</span>}
          <button className="btn btn-primary" onClick={run} disabled={running}>{done?"↺ Re-run":"▶ Run Validation"}</button>
          {rows.length>0&&<button className="btn btn-outline" onClick={reset}>Reset</button>}
        </div>
      </div>

      <div className="g2">
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head">
            <div className="panel-title">Check Results</div>
            <div className="panel-sub">{rows.length} / {CHECKS.length}</div>
          </div>
          <div ref={bodyRef} style={{ maxHeight:360, overflowY:"auto", padding:"0 18px" }}>
            <table className="vtable">
              <thead>
                <tr className="vtable-head">
                  <th style={{ width:200 }}>Check</th>
                  <th style={{ width:80, textAlign:"right" }}>SAS</th>
                  <th style={{ width:80, textAlign:"right" }}>PySpark</th>
                  <th style={{ width:60, textAlign:"right" }}>Δ</th>
                  <th style={{ width:50, textAlign:"right" }}>Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r,i)=>(
                  <tr key={i} className="vrow">
                    <td className="vr-name">{r.name}</td>
                    <td className="vr-num vr-sas">{r.sas}</td>
                    <td className="vr-num vr-py">{r.py}</td>
                    <td className="vr-delta">{r.delta}</td>
                    <td className="vr-pass">{r.pass?<span className="chip-pass">PASS</span>:<span className="chip-fail">FAIL</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
          <div className="panel" style={{ marginBottom:0 }}>
            <div className="panel-head"><div className="panel-title">Validation Coverage</div></div>
            <div className="panel-body">
              {[
                { lbl:"Row count parity",       pct:100, c:"var(--teal)" },
                { lbl:"Schema column match",    pct:100, c:"var(--teal)" },
                { lbl:"Distribution match p95", pct:97,  c:"var(--blue)" },
                { lbl:"Business rule pass rate",pct:99,  c:"var(--blue)" },
                { lbl:"Automation coverage",    pct:85,  c:"var(--amber)" },
              ].map(m=>(
                <div className="pbar" key={m.lbl}>
                  <div className="pbar-label">{m.lbl}</div>
                  <div className="pbar-track"><div className="pbar-fill" style={{ width:done?m.pct+"%":"0%", background:m.c }}/></div>
                  <div className="pbar-val" style={{ color:m.c }}>{m.pct}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel" style={{ marginBottom:0 }}>
            <div className="panel-head"><div className="panel-title">What We Validated Beyond Metadata</div></div>
            <div className="panel-body">
              <div style={{ fontSize:11, color:"var(--muted2)", lineHeight:2 }}>
                {["Statistical distributions (p25 · p50 · p75 · p95)","Null rate comparison per column","Business rule enforcement (score ∈ [0,1])","Decimal precision to 6 places","Silent corruption detection via row-level hash"].map(v=>(
                  <div key={v}>
                    <span style={{ color:"var(--teal)", marginRight:8 }}>✓</span>{v}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ DEVOPS ═══════════════ */
function DevOpsSlide() {
  const steps = [
    { color:"var(--rose)",   icon:"⌥",  title:"Legacy State",           desc:"SAS files on NAS. Changes tracked by comment-line edits. No branching. No traceability. Ad-hoc docs." },
    { color:"var(--blue)",   icon:"⚙️", title:"CIBC AI Bootstrap",      desc:"Used CIBC AI to generate initial Jira epics/stories from SAS code scan. Confluency pages auto-drafted." },
    { color:"var(--purple)", icon:"⑂",  title:"GitHub Migration",        desc:"Moved all SAS + PySpark scripts to GitHub. Branch strategy: main / dev / feature. All merges via PR." },
    { color:"var(--teal)",   icon:"✓",  title:"PR Review Workflow",     desc:"Code review enforced on all merges. GitHub Actions CI: linting, unit tests, schema diff on every PR." },
    { color:"var(--amber)",  icon:"📋", title:"Jira Tracking",          desc:"Epics → Stories → Sub-tasks mapped to 5 workstreams. Bi-weekly sprint cadence. Full burndown visible." },
    { color:"var(--green)",  icon:"📄", title:"Confluence Documentation",desc:"Living docs: migration runbook, validation results, agent pipeline design, schema registry." },
  ];
  const before = ["NAS-stored .sas files","Eyeball-diff workflow","Comment-line changelog","No traceability","Ad-hoc docs"];
  const after  = ["Git history + blame","PR review before merge","Jira epic/story tracking","Confluence auto-docs","AI-generated tickets"];
  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">DevOps & Governance</div>
          <div className="pg-sub">Replaced NAS file versioning with GitHub + Jira + Confluence, bootstrapped end-to-end using CIBC AI.</div>
        </div>
        <span className="badge badge-teal">✓ Full Audit Trail</span>
      </div>

      <div className="g2" style={{ marginBottom:14 }}>
        {/* before / after */}
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head"><div className="panel-title">Before → After</div></div>
          <div className="panel-body">
            <div style={{ display:"flex", gap:0 }}>
              <div style={{ flex:1, borderRight:"1px solid var(--border)", paddingRight:16 }}>
                <div style={{ fontSize:10, fontWeight:600, color:"var(--rose)", letterSpacing:.5, textTransform:"uppercase", marginBottom:8 }}>Before</div>
                {before.map(v=><div key={v} style={{ fontSize:11, color:"var(--muted2)", lineHeight:2.2 }}><span style={{ color:"var(--rose)", marginRight:6 }}>✗</span>{v}</div>)}
              </div>
              <div style={{ flex:1, paddingLeft:16 }}>
                <div style={{ fontSize:10, fontWeight:600, color:"var(--teal)", letterSpacing:.5, textTransform:"uppercase", marginBottom:8 }}>After</div>
                {after.map(v=><div key={v} style={{ fontSize:11, color:"var(--muted2)", lineHeight:2.2 }}><span style={{ color:"var(--teal)", marginRight:6 }}>✓</span>{v}</div>)}
              </div>
            </div>
          </div>
        </div>

        {/* tool stack */}
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head"><div className="panel-title">Tool Stack</div></div>
          <div className="panel-body">
            {[
              { icon:"⑂",  name:"GitHub",     desc:"Source control, PR reviews, Actions CI", color:"var(--text)" },
              { icon:"📋", name:"Jira",        desc:"Epic/story tracking, sprint management",  color:"var(--blue)" },
              { icon:"📄", name:"Confluence",  desc:"Living documentation, runbooks",          color:"var(--blue)" },
              { icon:"🤖", name:"CIBC AI",     desc:"Bootstrapped tickets, docs from SAS scan",color:"var(--purple)" },
              { icon:"⚙️", name:"GH Actions",  desc:"CI: lint, unit tests, schema diff",       color:"var(--amber)" },
            ].map(t=>(
              <div key={t.name} style={{ display:"flex", alignItems:"center", gap:10, padding:"8px 0", borderBottom:"1px solid var(--border)" }}>
                <span style={{ fontSize:16 }}>{t.icon}</span>
                <div>
                  <div style={{ fontSize:12, fontWeight:600, color:t.color }}>{t.name}</div>
                  <div style={{ fontSize:10, color:"var(--muted2)" }}>{t.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="panel">
        <div className="panel-head"><div className="panel-title">DevOps Rollout Timeline</div></div>
        <div className="panel-body">
          <div>
            {steps.map((s,i)=>(
              <div className="timeline-step" key={i}>
                {i<steps.length-1 && <div className="tl-line"/>}
                <div className="tl-orb" style={{ background:`${s.color}18`, border:`1px solid ${s.color}40` }}>
                  <span style={{ fontSize:13 }}>{s.icon}</span>
                </div>
                <div className="tl-content">
                  <div className="tl-title" style={{ color:s.color }}>{s.title}</div>
                  <div className="tl-desc">{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ IMPACT ═══════════════ */
function ImpactSlide() {
  const [vis, setVis] = useState(false);
  useEffect(()=>{ const t=setTimeout(()=>setVis(true),400); return()=>clearTimeout(t); },[]);
  const metrics = [
    { val:"5",     lbl:"Workstreams",       sub:"all delivered",  color:"var(--teal)" },
    { val:"100%",  lbl:"Format Parity",     sub:"zero drift",      color:"var(--blue)" },
    { val:"0",     lbl:"Disruptions",       sub:"consumers unaffected", color:"var(--teal)" },
    { val:"6",     lbl:"AI Agents",         sub:"in pipeline",    color:"var(--purple)" },
    { val:"Git",   lbl:"Replaced NAS VCS",  sub:"full audit trail",color:"var(--blue)" },
    { val:"⚡",    lbl:"EDGE Deadline",     sub:"on time",         color:"var(--amber)" },
  ];
  const quality = [
    { lbl:"NAS output parity",       pct:100, c:"var(--teal)" },
    { lbl:"Distribution match (p95)",pct:97,  c:"var(--blue)" },
    { lbl:"Business rule pass rate", pct:99,  c:"var(--blue)" },
    { lbl:"Automation coverage",     pct:85,  c:"var(--amber)" },
    { lbl:"PR coverage (all merges)",pct:100, c:"var(--teal)" },
  ];
  return (
    <div className="slide">
      <div className="pg-header">
        <div>
          <div className="pg-title">Impact Summary</div>
          <div className="pg-sub">End-to-end ownership — engineering, AI design, automation, project governance.</div>
        </div>
        <span className="badge badge-teal">✓ All Workstreams Complete</span>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"repeat(6,1fr)", gap:10, marginBottom:16 }}>
        {metrics.map(m=>(
          <div className="metric-card" key={m.lbl} style={{ padding:"14px 12px", textAlign:"center" }}>
            <div style={{ fontSize:24, fontWeight:600, color:m.color, marginBottom:4 }}>{m.val}</div>
            <div style={{ fontSize:10, fontWeight:600, color:"var(--muted)", letterSpacing:.5, textTransform:"uppercase", marginBottom:2 }}>{m.lbl}</div>
            <div style={{ fontSize:10, color:"var(--muted2)" }}>{m.sub}</div>
          </div>
        ))}
      </div>

      <div className="g2">
        <div className="panel" style={{ marginBottom:0 }}>
          <div className="panel-head"><div className="panel-title">Quality Metrics</div></div>
          <div className="panel-body">
            {quality.map(m=>(
              <div className="pbar" key={m.lbl}>
                <div className="pbar-label">{m.lbl}</div>
                <div className="pbar-track"><div className="pbar-fill" style={{ width:vis?m.pct+"%":"0%", background:m.c }}/></div>
                <div className="pbar-val" style={{ color:m.c }}>{m.pct}%</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          {[
            ["var(--blue)",   "01 · SAS→PySpark Migration",  "Decoupled rewrite. CSV bridge. CIBC EDGE deadline met. 100% NAS format parity."],
            ["var(--purple)", "02 · Agentic AI Pipeline",    "6-agent chain: audit → translate → review → fix → validate → document. Automated."],
            ["var(--teal)",   "03 · Validation Suite",       "Row count, schema, distributions, business rules. Silent corruption caught via parity diff."],
            ["var(--amber)",  "04 · Power Automate Flows",   "M365 bridge: email routing, Teams alerts, NAS concat. Zero new infra required."],
            ["var(--rose)",   "05 · DevOps & Governance",    "GitHub + Jira + Confluence via CIBC AI. All merges PR-reviewed. Full audit trail."],
          ].map(([c,t,d])=>(
            <div key={t} style={{ padding:"10px 14px", background:"var(--card2)", border:"1px solid var(--border)", borderLeft:`3px solid ${c}`, borderRadius:"0 8px 8px 0" }}>
              <div style={{ fontSize:11, fontWeight:600, color:c, marginBottom:3 }}>{t}</div>
              <div style={{ fontSize:10, color:"var(--muted2)", lineHeight:1.5 }}>{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════ SHELL ═══════════════ */
const SLIDE_COMPS = [OverviewSlide, MigrationSlide, AssetsSlide, AgenticSlide, ValidationSlide, DevOpsSlide, ImpactSlide];

function NarratorBar({ cur, onBack, onNext }) {
  const s = STORY[cur];
  const isLast = cur===N-1;
  const parse = (str) => str.split(/(<em>.*?<\/em>)/g).map((p,i)=>
    p.startsWith("<em>") ? <em key={i}>{p.replace(/<\/?em>/g,"")}</em> : p
  );
  return (
    <div className="narrator">
      <button className="nar-back" onClick={onBack} disabled={cur===0}>←</button>
      <div className="nar-body">
        <div className="nar-label">{isLast?"complete":"next"}</div>
        <div className="nar-msg">{isLast?<em>Done — use sidebar to revisit any section.</em>:parse(s.msg)}</div>
      </div>
      <div className="nar-right">
        <div className="nar-count"><span>{String(cur+1).padStart(2,"0")}</span>/ {String(N).padStart(2,"0")}</div>
        <div className="nar-cta">
          {isLast
            ? <span className="badge badge-teal">✓ Complete</span>
            : <button className="btn btn-primary" onClick={onNext}>{s.cta}</button>
          }
        </div>
      </div>
      <div className="nar-progress"><div className="nar-progress-fill" style={{ width:((cur+1)/N*100)+"%" }}/></div>
    </div>
  );
}

export default function MROPortfolioDashboard() {
  const [cur, setCur] = useState(0);
  const go = useCallback((n)=>{ if(n>=0&&n<N) setCur(n); },[]);

  useEffect(()=>{
    const h=(e)=>{
      if(e.key==="ArrowRight"||e.key==="ArrowDown") go(cur+1);
      if(e.key==="ArrowLeft"||e.key==="ArrowUp")   go(cur-1);
    };
    window.addEventListener("keydown",h);
    return()=>window.removeEventListener("keydown",h);
  },[cur,go]);

  const Comp = SLIDE_COMPS[cur];
  const sectionMap = {};
  SLIDES.forEach(s=>{ sectionMap[s.section]=sectionMap[s.section]||[]; sectionMap[s.section].push(s); });

  return (
    <div className="app">
      <style>{CSS}</style>

      {/* Top bar */}
      <div className="topbar">
        <div className="topbar-brand">
          <div className="brand-icon">⬡</div>
          <div><div className="brand-name">MRO Platform</div><div className="brand-sub">CIBC EDGE · 2025</div></div>
        </div>
        <div className="topbar-center">
          {["Data Engineering","Agentic AI","Validation","DevOps"].map((t,i)=>(
            <button key={t} className="topbar-tab" onClick={()=>go([1,3,4,5][i])}>{t}</button>
          ))}
        </div>
        <div className="topbar-right">
          <div className="status-pill"><div className="status-dot"/>CIBC EDGE Active</div>
          <div style={{ fontSize:11, color:"var(--muted)", fontFamily:"var(--mono)" }}>Azure Databricks</div>
        </div>
      </div>

      <div className="layout">
        {/* Sidebar */}
        <div className="sidebar">
          {Object.entries(sectionMap).map(([section,items])=>(
            <div key={section}>
              <div className="nav-section">{section}</div>
              {items.map(item=>{
                const idx = SLIDES.findIndex(s=>s.id===item.id);
                return (
                  <button key={item.id} className={`nav-item${idx===cur?" active":""}`} onClick={()=>go(idx)}>
                    <span className="nav-icon">{item.icon}</span>
                    {item.label}
                    {item.badge&&<span className={`nav-badge ${item.badgeCls||"nb-blue"}`}>{item.badge}</span>}
                  </button>
                );
              })}
              <div className="nav-divider"/>
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="content">
          <div key={cur} className="slide"><Comp go={go}/></div>
        </div>
      </div>

      <NarratorBar cur={cur} onBack={()=>go(cur-1)} onNext={()=>go(cur+1)}/>
    </div>
  );
}
