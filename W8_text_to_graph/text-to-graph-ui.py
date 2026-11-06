"""Local teaching UI; the independently built student service performs extraction."""




from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
from time import monotonic, perf_counter

from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rdflib import BNode, Graph, Literal, URIRef, XSD
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Text-to-graph evaluation UI")

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Text-to-graph · DTU</title>
  <style>
    :root { color-scheme:light; --red:#990000; --ink:#202020; --muted:#626262; --line:#d8d8d8; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#f5f5f3; font:15px/1.5 system-ui,sans-serif; }
    header { background:white; border-top:7px solid var(--red); border-bottom:1px solid var(--line); padding:22px max(24px,calc((100vw - 1400px)/2)); }
    header p { margin:5px 0 0; color:var(--muted); }
    h1 { margin:0; font-size:26px; letter-spacing:-.6px; } h1 span { color:var(--red); }
    h2 { margin:0 0 14px; font-size:18px; } h3 { margin:10px 0; font-size:15px; }
    main { max-width:1450px; margin:auto; padding:24px; }
    .card { background:white; border:1px solid var(--line); border-radius:5px; padding:20px; margin-bottom:20px; }
    .toolbar,.actions { display:flex; gap:10px; align-items:end; flex-wrap:wrap; }
    .toolbar > label:first-child { flex:1; min-width:270px; }
    label { display:block; font-weight:600; font-size:13px; }
    input,textarea,select,button { font:inherit; }
    input,textarea,select { border:1px solid #aaa; border-radius:3px; padding:8px 10px; background:white; color:var(--ink); }
    input,select { min-height:40px; } label > input,label > select { display:block; width:100%; margin-top:4px; }
    input[type=number] { width:90px; } textarea { width:100%; resize:vertical; line-height:1.45; margin:6px 0 10px; }
    textarea.code { font:13px/1.45 ui-monospace,monospace; tab-size:2; }
    button,.file-button { border:1px solid #aaa; background:white; color:var(--ink); padding:8px 13px; border-radius:3px; cursor:pointer; font-weight:600; font-size:13px; min-height:40px; }
    button.primary { background:var(--red); border-color:var(--red); color:white; }
    button:hover,.file-button:hover { filter:brightness(.94); } button:disabled { opacity:.45; cursor:default; }
    :focus-visible { outline:3px solid #c69531; outline-offset:2px; }
    .grid { display:grid; grid-template-columns:300px minmax(0,1fr); gap:20px; }
    .columns { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .subtle { color:var(--muted); font-size:13px; } .subtle p { margin:6px 0; }
    #cases { width:100%; height:240px; margin:8px 0 12px; } #cases option { padding:7px; white-space:normal; }
    #status { margin:0 0 18px; min-height:24px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .error { color:#990000; } .notice { background:#fff8e5; border-left:3px solid #b67b00; padding:10px; }
    .metrics { display:flex; gap:26px; flex-wrap:wrap; margin:14px 0; }
    .metric strong { display:block; font-size:27px; letter-spacing:-1px; line-height:1.1; }
    .metric span { font-size:12px; color:var(--muted); }
    table { border-collapse:collapse; width:100%; font-size:13px; }
    th,td { text-align:left; padding:8px; border-bottom:1px solid #ddd; vertical-align:top; overflow-wrap:anywhere; }
    th { background:#f3f3f1; } .scroll { overflow:auto; } .triples { table-layout:fixed; } .triples td { font-family:ui-monospace,monospace; font-size:12px; }
    details { border-top:1px solid var(--line); padding:12px 0; } summary { cursor:pointer; font-weight:600; }
    .actions { margin:8px 0; } .compact { min-height:30px; padding:4px 8px; }
    #batch-summary { margin:12px 0; } #name { width:100%; margin:5px 0 14px; }
    [hidden] { display:none !important; }
    @media(max-width:900px) { .grid { grid-template-columns:1fr; } .columns { grid-template-columns:1fr; } #cases { height:140px; } main { padding:14px; } .card { padding:15px; } }
  </style>
</head>
<body>
<header><h1><span>DTU</span> · Text-to-graph</h1><p>Inspect keyboard extractions and compare RDF triples against editable reference graphs.</p></header>
<main>
  <section class="card toolbar" aria-label="Service settings">
    <label>Student extraction endpoint<input id="endpoint" type="url" value="http://localhost:8000/v1/extract" spellcheck="false"></label>
    <label>Timeout (s)<input id="timeout" type="number" min="1" max="300" value="60"></label>
    <label>Parallel calls<input id="parallel" type="number" min="1" max="8" value="1"></label>
    <button id="run-all" class="primary">Run all cases</button><button id="stop" disabled>Stop after active calls</button>
  </section>
  <p id="status" role="status" aria-live="polite">Loading test cases…</p>
  <div class="grid">
    <aside class="card">
      <h2>Test dataset</h2>
      <p class="subtle" id="source"></p>
      <label for="cases">Select a case</label><select id="cases" size="8"></select>
      <div class="actions"><button id="add">Add case</button><button id="remove">Delete case</button></div>
      <div class="actions"><label class="file-button" for="import">Import .txt</label><input id="import" type="file" accept=".txt,text/plain" hidden><button id="export">Export .txt</button></div>
      <button id="reset">Reload server dataset</button>
      <p class="subtle">Edits are kept in this browser. Export your dataset to keep or share a file.</p>
      <details><summary>Scoring rules</summary><div class="subtle">
        <p>Unique RDF triples are compared after JSON-LD expansion. Local extraction URIs and blank nodes are aligned one to one; catalogue URIs stay fixed.</p>
        <p>Finite numerical values compare by value (7 = 7.0). Literal text, language tags and other datatypes retain RDF semantics.</p>
        <p>Every triple counts, including labels and type statements. Additional facts count as false positives, even when an exercise calls them optional.</p>
        <p>Both graphs empty: F1 = 1. Empty prediction with nonempty reference: precision = recall = F1 = 0.</p>
        <p>Alignment searches up to 100,000 states or 3 seconds. If unfinished, scores are labelled lower bounds.</p>
      </div></details>
    </aside>
    <section class="card">
      <h2>Extract &amp; compare</h2>
      <label for="name">Case name</label><input id="name">
      <label for="text">Input text</label><textarea id="text" rows="6" placeholder="Describe a musical keyboard or an offer…"></textarea>
      <div class="actions"><button id="extract" class="primary">Extract &amp; score</button><button id="save">Save case</button></div>
      <div class="columns">
        <div><label for="reference">Reference JSON-LD</label><textarea class="code" id="reference" rows="16" spellcheck="false"></textarea></div>
        <div><label for="prediction">Student response JSON-LD (or paste one)</label><textarea class="code" id="prediction" rows="16" spellcheck="false" placeholder="The response appears here. You can also paste a response for offline comparison."></textarea></div>
      </div>
      <button id="compare">Compare pasted response</button>
      <section id="score" hidden aria-label="Evaluation results">
        <p id="score-note" class="subtle"></p><div class="metrics" id="metrics"></div>
        <div id="diff"></div>
        <details><summary>Local-node alignment</summary><div id="alignment" class="scroll"></div></details>
      </section>
    </section>
  </div>
  <section class="card" id="batch" hidden>
    <h2>Dataset results</h2><p id="batch-summary" aria-live="polite"></p>
    <button id="report">Download report JSON</button>
    <div class="scroll"><table><thead><tr><th>Case</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>Time</th><th>Status</th></tr></thead><tbody id="batch-rows"></tbody></table></div>
  </section>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
const DATA_KEY = 'dtu-text-to-graph-cases-v1';
const SETTINGS_KEY = 'dtu-text-to-graph-settings-v1';
let cases = [], selected = 0, results = [], running = false, stopRequested = false, batchSettings = null;
const emptyGraph = () => ({'@context': {'@base':'https://example.org/keyboards/extraction/reference/','kb':'https://keyboards.wikibase.cloud/entity/','kbt':'https://keyboards.wikibase.cloud/prop/direct/','rdfs':'http://www.w3.org/2000/01/rdf-schema#'}, '@graph':[]});
const pretty = value => JSON.stringify(value, null, 2);
const pct = value => (100 * value).toFixed(1) + '%';

function status(message, error = false) { $('status').textContent = message; $('status').className = error ? 'error' : ''; }

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
  return result;
}

function persist() {
  try { localStorage.setItem(DATA_KEY, JSON.stringify(cases)); }
  catch (_) { status('Browser storage is unavailable. Export the dataset to keep edits.', true); }
}

function settings() {
  const timeout = Number($('timeout').value), parallel = Number($('parallel').value);
  if (!Number.isInteger(parallel) || parallel < 1 || parallel > 8) throw new Error('Parallel calls must be an integer from 1 to 8.');
  if (!Number.isFinite(timeout) || timeout < 1 || timeout > 300) throw new Error('Timeout must be from 1 to 300 seconds.');
  const endpoint = $('endpoint').value.trim();
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify({endpoint,timeout,parallel})); } catch (_) {}
  return {endpoint, timeout, parallel};
}

function updateOptions() {
  $('cases').replaceChildren();
  cases.forEach((item, index) => { const option = document.createElement('option'); option.value = index; option.textContent = (index + 1) + '. ' + item.name; $('cases').append(option); });
  $('cases').value = selected;
}

function clearResults() { results = []; $('batch').hidden = true; }

function commit() {
  const reference = JSON.parse($('reference').value);
  if (!reference || !Array.isArray(reference['@graph']) || !reference['@context']) throw new Error('Reference needs @context and an @graph array.');
  if (!$('text').value.trim()) throw new Error('Input text is empty.');
  const next = {name:$('name').value.trim() || 'Untitled case', text:$('text').value, reference};
  if (JSON.stringify(next) !== JSON.stringify(cases[selected])) clearResults();
  cases[selected] = next;
  persist(); updateOptions();
}

function showCase(index, result) {
  selected = index; $('cases').value = index;
  const item = cases[index];
  $('name').value = item.name; $('text').value = item.text; $('reference').value = pretty(item.reference);
  $('prediction').value = result && result.prediction !== undefined ? pretty(result.prediction) : '';
  $('score').hidden = true;
  if (result && result.score) showScore(result.score, result.elapsed_seconds);
  if (result && (result.error || result.validation_error)) status(result.error || result.validation_error, true);
}

function table(headers, rows, className = '') {
  const node = document.createElement('table'); node.className = className;
  const head = node.createTHead().insertRow(); headers.forEach(h => { const th = document.createElement('th'); th.textContent = h; head.append(th); });
  const body = node.createTBody();
  rows.forEach(values => { const row = body.insertRow(); values.forEach(value => { const td = row.insertCell(); td.textContent = value === null ? 'Unmatched' : value; }); });
  return node;
}

function showScore(score, seconds) {
  $('score').hidden = false; $('metrics').replaceChildren();
  [['Precision',pct(score.precision)],['Recall',pct(score.recall)],['F1',pct(score.f1)],['TP / FP / FN',`${score.tp} / ${score.fp} / ${score.fn}`]].forEach(([label,value]) => {
    const div = document.createElement('div'); div.className = 'metric';
    const strong = document.createElement('strong'); strong.textContent = value;
    const span = document.createElement('span'); span.textContent = label; div.append(strong,span); $('metrics').append(div);
  });
  $('score-note').textContent = `${score.predicted_triples} predicted · ${score.reference_triples} reference triples` + (seconds === undefined ? '' : ` · ${seconds.toFixed(2)} s HTTP time`) + (score.exact ? ' · optimal local-node alignment' : ' · alignment budget reached: scores are lower bounds');
  $('score-note').className = score.exact ? 'subtle' : 'notice';
  $('diff').replaceChildren();
  [['missing','Missing reference triples'],['extra','Extra predicted triples'],['matched','Matched triples']].forEach(([key,title]) => {
    const details = document.createElement('details'); details.open = key !== 'matched' && score[key].length > 0;
    const summary = document.createElement('summary'); summary.textContent = `${title} (${score[key].length})`;
    const wrap = document.createElement('div'); wrap.className = 'scroll';
    wrap.append(table(['Subject','Predicate','Object'],score[key],'triples')); details.append(summary,wrap); $('diff').append(details);
  });
  $('alignment').replaceChildren(table(['Prediction node','Reference node'],score.alignment.map(row => [row.prediction,row.reference])));
}

function busy(value) {
  running = value;
  ['run-all','extract','compare','save','add','remove','import','export','reset','cases','name','text','reference','prediction','endpoint','timeout','parallel'].forEach(id => { $(id).disabled = value; });
  $('stop').disabled = !value;
  $('report').disabled = value;
}

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function loadServer() {
  const data = await api('api/dataset'); cases = data.cases; selected = 0; clearResults(); persist(); updateOptions(); showCase(0);
  $('source').textContent = data.sample ? 'Bundled sample: three specification examples.' : 'Loaded from ' + data.source;
  status(data.sample ? 'Ready. Import keyboards.txt to use your full dataset.' : `Ready. ${cases.length} cases loaded.`);
}
$('cases').addEventListener('change', () => {
  const next = Number($('cases').value);
  try { commit(); showCase(next, results[next]); } catch (error) { $('cases').value = selected; status('Save valid reference JSON before changing case: ' + error.message, true); }
});
$('save').onclick = () => { try { commit(); status('Case saved in this browser.'); } catch (error) { status(error.message,true); } };

$('add').onclick = () => {
    try {
	commit();
	cases.push({name:'New case',text:'Describe a keyboard here.',reference:emptyGraph()});
	clearResults(); selected=cases.length-1; persist(); updateOptions(); showCase(selected); }
    catch (error) { status(error.message, true); } };

$('remove').onclick = () => {
  if (cases.length < 2) return status('Keep at least one case, or import a replacement dataset.',true);
  if (!confirm('Delete this case from the browser dataset?')) return;
  cases.splice(selected,1); selected = Math.min(selected,cases.length-1); clearResults(); persist(); updateOptions(); showCase(selected); status('Case deleted.');
};
$('reset').onclick = async () => { if (!confirm('Replace browser edits with the server dataset? Export first to keep edits.')) return; try { await loadServer(); } catch (error) { status(error.message,true); } };
$('import').onchange = async event => {
  try {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 5_000_000) throw new Error('Dataset file exceeds 5 MB.');
    const data = await api('api/dataset/parse',{content:await file.text()});
    cases=data.cases; selected=0; clearResults(); persist(); updateOptions(); showCase(0);
    $('source').textContent = 'Imported ' + file.name; status(`Imported ${cases.length} cases.`);
  } catch (error) { status(error.message,true); } finally { event.target.value=''; }
};
$('export').onclick = () => { try { commit(); download('keyboards.txt',cases.map(item => item.text.trim()+'\n\n'+pretty(item.reference)).join('\n\n')+'\n','text/plain;charset=utf-8'); status('Dataset exported.'); } catch (error) { status(error.message,true); } };
$('compare').onclick = async () => {
  try { commit(); const prediction = JSON.parse($('prediction').value); busy(true); status('Comparing RDF graphs…'); const score = await api('api/compare',{prediction,reference:cases[selected].reference}); showScore(score); status(score.exact ? 'Comparison complete.' : 'Comparison complete with lower-bound scores.'); }
  catch (error) { $('score').hidden=true; status(error.message,true); } finally { busy(false); }
};
$('extract').onclick = async () => {
  try {
    commit(); const config=settings(); busy(true); $('score').hidden=true; $('prediction').value=''; status('Calling student service…');
    const result = await api('api/extract',{...config,text:cases[selected].text,reference:cases[selected].reference});
    $('prediction').value=pretty(result.prediction);
    if (result.validation_error) throw new Error(result.validation_error);
    showScore(result.score,result.elapsed_seconds); status('Extraction and comparison complete.');
  } catch (error) { status(error.message,true); } finally { busy(false); }
};
function renderBatch() {
  const scored=results.filter(item=>item && item.score), attempted=results.filter(Boolean).length;
  const errors=results.filter(item=>item && !item.score).length;
  const tp=scored.reduce((sum,item)=>sum+item.score.tp,0), fp=scored.reduce((sum,item)=>sum+item.score.fp,0), fn=scored.reduce((sum,item)=>sum+item.score.fn,0);
  const micro=2*tp+fp+fn ? 2*tp/(2*tp+fp+fn) : 1;
  const precision=tp+fp ? tp/(tp+fp) : (fn ? 0 : 1), recall=tp+fn ? tp/(tp+fn) : 1;
  const macro=scored.length ? scored.reduce((sum,item)=>sum+item.score.f1,0)/scored.length : 0;
  const bounded=scored.some(item=>!item.score.exact);
  $('batch-summary').textContent=`${attempted}/${cases.length} attempted · ${scored.length} scored · ${errors} failed · ${cases.length-attempted} not run.` + (scored.length ? ` Scored cases only: micro P ${pct(precision)}, R ${pct(recall)}, F1 ${pct(micro)}; macro F1 ${pct(macro)}. TP/FP/FN ${tp}/${fp}/${fn}.` : '') + (bounded ? ' Alignment limits reached: aggregate scores are lower bounds.' : '');
  $('batch-rows').replaceChildren();
  cases.forEach((item,index) => {
    const result=results[index], score=result && result.score, row=document.createElement('tr');
    const cell=document.createElement('td'), button=document.createElement('button'); button.className='compact'; button.textContent=(index+1)+'. '+item.name; button.disabled=running;
    button.onclick=()=>{ try { commit(); showCase(index,results[index]); $('text').scrollIntoView({behavior:'smooth',block:'center'}); } catch (error) { status(error.message,true); } }; cell.append(button); row.append(cell);
    const values=score ? [pct(score.precision),pct(score.recall),pct(score.f1),`${score.tp} / ${score.fp} / ${score.fn}`,result.elapsed_seconds.toFixed(2)+' s',score.exact?'Scored':'Lower bound'] : ['—','—','—','—','—',result ? (result.error || result.validation_error || 'Failed') : 'Not run'];
    values.forEach(value=>{ const td=document.createElement('td'); td.textContent=value; row.append(td); }); $('batch-rows').append(row);
  });
}
$('run-all').onclick = async () => {
  try {
    commit(); const config=settings(); batchSettings={...config}; busy(true); stopRequested=false; results=new Array(cases.length); $('batch').hidden=false;
    let next=0; renderBatch();
    async function worker() {
      while (!stopRequested && next<cases.length) {
        const index=next++, item=cases[index]; status(`Running case ${index+1}/${cases.length}: ${item.name}`);
        try { results[index]=await api('api/extract',{...config,text:item.text,reference:item.reference}); }
        catch (error) { results[index]={error:error.message}; }
        renderBatch();
      }
    }
    await Promise.all(Array.from({length:Math.min(config.parallel,cases.length)},worker));
    status(stopRequested?'Batch stopped. Active calls have finished.':'Batch complete.');
  } catch (error) { status(error.message,true); }
  finally { busy(false); renderBatch(); }
};
$('stop').onclick = () => { stopRequested=true; $('stop').disabled=true; status('Stopping: waiting for active calls; no more cases will start.'); };
$('report').onclick = () => download('text-to-graph-report.json',pretty({created_at:new Date().toISOString(),settings:batchSettings,cases:cases.map((item,index)=>({...item,result:results[index] || null}))}),'application/json');
async function init() {
  try {
    const config=JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null');
    if (config) for (const id of ['endpoint','timeout','parallel']) $(id).value=config[id];
    const saved=JSON.parse(localStorage.getItem(DATA_KEY) || 'null');
    if (Array.isArray(saved) && saved.length && saved.every(item=>typeof item.text==='string' && typeof item.name==='string' && item.reference && Array.isArray(item.reference['@graph']))) {
      cases=saved; updateOptions(); showCase(0); $('source').textContent='Restored browser dataset.'; status(`Ready. ${cases.length} cases restored from this browser.`); return;
    }
  } catch (_) { /* A blocked or corrupt browser store does not prevent startup. */ }
  try { await loadServer(); } catch (error) { status(error.message,true); }
}
init();
</script>
</body>
</html>
"""



DATASET = """
Yamaha GS1 weighs 90 kg


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/df5a9e86-2ce1-4bfc-8596-86e01af8527c/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "kb:Q100",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P6": {
        "@id": "kb:Q2"
      },
      "kbt:P7": 90
    }
  ]
}


Roland Jupiter-08 has just 25 keys


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/f0f37691-c07a-459a-aaa5-42e8e262a927/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "model/1",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "rdfs:label": {
        "@value": "Roland Jupiter-08",
        "@language": "en"
      },
      "kbt:P1": 25,
      "kbt:P6": {
        "@id": "kb:Q16"
      }
    }
  ]
}


Yamaha XSR900 2027 has a motor on 890 cc and weighs 193 kg


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/8a6a742a-bf74-435d-b8ab-1b6ac0f768b3/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": []
}


Der er kun 32 tangenter på et Roland SH-101.


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/923d9fa7-a76f-4fff-ba1f-32a8801c7064/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "kb:Q18",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P1": 32,
      "kbt:P6": {
        "@id": "kb:Q16"
      }
    }
  ]
}


Morsø Musikhus sells a Yamaha PSR-E-583 for 4595 see http://momuhu.dk/index.php?route=product/product&path=59&product_id=1344


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/25ce35b9-d17b-4d1a-9d81-d568e212414b/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "offer/1",
      "kbt:P11": {
        "@id": "kb:Q42"
      },
      "kbt:P24": {
        "@id": "kb:Q106"
      },
      "kbt:P23": {
        "@id": "kb:Q70"
      },
      "kbt:P5": {
        "@id": "http://momuhu.dk/index.php?route=product/product&path=59&product_id=1344"
      },
      "kbt:P25": 4595
    }
  ]
}


Looking for a first keyboard that offers room to grow? The Yamaha PSR-E373 is a black portable keyboard with 61 touch-sensitive keys. Its 622 instrument voices and 205 accompaniment styles give you plenty to explore, while the lesson functions help you practise. Duo mode also lets two people play together.

It is reasonably easy to carry: the instrument weighs 4.6 kg without batteries. Allow 94.5 cm of space from left to right and 36.9 cm from front to back; the keyboard itself stands 11.8 cm high. Its maximum polyphony is 48 notes—different from the number of sounds available to choose from.

You can play through the two built-in speakers, each measuring 12 cm across. Their amplifiers are rated at 2.5 W per channel. For quiet practice, there is one standard stereo headphone socket. That same PHONES/OUTPUT socket can instead send the sound to external equipment, so these are two uses of a single physical connector.

There is also one stereo mini-jack AUX input for an external audio source. USB TO HOST supports MIDI and digital audio transfer in both directions, making it useful for connecting to a computer.

The keyboard runs from a compatible mains adaptor or six AA batteries. Yamaha specifies power consumption of 5 W when using the PA-130 adaptor. A music rest is included, and the song book is available to download.


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/11ce35b9-d17b-4d1a-9d81-d568e212414b/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "1",
      "rdfs:label": {
        "@value": "PSR-E373",
        "@language": "en"
      },
      "kbt:P1": 61,
      "kbt:P2": 945,
      "kbt:P3": 369,
      "kbt:P4": 118,
      "kbt:P6": {"@id": "kb:Q2"},
      "kbt:P7": 4.6,
      "kbt:P8": 48,
      "kbt:P9": {"@id": "kb:Q3"},
      "kbt:P10": 2,
      "kbt:P11": {"@id": "kb:Q14"},
      "kbt:P13": 1,
      "kbt:P16": 5,
      "kbt:P20": 1,
      "kbt:P22": 1
    }
  ]
}


The Yamaha PSR-E483 and PSR-E583 look like close relatives on paper, and choosing between these two keyboard models involves more than counting keys. Both have 61 touch-sensitive keys and can sound up to 64 notes simultaneously. The E483 offers 860 voices, while the E583 increases that selection to 890.

Neither model saves you any desk space over the other. Each measures 99.2 cm across, 40.4 cm from front to back and 13.6 cm high. The PSR-E483 is slightly lighter at 7.0 kg; the PSR-E583 weighs 7.3 kg. Those weights exclude batteries. If you are arranging delivery, avoid confusing the instrument dimensions with the packaging: both boxes measure 1068 × 500 × 216 mm in width, depth and height.

The sound systems are a more noticeable difference. Each keyboard contains two speakers with a diameter of 12 cm, but the E483 has two 6 W amplifier channels, compared with two 12 W channels on the E583. Despite that difference, Yamaha lists power consumption as 13 W for each model, using the PA-150C adaptor for the E483 and the PA-300C for the E583.

The physical audio connections are the same on both instruments: one headphone socket, two separate line-output sockets labelled L/L+R and R, one stereo mini-jack AUX input, and one microphone input. The stereo AUX connection is a single socket, not two inputs.

Both models also provide a USB Type-A connection for a flash drive and a USB Type-C connection to a computer. The computer connection supports a stereo USB audio interface at 44.1 kHz and 16 bits. The E583 adds a modulation wheel and support for an expression pedal, whereas the E483 has neither. Both can run on six AA batteries.


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/22ce35b9-d17b-4d1a-9d81-d568e212414b/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "model/PSR-E483",
      "rdfs:label": {
        "@value": "PSR-E483",
        "@language": "en"
      },
      "kbt:P1": 61,
      "kbt:P2": 992,
      "kbt:P3": 404,
      "kbt:P4": 136,
      "kbt:P6": {"@id": "kb:Q2"},
      "kbt:P7": 7.0,
      "kbt:P8": 64,
      "kbt:P10": 2,
      "kbt:P11": {"@id": "kb:Q14"},
      "kbt:P13": 1,
      "kbt:P16": 13,
      "kbt:P20": 2,
      "kbt:P21": 1,
      "kbt:P22": 1
    },
    {
      "@id": "kb:Q106",
      "rdfs:label": {
        "@value": "PSR-E583",
        "@language": "en"
      },
      "kbt:P1": 61,
      "kbt:P2": 992,
      "kbt:P3": 404,
      "kbt:P4": 136,
      "kbt:P6": {"@id": "kb:Q2"},
      "kbt:P7": 7.3,
      "kbt:P8": 64,
      "kbt:P10": 2,
      "kbt:P11": {"@id": "kb:Q14"},
      "kbt:P13": 1,
      "kbt:P16": 13,
      "kbt:P20": 2,
      "kbt:P21": 1,
      "kbt:P22": 1
    }
  ]
}


Jeg sælger mit røde Nord Stage 4 med 88 tangenter, da jeg ikke får det brugt nok. Det står i rigtig pæn stand med enkelte små brugsspor og har kun været brugt hjemme.

Der er separate sektioner til piano, orgel og synthesizer samt masser af muligheder for at justere lyde og effekter. Et godt instrument, men mere end jeg har brug for.

Kan afhentes og prøves i Odense efter aftale. Strømkabel medfølger. Prisen er uden stativ.

Sælges for 26.500 kr.

Annonce opdateret: 18. oktober 2026.


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/334d63d6-698f-4c3e-b82a-9e53407eda6f/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/"
  },
  "@graph": [
    {
      "@id": "kb:Q23",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P9": {
        "@id": "kb:Q104"
      },
      "kbt:P1": 88,
      "kbt:P6": {
        "@id": "kb:Q5"
      }
    },	
    {
      "@id": "offer/1",
      "kbt:P11": {"@id": "kb:Q42"},
      "kbt:P24": {"@id": "kb:Q23"},
      "kbt:P25": 26500,
      "kbt:P26": {
        "@value": "2026-10-18",
        "@type": "http://www.w3.org/2001/XMLSchema#date"
      }
    }
  ]
}


From https://www.inside.dtu.dk/services/meddelelser/koeb-salg-mv/roland-sh101-for-sale?msgid=4845f9db-c725-4faf-97b9-f460b85a1093

Jeg sælger min gamle roland sh101 for 2500. Jeg købte den for 1000 :)


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/2a4d63d6-698f-4c3e-b82a-9e53407eda6f/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "offer/1",
      "kbt:P11": {
        "@id": "kb:Q42"
      },
      "kbt:P24": {
        "@id": "kb:Q18"
      },
      "kbt:P5": {
        "@id": "https://www.inside.dtu.dk/services/meddelelser/koeb-salg-mv/roland-sh101-for-sale?msgid=4845f9db-c725-4faf-97b9-f460b85a1093"
      },
      "kbt:P25": 2500
    }
  ]
}


The three different Nord Stage 4 models all lack load speekers.


{
  "@context": {
    "@base": "https://example.org/keyboards/extraction/4590e1cf-ec6a-46a0-97c6-da2a216feaf1/",
    "kb": "https://keyboards.wikibase.cloud/entity/",
    "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
  "@graph": [
    {
      "@id": "kb:Q24",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P6": {
        "@id": "kb:Q5"
      },
      "kbt:P10": 0
    },
    {
      "@id": "kb:Q23",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P6": {
        "@id": "kb:Q5"
      },
      "kbt:P10": 0
    },
    {
      "@id": "kb:Q25",
      "kbt:P11": {
        "@id": "kb:Q14"
      },
      "kbt:P6": {
        "@id": "kb:Q5"
      },
      "kbt:P10": 0
    }
  ]
}
"""


def parse_dataset(content):
    """Read multiline text followed by a JSON-LD object, repeatedly.

    Parameters
    ----------
    content : str
        UTF-8 text containing text / JSON-LD pairs. Input text and JSON objects
        may themselves contain blank lines. A JSON object begins on a new line
        after a blank line; normal JSON decoding locates its end.

    Returns
    -------
    list of dict
        Records with ``name``, ``text`` and ``reference`` fields.
    """
    content = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    records, position = [], 0
    decoder = json.JSONDecoder()
    boundary = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*[ \t]*(?=\{)")
    while content[position:].strip():
        match = boundary.search(content, position)
        if match is None:
            raise ValueError(f"Case {len(records) + 1}: expected a blank line followed by JSON-LD.")
        text = content[position:match.start()].strip()
        if not text:
            raise ValueError(f"Case {len(records) + 1}: input text is empty.")
        try:
            reference, end = decoder.raw_decode(content, match.end())
        except json.JSONDecodeError as error:
            raise ValueError(f"Case {len(records) + 1}: {error}") from error
        if not isinstance(reference, dict) or not isinstance(reference.get("@graph"), list):
            raise ValueError(f"Case {len(records) + 1}: expected a JSON-LD object with @graph.")
        if end < len(content) and content[end:].strip() and not re.match(r"[ \t]*\n[ \t]*\n", content[end:]):
            raise ValueError(f"Case {len(records) + 1}: expected a blank line after JSON-LD.")
        records.append({"name": text.splitlines()[0][:80], "text": text, "reference": reference})
        position = end
    if not records:
        raise ValueError("The dataset is empty.")
    if len(records) > 500:
        raise ValueError("At most 500 test cases are supported.")
    return records


def dump_dataset(records):
    """Serialize editable cases in the original text / JSON-LD format."""
    return "\n\n".join(record["text"].strip() + "\n\n" +
                       json.dumps(record["reference"], indent=2, ensure_ascii=False)
                       for record in records) + "\n"


EXTRACTION_ROOT = "https://example.org/keyboards/extraction/"
FALLBACK_BASE = EXTRACTION_ROOT + "unbased/"
NUMERIC_TYPES = {
    XSD.integer, XSD.decimal, XSD.double, XSD.float, XSD.int, XSD.long,
    XSD.short, XSD.byte, XSD.nonNegativeInteger, XSD.positiveInteger,
    XSD.nonPositiveInteger, XSD.negativeInteger, XSD.unsignedInt,
    XSD.unsignedLong, XSD.unsignedShort, XSD.unsignedByte,
}


def check_contexts(value, *, root=True):
    """Reject remote contexts and nested graphs before invoking RDFLib."""
    if isinstance(value, list):
        for item in value:
            check_contexts(item, root=False)
    elif isinstance(value, dict):
        if "@import" in value:
            raise ValueError("JSON-LD @import is not supported; use an inline context.")
        if not root and "@graph" in value:
            raise ValueError("Nested/named graphs are not supported in this exercise.")
        for key, item in value.items():
            if key == "@context":
                contexts = item if isinstance(item, list) else [item]
                if any(c is not None and not isinstance(c, dict) for c in contexts):
                    raise ValueError("Use inline JSON-LD contexts; remote contexts are disabled.")
            check_contexts(item, root=False)


def parse_graph(document):
    """Expand JSON-LD and normalize local nodes and numerical literals.

    Parameters
    ----------
    document : dict
        JSON-LD object with an inline ``@context`` and an ``@graph`` array.

    Returns
    -------
    set of tuple
        Unique RDF triples. Local URIs become anonymous alignment variables.
        Equal finite numbers compare equally across RDF numerical datatypes.
    """
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON-LD object, not a JSON-encoded string.")
    if not isinstance(document.get("@context"), dict):
        raise ValueError("A shared inline @context object is required.")
    if not isinstance(document.get("@graph"), list):
        raise ValueError("An @graph array is required.")
    if "@id" in document:
        raise ValueError("A named root graph is not supported; omit the root @id.")
    check_contexts(document)
    # Keep the supported JSON-LD surface explicit: aliases for graph/context
    # keywords could otherwise bypass the checks above.
    def reject_aliases(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if not key.startswith("@") and isinstance(item, str) and item in {
                    "@context", "@graph", "@import", "@included"
                }:
                    raise ValueError(f"Aliases for {item} are not supported.")
                reject_aliases(item)
        elif isinstance(value, list):
            for item in value:
                reject_aliases(item)
    reject_aliases(document)
    graph = Graph()
    try:
        graph.parse(data=json.dumps(document, allow_nan=False), format="json-ld", publicID=FALLBACK_BASE)
    except Exception as error:
        raise ValueError(f"Invalid JSON-LD: {error}") from error
    if len(graph) > 1000:
        raise ValueError("At most 1,000 triples per graph are supported.")

    def normalize(term):
        if isinstance(term, URIRef) and str(term).startswith(EXTRACTION_ROOT):
            return BNode("uri:" + str(term))
        if isinstance(term, BNode):
            return BNode("blank:" + str(term))
        if isinstance(term, Literal) and term.datatype in NUMERIC_TYPES:
            try:
                number = Decimal(str(term))
                if not number.is_finite():
                    raise ValueError("Non-finite numbers are not supported.")
                # format(..., 'f') avoids scientific notation in xsd:decimal.
                lexical = format(number, "f") if number else "0"
                if "." in lexical:
                    lexical = lexical.rstrip("0").rstrip(".")
                return Literal(lexical, datatype=XSD.decimal, normalize=False)
            except InvalidOperation as error:
                raise ValueError("Invalid RDF numerical literal.") from error
        # RDF 1.1 simple literals and explicitly typed xsd:string are equivalent.
        if isinstance(term, Literal) and term.datatype == XSD.string:
            return Literal(str(term))
        return term
    return {tuple(normalize(term) for term in triple) for triple in graph}


def align_graphs(predicted, reference, *, max_states=100_000, max_seconds=3.0):
    """Maximize overlap under a partial injective mapping of local nodes.

    Branch-and-bound search preserves links between models and offers. If the
    search budget is exhausted, the returned overlap is a lower bound and
    ``exact`` is false; it must not be presented as an exact optimum.
    """
    gold_by_predicate = defaultdict(list)
    for triple in reference:
        gold_by_predicate[triple[1]].append(triple)
    nodes = {term for triple in predicted for term in triple if isinstance(term, BNode)}
    if len(nodes) > 100:
        raise ValueError("At most 100 local nodes per response are supported.")
    candidates = {node: set() for node in nodes}
    compatible = {}
    for triple in predicted:
        possible = []
        for target in gold_by_predicate[triple[1]]:
            pairs = {}
            reverse = {}
            for source, destination in zip(triple, target):
                if isinstance(source, BNode):
                    if not isinstance(destination, BNode):
                        break
                    if source in pairs and pairs[source] != destination:
                        break
                    if destination in reverse and reverse[destination] != source:
                        break
                    pairs[source] = destination
                    reverse[destination] = source
                elif source != destination:
                    break
            else:
                possible.append(pairs)
                for source, destination in pairs.items():
                    candidates[source].add(destination)
        compatible[triple] = possible
    degree = Counter(t for triple in predicted for t in triple if isinstance(t, BNode))
    order = sorted(nodes, key=lambda n: (len(candidates[n]), -degree[n], str(n)))
    mapping, used = {}, set()
    best_mapping = {node: None for node in nodes}
    best = sum(bool(options) for triple, options in compatible.items()
               if not any(isinstance(t, BNode) for t in triple))
    states, exhausted = 0, False
    deadline = monotonic() + max_seconds

    def bound():
        count = 0
        for options in compatible.values():
            for pairs in options:
                if all((mapping[source] == target if source in mapping else target not in used)
                       for source, target in pairs.items()):
                    count += 1
                    break
        return count

    def search(index):
        nonlocal best, best_mapping, states, exhausted
        if states >= max_states or monotonic() >= deadline:
            exhausted = True
            return
        states += 1
        upper = bound()
        if upper <= best:
            return
        if index == len(order):
            best, best_mapping = upper, dict(mapping)
            return
        source = order[index]
        # Try promising assignments first, followed by leaving this node unmatched.
        choices = []
        for target in sorted(candidates[source] - used, key=str):
            mapping[source] = target
            used.add(target)
            choices.append((bound(), target))
            used.remove(target)
        mapping.pop(source, None)
        for _, target in sorted(choices, key=lambda pair: -pair[0]) + [(0, None)]:
            mapping[source] = target
            if target is not None:
                used.add(target)
            search(index + 1)
            if target is not None:
                used.remove(target)
            del mapping[source]
            if exhausted:
                return

    search(0)
    aligned = {
        tuple(best_mapping.get(t) or BNode("unmatched:" + str(t)) if isinstance(t, BNode) else t
              for t in triple)
        for triple in predicted
    }
    return aligned, best_mapping, not exhausted or best == min(len(predicted), len(reference)), states


def display_term(term):
    """Return a readable RDF term without changing its comparison semantics."""
    if isinstance(term, BNode):
        return "_:" + str(term)
    if isinstance(term, URIRef):
        for prefix, uri in {
            "kb": "https://keyboards.wikibase.cloud/entity/",
            "kbt": "https://keyboards.wikibase.cloud/prop/direct/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        }.items():
            if str(term).startswith(uri):
                return prefix + ":" + str(term)[len(uri):]
    return term.n3()


def score_graphs(prediction, reference):
    """Compute triple precision, recall, F1 and an auditable aligned diff."""
    predicted, gold = parse_graph(prediction), parse_graph(reference)
    aligned, mapping, exact, states = align_graphs(predicted, gold)
    matched, extra, missing = aligned & gold, aligned - gold, gold - aligned
    tp, fp, fn = len(matched), len(extra), len(missing)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not gold else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0

    def rows(triples):
        return sorted([list(map(display_term, triple)) for triple in triples])
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "predicted_triples": len(predicted), "reference_triples": len(gold),
        "exact": exact, "search_states": states,
        "alignment": [{"prediction": display_term(source),
                       "reference": display_term(target) if target is not None else None}
                      for source, target in sorted(mapping.items(), key=lambda pair: str(pair[0]))],
        "matched": rows(matched), "extra": rows(extra), "missing": rows(missing),
    }


class ExtractRequest(BaseModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    text: str = Field(min_length=1, max_length=100_000)
    timeout: float = Field(default=60, ge=1, le=300)
    reference: dict | None = None


class CompareRequest(BaseModel):
    prediction: dict
    reference: dict


class DatasetRequest(BaseModel):
    content: str = Field(max_length=5_000_000)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse(HTML)


@app.get("/api/dataset")
def initial_dataset():
    print('init dataset')
    try:
        cases = parse_dataset(DATASET)
        for case in cases:
            parse_graph(case["reference"])
        return {"cases": cases}
    except (ValueError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/dataset/parse")
def import_dataset(payload: DatasetRequest):
    try:
        cases = parse_dataset(payload.content)
        for case in cases:
            parse_graph(case["reference"])
        return {"cases": cases}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/compare")
def compare(payload: CompareRequest):
    try:
        return score_graphs(payload.prediction, payload.reference)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/extract")
async def extract(payload: ExtractRequest):
    endpoint = payload.endpoint.strip()
    try:
        url = urlsplit(endpoint)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("Use a full http(s) extraction URL without credentials, query or fragment.")
        _ = url.port
        if payload.reference is not None:
            await run_in_threadpool(parse_graph, payload.reference)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    start = perf_counter()
    try:
        # Server-side requests avoid browser CORS and mixed-content restrictions.
        # TLS certificates remain verified. Never retry extraction implicitly.
        async with httpx.AsyncClient(timeout=payload.timeout, follow_redirects=False, trust_env=False) as client:
            async with client.stream("POST", endpoint, json={"text": payload.text}) as response:
                if response.status_code != 200:
                    raise HTTPException(502, f"Student service returned HTTP {response.status_code}.")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 2_000_000:
                        raise HTTPException(502, "Student response exceeds 2 MB.")
        elapsed = perf_counter() - start
        def reject_constant(value):
            raise ValueError(f"Non-finite JSON number: {value}")
        prediction = json.loads(body, parse_constant=reject_constant)
    except httpx.TimeoutException as error:
        raise HTTPException(504, f"Student service timed out after {payload.timeout:g} seconds.") from error
    except httpx.RequestError as error:
        raise HTTPException(502, f"Could not reach student service: {error}") from error
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(502, "Student service did not return valid JSON.") from error
    result = {"prediction": prediction, "elapsed_seconds": elapsed}
    try:
        await run_in_threadpool(parse_graph, prediction)
        if payload.reference is not None:
            result["score"] = await run_in_threadpool(score_graphs, prediction, payload.reference)
    except ValueError as error:
        # Keep the raw prediction visible even when its JSON-LD is invalid.
        result["validation_error"] = str(error)
    return result
