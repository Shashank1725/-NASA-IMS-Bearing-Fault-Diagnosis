const $ = (id) => document.getElementById(id);
let active = null;

function initThree() {
  const canvas = $('three-bg');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 1, 1800);
  camera.position.z = 520;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  const count = 850;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    positions[i * 3] = (Math.random() - .5) * 1200;
    positions[i * 3 + 1] = (Math.random() - .5) * 720;
    positions[i * 3 + 2] = (Math.random() - .5) * 1000;
    const color = new THREE.Color(Math.random() > .55 ? '#1de0bf' : '#5d8cff');
    colors.set([color.r, color.g, color.b], i * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({ size: 2, vertexColors: true, transparent: true, opacity: .45, blending: THREE.AdditiveBlending });
  const particles = new THREE.Points(geometry, material);
  scene.add(particles);
  let tx = 0; let ty = 0;
  addEventListener('pointermove', (event) => { tx = (event.clientX / innerWidth - .5) * 18; ty = (event.clientY / innerHeight - .5) * -10; });
  addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
  function animate(time) { requestAnimationFrame(animate); particles.rotation.y += .00025; particles.rotation.x += .00007; particles.position.x += (tx - particles.position.x) * .015; particles.position.y += (ty - particles.position.y) * .015; material.opacity = .39 + Math.sin(time * .0006) * .05; renderer.render(scene, camera); }
  requestAnimationFrame(animate);
}

function plot(id, data, layout = {}) { Plotly.react($(id), data, { paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#8ca2b8', family: 'DM Mono' }, margin: { l: 45, r: 20, t: 10, b: 40 }, ...layout }, { responsive: true, displayModeBar: false }); }
function format(value) { return value === null || value === undefined ? '—' : Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 }); }
function showError(message) { $('error').textContent = message; $('error').classList.remove('hidden'); }
function hideError() { $('error').classList.add('hidden'); }

function renderOverview(data) {
  const clean = data.missing_values === 0;
  const scored = data.scored_rows || 0;
  const predictions = data.predictions || [];
  const faulty = predictions.filter((row) => row.prediction === 1).length;
  const risk = predictions.filter((row) => row.fault_probability !== undefined).map((row) => row.fault_probability);
  const avgRisk = risk.length ? risk.reduce((a, b) => a + b, 0) / risk.length : null;
  $('overview-cards').innerHTML = [
    ['ROWS ANALYSED', format(data.rows), 'observations in active file'],
    ['SIGNALS DETECTED', format(data.numeric_columns.length), 'numeric features available'],
    ['DATA QUALITY', clean ? 'CLEAN' : `${format(data.missing_values)} MISSING`, clean ? 'no missing values detected' : 'review before scoring'],
    ['MODEL RISK', avgRisk === null ? '—' : `${(avgRisk * 100).toFixed(1)}%`, scored ? `${faulty} high-risk predictions` : 'waiting for scoring'],
  ].map(([label, value, note]) => `<div class="card"><label>${label}</label><strong>${value}</strong><small>${note}</small></div>`).join('');
}

function renderCharts(data) {
  const summary = data.summary || [];
  const select = $('feature-select');
  select.innerHTML = summary.map((item) => `<option value="${item.feature}">${item.feature}</option>`).join('');
  const drawMain = () => {
    const feature = select.value;
    const index = data.preview.map((_, i) => i);
    const values = data.preview.map((row) => Number(row[feature])).filter((v) => Number.isFinite(v));
    if ($('chart-select').value === 'distribution') plot('main-chart', [{ x: values, type: 'histogram', marker: { color: '#5d8cff' } }], { xaxis: { gridcolor: '#203247', title: feature }, yaxis: { gridcolor: '#203247' } });
    else plot('main-chart', [{ x: index.slice(0, values.length), y: values, type: 'scatter', mode: 'lines+markers', line: { color: '#1de0bf', width: 2 }, marker: { size: 4 } }], { xaxis: { gridcolor: '#203247', title: 'Preview row' }, yaxis: { gridcolor: '#203247', title: feature } });
  };
  select.onchange = drawMain; $('chart-select').onchange = drawMain; drawMain();
  const variable = [...summary].sort((a, b) => (b.std || 0) - (a.std || 0)).slice(0, 12).reverse();
  plot('variability-chart', [{ x: variable.map((x) => x.std), y: variable.map((x) => x.feature), type: 'bar', orientation: 'h', marker: { color: '#1de0bf' } }], { xaxis: { gridcolor: '#203247', title: 'Standard deviation' }, yaxis: { gridcolor: '#203247' } });
}

function renderRisk(data) {
  const rows = data.predictions || [];
  const probabilities = rows.map((row) => row.fault_probability).filter((v) => v !== undefined);
  const risk = probabilities.length ? probabilities.reduce((a, b) => a + b, 0) / probabilities.length : 0;
  plot('gauge', [{ type: 'indicator', mode: 'gauge+number', value: risk * 100, number: { suffix: '%', font: { color: '#e8f1f8', size: 34 } }, gauge: { axis: { range: [0, 100], ticksuffix: '%' }, bar: { color: '#ff647c' }, steps: [{ range: [0, 35], color: 'rgba(29,224,191,.3)' }, { range: [35, 70], color: 'rgba(255,193,93,.3)' }, { range: [70, 100], color: 'rgba(255,100,124,.3)' }] } }], { margin: { l: 20, r: 20, t: 15, b: 10 } });
  $('risk-note').textContent = probabilities.length ? `${probabilities.filter((v) => v >= .5).length} of ${probabilities.length} scored rows are above the 50% risk threshold.` : 'Model predictions are unavailable for this dataset.';
}

function renderEvaluation(data) {
  const evaluation = data.evaluation || {};
  $('evaluation-badge').textContent = evaluation.available ? 'VERIFIED LABELS' : 'NOT AVAILABLE';
  $('evaluation-badge').style.color = evaluation.available ? '#1de0bf' : '#ffc15d';
  $('evaluation-title').textContent = evaluation.available ? `Live results • ${evaluation.label_column}` : 'Dataset-specific results';
  $('evaluation-note').textContent = evaluation.reason || '';
  if (!evaluation.available) { $('metrics-grid').innerHTML = '<div class="metric"><label>Evaluation</label><strong>—</strong></div>'; $('confusion-chart').innerHTML = ''; return; }
  $('metrics-grid').innerHTML = [['Accuracy', evaluation.accuracy], ['Precision', evaluation.precision], ['Recall', evaluation.recall], ['F1 score', evaluation.f1]].map(([label, value]) => `<div class="metric"><label>${label}</label><strong>${(value * 100).toFixed(1)}%</strong></div>`).join('');
  const z = evaluation.confusion_matrix; plot('confusion-chart', [{ z, type: 'heatmap', colorscale: [[0, '#102d42'], [1, '#1de0bf']], showscale: false, text: z, texttemplate: '%{text}' }], { xaxis: { title: 'Predicted' }, yaxis: { title: 'Actual', autorange: 'reversed' }, margin: { l: 48, r: 10, t: 10, b: 35 } });
}

function renderTable(rows) {
  const columns = rows.length ? Object.keys(rows[0]) : [];
  $('table-head').innerHTML = `<tr>${columns.map((column) => `<th>${column.replaceAll('_', ' ')}</th>`).join('')}</tr>`;
  $('table-body').innerHTML = rows.slice(0, 100).map((row) => `<tr>${columns.map((column) => `<td class="${column === 'state' ? (row[column] === 'Healthy' ? 'state-healthy' : 'state-faulty') : ''}">${format(row[column])}</td>`).join('')}</tr>`).join('');
  $('scored-count').textContent = `${rows.length} scored rows • first 100 shown`;
}

function render(data) {
  active = data; $('dashboard').classList.remove('hidden'); $('dataset-name').textContent = data.filename; $('model-name').textContent = data.model.name || 'NO MODEL'; renderOverview(data); renderCharts(data); renderRisk(data); renderEvaluation(data); renderTable(data.predictions || []);
}
async function upload(file) {
  hideError(); $('file-meta').textContent = `ANALYSING ${file.name}...`; $('file-meta').classList.remove('hidden');
  const form = new FormData(); form.append('file', file);
  try { const response = await fetch('/api/analyze', { method: 'POST', body: form }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Upload failed'); render(data); $('file-meta').textContent = `${file.name} • ${file.size.toLocaleString()} bytes`; } catch (error) { showError(error.message); }
}

$('browse-btn').onclick = () => $('file-input').click(); $('file-input').onchange = (event) => event.target.files[0] && upload(event.target.files[0]);
$('drop-zone').ondragover = (event) => { event.preventDefault(); $('drop-zone').classList.add('drag'); }; $('drop-zone').ondragleave = () => $('drop-zone').classList.remove('drag'); $('drop-zone').ondrop = (event) => { event.preventDefault(); $('drop-zone').classList.remove('drag'); event.dataTransfer.files[0] && upload(event.dataTransfer.files[0]); };
$('reset-btn').onclick = () => { active = null; $('dashboard').classList.add('hidden'); $('file-meta').classList.add('hidden'); $('file-input').value = ''; };
fetch('/api/health').then((response) => response.json()).then((data) => { $('api-status').textContent = data.status.toUpperCase(); $('api-status').previousElementSibling.style.background = '#1de0bf'; $('model-name').textContent = data.model || 'NO MODEL'; }).catch(() => { $('api-status').textContent = 'OFFLINE'; });
initThree();
