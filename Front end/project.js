// -----------------------------------------------------------------------
// project.js — Live analysis page logic
// -----------------------------------------------------------------------

const API_BASE = 'http://localhost:5000';

// -----------------------------------------------------------------------
// DOM refs
// -----------------------------------------------------------------------
let audioFile = null;

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileInfo    = document.getElementById('fileInfo');
const fiName      = document.getElementById('fiName');
const fiMeta      = document.getElementById('fiMeta');
const fiRemove    = document.getElementById('fiRemove');
const audioWrap   = document.getElementById('audioWrap');
const audioPlayer = document.getElementById('audioPlayer');
const analyseBtn  = document.getElementById('analyseBtn');
const errBox      = document.getElementById('errBox');
const errMsg      = document.getElementById('errMsg');
const pipelineCard= document.getElementById('pipelineCard');
const emptyState  = document.getElementById('emptyState');
const resultsEl   = document.getElementById('results');

// -----------------------------------------------------------------------
// File handling
// -----------------------------------------------------------------------
function loadFile(f) {
  if (!f) return;
  if (!f.name.toLowerCase().endsWith('.wav')) {
    showErr('Only .wav files are supported by the pipeline.');
    return;
  }
  audioFile = f;
  fiName.textContent = f.name;
  fiMeta.textContent = `${(f.size / 1024).toFixed(1)} KB · audio/wav`;
  fileInfo.classList.add('show');
  analyseBtn.disabled = false;
  audioPlayer.src = URL.createObjectURL(f);
  audioWrap.classList.add('show');
  reset(); hideErr();
}

fileInput.addEventListener('change', e => loadFile(e.target.files[0]));

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});

fiRemove.addEventListener('click', () => {
  audioFile = null; fileInput.value = '';
  fileInfo.classList.remove('show');
  audioWrap.classList.remove('show');
  analyseBtn.disabled = true;
  pipelineCard.classList.remove('show');
  resetPipeline(); reset(); hideErr();
});

// -----------------------------------------------------------------------
// Run analysis
// -----------------------------------------------------------------------
analyseBtn.addEventListener('click', async () => {
  if (!audioFile) return;
  analyseBtn.disabled = true;
  hideErr(); reset();
  pipelineCard.classList.add('show');
  resetPipeline();
  emptyState.style.display = 'none';

  try {
    // Step 1 — local ingestion feedback
    step('ps1', 'ps1t', 'active', 'Reading file…');
    await wait(300);
    step('ps1', 'ps1t', 'done', `${audioFile.name} · ${(audioFile.size / 1024).toFixed(1)} KB`);

    // Steps 2–4 run server-side simultaneously
    step('ps2', 'ps2t', 'active', 'Detecting anomalies…');
    step('ps3', 'ps3t', 'active', 'Extracting timestamps…');
    step('ps4', 'ps4t', 'active', 'Running CNN classifier…');

    const data = await callAPI(audioFile);

    step('ps2', 'ps2t', 'done', `${data.anomalies.length} event${data.anomalies.length !== 1 ? 's' : ''} found`);
    step('ps3', 'ps3t', 'done', 'Timestamps computed');
    step('ps4', 'ps4t', 'done', `${data.predicted_class} · ${data.confidence}%`);

    render(data);

  } catch (e) {
    ['ps1','ps2','ps3','ps4'].forEach(id => {
      const el = document.getElementById(id);
      if (el.classList.contains('active')) el.classList.remove('active');
    });
    showErr(e.message || 'Analysis failed. Is the Flask server running?');
    analyseBtn.disabled = false;
  }
});

// -----------------------------------------------------------------------
// API call
// -----------------------------------------------------------------------
async function callAPI(file) {
  const fd = new FormData();
  fd.append('file', file);

  const res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd });
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || `Server error (${res.status})`);
  return json;
}

// -----------------------------------------------------------------------
// Render results
// -----------------------------------------------------------------------
function render(data) {

  // --- Anomaly table ---
  document.getElementById('anomalyCount').textContent = data.anomalies.length;
  const tbody = document.getElementById('anomalyBody');
  tbody.innerHTML = '';

  if (!data.anomalies.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:2.5rem;font-size:.83rem">No anomalies detected in this recording.</td></tr>`;
  } else {
    data.anomalies.forEach(a => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="mono">#${a.id}</span></td>
        <td><span class="mono">${a.start.toFixed(3)}s</span></td>
        <td><span class="mono">${a.end.toFixed(3)}s</span></td>
        <td><span class="mono">${a.duration.toFixed(3)}s</span></td>
        <td><span class="mono">${a.peak_db.toFixed(1)} dB</span></td>
        <td>
          <div class="score-row">
            <div class="score-track"><div class="score-fill" style="width:${a.confidence}%"></div></div>
            <span class="mono" style="min-width:36px">${a.confidence}%</span>
          </div>
        </td>
        <td><span class="cls-chip"><i class="fas fa-tag"></i>&nbsp;${a.class}</span></td>`;
      tbody.appendChild(tr);
    });
  }

  // --- Acoustic features ---
  const f  = data.features;
  const fg = document.getElementById('featGrid');
  fg.innerHTML = '';
  [
    { lbl: 'Duration',          val: f.duration,          unit: 'seconds' },
    { lbl: 'Sample Rate',       val: f.sample_rate,       unit: 'Hz'      },
    { lbl: 'Channels',          val: f.channels,          unit: 'ch'      },
    { lbl: 'RMS Level',         val: f.rms_db,            unit: 'dB'      },
    { lbl: 'Spectral Centroid', val: f.spectral_centroid, unit: 'Hz'      },
    { lbl: 'Spectral Rolloff',  val: f.spectral_rolloff,  unit: 'Hz'      },
    { lbl: 'Zero Crossing',     val: f.zcr,               unit: '/sec'    },
    { lbl: 'Bandwidth',         val: f.bandwidth,         unit: 'Hz'      },
  ].forEach(item => {
    const d = document.createElement('div');
    d.className = 'feat-card';
    d.innerHTML = `
      <div class="feat-lbl">${item.lbl}</div>
      <div class="feat-val">${item.val}</div>
      <div class="feat-unit">${item.unit}</div>`;
    fg.appendChild(d);
  });

  // --- Classification ---
  document.getElementById('clsName').textContent = data.predicted_class;
  document.getElementById('clsConf').textContent = data.confidence + '%';
  document.getElementById('clsDesc').textContent =
    data.predicted_class === 'Unknown'
      ? 'Confidence fell below the 60% threshold. The acoustic signature could not be reliably assigned to a known class.'
      : `Classified as "${data.predicted_class}" with ${data.confidence}% confidence. Anomalous activity detected in ${data.anomaly_ratio}% of the recording.`;

  // Probability bars (sorted highest → lowest)
  const pb     = document.getElementById('probBars');
  pb.innerHTML = '';
  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([name, prob]) => {
    const r = document.createElement('div');
    r.className = 'prob-row';
    r.innerHTML = `
      <div class="prob-name">${name}</div>
      <div class="prob-track"><div class="prob-fill" style="width:0%"></div></div>
      <div class="prob-pct">${prob.toFixed(1)}%</div>`;
    pb.appendChild(r);
    requestAnimationFrame(() =>
      setTimeout(() => r.querySelector('.prob-fill').style.width = prob + '%', 60)
    );
  });

  // --- Report ---
  const top3 = sorted.slice(0, 3)
    .map(([n, p]) => `<strong style="color:var(--text)">${n}</strong> (${p.toFixed(1)}%)`)
    .join(', ');

  document.getElementById('reportContent').innerHTML = `
    <p style="margin-bottom:.8rem;color:var(--teal);font-weight:600;font-family:'Syne',sans-serif">
      Analysis Complete
    </p>
    <p style="margin-bottom:.55rem">
      Detected <strong style="color:var(--text)">${data.anomalies.length} anomalous segment${data.anomalies.length !== 1 ? 's' : ''}</strong>
      covering <strong style="color:var(--text)">${data.anomaly_ratio}%</strong> of the recording.
    </p>
    <p style="margin-bottom:.55rem">
      CNN result: <strong style="color:var(--bio)">${data.predicted_class}</strong>
      — confidence <strong style="color:var(--text)">${data.confidence}%</strong>.
    </p>
    <p style="color:var(--muted);font-size:.82rem;margin-top:.8rem">
      Top candidates: ${top3}
    </p>`;

  resultsEl.classList.add('show');
  analyseBtn.disabled = false;
}

// -----------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------
function step(id, tid, state, msg) {
  const el = document.getElementById(id);
  el.classList.remove('active', 'done');
  if (state) el.classList.add(state);
  document.getElementById(tid).textContent = msg;
}

function resetPipeline() {
  ['ps1','ps2','ps3','ps4'].forEach(id => document.getElementById(id).classList.remove('active','done'));
  ['ps1t','ps2t','ps3t','ps4t'].forEach(id => document.getElementById(id).textContent = 'Waiting…');
}

function wait(ms)  { return new Promise(r => setTimeout(r, ms)); }
function showErr(m){ errMsg.textContent = m; errBox.classList.add('show'); }
function hideErr() { errBox.classList.remove('show'); }
function reset()   { resultsEl.classList.remove('show'); emptyState.style.display = ''; }
