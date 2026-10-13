(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const AUTO_DELAY_MS = 700;
  const STATE_LABEL = {
    idle: 'not loaded', downloading: 'downloading model', loading: 'loading model',
    checking: 'checking token budget', ready: 'ready', inferring: 'inference in progress', error: 'error',
  };
  const S = { status: null, game: null, last: null, auto: false, inFlight: false, gen: 0, timer: null, error: null };

  async function api(path, opts = {}) {
    const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      const err = new Error((data && data.error && data.error.message) || res.statusText || 'Request failed');
      err.code = data && data.error && data.error.code;
      err.status = res.status;
      throw err;
    }
    return data;
  }

  const engine = () => document.querySelector('input[name=engine]:checked').value;
  const modelState = () => (S.status ? S.status.model.state : 'idle');
  const engineReady = () => engine() === 'baseline' || modelState() === 'ready' || modelState() === 'inferring';
  const inProgress = () => S.game && S.game.status === 'in_progress';
  const canTurn = () => inProgress() && !S.inFlight && engineReady();

  // ---------- Laya status ----------
  function renderStatus() {
    const st = S.status;
    const chip = $('laya-chip');
    if (!st) { $('laya-chip-text').textContent = 'Server unreachable'; chip.className = 'chip chip-error'; return; }
    const m = st.model;
    const cls = m.state === 'ready' ? 'ready' : m.state === 'error' ? 'error' : m.state === 'idle' ? 'idle' : 'work';
    chip.className = 'chip chip-' + cls;
    const prefix = m.is_test_double ? 'TEST DOUBLE (not Laya) – ' : 'Laya: ';
    $('laya-chip-text').textContent = prefix + STATE_LABEL[m.state] + (m.device && m.state === 'ready' ? ' (' + m.device + ')' : '');
    const lines = [m.message];
    if (m.error) lines.push(m.error);
    if (m.device_note) lines.push(m.device_note);
    const hw = m.hardware || {};
    if (hw.torch) lines.push('PyTorch ' + hw.torch + ' · CUDA ' + (hw.cuda_available ? 'yes' : 'no') + ' · Apple MPS ' + (hw.mps_available ? 'yes' : 'no'));
    $('laya-detail').textContent = lines.filter(Boolean).join(' · ');
    $('retry-load').hidden = m.state !== 'error';
    const rev = m.resolved_revision || m.requested_revision;
    $('foot-model').textContent = 'Model: ' + m.model_id + (rev ? ' @ ' + rev.slice(0, 12) : '') + (m.device ? ' on ' + m.device : '') + '.';
    renderControls();
  }

  async function pollStatus() {
    try { S.status = await api('/api/status'); } catch (_) { S.status = null; }
    renderStatus();
    const st = S.status && S.status.model.state;
    setTimeout(pollStatus, st === 'ready' || st === 'error' ? 3000 : 1000);
  }

  // ---------- rendering ----------
  function renderControls() {
    const g = S.game;
    $('next').disabled = !canTurn() || S.auto;
    $('auto').disabled = !canTurn() && !S.auto ? true : S.auto;
    $('auto').textContent = S.auto ? 'Auto Play running…' : 'Auto Play';
    $('pause').disabled = !S.auto;
    $('start-word').disabled = false;
    const noLaya = !engineReady();
    $('next').title = noLaya ? 'Laya is not ready yet (see the status indicator).' : '';
    if (!g) return;
  }

  function renderGame() {
    const g = S.game;
    $('game').hidden = !g; $('decision').hidden = !g; $('history-card').hidden = !g; $('sees-card').hidden = !g;
    if (!g) return;
    $('pattern').textContent = g.pattern.join(' ');
    $('pattern').setAttribute('aria-label', 'Word pattern: ' + g.pattern.map((c) => (c === '_' ? 'blank' : c)).join(' '));
    $('mistakes').textContent = g.incorrect_count + ' of ' + g.max_incorrect + ' incorrect guesses used (' + g.remaining_incorrect + ' left)';
    for (let i = 1; i <= 6; i++) $('part-' + i).classList.toggle('on', i <= g.incorrect_count);
    $('hangman').setAttribute('aria-label', 'Hangman drawing: ' + g.incorrect_count + ' of ' + g.max_incorrect + ' parts drawn');
    const outcome = $('outcome');
    outcome.className = 'outcome';
    if (g.status === 'won') { outcome.textContent = 'Solved! The word was ' + g.secret + '.'; outcome.classList.add('won'); }
    else if (g.status === 'lost') { outcome.textContent = 'Out of guesses. The word was ' + g.secret + '.'; outcome.classList.add('lost'); }
    else outcome.textContent = 'The secret word is hidden.';
    const byLetter = {};
    g.guesses.forEach((x) => { byLetter[x.letter] = x; });
    const box = $('letters');
    box.textContent = '';
    for (const c of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {
      const d = document.createElement('div');
      const r = byLetter[c];
      d.className = 'letter' + (r ? (r.hit ? ' hit' : ' miss') : '');
      d.setAttribute('role', 'listitem');
      d.setAttribute('aria-label', c + (r ? (r.hit ? ': hit' : ': miss') : ': not guessed'));
      d.textContent = c;
      const small = document.createElement('small');
      small.textContent = r ? (r.hit ? 'hit' : 'miss') : ' ';
      d.appendChild(small);
      box.appendChild(d);
    }
    renderHistory();
    renderControls();
  }

  const srcLabel = (s) => (s === 'laya' ? 'Laya' : 'Baseline (not Laya)');
  const fmt = (v, d = 4) => (typeof v === 'number' ? v.toFixed(d) : '–');

  function renderHistory() {
    const body = $('history');
    body.textContent = '';
    (S.game ? S.game.turns : []).forEach((t) => {
      const tr = document.createElement('tr');
      const cells = [t.number, srcLabel(t.source), t.letter, null, t.pattern_after.split('').join(' '), t.incorrect_after, t.latency_ms.toFixed(1) + ' ms'];
      cells.forEach((v, i) => {
        const td = document.createElement('td');
        if (i === 3) {
          const s = document.createElement('span');
          s.className = 'tag ' + (t.hit ? 'hit' : 'miss');
          s.textContent = t.hit ? 'hit ×' + t.occurrences : 'miss';
          td.appendChild(s);
        } else td.textContent = v;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  function renderDecision() {
    const t = S.last;
    $('d-source').textContent = t ? srcLabel(t.source) : '–';
    $('d-letter').textContent = t ? t.letter : '–';
    $('d-latency').textContent = t ? t.latency_ms.toFixed(1) + ' ms' : '–';
    $('d-conf').textContent = t ? fmt(t.confidence) : '–';
    $('d-aconf').textContent = t ? fmt(t.answer_confidence) : '–';
    const box = $('probs');
    box.textContent = '';
    if (!t) { box.textContent = 'No guess yet.'; return; }
    if (!t.probabilities) { box.textContent = 'The baseline does not produce probabilities.'; return; }
    Object.entries(t.probabilities).sort((a, b) => b[1] - a[1]).forEach(([letter, p]) => {
      const row = document.createElement('div');
      row.className = 'prob' + (letter === t.letter ? ' sel' : '');
      const l = document.createElement('b'); l.textContent = letter;
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('i'); fill.style.width = (p * 100).toFixed(2) + '%'; bar.appendChild(fill);
      const v = document.createElement('span'); v.textContent = p.toFixed(4);
      row.append(l, bar, v);
      box.appendChild(row);
    });
  }

  async function renderSees() {
    const g = S.game;
    if (!g) return;
    const pre = $('sees-json');
    try {
      if (g.status === 'in_progress') {
        const r = await api('/api/games/' + g.id + '/decision-input');
        $('sees-which').textContent = engine() === 'laya' ? 'next' : 'next (the baseline ignores this request)';
        pre.textContent = JSON.stringify({ decision_input: r.decision_input, laya_request: r.laya_request }, null, 2);
      } else {
        const last = S.last && S.last.request;
        $('sees-which').textContent = 'last';
        pre.textContent = last ? JSON.stringify({ laya_request: last }, null, 2) : 'The game is over.';
      }
    } catch (e) { pre.textContent = 'Could not load: ' + e.message; }
  }

  function showError(e) {
    S.error = e;
    $('error').hidden = false;
    $('error-text').textContent = e.message + (e.code ? '  [' + e.code + ']' : '');
    $('error-retry').hidden = !(e.code === 'model_not_ready' || e.code === 'invalid_model_response' || e.code === 'inference_failed' || e.status === undefined);
  }
  function clearError() { S.error = null; $('error').hidden = true; }

  // ---------- actions ----------
  async function takeTurn() {
    if (!canTurn()) return false;
    const myGen = S.gen, sid = S.game.id;
    S.inFlight = true; clearError(); renderControls();
    try {
      const r = await api('/api/games/' + sid + '/turn', { method: 'POST', body: JSON.stringify({ engine: engine() }) });
      if (myGen !== S.gen) return false; // a new game started while this guess was running: ignore it
      S.game = r.game; S.last = r.turn;
      renderGame(); renderDecision(); renderSees();
      const t = r.turn;
      $('live').textContent = srcLabel(t.source) + ' guessed ' + t.letter + ': ' + (t.hit ? t.occurrences + ' match' + (t.occurrences > 1 ? 'es' : '') : 'no match') + '. ' + $('outcome').textContent;
      return true;
    } catch (e) {
      if (myGen !== S.gen) return false;
      stopAuto(); showError(e);
      pollStatusOnce();
      return false;
    } finally {
      if (myGen === S.gen) { S.inFlight = false; renderControls(); }
    }
  }

  async function pollStatusOnce() { try { S.status = await api('/api/status'); renderStatus(); } catch (_) { /* polling continues */ } }

  function startAuto() { if (S.auto || !canTurn()) return; S.auto = true; renderControls(); loop(); }
  function stopAuto() { S.auto = false; clearTimeout(S.timer); S.timer = null; renderControls(); }
  async function loop() {
    if (!S.auto) return;
    const myGen = S.gen;
    const ok = await takeTurn();
    if (myGen !== S.gen || !S.auto) return;           // New Game or Pause happened: schedule nothing
    if (!ok || !inProgress()) { stopAuto(); return; }  // error or game over: stop
    S.timer = setTimeout(loop, AUTO_DELAY_MS);
  }

  async function startGame(body) {
    $('word-error').textContent = '';
    stopAuto();
    try {
      const prev = S.game ? S.game.id : null;
      const game = await api('/api/games', { method: 'POST', body: JSON.stringify({ ...body, previous_id: prev }) });
      S.gen += 1; S.inFlight = false; S.game = game; S.last = null; clearError();
      $('word').value = '';                 // the secret word never stays on screen
      $('show-word').checked = false; $('word').type = 'password';
      renderGame(); renderDecision(); renderSees();
      $('live').textContent = 'New game started. The word has ' + game.length + ' letters.';
      $('game').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) { $('word-error').textContent = e.message; $('word').focus(); }
  }

  function newGame() {
    stopAuto(); S.gen += 1; S.inFlight = false; clearError();
    if (S.game) api('/api/games/' + S.game.id, { method: 'DELETE' }).catch(() => {});
    S.game = null; S.last = null;
    renderGame(); renderDecision();
    $('word').focus();
  }

  $('word-form').addEventListener('submit', (ev) => { ev.preventDefault(); startGame({ word: $('word').value }); });
  $('start-random').addEventListener('click', () => startGame({ random: true }));
  $('show-word').addEventListener('change', (ev) => { $('word').type = ev.target.checked ? 'text' : 'password'; });
  $('next').addEventListener('click', () => { takeTurn(); });
  $('auto').addEventListener('click', startAuto);
  $('pause').addEventListener('click', stopAuto);
  $('new').addEventListener('click', newGame);
  $('error-retry').addEventListener('click', () => { clearError(); takeTurn(); });
  $('retry-load').addEventListener('click', async () => { try { await api('/api/model/load', { method: 'POST' }); } catch (_) { /* shown by polling */ } pollStatusOnce(); });
  document.querySelectorAll('input[name=engine]').forEach((r) => r.addEventListener('change', () => { renderControls(); renderSees(); }));

  renderGame(); renderDecision(); pollStatus();
  window.__layaHangman = S; // for the browser test only
})();
