/* CrossModal Align Lab — Living Borderless White-Box Interactive Engine */
(() => {
  const PALETTE = {
    ink: "#1d1d1f",
    inkSoft: "#424245",
    muted: "#6e6e73",
    faint: "#86868b",
    blue: "#0071e3",
    teal: "#2b9c94",
    coral: "#e66a4e",
    violet: "#7b6bb0",
    amber: "#b7791f"
  };

  const CANVAS_SANS = '"Google Sans Text", "DM Sans", "Noto Sans SC", -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  const CANVAS_MONO = '"Roboto Mono", "SFMono-Regular", Consolas, monospace';

  const state = {
    bundle: null,
    clock: 0,
    reveals: { trap: 0, oov: 0, temporal: 0, domain: 0 },
    scrollAnim: new Map(),
    // 01 Routing Trap
    shiftVal: -1.0,
    autoShift: true,
    autoShiftPhase: 0,
    // 02 OOV Arena
    activeOovIdx: 0,
    activeHeadKey: "text_anchored",
    autoOov: true,
    autoOovTimer: 0,
    // 03 Temporal Microscope
    speedVal: 1.0,
    autoSpeed: true,
    autoSpeedPhase: 0,
    attnMode: "qformer",
    // 04 Domain + FSQ
    domainMode: "conditional",
    useFsq: true,
    autoDomain: true,
    autoDomainTimer: 0,
    pointAnim: []
  };

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function easeInOut(t) { const x = clamp(t, 0, 1); return -(Math.cos(Math.PI * x) - 1) * 0.5; }
  function easeOut(t) { const x = clamp(t, 0, 1); return 1 - Math.pow(1 - x, 3); }

  function prepareCanvas(id) {
    const canvas = document.getElementById(id);
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(10, Math.floor(rect.width));
    const h = Math.max(10, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
  }

  function drawPill(ctx, text, cx, cy, bg, fg) {
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    const tw = ctx.measureText(text).width;
    const pw = tw + 16;
    const ph = 20;
    drawRoundedRect(ctx, cx - pw * 0.5, cy - ph * 0.5, pw, ph, ph * 0.5);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.fillStyle = fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cx, cy + 0.5);
    ctx.restore();
  }

  let toastTimer = null;
  function showToast(msg) {
    const box = document.getElementById("studioToast");
    const text = document.getElementById("studioToastText");
    if (!box || !text) return;
    text.textContent = msg;
    box.classList.add("is-visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => box.classList.remove("is-visible"), 3200);
  }

  // =========================================================================
  // Scroll-Linked Apple / OpenAI / Gemini Fade-In & Fade-Out
  // =========================================================================
  function updateScrollChoreography(dt) {
    const vh = window.innerHeight || 800;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.querySelectorAll("[data-scroll-stage]").forEach((sec) => {
      const key = sec.dataset.scrollStage;
      const rect = sec.getBoundingClientRect();
      const center = rect.top + rect.height * 0.5;
      const norm = (center - vh * 0.5) / (vh * 0.72);
      if (rect.bottom > vh * 0.08 && rect.top < vh * 0.92) {
        state.reveals[key] = Math.min(1, state.reveals[key] + dt / 1100);
      }
      if (reduce) { sec.style.opacity = "1"; sec.style.transform = "none"; return; }
      let o = 1, y = 0, s = 1;
      if (norm > 0.25) {
        const t = clamp((norm - 0.25) / 0.85, 0, 1);
        o = 1 - t * 0.92; y = t * 42; s = 1 - t * 0.016;
      } else if (norm < -0.35) {
        const t = clamp((-norm - 0.35) / 0.85, 0, 1);
        o = 1 - t * 0.84; y = -t * 22; s = 1 - t * 0.01;
      }
      const prev = state.scrollAnim.get(sec) || { o, y, s };
      const damp = 1 - Math.exp(-dt * 0.012);
      prev.o += (o - prev.o) * damp;
      prev.y += (y - prev.y) * damp;
      prev.s += (s - prev.s) * damp;
      state.scrollAnim.set(sec, prev);
      sec.style.opacity = prev.o.toFixed(3);
      sec.style.transform = `translate3d(0, ${prev.y.toFixed(2)}px, 0) scale(${prev.s.toFixed(4)})`;
    });
  }

  // =========================================================================
  // Autonomous Living Choreography (Zero Hover Required)
  // =========================================================================
  function stepAutonomous(dt) {
    // 1. Routing Trap smooth shift sweep 0.0 -> -1.0 (hold) -> 0.0
    if (state.autoShift && state.reveals.trap > 0.4) {
      state.autoShiftPhase = (state.autoShiftPhase + dt / 8000) % 1.0;
      const p = state.autoShiftPhase;
      let s = 0.0;
      if (p < 0.14) s = 0.0;
      else if (p < 0.44) s = -1.0 * easeInOut((p - 0.14) / 0.30);
      else if (p < 0.78) s = -1.0;
      else s = -1.0 * (1 - easeInOut((p - 0.78) / 0.22));
      state.shiftVal = s;
      const sl = document.getElementById("sliderShift");
      if (sl) sl.value = s.toFixed(2);
      updateTrapUI();
    }

    // 2. OOV Arena autonomous tour across OOV words & heads
    if (state.autoOov && state.reveals.oov > 0.5 && state.bundle) {
      state.autoOovTimer += dt;
      if (state.autoOovTimer >= 3600) {
        state.autoOovTimer = 0;
        const cycle = ["fixed_slot", "text_anchored", "lm_bridged"];
        const nextHead = cycle[(cycle.indexOf(state.activeHeadKey) + 1) % cycle.length];
        state.activeOovIdx = (state.activeOovIdx + 1) % state.bundle.oov_arena.oov_words.length;
        setHeadMode(nextHead, false);
        renderOovButtons();
      }
    }

    // 3. Temporal speed rubber-band stretch 0.6x <-> 1.8x
    if (state.autoSpeed && state.reveals.temporal > 0.4) {
      state.autoSpeedPhase = (state.autoSpeedPhase + dt / 6800) % 1.0;
      const wave = 0.5 - 0.5 * Math.cos(state.autoSpeedPhase * Math.PI * 2);
      state.speedVal = 0.60 + wave * 1.25;
      const sl = document.getElementById("sliderSpeed");
      if (sl) sl.value = state.speedVal.toFixed(2);
      const r = document.getElementById("speedReadout");
      if (r) r.textContent = `${state.speedVal.toFixed(2)}x · 自动伸缩演示`;
    }

    // 4. Domain alignment mode tour (none -> marginal -> conditional)
    if (state.autoDomain && state.reveals.domain > 0.5) {
      state.autoDomainTimer += dt;
      if (state.autoDomainTimer >= 3800) {
        state.autoDomainTimer = 0;
        const seq = ["none", "marginal", "conditional"];
        const next = seq[(seq.indexOf(state.domainMode) + 1) % seq.length];
        setDomainMode(next, false);
      }
    }
  }

  // =========================================================================
  // 01 / Routing Trap Collapse Simulator
  // =========================================================================
  function getTrapMetricsAt(shiftVal) {
    if (!state.bundle) return { single_track_recall5: 100, gate_auroc: 0.999, dual_track_hit_rate: 4, dual_track_hits: 2, n_positive: 50 };
    const sweep = state.bundle.routing_trap.sweep;
    let best = sweep[0];
    let bestD = Infinity;
    for (const item of sweep) {
      const d = Math.abs(item.shift - shiftVal);
      if (d < bestD) { bestD = d; best = item; }
    }
    return best;
  }

  function updateTrapUI() {
    const m = getTrapMetricsAt(state.shiftVal);
    const readout = document.getElementById("shiftReadout");
    if (readout) {
      readout.textContent = `shift = ${state.shiftVal.toFixed(2)} · ${state.autoShift ? "自动推演" : "手动调节"}`;
    }
    document.getElementById("gaugeRecall5").textContent = `${m.single_track_recall5.toFixed(1)}%`;
    document.getElementById("gaugeAuroc").textContent = m.gate_auroc.toFixed(3);
    const dualEl = document.getElementById("gaugeDualHit");
    dualEl.textContent = `${m.dual_track_hit_rate.toFixed(1)}% (${m.dual_track_hits}/${m.n_positive})`;
    dualEl.style.color = m.dual_track_hit_rate < 40 ? PALETTE.coral : (m.dual_track_hit_rate < 85 ? PALETTE.amber : PALETTE.teal);
  }

  function drawRoutingTrap() {
    const env = prepareCanvas("canvasTrap");
    if (!env || !state.bundle) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const rt = state.bundle.routing_trap;
    const padL = 36, padR = 36, padT = 42, padB = 42;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    const xMin = -2.2, xMax = 2.4;
    const xOf = (v) => padL + ((v - xMin) / (xMax - xMin)) * plotW;

    const muPos = rt.pos_mean_base + state.shiftVal;
    const muNeg = rt.neg_mean_base + state.shiftVal;
    const sigma = rt.std;
    const thr = rt.threshold;
    const thrX = xOf(thr);

    const gauss = (x, mu) => Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));

    // Baseline & x-axis ticks
    ctx.strokeStyle = "rgba(29, 29, 31, 0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    ctx.save();
    ctx.font = `500 9.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.textAlign = "center";
    for (const tk of [-2, -1, 0, 1, 2]) {
      const tx = xOf(tk);
      ctx.fillText(tk.toFixed(1), tx, h - padB + 18);
    }
    ctx.restore();

    // Negative distribution (soft muted gray peak)
    const steps = 160;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    for (let i = 0; i <= steps; i++) {
      const xv = xMin + (xMax - xMin) * (i / steps);
      const yv = h - padB - gauss(xv, muNeg) * plotH * 0.72;
      ctx.lineTo(xOf(xv), yv);
    }
    ctx.lineTo(w - padR, h - padB);
    ctx.closePath();
    ctx.fillStyle = "rgba(134, 134, 139, 0.14)";
    ctx.fill();
    ctx.strokeStyle = "rgba(134, 134, 139, 0.55)";
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.restore();

    // Positive distribution split at threshold wall: Left = Rejected (Coral), Right = Accepted (Teal)
    const drawPosHalf = (clipX0, clipX1, fillCol, strokeCol) => {
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX0, 0, Math.max(0, clipX1 - clipX0), h);
      ctx.clip();

      ctx.beginPath();
      ctx.moveTo(padL, h - padB);
      for (let i = 0; i <= steps; i++) {
        const xv = xMin + (xMax - xMin) * (i / steps);
        const yv = h - padB - gauss(xv, muPos) * plotH * 0.86;
        ctx.lineTo(xOf(xv), yv);
      }
      ctx.lineTo(w - padR, h - padB);
      ctx.closePath();
      ctx.fillStyle = fillCol;
      ctx.fill();
      ctx.strokeStyle = strokeCol;
      ctx.lineWidth = 2.2;
      ctx.stroke();
      ctx.restore();
    };

    drawPosHalf(0, thrX, "rgba(230, 106, 78, 0.24)", PALETTE.coral);
    drawPosHalf(thrX, w, "rgba(43, 156, 148, 0.24)", PALETTE.teal);

    // Locked red threshold wall (target_fpr = 0.01)
    ctx.save();
    ctx.strokeStyle = PALETTE.coral;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(thrX, padT - 10);
    ctx.lineTo(thrX, h - padB);
    ctx.stroke();
    ctx.setLineDash([]);
    drawPill(ctx, `锁死阈值墙 ${thr.toFixed(3)} (FPR=1%)`, thrX, padT - 16, PALETTE.coral, "#ffffff");
    ctx.restore();

    // Dynamic callout over positive peak
    const m = getTrapMetricsAt(state.shiftVal);
    const peakX = clamp(xOf(muPos), padL + 74, w - padR - 74);
    const isCollapsed = m.dual_track_hit_rate < 40;
    drawPill(
      ctx,
      isCollapsed ? `塌方：仅 ${m.dual_track_hit_rate.toFixed(0)}% 通过` : `正样本 ${m.dual_track_hit_rate.toFixed(0)}% 通过`,
      peakX,
      h - padB - plotH * 0.90,
      isCollapsed ? "rgba(230, 106, 78, 0.95)" : "rgba(43, 156, 148, 0.94)",
      "#ffffff"
    );
  }

  // =========================================================================
  // 02 / Compositional Syllable LEGO x 4-Head OOV Arena
  // =========================================================================
  function renderOovButtons() {
    const container = document.getElementById("oovWordButtons");
    if (!container || !state.bundle) return;
    container.innerHTML = "";
    state.bundle.oov_arena.oov_words.forEach((w, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `soft-button${idx === state.activeOovIdx ? " is-on" : ""}`;
      btn.textContent = `${w.name}「${w.gloss}」`;
      btn.addEventListener("click", () => {
        state.autoOov = false;
        state.activeOovIdx = idx;
        renderOovButtons();
      });
      container.appendChild(btn);
    });
  }

  function setHeadMode(headKey, userInitiated = true) {
    if (userInitiated) state.autoOov = false;
    state.activeHeadKey = headKey;
    document.querySelectorAll("[data-head-key]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.headKey === headKey);
    });
    if (state.bundle) {
      const h = state.bundle.oov_arena.heads[headKey];
      document.getElementById("headDescText").textContent = h.note;
      const rEl = document.getElementById("oovRecallReadout");
      rEl.textContent = `${h.oov_recall1.toFixed(1)}%`;
      rEl.style.color = h.open_vocab ? PALETTE.teal : PALETTE.coral;
    }
  }

  function drawOovArena() {
    const env = prepareCanvas("canvasOov");
    if (!env || !state.bundle) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const arena = state.bundle.oov_arena;
    const sylById = new Map(arena.syllables.map(s => [s.id, s]));
    const activeOov = arena.oov_words[state.activeOovIdx];
    const head = arena.heads[state.activeHeadKey];

    const splitX = w * 0.52;

    // LEFT HALF: Compositional Syllable LEGO Provenance
    ctx.save();
    ctx.textAlign = "left";
    ctx.font = `600 10.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("SEEN 训练词（共享音节库）", 12, 20);
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText(`未见新词：${activeOov.name}「${activeOov.gloss}」`, 12, h - 78);
    ctx.restore();

    const blockW = Math.min(64, (splitX - 135) / 3);
    const blockH = 24;
    const seenPositions = new Map(); // key `${pos}_${sylId}` -> {x, y}

    arena.seen_words.forEach((word, rowIdx) => {
      const ry = 42 + rowIdx * 40;
      ctx.save();
      ctx.textAlign = "left";
      ctx.font = `500 11px ${CANVAS_SANS}`;
      ctx.fillStyle = PALETTE.inkSoft;
      ctx.textBaseline = "middle";
      ctx.fillText(`${word.gloss}`, 12, ry + blockH * 0.5);

      word.syllables.forEach((sid, pos) => {
        const syl = sylById.get(sid);
        const bx = 56 + pos * (blockW + 8);
        const isMatch = activeOov.syllables[pos] === sid;
        if (isMatch) seenPositions.set(`${pos}_${sid}`, { x: bx + blockW * 0.5, y: ry + blockH });

        drawRoundedRect(ctx, bx, ry, blockW, blockH, 7);
        ctx.fillStyle = isMatch ? syl.color : "rgba(29, 29, 31, 0.055)";
        ctx.fill();
        ctx.font = `600 10px ${CANVAS_MONO}`;
        ctx.fillStyle = isMatch ? "#ffffff" : PALETTE.faint;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(syl.code, bx + blockW * 0.5, ry + blockH * 0.5);
      });
      ctx.restore();
    });

    // OOV Word 3-block LEGO at bottom left + Bezier provenance threads
    const oovY = h - 56;
    activeOov.syllables.forEach((sid, pos) => {
      const syl = sylById.get(sid);
      const bx = 56 + pos * (blockW + 8);
      const src = seenPositions.get(`${pos}_${sid}`);
      if (src) {
        ctx.save();
        ctx.strokeStyle = syl.color;
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        const midY = (src.y + oovY) * 0.5;
        ctx.bezierCurveTo(src.x, midY, bx + blockW * 0.5, midY, bx + blockW * 0.5, oovY);
        ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      drawRoundedRect(ctx, bx, oovY, blockW, 30, 8);
      ctx.fillStyle = syl.color;
      ctx.fill();
      ctx.font = `600 10.5px ${CANVAS_MONO}`;
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(syl.label, bx + blockW * 0.5, oovY + 15);
      ctx.restore();
    });

    // RIGHT HALF: 4-Head Similarity Matrix + CosFace Temperature Annealing
    const matX = splitX + 34;
    const matY = 48;
    const matSize = Math.min(w - matX - 24, h - 130, 240);
    const cell = matSize / 4;

    ctx.save();
    ctx.font = `600 10.5px ${CANVAS_MONO}`;
    ctx.fillStyle = head.open_vocab ? PALETTE.blue : PALETTE.coral;
    ctx.fillText(`${head.name} · OOV 相似度矩阵 (${head.oov_recall1.toFixed(1)}%)`, matX, 20);
    ctx.restore();

    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const val = head.matrix[r][c];
        const cx = matX + c * cell;
        const cy = matY + r * cell;
        drawRoundedRect(ctx, cx + 2, cy + 2, cell - 4, cell - 4, 8);
        if (!head.open_vocab) {
          ctx.fillStyle = "rgba(29, 29, 31, 0.055)";
          ctx.fill();
        } else {
          const alpha = 0.08 + val * 0.82;
          ctx.fillStyle = r === c
            ? `rgba(0, 113, 227, ${alpha.toFixed(2)})`
            : `rgba(134, 134, 139, ${(alpha * 0.45).toFixed(2)})`;
          ctx.fill();
          ctx.font = `600 10.5px ${CANVAS_MONO}`;
          ctx.fillStyle = (r === c && val > 0.5) ? "#ffffff" : PALETTE.inkSoft;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(val.toFixed(2), cx + cell * 0.5, cy + cell * 0.5);
        }
      }
    }

    if (!head.open_vocab) {
      drawPill(ctx, "无锚点槽位 · 闭集 OOV 失效 (0%)", matX + matSize * 0.5, matY + matSize * 0.5, PALETTE.coral, "#ffffff");
    }

    // CosFace 3-sub-center temperature annealing indicator below matrix
    const annealP = 0.5 - 0.5 * Math.cos(state.clock * 1.1);
    const temp = 10.0 + annealP * 30.0;
    const tEl = document.getElementById("tempReadout");
    if (tEl) tEl.textContent = `${temp.toFixed(1)} (10 → 40)`;

    const subY = matY + matSize + 44;
    ctx.save();
    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText(`CosFace 3 子中心温度退火: T = ${temp.toFixed(1)}`, matX, subY - 22);
    for (let k = 0; k < 3; k++) {
      const sx = matX + 32 + k * 66;
      const rSoft = 18 - annealP * 9;
      ctx.fillStyle = `rgba(0, 113, 227, ${(0.14 + annealP * 0.16).toFixed(2)})`;
      ctx.beginPath();
      ctx.arc(sx, subY, rSoft, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = PALETTE.blue;
      ctx.beginPath();
      ctx.arc(sx, subY, 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // =========================================================================
  // 03 / Dilated Receptive Field x Speed-Invariant Attention Microscope
  // =========================================================================
  function drawTemporalMicroscope() {
    const env = prepareCanvas("canvasTemporal");
    if (!env) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const padL = 96, padR = 24;
    const maxW = w - padL - padR;
    // Active utterance visual span stretches inversely with speaking speed!
    const stretch = clamp(1.0 / state.speedVal, 0.48, 1.0);
    const spanW = maxW * stretch;

    const topY = 26, topH = h * 0.27;
    const midY = topY + topH + 34, midH = h * 0.24;
    const botY = midY + midH + 34, botH = h - botY - 24;

    // Layer labels
    ctx.save();
    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText(state.attnMode === "qformer" ? "Q-Former" : "Pairwise", 6, topY + 18);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("跨模态注意力", 6, topY + 34);

    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.teal;
    ctx.fillText("PE [0, 1]", 6, midY + 18);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("位置编码对比", 6, midY + 34);

    ctx.font = `600 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.violet;
    ctx.fillText("d=(1,2,4)", 6, botY + 18);
    ctx.fillStyle = PALETTE.faint;
    ctx.font = `500 9.5px ${CANVAS_SANS}`;
    ctx.fillText("300ms 感受野", 6, botY + 34);
    ctx.restore();

    // TOP: Attention Heatmap (4 Q-Former queries vs 3 Pairwise syllable tokens)
    const nRows = state.attnMode === "qformer" ? 4 : 3;
    const nCols = 24;
    const cw = spanW / nCols;
    const ch = topH / nRows;
    for (let r = 0; r < nRows; r++) {
      const center = (r + 0.5) / nRows;
      for (let c = 0; c < nCols; c++) {
        const u = (c + 0.5) / nCols;
        const wgt = Math.exp(-0.5 * Math.pow((u - center) / 0.14, 2));
        ctx.fillStyle = state.attnMode === "qformer"
          ? `rgba(0, 113, 227, ${(0.06 + wgt * 0.78).toFixed(3)})`
          : `rgba(43, 156, 148, ${(0.06 + wgt * 0.82).toFixed(3)})`;
        drawRoundedRect(ctx, padL + c * cw + 1, topY + r * ch + 1, Math.max(2, cw - 2), Math.max(4, ch - 2), 3);
        ctx.fill();
      }
    }

    // MIDDLE: Absolute PE (drifts with speed!) vs Normalized Relative PE (locked to [0, 1]!)
    const midCenter = midY + midH * 0.5;
    ctx.strokeStyle = "rgba(29, 29, 31, 0.06)";
    ctx.beginPath();
    ctx.moveTo(padL, midCenter);
    ctx.lineTo(padL + maxW, midCenter);
    ctx.stroke();

    // Absolute PE (coral dashed — frequency stays fixed in raw ms, so end phase shifts wildly!)
    ctx.save();
    ctx.strokeStyle = "rgba(230, 106, 78, 0.65)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const px = padL + (i / 100) * spanW;
      const py = midCenter - Math.sin((px - padL) * 0.055) * (midH * 0.42);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Normalized Relative PE (teal solid — always exactly 2.5 cycles across [0, 1]!)
    ctx.strokeStyle = PALETTE.teal;
    ctx.lineWidth = 2.3;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const r = i / 100;
      const px = padL + r * spanW;
      const py = midCenter - Math.sin(r * Math.PI * 4.0) * (midH * 0.42);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();

    // BOTTOM: Dilated Receptive Field Pyramid (d=1, d=2, d=4 covering 15 frames = 300ms)
    const nFrames = 21;
    const fStep = spanW / (nFrames - 1);
    const focusF = Math.round(10 + 3 * Math.sin(state.clock * 1.1));
    const yL3 = botY + 12;
    const yL2 = botY + botH * 0.48;
    const yL1 = botY + botH - 10;

    // Draw frame dots
    for (let f = 0; f < nFrames; f++) {
      const fx = padL + f * fStep;
      const inRf = Math.abs(f - focusF) <= 7;
      ctx.fillStyle = inRf ? PALETTE.blue : "rgba(29, 29, 31, 0.14)";
      ctx.beginPath();
      ctx.arc(fx, yL1, inRf ? 3.2 : 2.0, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pyramid connections: d=4 -> d=2 -> d=1
    const topX = padL + focusF * fStep;
    const d4Nodes = [focusF - 4, focusF, focusF + 4].filter(x => x >= 0 && x < nFrames);
    ctx.strokeStyle = "rgba(123, 107, 176, 0.50)";
    ctx.lineWidth = 1.6;
    for (const n2 of d4Nodes) {
      const x2 = padL + n2 * fStep;
      ctx.beginPath(); ctx.moveTo(topX, yL3); ctx.lineTo(x2, yL2); ctx.stroke();
      for (const off of [-2, 0, 2]) {
        const n1 = n2 + off;
        if (n1 >= 0 && n1 < nFrames) {
          ctx.strokeStyle = "rgba(0, 113, 227, 0.36)";
          ctx.beginPath(); ctx.moveTo(x2, yL2); ctx.lineTo(padL + n1 * fStep, yL1); ctx.stroke();
        }
      }
      ctx.fillStyle = PALETTE.violet;
      ctx.beginPath(); ctx.arc(x2, yL2, 3.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = PALETTE.blue;
    ctx.beginPath(); ctx.arc(topX, yL3, 4.6, 0, Math.PI * 2); ctx.fill();
  }

  // =========================================================================
  // 04 / Class-Conditional Domain Alignment x FSQ Discrete Lattice
  // =========================================================================
  function setDomainMode(mode, userInitiated = true) {
    if (userInitiated) state.autoDomain = false;
    state.domainMode = mode;
    document.querySelectorAll("[data-domain-mode]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.domainMode === mode);
    });
  }

  function drawShape(ctx, cls, x, y, r, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    if (cls === 0) {
      ctx.arc(x, y, r, 0, Math.PI * 2);
    } else if (cls === 1) {
      ctx.moveTo(x, y - r * 1.15);
      ctx.lineTo(x + r * 1.1, y + r * 0.9);
      ctx.lineTo(x - r * 1.1, y + r * 0.9);
      ctx.closePath();
    } else {
      ctx.rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8);
    }
    ctx.fill();
  }

  function drawDomainFsq() {
    const env = prepareCanvas("canvasDomain");
    if (!env || !state.bundle) return;
    const { ctx, w, h } = env;
    ctx.clearRect(0, 0, w, h);

    const df = state.bundle.domain_fsq;
    const sphereCX = w * 0.34;
    const sphereCY = h * 0.50;
    const sphereR = Math.min(w * 0.28, h * 0.42);
    const domColors = [PALETTE.blue, PALETTE.teal, PALETTE.coral];

    // Unit sphere subtle guide circle
    ctx.strokeStyle = "rgba(29, 29, 31, 0.08)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(sphereCX, sphereCY, sphereR, 0, Math.PI * 2);
    ctx.stroke();

    // Optional FSQ [-1, 1] discrete lattice overlay
    if (state.useFsq) {
      ctx.save();
      ctx.strokeStyle = "rgba(0, 113, 227, 0.11)";
      ctx.lineWidth = 1;
      for (const g of df.fsq_grid_2d) {
        const gx = sphereCX + g * sphereR;
        const gy = sphereCY - g * sphereR;
        ctx.beginPath(); ctx.moveTo(gx, sphereCY - sphereR); ctx.lineTo(gx, sphereCY + sphereR); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(sphereCX - sphereR, gy); ctx.lineTo(sphereCX + sphereR, gy); ctx.stroke();
      }
      ctx.restore();
    }

    // Smoothly animate every point toward its active mode (+ FSQ snap if enabled)
    if (state.pointAnim.length !== df.points.length) {
      state.pointAnim = df.points.map(p => ({ x: p.conditional[0], y: p.conditional[1] }));
    }

    const fsqKey = state.domainMode === "none" ? "fsq_none" : (state.domainMode === "marginal" ? "fsq_marg" : "fsq_cond");
    df.points.forEach((p, idx) => {
      const target = state.useFsq ? p[fsqKey] : p[state.domainMode];
      const cur = state.pointAnim[idx];
      cur.x += (target[0] - cur.x) * 0.14;
      cur.y += (target[1] - cur.y) * 0.14;

      const px = sphereCX + cur.x * sphereR;
      const py = sphereCY - cur.y * sphereR;

      if (state.useFsq) {
        const orig = p[state.domainMode];
        const ox = sphereCX + orig[0] * sphereR;
        const oy = sphereCY - orig[1] * sphereR;
        ctx.strokeStyle = "rgba(29, 29, 31, 0.12)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(px, py); ctx.stroke();
      }

      drawShape(ctx, p.cls, px, py, 5.2, domColors[p.dom]);
    });

    // Mode pill above sphere
    const modeLabels = {
      none: "无域正则 · 按颜色分块（跨域错配）",
      marginal: "边缘对齐 · 颜色拉齐但形状坍缩！",
      conditional: "类条件对齐 · 同形跨色重合 ✓"
    };
    drawPill(
      ctx,
      modeLabels[state.domainMode],
      sphereCX,
      20,
      state.domainMode === "marginal" ? PALETTE.coral : (state.domainMode === "conditional" ? PALETTE.teal : "rgba(29, 29, 31, 0.08)"),
      state.domainMode === "none" ? PALETTE.inkSoft : "#ffffff"
    );

    // RIGHT PANEL: Live dann_alpha_schedule S-curve
    const rx0 = w * 0.68, rx1 = w - 24;
    const ry0 = 64, ry1 = h - 54;
    const rw = rx1 - rx0, rh = ry1 - ry0;

    ctx.save();
    ctx.font = `600 10.5px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.inkSoft;
    ctx.fillText("dann_alpha_schedule (γ = 10.0)", rx0, 34);

    ctx.strokeStyle = "rgba(29, 29, 31, 0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(rx0, ry0); ctx.lineTo(rx0, ry1); ctx.lineTo(rx1, ry1);
    ctx.stroke();

    ctx.strokeStyle = PALETTE.blue;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    df.dann_curve.forEach((v, i) => {
      const cx = rx0 + (i / (df.dann_curve.length - 1)) * rw;
      const cy = ry1 - v * rh;
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    // Live animated training progress dot along S-curve
    const prog = (state.clock * 0.22) % 1.0;
    const alphaVal = 2.0 / (1.0 + Math.exp(-10.0 * prog)) - 1.0;
    const dx = rx0 + prog * rw;
    const dy = ry1 - alphaVal * rh;
    ctx.fillStyle = "rgba(0, 113, 227, 0.20)";
    ctx.beginPath(); ctx.arc(dx, dy, 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = PALETTE.blue;
    ctx.beginPath(); ctx.arc(dx, dy, 4, 0, Math.PI * 2); ctx.fill();

    ctx.font = `500 10px ${CANVAS_MONO}`;
    ctx.fillStyle = PALETTE.blue;
    ctx.fillText(`α = ${alphaVal.toFixed(2)}`, Math.min(rx1 - 56, dx + 8), Math.max(ry0 + 12, dy - 10));
    ctx.fillStyle = PALETTE.faint;
    ctx.fillText("训练进度 p ∈ [0, 1]", rx0 + rw * 0.32, ry1 + 22);
    ctx.restore();
  }

  // =========================================================================
  // Event Wiring & 60 FPS Loop
  // =========================================================================
  function bindEvents() {
    document.getElementById("sliderShift")?.addEventListener("input", (e) => {
      state.autoShift = false;
      state.shiftVal = parseFloat(e.target.value);
      updateTrapUI();
    });

    document.querySelectorAll("[data-head-key]").forEach((btn) => {
      btn.addEventListener("click", () => {
        setHeadMode(btn.dataset.headKey, true);
        showToast(`已切换对齐头：${state.bundle.oov_arena.heads[btn.dataset.headKey].name}`);
      });
    });

    document.getElementById("sliderSpeed")?.addEventListener("input", (e) => {
      state.autoSpeed = false;
      state.speedVal = parseFloat(e.target.value);
      document.getElementById("speedReadout").textContent = `${state.speedVal.toFixed(2)}x · 手动调节`;
    });

    document.querySelectorAll("[data-attn-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.attnMode = btn.dataset.attnMode;
        document.querySelectorAll("[data-attn-mode]").forEach(b => b.classList.toggle("is-active", b === btn));
        document.getElementById("attnBadge").textContent = state.attnMode === "qformer"
          ? "4 Latent Queries 自动分工"
          : "零死参数 · 单次文本编码复用";
      });
    });

    document.querySelectorAll("[data-domain-mode]").forEach((btn) => {
      btn.addEventListener("click", () => setDomainMode(btn.dataset.domainMode, true));
    });

    const fsqBtn = document.getElementById("btnToggleFsq");
    fsqBtn?.addEventListener("click", () => {
      state.useFsq = !state.useFsq;
      fsqBtn.classList.toggle("is-on", state.useFsq);
      fsqBtn.textContent = `FSQ 离散晶格：${state.useFsq ? "开" : "关"}`;
    });
  }

  let lastTime = performance.now();
  function loop(now) {
    const dt = Math.min(50, Math.max(0, now - lastTime));
    lastTime = now;
    state.clock += dt * 0.001;

    updateScrollChoreography(dt);
    stepAutonomous(dt);

    if (state.bundle) {
      drawRoutingTrap();
      drawOovArena();
      drawTemporalMicroscope();
      drawDomainFsq();
    }
    requestAnimationFrame(loop);
  }

  async function init() {
    bindEvents();
    try {
      const resp = await fetch("data/lab_bundle.json");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      state.bundle = await resp.json();
      renderOovButtons();
      setHeadMode("text_anchored", false);
      updateTrapUI();
      requestAnimationFrame(loop);
    } catch (err) {
      showToast(`数据加载失败：${err.message}`);
    }
  }

  window.addEventListener("load", init);
})();
