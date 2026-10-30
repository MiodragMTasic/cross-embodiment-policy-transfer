// Study notebook: Miodrag's handwritten notes as a double ring steno pad. A page turns the way the
// pad was made: it lifts off the stack, goes over the rings and hangs behind the back board.
// Pages come from window.NOTES (assets/notes/notes.js, written by _build/add_notes.sh): photos cut
// out of their background, all framed on the same 15.2 x 22.9 cm canvas, the first one the cover.
// Plain JS and CSS 3D transforms, no library; styles in assets/notebook.css.
//
// Click or tap a page for the next one, the rings for the one before; arrow keys when focused; on
// a desktop the page can also be dragged upward. Touch never drags, so the page always scrolls.
// "Read this page" opens the page's typed transcript under the pad (pages[i].transcript).
(function () {
  "use strict";
  var root = document.querySelector(".steno");
  var data = window.NOTES;
  if (!root || !data || !data.pages || !data.pages.length) return;

  var pages = data.pages;
  var N = pages.length;
  var hasCover = pages[0].kind === "cover";
  var book = root.querySelector(".steno-book");
  var pad = root.querySelector(".steno-pad");
  var prevBtn = root.querySelector(".steno-prev");
  var nextBtn = root.querySelector(".steno-next");
  var count = root.querySelector(".steno-count");
  var status = root.querySelector(".steno-status");
  var readBtn = root.querySelector(".steno-read");
  var panel = root.querySelector(".steno-text");
  var textBody = root.querySelector(".steno-text-body");
  var reading = false; // the transcript stays open across page turns

  var RAD = Math.PI / 180;
  var HINGE = 0.45 / data.heightCm; // the wire's axis, as a fraction of the page height
  var ORIGIN = 0.4; // perspective-origin y in notebook.css
  var GONE = 270; // a turned page hangs behind the pad, edge-on to the reader
  var TURN_MS = 760; // a whole turn; partial turns (after a drag) take their share
  var motion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;

  var cur = 0; // the sheet on top
  var edgeOn = 83; // angle at which a sheet is edge-on to the eye; measured before each turn
  var geo = { h: 1, p: 2400 }; // pad height and perspective, in px
  var raf = 0;

  // ---------- building the pad ----------
  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }
  function img(cls, parent) {
    var i = el("img", cls, parent);
    i.width = data.width;
    i.height = data.height;
    i.decoding = "async";
    i.draggable = false;
    i.alt = "";
    return i;
  }

  el("div", "steno-board", pad);
  var edges = el("div", "steno-edges", pad); // the rest of the stack under the top sheet
  edges.style.zIndex = 1;
  var cast = el("div", "steno-cast", pad);

  var sheets = pages.map(function (p, i) {
    var s = {
      el: el("div", "steno-sheet", pad),
      angle: 0,
      anim: null,
      held: false,
      src: p.src,
    };
    s.face = img("steno-face", s.el);
    s.face.alt = p.alt || "";
    s.face.loading = "lazy";
    s.back = el("div", "steno-back", s.el);
    s.back.setAttribute("aria-hidden", "true");
    if (p.kind === "cover") {
      // the inside of the cover is bare kraft, cut to the cover's shape
      el("div", "steno-inside", s.back).style.top =
        (p.tearTop || 0) * 100 + "%";
    } else {
      s.paper = img("steno-paper", s.back);
      s.lines = el("div", "steno-lines", s.back);
    }
    s.shade = img("steno-shade", s.el);
    s.shade.setAttribute("aria-hidden", "true");
    // the turned-up bottom corner that says the page turns; drawn only on the top sheet
    el("div", "steno-ear", s.el).setAttribute("aria-hidden", "true");
    if (p.kind === "cover") s.el.classList.add("is-cover");
    return s;
  });

  // twin-loop wire through 16 slots, drawn in centimetres of the page; the top edge is y = 0
  (function rings() {
    var n = 16,
      first = 0.9,
      W = data.widthCm || 15.2;
    var pitch = (W - 2 * first) / (n - 1);
    var slots = "",
      shadow = "",
      wire = "",
      hi = "";
    for (var k = 0; k < n; k++) {
      var c = first + k * pitch;
      slots +=
        '<rect class="slot" x="' +
        (c - 0.21).toFixed(3) +
        '" y="0.3" width="0.42" height="0.3" rx="0.035"/>' +
        '<rect class="slot-lip" x="' +
        (c - 0.19).toFixed(3) +
        '" y="0.565" width="0.38" height="0.025"/>';
      [-0.12, 0.07].forEach(function (dx) {
        var x = c + dx;
        var d =
          "M" +
          x.toFixed(3) +
          " 0.57V-0.045A0.075 0.075 0 0 1 " +
          (x + 0.15).toFixed(3) +
          " -0.045V0.03";
        wire += '<path d="' + d + '"/>';
        shadow += '<path d="' + d + '" transform="translate(0.025 0.035)"/>';
        hi += '<path d="M' + (x - 0.013).toFixed(3) + ' 0.5V-0.04"/>';
      });
    }
    pad.insertAdjacentHTML(
      "beforeend",
      '<svg class="steno-rings" viewBox="0 -0.25 ' +
        W +
        ' 0.9" aria-hidden="true" focusable="false">' +
        '<defs><linearGradient id="steno-slot" x1="0" y1="0" x2="0" y2="1">' +
        '<stop offset="0" stop-color="#2a1d14"/>' +
        '<stop offset="0.45" stop-color="#7d6d5f"/>' +
        '<stop offset="1" stop-color="#ddd7cc"/>' +
        "</linearGradient></defs>" +
        slots +
        '<g class="wire-shadow">' +
        shadow +
        "</g>" +
        '<g class="wire">' +
        wire +
        "</g>" +
        '<g class="wire-hi">' +
        hi +
        "</g></svg>",
    );
  })();
  var top = el("div", "steno-top", pad);
  top.setAttribute("aria-hidden", "true");
  top.title = "Previous page";

  function load(i, both) {
    var s = sheets[i];
    if (!s) return;
    if (!s.face.getAttribute("src")) s.face.src = s.src;
    if (both && !s.shade.getAttribute("src")) {
      s.shade.src = s.src;
      if (s.paper) {
        s.paper.src = s.src;
        // the ruling is cut to the sheet's shape with the photo as a mask; browsers refuse masks
        // from file://, so a page opened from disk turns plain backs instead
        if (/^https?:$/.test(location.protocol)) {
          var mask = 'url("' + s.src + '")';
          s.lines.style.webkitMaskImage = mask;
          s.lines.style.maskImage = mask;
        } else s.lines.hidden = true;
      }
    }
  }

  // ---------- drawing ----------
  function measure() {
    var h = pad.offsetHeight || 1;
    var p = parseFloat(getComputedStyle(pad).perspective) || 2400;
    geo = { h: h, p: p };
    edgeOn = Math.atan2(p, (ORIGIN - HINGE) * h) / RAD;
  }

  function pose(s, i) {
    var a = s.angle;
    s.el.style.transform = a ? "rotateX(" + a.toFixed(2) + "deg)" : "";
    var front = a < edgeOn || a > edgeOn + 180;
    s.face.style.opacity = front ? "" : "0";
    s.back.style.opacity = front ? "" : "1";
    var sin = Math.sin(a * RAD);
    s.shade.style.opacity = a ? (0.34 * sin * sin).toFixed(3) : "";
    // above the stack in book order until it passes upright, then behind the back board
    s.el.style.zIndex = a > 180 ? i - 200 : 2 * (100 + N - i);
  }

  // the shadow of the lowest sheet that is lifting, thrown on the sheet beneath it: it ends in a
  // soft edge just past where the lifted sheet's free edge appears, so it sweeps up the page
  function shadow() {
    for (var i = N - 2; i >= 0; i--) {
      var a = sheets[i].angle;
      if (a > 0 && a < 125) {
        var h = geo.h,
          yh = HINGE * h,
          oy = ORIGIN * h;
        var edge =
          oy +
          (yh + h * Math.cos(a * RAD) - oy) *
            (geo.p / (geo.p - h * Math.sin(a * RAD)));
        var reach = Math.max(0, Math.min(1, (edge - yh) / (0.8 * (h - yh))));
        var o = Math.min(1, a / 12) * Math.max(0, Math.min(1, (125 - a) / 45));
        cast.style.transform = "scaleY(" + reach.toFixed(4) + ")";
        cast.style.opacity = o.toFixed(3);
        cast.style.zIndex = 2 * (100 + N - i) - 1;
        return;
      }
    }
    cast.style.opacity = "0";
  }

  function render() {
    pad.style.setProperty("--left", N - 1 - cur);
    pad.classList.toggle("is-last", cur >= N - 1);
    edges.hidden = cur >= N - 1;
    sheets.forEach(function (s, i) {
      var live = s.anim || s.held || s.fading;
      if (!live) s.angle = i < cur ? GONE : 0;
      pose(s, i);
      s.el.classList.toggle("is-top", i === cur && !live && cur < N - 1);
      s.el.style.visibility =
        live || i === cur || i === cur + 1 ? "" : "hidden";
      if (i === cur) s.el.removeAttribute("aria-hidden");
      else s.el.setAttribute("aria-hidden", "true");
    });
    // the cover photo stops below its ring slots; while it is shut, nothing under it shows above it
    if (hasCover && N > 1) {
      var shut = cur === 0 && !sheets[0].anim && !sheets[0].held;
      sheets[1].el.style.clipPath = shut
        ? "inset(" + ((pages[0].tearTop || 0) * 100 + 0.4) + "% 0 0 0)"
        : "";
    }
    shadow();
  }

  // ---------- turning ----------
  // paper: a slow lift, quick over the top, a soft fall behind the pad
  function ease(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function turn(i, to) {
    var s = sheets[i];
    load(i, true);
    measure();
    s.held = false;
    var from = s.angle;
    if (from === to) return render();
    s.anim = {
      from: from,
      to: to,
      t0: performance.now(),
      ms: Math.max(180, (TURN_MS * Math.abs(to - from)) / GONE),
    };
    s.el.classList.add("is-turning");
    render();
    if (!raf) raf = requestAnimationFrame(frame);
  }

  function frame(now) {
    raf = 0;
    var busy = false;
    sheets.forEach(function (s) {
      if (!s.anim) return;
      var t = Math.min(1, Math.max(0, (now - s.anim.t0) / s.anim.ms));
      s.angle = s.anim.from + (s.anim.to - s.anim.from) * ease(t);
      if (t >= 1) {
        s.angle = s.anim.to;
        s.anim = null;
        s.el.classList.remove("is-turning");
      } else busy = true;
    });
    render();
    if (busy) raf = requestAnimationFrame(frame);
  }

  // reduced motion: the top sheet fades away, or fades back in
  function fade(i, show) {
    var s = sheets[i];
    load(i);
    s.fading = true;
    s.angle = 0;
    render();
    var a = s.el.animate(
      [{ opacity: show ? 0 : 1 }, { opacity: show ? 1 : 0 }],
      { duration: 240, easing: "ease" },
    );
    a.onfinish = a.oncancel = function () {
      s.fading = false;
      render();
    };
  }

  function step(dir) {
    var can = dir > 0 ? cur < N - 1 : cur > 0;
    stopPeek(can && dir > 0); // a turn forward carries on from the lifted page
    if (!can) return false;
    var i = dir > 0 ? cur : cur - 1;
    cur += dir;
    if (motion && motion.matches) fade(i, dir < 0);
    else turn(i, dir > 0 ? GONE : 0);
    update(true);
    turned();
    return true;
  }

  // ---------- first visit: a line on the cover, and one small lift of the page ----------
  // Both are once per tab session. The line goes for good at the first turn, however it is made.
  function remembered(key) {
    try {
      return window.sessionStorage.getItem(key) === "1";
    } catch (e) {
      return false;
    }
  }
  function remember(key) {
    try {
      window.sessionStorage.setItem(key, "1");
    } catch (e) {}
  }

  var hint = null;
  if (N > 1 && !remembered("steno-turned")) {
    hint = el("div", "steno-hint", pad);
    hint.setAttribute("aria-hidden", "true");
    var fine =
      window.matchMedia &&
      window.matchMedia("(hover: hover) and (pointer: fine)").matches;
    hint.innerHTML =
      '<svg viewBox="0 0 24 24"><path d="M5 15.5c1.4-5.2 6.2-8.3 11.3-7.1"/>' +
      '<path d="M13.2 5.2l3.3 3.2-3.6 2.8"/></svg>' +
      (fine ? "Click the page to turn it" : "Tap the page to turn it");
  }

  function turned() {
    remember("steno-turned");
    if (!hint) return;
    var h = hint;
    hint = null;
    h.classList.add("is-gone");
    setTimeout(
      function () {
        h.remove();
      },
      motion && motion.matches ? 0 : 200,
    );
  }

  // the top sheet lifts toward the reader and settles back, the first time the pad is in view
  var peeking = null;
  var PEEK = { deg: 30, up: 560, hold: 160, down: 720 };
  function peek() {
    var s = sheets[cur];
    if (
      peeking ||
      remembered("steno-peeked") ||
      remembered("steno-turned") ||
      (motion && motion.matches) ||
      cur >= N - 1 ||
      s.anim ||
      s.held ||
      book.closest("[inert]")
    )
      return;
    remember("steno-peeked");
    load(cur, true);
    measure();
    var i = cur,
      t0 = performance.now();
    peeking = { i: i };
    s.held = true;
    s.el.classList.add("is-turning");
    var token = peeking;
    requestAnimationFrame(function tick(now) {
      if (peeking !== token) return;
      var t = now - t0,
        a;
      if (t < PEEK.up) a = 1 - Math.pow(1 - t / PEEK.up, 3);
      else if (t < PEEK.up + PEEK.hold) a = 1;
      else if (t < PEEK.up + PEEK.hold + PEEK.down)
        a = 1 - ease((t - PEEK.up - PEEK.hold) / PEEK.down);
      else {
        peeking = null;
        s.held = false;
        s.angle = 0;
        s.el.classList.remove("is-turning");
        render();
        return;
      }
      s.angle = PEEK.deg * a;
      render();
      requestAnimationFrame(tick);
    });
  }
  // any real turn or drag takes over from the lift where it is; otherwise the page settles back
  function stopPeek(handOver) {
    if (!peeking) return;
    var i = peeking.i;
    peeking = null;
    sheets[i].held = false;
    if (!handOver) turn(i, 0);
  }

  // Home / End riffle through the pages in between
  var riffle = 0;
  function jump(target) {
    clearTimeout(riffle);
    if (target === cur) return false;
    var dir = target > cur ? 1 : -1;
    (function tick() {
      if (cur !== target && step(dir))
        riffle = setTimeout(tick, motion && motion.matches ? 0 : 70);
    })();
    return true;
  }

  function name(i, spoken) {
    if (hasCover && i === 0) return "Cover";
    var n = hasCover ? i : i + 1,
      of = hasCover ? N - 1 : N;
    return spoken ? "Page " + n + " of " + of : n + " / " + of;
  }

  function update(announce) {
    count.textContent = name(cur);
    if (announce) status.textContent = name(cur, true);
    prevBtn.setAttribute("aria-disabled", String(cur <= 0));
    nextBtn.setAttribute("aria-disabled", String(cur >= N - 1));
    top.hidden = cur <= 0;
    for (var i = cur - 1; i <= cur + 2; i++) load(i);
    showText();
  }

  // ---------- transcript ----------
  // one div per written line; a wrapped line hangs under its tree branch, or two places in from
  // where it starts, so each written line stays one block. Marks (website/_build/
  // NOTEBOOK_TRANSCRIPT.md has the key): ~~word~~ struck; ^{word} / _{word} written above /
  // below the line; the reader's notes ([?], [word?], [diagram: ...], never code like [0.50 or
  // ["key") a shade lighter.
  var MARKS = /(~~[^~]+~~|[\^_]\{[^}]*\}|\[[a-z?][^\[\]"]*\])/i;
  function inline(parent, text) {
    text.split(MARKS).forEach(function (part, k) {
      var c = part.charAt(0);
      if (!(k % 2)) {
        if (part) parent.appendChild(document.createTextNode(part));
      } else if (c === "~") {
        el("span", "steno-sr", parent).textContent = "(crossed out) ";
        el("s", "", parent).textContent = part.slice(2, -2);
      } else if (c === "^" || c === "_") {
        inline(el(c === "^" ? "sup" : "sub", "", parent), part.slice(2, -1));
      } else el("span", "steno-mark", parent).textContent = part;
    });
  }
  function fill(text) {
    textBody.textContent = "";
    text.split("\n").forEach(function (line) {
      var d = el("div", "", textBody);
      var m = /^[ │]*([├└] )?/.exec(line);
      var hang = m[0].length + (m[1] ? 0 : 2);
      d.style.paddingLeft = hang + "ch";
      d.style.textIndent = -hang + "ch";
      inline(d, line);
      if (!line) d.innerHTML = "&nbsp;";
    });
  }

  function showText() {
    if (!readBtn || !panel || !textBody) return;
    var text = pages[cur].transcript || "";
    var open = reading && !!text;
    readBtn.setAttribute("aria-disabled", String(!text));
    readBtn.setAttribute("aria-expanded", String(open));
    if (open && panel.getAttribute("data-page") !== String(cur)) {
      fill(text);
      panel.setAttribute("data-page", cur);
    }
    panel.hidden = !open;
  }

  // ---------- input ----------
  function onRings(y) {
    var r = pad.getBoundingClientRect();
    return y < r.top + 0.065 * r.height;
  }

  prevBtn.addEventListener("click", function () {
    step(-1);
  });
  nextBtn.addEventListener("click", function () {
    step(1);
  });
  if (readBtn) {
    readBtn.hidden = !pages.some(function (p) {
      return p.transcript;
    });
    readBtn.addEventListener("click", function () {
      if (readBtn.getAttribute("aria-disabled") === "true") return;
      reading = !reading;
      showText();
    });
  }

  var drag = null,
    swallowClick = false;
  book.addEventListener("click", function (e) {
    if (swallowClick) {
      swallowClick = false;
      return;
    }
    step(onRings(e.clientY) ? -1 : 1);
  });

  book.addEventListener("keydown", function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var k = e.key,
      moved = false;
    if (k === "ArrowRight" || k === "ArrowDown") moved = step(1);
    else if (k === "ArrowLeft" || k === "ArrowUp") moved = step(-1);
    else if (k === "Home") moved = jump(0);
    else if (k === "End") moved = jump(N - 1);
    if (moved) e.preventDefault();
  });

  // mouse or pen: drag the page up and over; let go past the top, or flick, to finish the turn
  book.addEventListener("pointerdown", function (e) {
    if (
      e.pointerType === "touch" ||
      e.button !== 0 ||
      cur >= N - 1 ||
      onRings(e.clientY)
    )
      return;
    drag = {
      id: e.pointerId,
      y0: e.clientY,
      i: cur,
      on: false,
      v: 0,
      y: e.clientY,
      t: e.timeStamp,
    };
  });
  book.addEventListener("pointermove", function (e) {
    if (!drag || e.pointerId !== drag.id) return;
    var s = sheets[drag.i];
    if (!drag.on) {
      if (drag.y0 - e.clientY < 6) return;
      drag.on = true;
      book.setPointerCapture(e.pointerId);
      book.classList.add("is-dragging");
      var r = pad.getBoundingClientRect();
      drag.hinge = r.top + HINGE * r.height;
      drag.r = Math.max(40, drag.y0 - drag.hinge);
      stopPeek(true);
      s.anim = null;
      s.held = true;
      load(drag.i, true);
      measure();
      s.el.classList.add("is-turning");
    }
    var c = Math.max(-1, Math.min(1, (e.clientY - drag.hinge) / drag.r));
    s.angle = Math.acos(c) / RAD;
    var dt = e.timeStamp - drag.t;
    if (dt > 0) drag.v = 0.7 * drag.v + 0.3 * ((drag.y - e.clientY) / dt);
    drag.y = e.clientY;
    drag.t = e.timeStamp;
    render();
  });
  function release(e) {
    if (!drag || e.pointerId !== drag.id) return;
    var d = drag;
    drag = null;
    if (!d.on) return;
    swallowClick = e.type === "pointerup";
    setTimeout(function () {
      swallowClick = false;
    }, 0);
    book.classList.remove("is-dragging");
    var s = sheets[d.i];
    s.held = false;
    if (e.type === "pointerup" && d.i === cur && (s.angle > 75 || d.v > 0.5)) {
      cur++;
      update(true);
      turn(d.i, GONE);
      turned();
    } else turn(d.i, 0);
  }
  book.addEventListener("pointerup", release);
  book.addEventListener("pointercancel", release);

  root.hidden = false;
  update(false);
  render();

  // the lift waits until most of the pad is on screen (or most of the screen, for a pad taller
  // than it), then a beat for the scroll to settle
  if (
    "IntersectionObserver" in window &&
    !(motion && motion.matches) &&
    !remembered("steno-peeked") &&
    !remembered("steno-turned")
  ) {
    var steps = [];
    for (var k = 0; k <= 20; k++) steps.push(k / 20);
    var io = new IntersectionObserver(
      function (entries) {
        var e = entries[entries.length - 1];
        var need =
          0.6 * Math.min(e.boundingClientRect.height, window.innerHeight);
        if (!e.isIntersecting || e.intersectionRect.height < need) return;
        if (book.closest("[inert]")) return;
        io.disconnect();
        setTimeout(peek, 450);
      },
      { threshold: steps },
    );
    io.observe(pad);
  }
})();
