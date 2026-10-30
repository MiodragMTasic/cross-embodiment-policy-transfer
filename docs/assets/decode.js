// Byline name decode, ported to vanilla JS from the DecodeText component on Miodrag's miodragdev site
// (github.com/MiodragMTasic/miodrag-architect-website, src/components/DecodeText.tsx), used as
// trigger="mount" with redecodeOnHover. Same charset, hues, timing, hash and keyframes: each glyph
// scrambles through POOL in colour, then locks into the real character left to right in about a
// second, and locked glyphs keep rare colour pops (.decode-lock / .chroma-char in site.css).
// Once a decode has finished, hovering the title runs it again; so does a single tap on a touch
// screen, or keyboard focus if the title is ever focusable. The title's text stays in the HTML
// for no-JS and reduced motion, and screen readers get it from aria-label.
(function () {
  var el = document.querySelector("[data-decode]");
  if (!el) return;
  var reduce =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)");
  if (reduce && reduce.matches) return;

  var POOL = "01<>/#%+=*$&ABCDEF0123456789▒░";
  // his HUES #6eb7f7 #e08cb6 #c9a24d #5fc998, darkened in site.css to 3:1 on this white page
  var HUES = [
    "var(--decode-1)",
    "var(--decode-2)",
    "var(--decode-3)",
    "var(--decode-4)",
  ];
  var text = el.textContent.trim().replace(/\s+/g, " ");
  var chars = Array.from(text);
  var n = chars.length;
  var slots = []; // one per non-space character: { i, ch, word, node, glyph, locked }

  el.setAttribute("role", "text");
  el.setAttribute("aria-label", text);
  el.textContent = "";
  var word = null;
  chars.forEach(function (ch, i) {
    if (ch === " ") {
      var space = document.createElement("span");
      space.setAttribute("aria-hidden", "true");
      space.textContent = " ";
      el.appendChild(space);
      word = null;
      return;
    }
    if (!word) {
      // keeps each word on one line; per-glyph inline-blocks would otherwise break anywhere
      word = document.createElement("span");
      word.className = "decode-word";
      word.setAttribute("aria-hidden", "true");
      el.appendChild(word);
    }
    var slot = { i: i, ch: ch, word: word, node: null, glyph: null };
    slot.node = document.createElement("span");
    word.appendChild(slot.node);
    slots.push(slot);
  });

  // a scrambling glyph, drawn over an invisible copy of the real character so nothing moves
  function scrambleNode(s) {
    var node = document.createElement("span");
    node.className = "decode-cell";
    var ghost = document.createElement("span");
    ghost.className = "decode-ghost";
    ghost.textContent = s.ch;
    s.glyph = document.createElement("span");
    s.glyph.className = "decode-glyph";
    node.appendChild(ghost);
    node.appendChild(s.glyph);
    return node;
  }

  // a locked glyph: a fresh element, so the lock-in snap replays on every decode
  function lockedNode(s) {
    var node = document.createElement("span");
    node.className = "decode-lock chroma-char";
    var h = (((s.i + 1) * 2654435761) % 1000) / 1000;
    node.style.animationDelay = "0s, " + (-h * 14).toFixed(2) + "s";
    node.style.animationDuration = "0.4s, " + (11 + h * 9).toFixed(2) + "s";
    node.textContent = s.ch;
    return node;
  }

  var progress = 0;
  var tick = 0; // like the original, keeps counting across decodes
  var timer = null;

  function render() {
    slots.forEach(function (s) {
      var locked = s.i < progress;
      if (locked !== s.locked) {
        var node = locked ? lockedNode(s) : scrambleNode(s);
        s.word.replaceChild(node, s.node);
        s.node = node;
        s.locked = locked;
      }
      if (!locked) {
        s.glyph.textContent =
          POOL[(s.i * 31 + tick * 17 + s.i * tick) % POOL.length];
        s.glyph.style.color = HUES[(s.i + tick) % HUES.length];
      }
    });
  }

  function run() {
    clearInterval(timer);
    progress = 0;
    render();
    var stepMs = 34;
    var perTick = n / (700 / stepMs);
    var p = -n * 0.4;
    timer = setInterval(function () {
      p += perTick;
      tick += 1;
      progress = p;
      render();
      if (p >= n) clearInterval(timer);
    }, stepMs);
  }

  // re-decode only after the previous decode has finished, as the original's onMouseEnter;
  // a touch tap fires pointerup (not pointerenter's mouse case), so it never runs twice
  function again() {
    if (progress < n) return;
    if (reduce && reduce.matches) return;
    run();
  }
  el.addEventListener("pointerenter", function (e) {
    if (e.pointerType !== "touch") again();
  });
  el.addEventListener("pointerup", function (e) {
    if (e.pointerType === "touch") again();
  });
  el.addEventListener("focusin", again);

  run();
})();
