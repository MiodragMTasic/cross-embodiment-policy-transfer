// Video handling adapted from the keep-playing script in keunhong.github.io (CC BY-SA 4.0);
// changed to play only while on screen and to respect reduced motion.
(function () {
  // A slot still holding its {{TOKEN}} is a pending value, not a result.
  document.querySelectorAll(".slot").forEach(function (el) {
    if (/^\s*\{\{[A-Z0-9_]+\}\}\s*$/.test(el.textContent)) {
      el.classList.add("pending");
      el.title = "Pending: filled from the outputs listed in PLACEHOLDERS.md";
    }
  });

  // Keep title as a no-JS fallback; enhanced terms get a persistent accessible
  // description and a visible, hoverable definition on pointer, keyboard or touch.
  var terms = document.querySelectorAll("dfn[title], abbr[title]");
  var activeTerm = null;
  var closeTimer = null;
  var pinned = false;

  function cancelClose() {
    window.clearTimeout(closeTimer);
  }

  function hideDefinition() {
    cancelClose();
    if (activeTerm) {
      activeTerm.definition.hidden = true;
      activeTerm.element.classList.remove("term-help-open");
    }
    activeTerm = null;
    pinned = false;
  }

  function placeDefinition() {
    if (!activeTerm) return;
    var box = activeTerm.definition;
    var rect = activeTerm.element.getBoundingClientRect();
    var width = document.documentElement.clientWidth;
    var height = window.innerHeight;
    var margin = 12;
    if (rect.bottom < 0 || rect.top > height) {
      hideDefinition();
      return;
    }
    box.style.maxHeight = Math.max(0, height - margin * 2) + "px";
    var bounds = box.getBoundingClientRect();
    var left = Math.max(margin, Math.min(
      rect.left + rect.width / 2 - bounds.width / 2,
      width - bounds.width - margin
    ));
    var below = rect.bottom + 8;
    var top = below + bounds.height <= height - margin
      ? below : rect.top - bounds.height - 8;
    top = Math.max(margin, Math.min(top, height - bounds.height - margin));
    box.style.left = left + "px";
    box.style.top = top + "px";
  }

  function showDefinition(term) {
    cancelClose();
    if (activeTerm !== term) {
      hideDefinition();
      activeTerm = term;
    }
    term.definition.hidden = false;
    term.element.classList.add("term-help-open");
    placeDefinition();
  }

  function scheduleClose(term) {
    cancelClose();
    closeTimer = window.setTimeout(function () {
      if (activeTerm === term && !pinned && document.activeElement !== term.element) {
        hideDefinition();
      }
    }, 160);
  }

  terms.forEach(function (el, index) {
    var description = el.getAttribute("title");
    if (!description) return;
    var definition = document.createElement("div");
    var id = "term-definition-" + index;
    while (document.getElementById(id)) id += "-help";
    definition.id = id;
    definition.className = "term-definition";
    definition.setAttribute("role", "tooltip");
    definition.hidden = true;
    definition.textContent = description;
    document.body.appendChild(definition);
    var term = { element: el, definition: definition };
    var priorDescription = el.getAttribute("aria-describedby");
    el.setAttribute("aria-describedby", priorDescription ? priorDescription + " " + id : id);
    el.classList.add("term-help");
    el.tabIndex = 0;
    el.removeAttribute("title");

    el.addEventListener("pointerenter", function (event) {
      if (event.pointerType !== "touch") showDefinition(term);
    });
    el.addEventListener("pointerleave", function () { scheduleClose(term); });
    el.addEventListener("focus", function () { showDefinition(term); });
    el.addEventListener("blur", function () {
      if (activeTerm === term) {
        pinned = false;
        scheduleClose(term);
      }
    });
    el.addEventListener("click", function () {
      if (activeTerm === term && pinned) {
        hideDefinition();
      } else {
        showDefinition(term);
        pinned = true;
      }
    });
    el.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        el.click();
      }
    });
    definition.addEventListener("pointerenter", cancelClose);
    definition.addEventListener("pointerleave", function () { scheduleClose(term); });
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && activeTerm) hideDefinition();
  });
  document.addEventListener("pointerdown", function (event) {
    if (activeTerm && !activeTerm.element.contains(event.target) &&
        !activeTerm.definition.contains(event.target)) hideDefinition();
  });
  window.addEventListener("resize", placeDefinition);
  window.addEventListener("scroll", placeDefinition, true);

  var reduce =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var videos = Array.prototype.slice.call(
    document.querySelectorAll("video[data-loop]"),
  );

  if (reduce) {
    videos.forEach(function (v) {
      v.removeAttribute("autoplay");
      v.pause();
      v.controls = true;
    });
    return;
  }

  function play(v) {
    var p = v.play();
    if (p && p.catch) p.catch(function () {});
  }

  if (!("IntersectionObserver" in window)) {
    videos.forEach(play);
    return;
  }

  var onScreen = new Set();
  var io = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          onScreen.add(e.target);
          if (!document.hidden) play(e.target);
        } else {
          onScreen.delete(e.target);
          e.target.pause();
        }
      });
    },
    { rootMargin: "120px 0px" },
  );
  videos.forEach(function (v) {
    io.observe(v);
    v.addEventListener("stalled", function () {
      if (!document.hidden && onScreen.has(v)) play(v);
    });
  });
  document.addEventListener("visibilitychange", function () {
    videos.forEach(function (v) {
      if (document.hidden) v.pause();
      else if (onScreen.has(v)) play(v);
    });
  });
})();
