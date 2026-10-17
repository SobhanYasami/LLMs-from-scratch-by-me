// Site behaviour: language tabs, theme, mobile nav, nav filter, "on this page", copy buttons.
(function () {
  var doc = document, root = doc.documentElement;
  function get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function $all(s, c) { return Array.prototype.slice.call((c || doc).querySelectorAll(s)); }

  // ---- code language tabs: one choice is shared by every block and remembered
  function show(lang) {
    $all(".tabs").forEach(function (t) {
      $all("[data-lang]", t).forEach(function (el) {
        var on = el.dataset.lang === lang;
        el.classList.toggle("on", on);
        if (el.getAttribute("role") === "tab") {
          el.setAttribute("aria-selected", on);
          el.tabIndex = on ? 0 : -1;
        }
      });
    });
  }
  $all(".tabbar button").forEach(function (b) {
    b.addEventListener("click", function () { set("lang", b.dataset.lang); show(b.dataset.lang); });
    b.addEventListener("keydown", function (e) {
      var k = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!k) return;
      var tabs = $all("button", b.parentNode), n = tabs[(tabs.indexOf(b) + k + tabs.length) % tabs.length];
      n.focus(); n.click(); e.preventDefault();
    });
  });
  show(get("lang") || "python");

  // ---- theme: light / dark, overrides the OS setting (syntax colours follow via CSS)
  var tb = doc.querySelector(".theme-btn");
  if (tb) tb.addEventListener("click", function () {
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    set("theme", root.dataset.theme);
  });

  // ---- mobile nav drawer
  var mb = doc.querySelector(".menu-btn"), scrim = doc.querySelector(".scrim");
  function nav(open) {
    doc.body.classList.toggle("nav-open", open);
    if (mb) mb.setAttribute("aria-expanded", open);
    if (scrim) scrim.hidden = !open;
  }
  if (mb) mb.addEventListener("click", function () { nav(!doc.body.classList.contains("nav-open")); });
  if (scrim) scrim.addEventListener("click", function () { nav(false); });
  doc.addEventListener("keydown", function (e) { if (e.key === "Escape") nav(false); });

  // ---- nav filter
  var f = doc.querySelector(".filter"), links = doc.querySelector(".navlinks");
  if (f && links) {
    var none = doc.createElement("div");
    none.className = "nomatch"; none.hidden = true; none.textContent = f.dataset.nomatch;
    links.appendChild(none);
    f.addEventListener("input", function () {
      var q = f.value.trim().toLowerCase(), any = false;
      $all("a", links).forEach(function (a) {
        var hit = !q || a.textContent.toLowerCase().indexOf(q) > -1;
        a.hidden = !hit; if (hit) any = true;
      });
      $all("h4", links).forEach(function (h) { h.hidden = !!q; });
      none.hidden = any;
    });
  }
  var cur = doc.querySelector("nav.side a.on");
  if (cur && cur.scrollIntoView) { var s = cur.closest("nav.side"); if (s) s.scrollTop = Math.max(0, cur.offsetTop - s.clientHeight / 3); }

  // ---- on this page: highlight the section being read
  var toc = doc.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    var map = {};
    $all("a", toc).forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          $all("a.on", toc).forEach(function (a) { a.classList.remove("on"); });
          if (map[e.target.id]) map[e.target.id].classList.add("on");
        }
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    $all("main h2[id]").forEach(function (h) { io.observe(h); });
  }

  // ---- syntax highlighting + copy buttons
  var I = window.I18N || { copy: "Copy", copied: "Copied" };
  $all(".codewrap").forEach(function (wrap) {
    var pre = wrap.querySelector("pre");
    if (!pre) return;
    var b = doc.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = I.copy;
    b.addEventListener("click", function () {
      var txt = pre.innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () {
        b.textContent = I.copied; setTimeout(function () { b.textContent = I.copy; }, 1500);
      }, function () {});
    });
    wrap.appendChild(b);
  });
})();
