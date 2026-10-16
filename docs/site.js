// Language tabs: one choice (Python/Go/Rust) is shared by every code block and remembered.
(function () {
  var KEY = "lang";
  function get() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function set(v) { try { localStorage.setItem(KEY, v); } catch (e) {} }
  function show(lang) {
    document.querySelectorAll(".tabs").forEach(function (t) {
      t.querySelectorAll("[data-lang]").forEach(function (el) {
        el.classList.toggle("on", el.dataset.lang === lang);
      });
    });
  }
  document.querySelectorAll(".tabbar button").forEach(function (b) {
    b.addEventListener("click", function () { set(b.dataset.lang); show(b.dataset.lang); });
  });
  show(get() || "python");
  if (window.hljs) hljs.highlightAll();
})();
