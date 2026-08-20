// Scrollspy for the side rail, reveal on scroll, and a three-state theme switch.
(function () {
  var root = document.documentElement;

  // theme: system -> light -> dark -> system
  var btn = document.getElementById('theme');
  var order = [null, 'light', 'dark'];
  var label = { 'null': 'Theme: auto', 'light': 'Theme: light', 'dark': 'Theme: dark' };
  var saved = null;
  try { saved = localStorage.getItem('theme'); } catch (e) {}
  function apply(v) {
    if (v) { root.setAttribute('data-theme', v); } else { root.removeAttribute('data-theme'); }
    if (btn) { btn.textContent = label[String(v)]; }
    try { v ? localStorage.setItem('theme', v) : localStorage.removeItem('theme'); } catch (e) {}
  }
  apply(saved);
  if (btn) {
    btn.addEventListener('click', function () {
      var cur = root.getAttribute('data-theme') || null;
      apply(order[(order.indexOf(cur) + 1) % order.length]);
    });
  }

  // rail scrollspy
  var links = Array.prototype.slice.call(document.querySelectorAll('.rail a'));
  var targets = links.map(function (a) { return document.querySelector(a.getAttribute('href')); })
                     .filter(Boolean);
  if (targets.length && 'IntersectionObserver' in window) {
    var seen = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen[e.target.id] = e.isIntersecting ? e.intersectionRatio : 0; });
      var best = null, bv = 0;
      Object.keys(seen).forEach(function (k) { if (seen[k] > bv) { bv = seen[k]; best = k; } });
      links.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + best); });
    }, { rootMargin: '-15% 0px -60% 0px', threshold: [0, 0.2, 0.5, 1] });
    targets.forEach(function (t) { io.observe(t); });
  }

  // reveal
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var blocks = document.querySelectorAll('.figblock, .stats, .tablewrap, .card');
  if (reduce || !('IntersectionObserver' in window)) {
    blocks.forEach(function (b) { b.classList.add('in'); });
  } else {
    blocks.forEach(function (b) { b.classList.add('reveal'); });
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); ro.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });
    blocks.forEach(function (b) { ro.observe(b); });
  }
})();
