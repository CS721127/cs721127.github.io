/**
 * iframe-guest.js
 * ────────────────────────────────────────────────────────────────
 * Drop ONE <script src="/asset/js/iframe-guest.js"></script> tag
 * into every page on your site (before </body>).
 *
 * When the page is loaded inside the shell's iframe it will:
 *   1. Hide the page's own <header> and <nav> (shell provides them)
 *   2. Intercept ALL same-origin internal <a> clicks and route them
 *      through the shell via postMessage — so the iframe src changes
 *      but the shell (and its audio) never reloads.
 *
 * When the page is loaded directly (not in iframe) it does nothing,
 * so every page still works standalone as before.
 * ────────────────────────────────────────────────────────────────
 */
(function () {
  const inIframe = window.self !== window.top;
  if (!inIframe) return;   // standalone visit — do nothing

  const ORIGIN = window.location.origin;  // e.g. https://cs721127.github.io

  /* ── 1. Hide the page's own header + nav ───────────────────── */
  // Disabled: shell header/nav are hidden, pages show their own chrome.
  function hideChrome() { /* no-op */ }

  /* ── 2. Intercept internal links ───────────────────────────── */
  function interceptLinks() {
    document.addEventListener('click', function (e) {
      const anchor = e.target.closest('a');
      if (!anchor) return;

      const href = anchor.getAttribute('href');
      if (!href) return;

      // Skip: external links, mailto, tel, hash-only, explicit _blank
      if (
        href.startsWith('mailto:') ||
        href.startsWith('tel:')    ||
        href.startsWith('#')       ||
        anchor.target === '_blank' ||
        anchor.target === '_top'
      ) return;

      // Resolve to absolute URL
      let abs;
      try { abs = new URL(href, window.location.href).href; }
      catch { return; }

      // Only intercept same-origin links
      if (!abs.startsWith(ORIGIN)) return;

      e.preventDefault();

      // Send navigation request to the shell
      window.parent.postMessage({
        type: 'navigate',
        src:  abs,
        // Try to extract the top-level section for active-nav highlighting
        // e.g. /blog/my_own_story/ → "blog"
        page: abs.replace(ORIGIN, '').split('/').filter(Boolean)[0] || 'home'
      }, ORIGIN);
    }, true /* capture — fires before any other handler */);
  }

  /* ── Run ────────────────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      hideChrome();
      interceptLinks();
    });
  } else {
    hideChrome();
    interceptLinks();
  }

})();