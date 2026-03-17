/* ─────────────────────────────────────────────────────────────────────────────
   audio.js  —  Audio player logic for the article audio bar
   Place at: /asset/js/audio.js

   Usage
   ─────
   A) Via HTML data attributes on #audioPlayerBar:
        data-src="/asset/audio/on-the-way.mp3"
        data-title="On the Way — Cen Sun"
        data-sub="Personal Essay Collection · 2024"

   B) Via JS public API anywhere on the page:
        window.audioPlayer.load('/asset/audio/file.mp3', 'Title', 'Subtitle')
───────────────────────────────────────────────────────────────────────────── */

(function () {
  'use strict';

  /* ── Element references ───────────────────────────────────────────────────── */
  const bar       = document.getElementById('audioPlayerBar');
  const audio     = document.getElementById('audioEl');
  const btnPlay   = document.getElementById('btnPlay');
  const iconPlay  = document.getElementById('iconPlay');
  const iconPause = document.getElementById('iconPause');
  const btnSkipB  = document.getElementById('btnSkipBack');
  const btnSkipF  = document.getElementById('btnSkipFwd');
  const btnSpeed  = document.getElementById('btnSpeed');
  const seekEl    = document.getElementById('audioSeek');
  const volEl     = document.getElementById('audioVolume');
  const curEl     = document.getElementById('audioCurrentTime');
  const durEl     = document.getElementById('audioDuration');
  const artwork   = document.getElementById('audioArtwork');
  const titleEl   = document.getElementById('audioTitle');
  const subEl     = document.getElementById('audioSub');

  if (!bar || !audio) return; // guard: player not on this page

  /* ── Constants ────────────────────────────────────────────────────────────── */
  const SPEEDS = [0.75, 1, 1.25, 1.5, 1.75, 2];

  /* ── State ────────────────────────────────────────────────────────────────── */
  let speedIdx = 1;   // default 1×
  let seeking  = false;

  /* ── Helpers ──────────────────────────────────────────────────────────────── */
  function fmtTime(s) {
    if (!isFinite(s) || s < 0) return '0:00';
    const m   = Math.floor(s / 60);
    const sec = Math.floor(s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  }

  function setPlayState(playing) {
    iconPlay.style.display  = playing ? 'none' : '';
    iconPause.style.display = playing ? ''     : 'none';
    artwork.classList.toggle('is-playing', playing);
  }

  function updateSeekGradient() {
    const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
    seekEl.style.background =
      `linear-gradient(to right, var(--ocean, #3a7bd5) ${pct}%, var(--border, #e8e3da) ${pct}%)`;
  }

  function updateVolGradient() {
    const pct = volEl.value * 100;
    volEl.style.background =
      `linear-gradient(to right, var(--ocean, #3a7bd5) ${pct}%, var(--border, #e8e3da) ${pct}%)`;
  }

  /* ── Load audio ───────────────────────────────────────────────────────────── */
  function loadAudio(src, title, sub) {
    if (title) titleEl.textContent = title;
    if (sub)   subEl.textContent   = sub;
    if (!src)  return;

    audio.src = src;
    // Reset UI state
    seekEl.value      = 0;
    curEl.textContent = '0:00';
    durEl.textContent = '0:00';
    updateSeekGradient();
    setPlayState(false);
  }

  // Initialise from data attributes
  loadAudio(
    bar.dataset.src   || '',
    bar.dataset.title || '',
    bar.dataset.sub   || ''
  );

  /* ── Playback ─────────────────────────────────────────────────────────────── */
  btnPlay.addEventListener('click', () => {
    if (!audio.src || audio.src === window.location.href) return;
    audio.paused ? audio.play() : audio.pause();
  });

  audio.addEventListener('play',  () => setPlayState(true));
  audio.addEventListener('pause', () => setPlayState(false));
  audio.addEventListener('ended', () => {
    setPlayState(false);
    seekEl.value = 0;
    updateSeekGradient();
  });

  /* ── Duration / time update ───────────────────────────────────────────────── */
  audio.addEventListener('loadedmetadata', () => {
    durEl.textContent = fmtTime(audio.duration);
    seekEl.max        = audio.duration || 100;
  });

  audio.addEventListener('timeupdate', () => {
    if (seeking) return;
    curEl.textContent = fmtTime(audio.currentTime);
    seekEl.value      = audio.currentTime;
    updateSeekGradient();
  });

  /* ── Seeking ──────────────────────────────────────────────────────────────── */
  function onSeekStart() { seeking = true; }
  function onSeekEnd()   { audio.currentTime = parseFloat(seekEl.value); seeking = false; }

  seekEl.addEventListener('mousedown',  onSeekStart);
  seekEl.addEventListener('touchstart', onSeekStart, { passive: true });
  seekEl.addEventListener('mouseup',    onSeekEnd);
  seekEl.addEventListener('touchend',   onSeekEnd);
  seekEl.addEventListener('input', () => {
    curEl.textContent = fmtTime(parseFloat(seekEl.value));
    updateSeekGradient();
  });

  /* ── Skip buttons ─────────────────────────────────────────────────────────── */
  btnSkipB.addEventListener('click', () => {
    audio.currentTime = Math.max(0, audio.currentTime - 10);
  });
  btnSkipF.addEventListener('click', () => {
    audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + 10);
  });

  /* ── Playback speed ───────────────────────────────────────────────────────── */
  btnSpeed.addEventListener('click', () => {
    speedIdx           = (speedIdx + 1) % SPEEDS.length;
    const spd          = SPEEDS[speedIdx];
    audio.playbackRate = spd;
    btnSpeed.textContent = spd === 1 ? '1×' : `${spd}×`;
  });

  /* ── Volume ───────────────────────────────────────────────────────────────── */
  volEl.addEventListener('input', () => {
    audio.volume = parseFloat(volEl.value);
    updateVolGradient();
  });
  updateVolGradient(); // initialise gradient on load

  /* ── Public API ───────────────────────────────────────────────────────────── */
  window.audioPlayer = { load: loadAudio };

})();