/* ══════════════════════════════════════════════════════════
   Imagine — Common JS (Firebase + i18n + Nav)
   ══════════════════════════════════════════════════════════ */

/* ── Firebase Config ─────────────────────────────────── */
const firebaseConfig = {
  apiKey: "AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc",
  authDomain: "imagine-b1e9c.firebaseapp.com",
  projectId: "imagine-b1e9c",
  storageBucket: "imagine-b1e9c.firebasestorage.app",
  messagingSenderId: "978580126686",
  appId: "1:978580126686:web:df4b17033cc8daca55fb4f",
  measurementId: "G-MRCCGQHLBQ"
};

/* ── Firebase Init (compat mode for CDN) ─────────────── */
let db = null;
let auth = null;

function initFirebase() {
  if (typeof firebase === 'undefined') return;
  firebase.initializeApp(firebaseConfig);
  db = firebase.firestore();
  auth = firebase.auth();
}

function getDb() { return db; }
function getAuth() { return auth; }

/* ── i18n System ─────────────────────────────────────── */
const commonI18n = {
  en: {
    "nav.releases": "Releases",
    "nav.guide": "Guide",
    "nav.qna": "Q&A",
    "nav.download": "Download",
    "common.loading": "Loading...",
    "common.empty": "No content yet",
    "common.error": "Failed to load content"
  },
  ko: {
    "nav.releases": "\uB9B4\uB9AC\uC2A4",
    "nav.guide": "\uAC00\uC774\uB4DC",
    "nav.qna": "Q&A",
    "nav.download": "\uB2E4\uC6B4\uB85C\uB4DC",
    "common.loading": "\uB85C\uB529 \uC911...",
    "common.empty": "\uC544\uC9C1 \uCF58\uD150\uCE20\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4",
    "common.error": "\uCF58\uD150\uCE20 \uB85C\uB4DC \uC2E4\uD328"
  }
};

let pageI18n = { en: {}, ko: {} };
let currentLang = 'en';

function registerPageI18n(keys) {
  if (keys.en) Object.assign(pageI18n.en, keys.en);
  if (keys.ko) Object.assign(pageI18n.ko, keys.ko);
}

function t(key) {
  return pageI18n[currentLang][key]
    || commonI18n[currentLang][key]
    || key;
}

function setLang(lang) {
  currentLang = lang;
  const allKeys = { ...commonI18n[lang], ...pageI18n[lang] };
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (allKeys[key]) el.textContent = allKeys[key];
  });
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.textContent.trim().toLowerCase() === lang);
  });
  document.documentElement.lang = lang === 'ko' ? 'ko' : 'en';
  localStorage.setItem('imagine-lang', lang);

  // Notify page-specific handler
  if (typeof onLangChange === 'function') onLangChange(lang);
}

function getCurrentLang() { return currentLang; }

function initLang() {
  const saved = localStorage.getItem('imagine-lang');
  if (saved && commonI18n[saved]) { setLang(saved); return; }
  const browserLang = (navigator.language || '').toLowerCase();
  if (browserLang.startsWith('ko')) setLang('ko');
}

/* ── Scroll Animations ───────────────────────────────── */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('[data-anim]').forEach((el, i) => {
    el.style.transitionDelay = `${i % 4 * 80}ms`;
    el.style.transitionDuration = '0.6s';
    el.style.transitionTimingFunction = 'cubic-bezier(0.16, 1, 0.3, 1)';
    observer.observe(el);
  });
}

/* ── Active Nav Link ─────────────────────────────────── */
function initActiveNavLink() {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a.nav-page-link').forEach(link => {
    const href = link.getAttribute('href').split('#')[0];
    if (href === path || (path === 'index.html' && href === 'index.html#download')) {
      link.classList.add('active');
    }
  });
}

/* ── Page Init ───────────────────────────────────────── */
function initCommon() {
  initFirebase();
  initActiveNavLink();
  initLang();
  initScrollAnimations();
}

// Auto-init on DOMContentLoaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCommon);
} else {
  initCommon();
}
