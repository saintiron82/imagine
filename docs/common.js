/* ══════════════════════════════════════════════════════════
   Imagine — Common JS (Firebase + i18n + Nav)
   ══════════════════════════════════════════════════════════ */

/* ── Admin Config ────────────────────────────────────── */
const ADMIN_EMAIL = 'saintiron82@gmail.com';

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
  if (typeof firebase.firestore === 'function') db = firebase.firestore();
  if (typeof firebase.auth === 'function') auth = firebase.auth();
}

function getDb() { return db; }
function getAuth() { return auth; }
function isAdmin(user) { return user && user.email === ADMIN_EMAIL; }

/* ── Auth Gate (login required pages) ────────────────── */
let _authResolve = null;
let _currentAuthUser = null;

function getCurrentUser() { return _currentAuthUser; }

function requireAuth() {
  return new Promise((resolve) => {
    _authResolve = resolve;
    const authObj = getAuth();
    if (!authObj) { resolve(null); return; }

    authObj.onAuthStateChanged(user => {
      _currentAuthUser = user;
      if (user) {
        hideAuthGate();
        resolve(user);
      } else {
        showAuthGate();
      }
    });
  });
}

function showAuthGate() {
  if (document.getElementById('auth-gate')) return;
  const lang = getCurrentLang();
  const isKo = lang === 'ko';

  const overlay = document.createElement('div');
  overlay.id = 'auth-gate';
  overlay.className = 'auth-gate';
  overlay.innerHTML = `
    <div class="auth-gate-card">
      <img src="icon.svg" alt="Imagine" style="width:64px;height:64px;margin-bottom:20px;filter:drop-shadow(0 0 20px rgba(56,189,248,0.3));">
      <h2>${isKo ? '\ub85c\uadf8\uc778\uc774 \ud544\uc694\ud569\ub2c8\ub2e4' : 'Sign In Required'}</h2>
      <p>${isKo ? '\uc774 \ud398\uc774\uc9c0\ub97c \ubcf4\ub824\uba74 Google\ub85c \ub85c\uadf8\uc778\ud558\uc138\uc694' : 'Sign in with Google to view this page'}</p>
      <button class="google-signin-btn" onclick="authGateSignIn()">
        <svg viewBox="0 0 24 24" width="20" height="20"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>
        <span>${isKo ? 'Google\ub85c \ub85c\uadf8\uc778' : 'Sign in with Google'}</span>
      </button>
    </div>`;
  document.body.appendChild(overlay);

  // Hide main content
  document.querySelectorAll('body > *:not(#auth-gate):not(.ambient):not(nav)').forEach(el => {
    el.style.display = 'none';
  });
}

function hideAuthGate() {
  const gate = document.getElementById('auth-gate');
  if (gate) gate.remove();
  // Restore main content
  document.querySelectorAll('body > *').forEach(el => {
    el.style.display = '';
  });
}

function authGateSignIn() {
  const authObj = getAuth();
  if (!authObj) return;
  const provider = new firebase.auth.GoogleAuthProvider();
  authObj.signInWithPopup(provider).catch(err => {
    console.error('Sign in failed:', err);
  });
}

/* ── i18n System ─────────────────────────────────────── */
const commonI18n = {
  en: {
    "nav.board": "Board",
    "nav.download": "Download",
    "nav.signin": "Sign in",
    "nav.signout": "Sign out",
    "common.loading": "Loading...",
    "common.empty": "No content yet",
    "common.error": "Failed to load content"
  },
  ko: {
    "nav.board": "\uAC8C\uC2DC\uD310",
    "nav.download": "\uB2E4\uC6B4\uB85C\uB4DC",
    "nav.signin": "\uB85C\uADF8\uC778",
    "nav.signout": "\uB85C\uADF8\uC544\uC6C3",
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

/* ── Nav Auth State ──────────────────────────────────── */
function initNavAuth() {
  const authObj = getAuth();
  if (!authObj) return;

  authObj.onAuthStateChanged(user => {
    const existing = document.querySelector('.nav-user');
    if (existing) existing.remove();

    const navLinks = document.querySelector('.nav-links');
    if (!navLinks) return;

    const div = document.createElement('div');
    div.className = 'nav-user';

    if (user) {
      const photo = user.photoURL
        ? `<img src="${user.photoURL}" alt="" class="nav-user-avatar" onerror="this.style.display='none'">`
        : '';
      div.innerHTML = `${photo}<button class="nav-signout" onclick="getAuth().signOut()">${t('nav.signout')}</button>`;
    } else {
      div.innerHTML = `<button class="nav-signin-btn" onclick="authGateSignIn()">${t('nav.signin')}</button>`;
    }

    navLinks.appendChild(div);
  });
}

/* ── Page Init ───────────────────────────────────────── */
function initCommon() {
  initFirebase();
  initActiveNavLink();
  initLang();
  initScrollAnimations();
  initNavAuth();
}

// Auto-init on DOMContentLoaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCommon);
} else {
  initCommon();
}
