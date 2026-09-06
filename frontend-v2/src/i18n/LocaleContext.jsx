import React, { createContext, useContext, useState, useCallback } from 'react';
import enUS from './locales/en-US.json';
import koKR from './locales/ko-KR.json';

const locales = {
  'en-US': enUS,
  'ko-KR': koKR,
};

const STORAGE_KEY = 'imageparser-locale';
const DEFAULT_LOCALE = 'en-US';

function getInitialLocale() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && locales[stored]) return stored;
  } catch {}
  return DEFAULT_LOCALE;
}

const LocaleContext = createContext(null);

export function LocaleProvider({ children }) {
  const [locale, setLocaleState] = useState(getInitialLocale);

  const setLocale = useCallback((newLocale) => {
    if (locales[newLocale]) {
      setLocaleState(newLocale);
      try { localStorage.setItem(STORAGE_KEY, newLocale); } catch {}
    }
  }, []);

  const t = useCallback((key, params) => {
    const dict = locales[locale] || locales[DEFAULT_LOCALE];
    let text = dict[key];

    if (text === undefined) {
      if (import.meta.env.DEV) {
        console.warn(`[i18n] Missing key: "${key}" for locale "${locale}"`);
      }
      return key;
    }

    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        text = text.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v));
      });
    }

    return text;
  }, [locale]);

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t, availableLocales: Object.keys(locales) }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const ctx = useContext(LocaleContext);
  if (!ctx) throw new Error('useLocale must be used within <LocaleProvider>');
  return ctx;
}
