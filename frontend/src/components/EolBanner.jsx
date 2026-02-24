import React, { useState, useEffect } from 'react';
import { AlertTriangle, X, RefreshCw } from 'lucide-react';
import { useLocale } from '../i18n';

const EOL_URL = 'https://raw.githubusercontent.com/saintiron82/imagine/main/eol.json';

function compareVersions(a, b) {
    const pa = a.split('.').map(Number);
    const pb = b.split('.').map(Number);
    for (let i = 0; i < 3; i++) {
        const va = pa[i] || 0;
        const vb = pb[i] || 0;
        if (va < vb) return -1;
        if (va > vb) return 1;
    }
    return 0;
}

const EolBanner = () => {
    const { t } = useLocale();
    const [eolInfo, setEolInfo] = useState(null);
    const [dismissed, setDismissed] = useState(false);

    useEffect(() => {
        const checkEol = async () => {
            try {
                const currentVersion = await window.electron?.updater?.getVersion();
                if (!currentVersion) return;

                const res = await fetch(EOL_URL, { signal: AbortSignal.timeout(5000) });
                if (!res.ok) return;

                const data = await res.json();
                const isEol = (data.eol_versions || []).includes(currentVersion) ||
                    (data.min_recommended && compareVersions(currentVersion, data.min_recommended) < 0);

                if (isEol) {
                    setEolInfo({ currentVersion, latestVersion: data.latest || data.min_recommended });
                }
            } catch {
                // Network error — silently ignore
            }
        };
        checkEol();
    }, []);

    if (!eolInfo || dismissed) return null;

    return (
        <div className="bg-yellow-900/90 border-b border-yellow-600 px-4 py-2 flex items-center gap-2 text-sm shrink-0 z-20">
            <AlertTriangle size={16} className="text-yellow-400 flex-shrink-0" />
            <span className="text-yellow-200 flex-1">
                {t('update.eol_warning', { version: eolInfo.currentVersion, latest: eolInfo.latestVersion })}
            </span>
            <button
                onClick={() => window.electron?.updater?.check()}
                className="px-2.5 py-1 text-xs font-medium bg-yellow-700 hover:bg-yellow-600 text-yellow-100 rounded transition-colors flex items-center gap-1"
            >
                <RefreshCw size={12} />
                {t('update.check_now')}
            </button>
            <button onClick={() => setDismissed(true)} className="text-yellow-400/60 hover:text-yellow-300 transition-colors p-0.5">
                <X size={14} />
            </button>
        </div>
    );
};

export default EolBanner;
