import React, { useState, useEffect } from 'react';
import { Loader2, Square } from 'lucide-react';
import { useLocale } from '../i18n';

const ProcessingIndicator = ({ isProcessing, processed = 0, total = 0, currentFile = '', onStop }) => {
  const { t } = useLocale();
  const [dots, setDots] = useState('');

  // Animated dots effect
  useEffect(() => {
    if (!isProcessing) {
      setDots('');
      return;
    }

    const interval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);

    return () => clearInterval(interval);
  }, [isProcessing]);

  if (!isProcessing) return null;

  const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-sm animate-in fade-in slide-in-from-bottom-4">
      <div className="bg-gray-800 rounded-lg shadow-2xl p-4 border border-gray-700 backdrop-blur-sm">
        {/* Header with Spinner */}
        <div className="flex items-center gap-3 mb-3">
          <Loader2 className="animate-spin text-blue-400 flex-shrink-0" size={24} />
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-bold text-white">
              {t('status.processing')}{dots}
            </h3>
            {currentFile && (
              <p className="text-xs text-gray-400 truncate">
                {currentFile}
              </p>
            )}
          </div>
          {onStop && (
            <button
              onClick={onStop}
              className="p-1.5 rounded hover:bg-red-900/50 text-red-400 hover:text-red-300 transition-colors flex-shrink-0"
              title={t('action.stop_processing')}
            >
              <Square size={16} />
            </button>
          )}
        </div>

        {/* Progress Bar */}
        {total > 0 && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-gray-400">
              <span>{t('status.files_progress', { processed, total })}</span>
              <span>{percentage}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-300 ease-out rounded-full"
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProcessingIndicator;
