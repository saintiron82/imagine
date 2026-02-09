import React, { useState, useRef, useCallback } from 'react';
import { Upload, X, ImageIcon, Clipboard } from 'lucide-react';
import { useLocale } from '../i18n';

const MAX_SIZE = 512;

function resizeToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                let w = img.width;
                let h = img.height;
                if (w > MAX_SIZE || h > MAX_SIZE) {
                    const ratio = Math.min(MAX_SIZE / w, MAX_SIZE / h);
                    w = Math.round(w * ratio);
                    h = Math.round(h * ratio);
                }
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL('image/jpeg', 0.85));
            };
            img.onerror = () => reject(new Error('Invalid image'));
            img.src = e.target.result;
        };
        reader.onerror = () => reject(new Error('File read error'));
        reader.readAsDataURL(file);
    });
}

export default function ImageSearchInput({ queryImage, onImageChange }) {
    const { t } = useLocale();
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const processFile = useCallback(async (file) => {
        if (!file || !file.type.startsWith('image/')) {
            setError(t('msg.invalid_image'));
            return;
        }
        setError(null);
        try {
            const base64 = await resizeToBase64(file);
            onImageChange(base64);
        } catch {
            setError(t('msg.invalid_image'));
        }
    }, [onImageChange, t]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) processFile(file);
    }, [processFile]);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handlePaste = useCallback((e) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                processFile(item.getAsFile());
                return;
            }
        }
    }, [processFile]);

    const handleFileSelect = useCallback((e) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
        e.target.value = '';
    }, [processFile]);

    const handleRemove = useCallback(() => {
        onImageChange(null);
        setError(null);
    }, [onImageChange]);

    // If image is selected, show preview
    if (queryImage) {
        return (
            <div className="w-full max-w-2xl mt-3">
                <div className="relative inline-block">
                    <img
                        src={queryImage}
                        alt="Query"
                        className="h-32 w-auto rounded-lg border border-gray-600 object-contain bg-gray-800"
                    />
                    <button
                        onClick={handleRemove}
                        className="absolute -top-2 -right-2 bg-red-600 hover:bg-red-500 text-white rounded-full p-1 transition-colors"
                        title={t('action.remove_image')}
                    >
                        <X size={14} />
                    </button>
                </div>
            </div>
        );
    }

    // Drop zone
    return (
        <div className="w-full max-w-2xl mt-3" onPaste={handlePaste} tabIndex={0}>
            <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`flex flex-col items-center justify-center gap-2 px-6 py-8 rounded-lg border-2 border-dashed cursor-pointer transition-all
                    ${isDragging
                        ? 'border-blue-400 bg-blue-900/20'
                        : 'border-gray-600 hover:border-gray-500 bg-gray-800/50 hover:bg-gray-800'
                    }`}
            >
                <div className="flex items-center gap-3 text-gray-400">
                    <ImageIcon size={24} />
                    <Upload size={20} />
                    <Clipboard size={18} />
                </div>
                <p className="text-sm text-gray-400 text-center">
                    {t('placeholder.drag_image')}
                </p>
                <button
                    type="button"
                    className="mt-1 px-4 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs rounded transition-colors"
                    onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                >
                    {t('action.upload_image')}
                </button>
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
            />

            {error && (
                <div className="mt-2 px-3 py-1.5 bg-red-900/30 border border-red-700 rounded text-red-400 text-xs">
                    {error}
                </div>
            )}
        </div>
    );
}
