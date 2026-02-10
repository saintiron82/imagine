import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, X, ImageIcon, Clipboard, Plus, Link, Loader2 } from 'lucide-react';
import { useLocale } from '../i18n';

const MAX_SIZE = 512;
const MAX_IMAGES = 10;
const URL_PATTERN = /^https?:\/\/.+/i;

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

async function fetchImageFromUrl(url) {
    // Use Electron main process to bypass CORS
    if (window.electron?.pipeline?.fetchImageUrl) {
        const result = await window.electron.pipeline.fetchImageUrl(url);
        if (!result.success) throw new Error(result.error);
        // result.data is a full data:image/...;base64,... string — resize it
        return new Promise((resolve, reject) => {
            const img = new window.Image();
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
            img.onerror = () => reject(new Error('Invalid image data'));
            img.src = result.data;
        });
    }
    // Fallback: direct load (may fail on CORS-restricted sites)
    return new Promise((resolve, reject) => {
        const img = new window.Image();
        img.crossOrigin = 'anonymous';
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
        img.onerror = () => reject(new Error('Failed to load image from URL'));
        img.src = url;
    });
}

export default function ImageSearchInput({ queryImages, onImagesChange }) {
    const { t } = useLocale();
    const [isDragging, setIsDragging] = useState(false);
    const [isLoadingUrl, setIsLoadingUrl] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const processFiles = useCallback(async (files) => {
        const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
        if (imageFiles.length === 0) {
            setError(t('msg.invalid_image'));
            return;
        }

        const remaining = MAX_IMAGES - queryImages.length;
        if (remaining <= 0) {
            setError(t('msg.max_images_reached'));
            return;
        }

        const toProcess = imageFiles.slice(0, remaining);
        setError(null);

        try {
            const newImages = await Promise.all(toProcess.map(f => resizeToBase64(f)));
            onImagesChange([...queryImages, ...newImages]);
        } catch {
            setError(t('msg.invalid_image'));
        }
    }, [queryImages, onImagesChange, t]);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
        const files = e.dataTransfer.files;
        if (files?.length > 0) processFiles(files);
    }, [processFiles]);

    const handleDragOver = useCallback((e) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const processUrl = useCallback(async (url) => {
        const remaining = MAX_IMAGES - queryImages.length;
        if (remaining <= 0) {
            setError(t('msg.max_images_reached'));
            return;
        }
        setIsLoadingUrl(true);
        setError(null);
        try {
            const base64 = await fetchImageFromUrl(url.trim());
            onImagesChange([...queryImages, base64]);
        } catch {
            setError(t('msg.url_fetch_failed'));
        } finally {
            setIsLoadingUrl(false);
        }
    }, [queryImages, onImagesChange, t]);

    // Global paste listener — captures image paste regardless of focus
    useEffect(() => {
        const handleGlobalPaste = (e) => {
            const items = e.clipboardData?.items;
            if (!items) return;

            // Check for image blobs first (always capture, regardless of focus)
            const imageFiles = [];
            for (const item of items) {
                if (item.type.startsWith('image/')) {
                    imageFiles.push(item.getAsFile());
                }
            }
            if (imageFiles.length > 0) {
                e.preventDefault();
                processFiles(imageFiles);
                return;
            }

            // Check for pasted URL text (only when not in text input/textarea)
            const tag = document.activeElement?.tagName;
            if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
                const text = e.clipboardData.getData('text/plain')?.trim();
                if (text && URL_PATTERN.test(text)) {
                    e.preventDefault();
                    processUrl(text);
                }
            }
        };

        document.addEventListener('paste', handleGlobalPaste);
        return () => document.removeEventListener('paste', handleGlobalPaste);
    }, [processFiles, processUrl]);

    const handleFileSelect = useCallback((e) => {
        const files = e.target.files;
        if (files?.length > 0) processFiles(files);
        e.target.value = '';
    }, [processFiles]);

    const handleRemoveImage = useCallback((index) => {
        const updated = queryImages.filter((_, i) => i !== index);
        onImagesChange(updated);
        setError(null);
    }, [queryImages, onImagesChange]);

    const hasImages = queryImages.length > 0;
    const canAddMore = queryImages.length < MAX_IMAGES;

    // If images are selected, show preview list + optional add zone
    if (hasImages) {
        return (
            <div className="w-full max-w-2xl mt-3">
                <div className="flex items-start gap-2 overflow-x-auto pb-2 custom-scrollbar">
                    {queryImages.map((img, index) => (
                        <div key={index} className="relative shrink-0">
                            <img
                                src={img}
                                alt={`Query ${index + 1}`}
                                className="h-28 w-auto rounded-lg border border-gray-600 object-contain bg-gray-800"
                            />
                            <button
                                onClick={() => handleRemoveImage(index)}
                                className="absolute -top-1.5 -right-1.5 bg-red-600 hover:bg-red-500 text-white rounded-full p-0.5 transition-colors"
                                title={t('action.remove_image')}
                            >
                                <X size={12} />
                            </button>
                        </div>
                    ))}

                    {/* Add more button */}
                    {canAddMore && (
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            className={`shrink-0 h-28 w-20 rounded-lg border-2 border-dashed flex flex-col items-center justify-center gap-1 cursor-pointer transition-all ${
                                isDragging
                                    ? 'border-blue-400 bg-blue-900/20'
                                    : 'border-gray-600 hover:border-gray-500 bg-gray-800/50 hover:bg-gray-800'
                            }`}
                        >
                            <Plus size={18} className="text-gray-400" />
                            <span className="text-[10px] text-gray-500">{t('action.add_more_images')}</span>
                        </button>
                    )}
                </div>

                <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] text-gray-500">
                        {t('label.image_count', { count: queryImages.length })}
                    </span>
                    {isLoadingUrl && (
                        <span className="flex items-center gap-1 text-[10px] text-blue-400">
                            <Loader2 size={10} className="animate-spin" />
                            {t('status.fetching_url')}
                        </span>
                    )}
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    multiple
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

    // Drop zone (no images yet)
    return (
        <div className="w-full max-w-2xl mt-3">
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
                {isLoadingUrl ? (
                    <Loader2 size={24} className="text-blue-400 animate-spin" />
                ) : (
                    <div className="flex items-center gap-3 text-gray-400">
                        <ImageIcon size={24} />
                        <Upload size={20} />
                        <Clipboard size={18} />
                        <Link size={18} />
                    </div>
                )}
                <p className="text-sm text-gray-400 text-center">
                    {isLoadingUrl ? t('status.fetching_url') : t('placeholder.drag_images_or_url')}
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
                multiple
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
