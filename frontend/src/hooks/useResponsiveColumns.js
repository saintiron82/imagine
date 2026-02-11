import { useState, useEffect, useCallback, useRef } from 'react';

const DEFAULT_GAP = 16; // gap-4 = 16px

/**
 * Responsive column count hook using ResizeObserver.
 * Returns { columnCount, cardWidth, rowHeight } based on container width.
 *
 * Uses the *maximum observed width* to avoid scrollbar-induced oscillation:
 * when a vertical scrollbar appears/disappears, the container width fluctuates
 * by ~15-48px, causing an infinite layout → resize → layout loop.
 * By latching to the wider (scrollbar-hidden) width we get a stable layout.
 *
 * @param {React.RefObject} containerRef - ref to the scrollable container
 * @param {object} options
 * @param {number[]} options.breakpoints - [minCols, ...breakpoints] mapping widths to cols
 * @param {number} options.gap - gap between cards in px
 * @param {number} options.cardAspectTotal - total card height as ratio of cardWidth
 */
export function useResponsiveColumns(containerRef, options = {}) {
    const {
        breakpoints = [2, 640, 3, 900, 4, 1200, 5],
        gap = DEFAULT_GAP,
        cardAspectTotal = 1.35,
    } = options;

    const [layout, setLayout] = useState({ columnCount: 2, cardWidth: 200, rowHeight: 270 });
    const maxWidthRef = useRef(0);
    const resetTimer = useRef(null);

    const calcLayout = useCallback((width) => {
        if (width <= 0) return;

        // Track the maximum width seen recently.
        // A smaller width within ~60px of the max is likely just the scrollbar
        // appearing — ignore it and keep using the wider measurement.
        const SCROLLBAR_TOLERANCE = 60;

        if (width >= maxWidthRef.current) {
            maxWidthRef.current = width;
        } else if (maxWidthRef.current - width < SCROLLBAR_TOLERANCE) {
            // Scrollbar appeared — use the cached wider width instead
            width = maxWidthRef.current;
        } else {
            // Genuine resize (e.g. window resized significantly) — accept new width
            maxWidthRef.current = width;
        }

        // Debounce reset: after 500ms of no resize events,
        // clear maxWidth so the next genuine resize is accepted cleanly.
        if (resetTimer.current) clearTimeout(resetTimer.current);
        resetTimer.current = setTimeout(() => { maxWidthRef.current = 0; }, 500);

        let cols = breakpoints[0];
        for (let i = 1; i < breakpoints.length - 1; i += 2) {
            if (width >= breakpoints[i]) {
                cols = breakpoints[i + 1];
            }
        }

        const cardW = (width - gap * (cols - 1)) / cols;
        const rowH = Math.ceil(cardW * cardAspectTotal) + gap;

        setLayout(prev => {
            if (prev.columnCount === cols && Math.abs(prev.cardWidth - cardW) < 1) return prev;
            return { columnCount: cols, cardWidth: cardW, rowHeight: rowH };
        });
    }, [breakpoints, gap, cardAspectTotal]);

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        maxWidthRef.current = 0;
        calcLayout(el.clientWidth);

        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                calcLayout(entry.contentRect.width);
            }
        });
        observer.observe(el);

        return () => {
            observer.disconnect();
            if (resetTimer.current) clearTimeout(resetTimer.current);
        };
    }, [containerRef, calcLayout]);

    return layout;
}
