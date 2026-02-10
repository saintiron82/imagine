import { useState, useEffect, useCallback } from 'react';

const DEFAULT_GAP = 16; // gap-4 = 16px

/**
 * Responsive column count hook using ResizeObserver.
 * Returns { columnCount, cardWidth, rowHeight } based on container width.
 *
 * @param {React.RefObject} containerRef - ref to the scrollable container
 * @param {object} options
 * @param {number[]} options.breakpoints - [minCols, ...breakpoints] mapping widths to cols
 * @param {number} options.gap - gap between cards in px
 * @param {number} options.cardAspectTotal - total card height as ratio of cardWidth (1.0 = square image + text below)
 */
export function useResponsiveColumns(containerRef, options = {}) {
    const {
        // Default breakpoints: width thresholds → column count
        // [cols_at_0, threshold1, cols1, threshold2, cols2, ...]
        breakpoints = [2, 640, 3, 900, 4, 1200, 5],
        gap = DEFAULT_GAP,
        cardAspectTotal = 1.35, // aspect-square (1.0) + text/info below (~0.35)
    } = options;

    const [layout, setLayout] = useState({ columnCount: 2, cardWidth: 200, rowHeight: 270 });

    const calcLayout = useCallback((width) => {
        if (width <= 0) return;

        // Parse breakpoints: [defaultCols, bp1, cols1, bp2, cols2, ...]
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

        // Initial calc
        calcLayout(el.clientWidth);

        const observer = new ResizeObserver(entries => {
            for (const entry of entries) {
                calcLayout(entry.contentRect.width);
            }
        });
        observer.observe(el);

        return () => observer.disconnect();
    }, [containerRef, calcLayout]);

    return layout;
}
