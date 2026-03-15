/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            animation: {
                'spin-slow': 'spin 3s linear infinite',
                'conveyor': 'conveyor 2s linear infinite',
                'item-flow': 'item-flow 1.5s linear infinite',
                'machine-pulse': 'machine-pulse 2s ease-in-out infinite',
            },
            keyframes: {
                conveyor: {
                    '0%': { transform: 'translateX(0)' },
                    '100%': { transform: 'translateX(-50%)' },
                },
                'item-flow': {
                    '0%': { left: '-6px', opacity: '0' },
                    '10%': { opacity: '1' },
                    '90%': { opacity: '1' },
                    '100%': { left: 'calc(100% + 6px)', opacity: '0' },
                },
                'machine-pulse': {
                    '0%, 100%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.15)' },
                },
            },
        },
    },
    plugins: [],
}
