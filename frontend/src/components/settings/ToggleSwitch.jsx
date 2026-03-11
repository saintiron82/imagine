import React from 'react';

const ToggleSwitch = ({ checked, onChange, disabled = false, size = 'md' }) => {
    const sizes = {
        sm: { track: 'h-5 w-9', thumb: 'h-3 w-3', on: 'translate-x-[18px]', off: 'translate-x-1' },
        md: { track: 'h-6 w-11', thumb: 'h-4 w-4', on: 'translate-x-6', off: 'translate-x-1' },
    };
    const s = sizes[size] || sizes.md;

    return (
        <button
            type="button"
            role="switch"
            aria-checked={checked}
            onClick={() => !disabled && onChange(!checked)}
            disabled={disabled}
            className={`relative inline-flex ${s.track} items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed ${
                checked ? 'bg-blue-600' : 'bg-gray-600'
            }`}
        >
            <span
                className={`inline-block ${s.thumb} transform rounded-full bg-white transition-transform ${
                    checked ? s.on : s.off
                }`}
            />
        </button>
    );
};

export default ToggleSwitch;
