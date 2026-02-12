#!/usr/bin/env node
/**
 * Patch Electron.app Info.plist for macOS dev mode.
 * Changes dock/menu bar name from "Electron" to "Imagine".
 * Runs automatically via postinstall hook.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

if (process.platform !== 'darwin') process.exit(0);

const plist = path.join(
    __dirname, '..', 'node_modules', 'electron', 'dist',
    'Electron.app', 'Contents', 'Info.plist'
);

if (!fs.existsSync(plist)) {
    console.log('[patch] Electron.app not found, skipping');
    process.exit(0);
}

try {
    execSync(`defaults write "${plist}" CFBundleName "Imagine"`);
    execSync(`defaults write "${plist}" CFBundleDisplayName "Imagine"`);
    execSync(`defaults write "${plist}" CFBundleIdentifier "com.imagine.app"`);
    console.log('[patch] Electron.app renamed to "Imagine" for dev mode');
} catch (err) {
    console.warn('[patch] Failed to patch Info.plist:', err.message);
}
