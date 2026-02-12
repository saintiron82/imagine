#!/usr/bin/env node
/**
 * Patch Electron.app for macOS dev mode.
 * - Renames executable from "Electron" to "Imagine"
 * - Updates Info.plist (CFBundleName, CFBundleDisplayName, CFBundleExecutable)
 * - Dock and menu bar show "Imagine" instead of "Electron"
 * Runs automatically via postinstall hook.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

if (process.platform !== 'darwin') process.exit(0);

const electronApp = path.join(
    __dirname, '..', 'node_modules', 'electron', 'dist', 'Electron.app'
);
const plist = path.join(electronApp, 'Contents', 'Info.plist');
const oldExe = path.join(electronApp, 'Contents', 'MacOS', 'Electron');
const newExe = path.join(electronApp, 'Contents', 'MacOS', 'Imagine');

if (!fs.existsSync(plist)) {
    console.log('[patch] Electron.app not found, skipping');
    process.exit(0);
}

try {
    // Rename executable binary
    if (fs.existsSync(oldExe) && !fs.existsSync(newExe)) {
        fs.renameSync(oldExe, newExe);
        console.log('[patch] Renamed executable: Electron -> Imagine');
    }

    // Patch Info.plist
    const pb = (cmd) => execSync(
        `/usr/libexec/PlistBuddy -c "${cmd}" "${plist}"`,
        { stdio: 'pipe' }
    );
    pb('Set :CFBundleName Imagine');
    pb('Set :CFBundleDisplayName Imagine');
    pb('Set :CFBundleExecutable Imagine');
    pb('Set :CFBundleIdentifier com.imagine.app');

    // Update path.txt so electron module resolves the renamed binary
    const pathTxt = path.join(electronApp, '..', '..', 'path.txt');
    if (fs.existsSync(pathTxt)) {
        fs.writeFileSync(pathTxt, 'Electron.app/Contents/MacOS/Imagine');
    }

    console.log('[patch] Electron.app patched to "Imagine" for dev mode');
} catch (err) {
    console.warn('[patch] Failed to patch Electron.app:', err.message);
}
