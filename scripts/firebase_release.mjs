#!/usr/bin/env node
/**
 * Firebase release script — upload to Storage + register in Firestore.
 *
 * Uses Firebase CLI refresh token for OAuth2 access token.
 * Calls REST APIs directly (no service account key needed).
 *
 * Usage:
 *   node scripts/firebase_release.mjs --version "v0.6.4.20260309_01" \
 *     --macos "frontend/dist-electron/Imagine-0.6.4-arm64.dmg" \
 *     --notes "Release notes here"
 */

import { readFileSync, existsSync, statSync } from 'fs';
import { basename, join } from 'path';
import { homedir } from 'os';
import { parseArgs } from 'util';

// ── Parse CLI args ──────────────────────────────────────
const { values } = parseArgs({
  options: {
    version:  { type: 'string', short: 'v' },
    notes:    { type: 'string', short: 'n', default: '' },
    macos:    { type: 'string', default: '' },
    windows:  { type: 'string', default: '' },
    linux:    { type: 'string', default: '' },
    'dry-run':{ type: 'boolean', default: false },
  },
  strict: true,
});

if (!values.version) {
  console.error('Usage: node firebase_release.mjs --version "vX.X.X" [--macos path] [--windows path] [--notes "..."]');
  process.exit(1);
}

const VERSION = values.version;
const NOTES = values.notes;
const DRY_RUN = values['dry-run'];
const BUCKET = 'imagine-b1e9c.firebasestorage.app';
const PROJECT_ID = 'imagine-b1e9c';

// Firebase CLI OAuth2 client (public, embedded in firebase-tools)
const FIREBASE_CLIENT_ID = '563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com';
const FIREBASE_CLIENT_SECRET = 'j9iVZfS8kkCEFUPaAeJV0sAi';

// ── Get access token from Firebase CLI refresh token ─────
async function getAccessToken() {
  const configPath = join(homedir(), '.config', 'configstore', 'firebase-tools.json');
  if (!existsSync(configPath)) {
    throw new Error('Firebase CLI not logged in. Run: firebase login');
  }
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  const refreshToken = config.tokens && config.tokens.refresh_token;
  if (!refreshToken) {
    throw new Error('No refresh token. Run: firebase login');
  }

  const resp = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      client_id: FIREBASE_CLIENT_ID,
      client_secret: FIREBASE_CLIENT_SECRET,
      refresh_token: refreshToken,
    }),
  });

  if (!resp.ok) {
    throw new Error(`Token refresh failed: ${resp.status} ${await resp.text()}`);
  }
  const data = await resp.json();
  return data.access_token;
}

// ── Upload file to Firebase Storage (REST API) ──────────
async function uploadFile(localPath, platform, token) {
  if (!localPath || !existsSync(localPath)) {
    console.log(`  [SKIP] ${platform}: no file`);
    return null;
  }

  const fileName = basename(localPath);
  const destPath = `releases/${VERSION}/${fileName}`;
  const encodedPath = encodeURIComponent(destPath);
  const fileSize = statSync(localPath).size;
  const sizeMB = (fileSize / 1024 / 1024).toFixed(1);

  if (DRY_RUN) {
    console.log(`  [DRY] Would upload ${localPath} (${sizeMB} MB) -> ${destPath}`);
    return `https://firebasestorage.googleapis.com/v0/b/${BUCKET}/o/${encodedPath}?alt=media`;
  }

  const contentType = localPath.endsWith('.dmg') ? 'application/x-apple-diskimage'
    : localPath.endsWith('.zip') ? 'application/zip'
    : localPath.endsWith('.exe') ? 'application/x-msdownload'
    : 'application/octet-stream';

  console.log(`  Uploading ${fileName} (${sizeMB} MB, ${platform})...`);

  const fileData = readFileSync(localPath);
  const url = `https://firebasestorage.googleapis.com/v0/b/${BUCKET}/o?uploadType=media&name=${encodedPath}`;

  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': contentType,
      'Content-Length': String(fileSize),
    },
    body: fileData,
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Upload failed (${resp.status}): ${err}`);
  }

  const result = await resp.json();
  const downloadUrl = `https://firebasestorage.googleapis.com/v0/b/${BUCKET}/o/${encodedPath}?alt=media`;
  console.log(`  Done: ${downloadUrl}`);
  return downloadUrl;
}

// ── Register in Firestore (REST API) ────────────────────
async function registerRelease(assets, token) {
  const now = new Date().toISOString();
  const docId = VERSION.replace(/\./g, '_');

  // Build Firestore document fields
  const fields = {
    version: { stringValue: VERSION },
    date: { stringValue: now },
    notes: { stringValue: NOTES },
    created_at: { stringValue: now },
    assets: {
      mapValue: {
        fields: {}
      }
    }
  };

  for (const [platform, url] of Object.entries(assets)) {
    fields.assets.mapValue.fields[platform] = { stringValue: url };
  }

  if (DRY_RUN) {
    console.log('\n[DRY] Would write to Firestore:');
    console.log(JSON.stringify({ version: VERSION, date: now, notes: NOTES, assets }, null, 2));
    return;
  }

  const url = `https://firestore.googleapis.com/v1/projects/${PROJECT_ID}/databases/(default)/documents/releases/${docId}`;
  const resp = await fetch(url, {
    method: 'PATCH',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ fields }),
  });

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Firestore write failed (${resp.status}): ${err}`);
  }

  console.log(`\nRegistered release ${VERSION} in Firestore`);
}

// ── Main ────────────────────────────────────────────────
async function main() {
  console.log(`\nFirebase Release: ${VERSION}\n`);

  console.log('Getting access token...');
  const token = await getAccessToken();
  console.log('Authenticated.\n');

  const assets = {};

  const macUrl = await uploadFile(values.macos, 'macos', token);
  if (macUrl) assets.macos = macUrl;

  const winUrl = await uploadFile(values.windows, 'windows', token);
  if (winUrl) assets.windows = winUrl;

  const linuxUrl = await uploadFile(values.linux, 'linux', token);
  if (linuxUrl) assets.linux = linuxUrl;

  await registerRelease(assets, token);

  console.log('\nDone.\n');
}

main().catch(err => {
  console.error('Release failed:', err.message || err);
  process.exit(1);
});
