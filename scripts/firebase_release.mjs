#!/usr/bin/env node
/**
 * Hybrid release script — GitHub Releases (files) + Firestore (metadata).
 *
 * 1. Uploads macOS DMG/zip to existing GitHub Release via `gh release upload`
 * 2. Reads GitHub Release asset URLs
 * 3. Registers release metadata in Firestore (REST API, Firebase CLI token)
 *
 * Usage:
 *   node scripts/firebase_release.mjs --version "v0.6.4.20260309_01" \
 *     --macos "frontend/dist-electron/Imagine-0.6.4-arm64.dmg" \
 *     --notes "Release notes here"
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import { execSync } from 'child_process';
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
    'skip-upload': { type: 'boolean', default: false },
  },
  strict: true,
});

if (!values.version) {
  console.error('Usage: node firebase_release.mjs --version "vX.X.X" [--macos path] [--notes "..."]');
  process.exit(1);
}

const VERSION = values.version;
const NOTES = values.notes;
const DRY_RUN = values['dry-run'];
const SKIP_UPLOAD = values['skip-upload'];
const PROJECT_ID = 'imagine-b1e9c';

// Firebase CLI OAuth2 client (public, embedded in firebase-tools)
const FIREBASE_CLIENT_ID = '563584335869-fgrhgmd47bqnekij5i8b5pr03ho849e6.apps.googleusercontent.com';
const FIREBASE_CLIENT_SECRET = 'j9iVZfS8kkCEFUPaAeJV0sAi';

// ── Get Firebase access token ────────────────────────────
async function getFirebaseAccessToken() {
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
  return (await resp.json()).access_token;
}

// ── Upload file to GitHub Release ────────────────────────
function uploadToGitHub(localPath, platform) {
  if (!localPath || !existsSync(localPath)) {
    console.log(`  [SKIP] ${platform}: no file`);
    return;
  }

  if (DRY_RUN) {
    console.log(`  [DRY] Would upload ${localPath} to GitHub Release ${VERSION}`);
    return;
  }

  if (SKIP_UPLOAD) {
    console.log(`  [SKIP-UPLOAD] ${platform}: ${localPath}`);
    return;
  }

  console.log(`  Uploading ${localPath} to GitHub Release ${VERSION}...`);
  try {
    execSync(`gh release upload "${VERSION}" "${localPath}" --clobber`, {
      stdio: 'inherit',
    });
    console.log(`  Done: ${platform} uploaded to GitHub Release`);
  } catch (err) {
    console.error(`  WARNING: GitHub upload failed for ${platform}: ${err.message}`);
    console.error(`  You may need to create the release first: gh release create "${VERSION}"`);
  }
}

// ── Get GitHub Release asset URLs ────────────────────────
function getGitHubAssetUrls() {
  try {
    const output = execSync(`gh release view "${VERSION}" --json assets --jq '.assets[].url'`, {
      encoding: 'utf8',
    }).trim();

    const urls = {};
    for (const url of output.split('\n').filter(Boolean)) {
      const lower = url.toLowerCase();
      if (lower.includes('arm64.dmg') || lower.includes('mac')) {
        urls.macos = url;
      } else if (lower.includes('win') || lower.includes('.exe')) {
        urls.windows = url;
      } else if (lower.includes('linux') || lower.includes('.appimage')) {
        urls.linux = url;
      }
    }
    return urls;
  } catch {
    console.warn('  Could not fetch GitHub Release assets (release may not exist yet)');
    return {};
  }
}

// ── Register in Firestore (REST API) ────────────────────
async function registerInFirestore(assets, token) {
  const now = new Date().toISOString();
  const docId = VERSION.replace(/\./g, '_');

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
    if (url) {
      fields.assets.mapValue.fields[platform] = { stringValue: url };
    }
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
  console.log(`\nHybrid Release: ${VERSION}\n`);

  // Step 1: Upload files to GitHub Release
  console.log('Step 1: Upload to GitHub Releases');
  uploadToGitHub(values.macos, 'macos');
  uploadToGitHub(values.windows, 'windows');
  uploadToGitHub(values.linux, 'linux');

  // Step 2: Get asset URLs from GitHub Release
  console.log('\nStep 2: Fetch GitHub Release asset URLs');
  const assets = DRY_RUN ? {} : getGitHubAssetUrls();
  console.log('  Assets:', JSON.stringify(assets, null, 2));

  // Step 3: Update Firestore metadata
  console.log('\nStep 3: Update Firestore metadata');
  console.log('  Getting Firebase access token...');
  const token = await getFirebaseAccessToken();
  console.log('  Authenticated.');

  await registerInFirestore(assets, token);

  console.log('\nDone.\n');
}

main().catch(err => {
  console.error('Release failed:', err.message || err);
  process.exit(1);
});
