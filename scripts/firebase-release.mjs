#!/usr/bin/env node
/**
 * Firebase Release Script
 *
 * Uploads build artifacts to Firebase Storage and creates/updates
 * a Firestore release document. Designed to run in GitHub Actions.
 *
 * Usage:
 *   node scripts/firebase-release.mjs \
 *     --tag v0.6.3.20260306_02 \
 *     --notes release_notes.md \
 *     --windows artifacts/win/Imagine-win.zip
 *
 * Environment:
 *   FIREBASE_SERVICE_ACCOUNT - JSON string of service account key
 */

import { initializeApp, cert } from 'firebase-admin/app';
import { getStorage } from 'firebase-admin/storage';
import { getFirestore } from 'firebase-admin/firestore';
import { readFileSync, statSync } from 'fs';
import { basename } from 'path';
import { parseArgs } from 'util';

/* ── Parse CLI args ────────────────────────────────────── */
const { values: args } = parseArgs({
  options: {
    tag:     { type: 'string' },
    notes:   { type: 'string' },
    windows: { type: 'string' },
    macos:   { type: 'string' },
    linux:   { type: 'string' },
  },
});

if (!args.tag) {
  console.error('ERROR: --tag is required');
  process.exit(1);
}

/* ── Firebase init ─────────────────────────────────────── */
const saJson = process.env.FIREBASE_SERVICE_ACCOUNT;
if (!saJson) {
  console.error('ERROR: FIREBASE_SERVICE_ACCOUNT env var is required');
  process.exit(1);
}

const serviceAccount = JSON.parse(saJson);
const BUCKET = 'imagine-b1e9c.firebasestorage.app';

initializeApp({
  credential: cert(serviceAccount),
  storageBucket: BUCKET,
});

const bucket = getStorage().bucket();
const db = getFirestore();

/* ── Upload helper ─────────────────────────────────────── */
async function uploadArtifact(localPath, tag) {
  const fileName = basename(localPath);
  const destPath = `releases/${tag}/${fileName}`;

  console.log(`Uploading ${localPath} → gs://${BUCKET}/${destPath}`);

  const size = statSync(localPath).size;
  console.log(`  File size: ${(size / 1024 / 1024).toFixed(1)} MB`);

  await bucket.upload(localPath, {
    destination: destPath,
    metadata: {
      contentType: 'application/zip',
      metadata: { version: tag },
    },
  });

  // Make publicly accessible
  const file = bucket.file(destPath);
  await file.makePublic();

  const publicUrl = `https://storage.googleapis.com/${BUCKET}/${destPath}`;
  console.log(`  Public URL: ${publicUrl}`);
  return publicUrl;
}

/* ── Main ──────────────────────────────────────────────── */
async function main() {
  const tag = args.tag;
  const version = tag.startsWith('v') ? tag.slice(1) : tag;

  console.log(`\n=== Firebase Release: ${tag} ===\n`);

  // Upload platform artifacts
  const assets = { windows: '', macos: '', linux: '' };

  if (args.windows) {
    assets.windows = await uploadArtifact(args.windows, tag);
  }
  if (args.macos) {
    assets.macos = await uploadArtifact(args.macos, tag);
  }
  if (args.linux) {
    assets.linux = await uploadArtifact(args.linux, tag);
  }

  // Read release notes
  let notes = '';
  if (args.notes) {
    try {
      notes = readFileSync(args.notes, 'utf-8');
    } catch (e) {
      console.warn(`Warning: Could not read notes file: ${args.notes}`);
    }
  }

  // Upsert Firestore release document
  const releaseDoc = {
    version: tag,
    date: new Date().toISOString().split('T')[0],
    notes: notes,
    assets: assets,
    updated_at: new Date().toISOString(),
  };

  const docId = tag.replace(/[/.]/g, '_');
  console.log(`\nWriting Firestore: releases/${docId}`);
  await db.collection('releases').doc(docId).set(releaseDoc, { merge: true });

  console.log('\nRelease document:');
  console.log(JSON.stringify(releaseDoc, null, 2));
  console.log('\n=== Done ===\n');
}

main().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
