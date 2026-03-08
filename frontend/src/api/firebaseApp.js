/**
 * Firebase SDK initialization — App + Auth instances.
 *
 * Firebase project: imagine-b1e9c
 * Uses browserLocalPersistence for automatic re-login (cookie-like behavior).
 */

import { initializeApp } from 'firebase/app';
import { getAuth, setPersistence, browserLocalPersistence } from 'firebase/auth';

const firebaseConfig = {
  apiKey: "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  // TODO: Replace with real key
  authDomain: "imagine-b1e9c.firebaseapp.com",
  databaseURL: "https://imagine-b1e9c-default-rtdb.firebaseio.com",
  projectId: "imagine-b1e9c",
  storageBucket: "imagine-b1e9c.appspot.com",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Persist auth state across browser restarts (IndexedDB-backed)
setPersistence(auth, browserLocalPersistence);

export { app, auth };
