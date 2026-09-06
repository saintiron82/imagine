/**
 * Firebase Auth wrapper — sign up, sign in, sign out, state observation.
 */

import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  signInWithCredential,
  GoogleAuthProvider,
  signOut as firebaseSignOut,
  onAuthStateChanged as firebaseOnAuthStateChanged,
  updateProfile,
} from 'firebase/auth';
import { auth } from './firebaseApp';

const googleProvider = new GoogleAuthProvider();
const isElectron = !!window.electron;

/**
 * Create a new Firebase account.
 * @param {string} email
 * @param {string} password
 * @param {string} [displayName]
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signUp(email, password, displayName) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  if (displayName) {
    await updateProfile(cred.user, { displayName });
  }
  return cred;
}

/**
 * Sign in with Google.
 * In Electron: opens a BrowserWindow via IPC to avoid unauthorized-domain errors.
 * In browser: uses Firebase signInWithPopup.
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signInWithGoogle() {
  if (isElectron && window.electron?.auth?.googleOAuth) {
    const { idToken } = await window.electron.auth.googleOAuth();
    const credential = GoogleAuthProvider.credential(idToken);
    return signInWithCredential(auth, credential);
  }
  return signInWithPopup(auth, googleProvider);
}

/**
 * Sign in with email and password.
 * @param {string} email
 * @param {string} password
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signIn(email, password) {
  return signInWithEmailAndPassword(auth, email, password);
}

/**
 * Sign out from Firebase.
 */
export async function signOut() {
  return firebaseSignOut(auth);
}

/**
 * Get the current Firebase ID token (auto-refreshes if expired).
 * @returns {Promise<string|null>}
 */
export async function getIdToken() {
  const user = auth.currentUser;
  if (!user) return null;
  return user.getIdToken(/* forceRefresh */ false);
}

/**
 * Subscribe to auth state changes.
 * @param {(user: import('firebase/auth').User | null) => void} callback
 * @returns {() => void} unsubscribe function
 */
export function onAuthStateChanged(callback) {
  return firebaseOnAuthStateChanged(auth, callback);
}
