/**
 * Firebase Auth client — handles personal identity (Layer 1).
 *
 * Supports Google sign-in and email/password authentication.
 * Server authorization (Layer 2) is handled separately via server_password.
 */
import { auth } from './firebaseConfig';
import {
  signInWithPopup,
  GoogleAuthProvider,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged as firebaseOnAuthStateChanged,
  updateProfile,
} from 'firebase/auth';

const googleProvider = new GoogleAuthProvider();

/**
 * Sign in with Google popup.
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signInWithGoogle() {
  return signInWithPopup(auth, googleProvider);
}

/**
 * Sign in with email and password.
 * @param {string} email
 * @param {string} password
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signInWithEmail(email, password) {
  return signInWithEmailAndPassword(auth, email, password);
}

/**
 * Create account with email and password.
 * @param {string} email
 * @param {string} password
 * @param {string} [displayName]
 * @returns {Promise<import('firebase/auth').UserCredential>}
 */
export async function signUpWithEmail(email, password, displayName) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  if (displayName) {
    await updateProfile(cred.user, { displayName });
  }
  return cred;
}

/**
 * Sign out from Firebase.
 * @returns {Promise<void>}
 */
export async function signOut() {
  return firebaseSignOut(auth);
}

/**
 * Get current user's Firebase ID token.
 * @returns {Promise<string|null>}
 */
export async function getIdToken() {
  const user = auth.currentUser;
  if (!user) return null;
  return user.getIdToken();
}

/**
 * Get current Firebase user.
 * @returns {import('firebase/auth').User|null}
 */
export function getCurrentUser() {
  return auth.currentUser;
}

/**
 * Listen to auth state changes.
 * @param {function} callback - Called with (user) on auth state change
 * @returns {function} Unsubscribe function
 */
export function onAuthStateChanged(callback) {
  return firebaseOnAuthStateChanged(auth, callback);
}
