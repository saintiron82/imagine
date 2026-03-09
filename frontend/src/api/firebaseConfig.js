/**
 * Firebase app initialization.
 * Project: imagine-b1e9c
 */
import { initializeApp } from 'firebase/app';
import { getAuth, setPersistence, browserLocalPersistence } from 'firebase/auth';

const firebaseConfig = {
  apiKey: "AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc",
  authDomain: "imagine-b1e9c.firebaseapp.com",
  databaseURL: "https://imagine-b1e9c-default-rtdb.firebaseio.com",
  projectId: "imagine-b1e9c",
  storageBucket: "imagine-b1e9c.firebasestorage.app",
  messagingSenderId: "978580126686",
  appId: "1:978580126686:web:df4b17033cc8daca55fb4f",
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Persist auth state across browser restarts (IndexedDB-backed)
setPersistence(auth, browserLocalPersistence);

export { auth };
export default app;
