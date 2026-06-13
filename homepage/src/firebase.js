/**
 * Firebase — 신원(Auth) + 라이선스 원장(Firestore groups/{key}).
 * 앱(frontend)과 같은 프로젝트 imagine-b1e9c 를 공유한다 — 홈페이지는 구매로 이
 * 원장에 라이선스를 쓰고(서버측 webhook), 앱은 lookupGroup 으로 읽는다.
 */
import { initializeApp } from 'firebase/app'
import { getAuth, setPersistence, browserLocalPersistence } from 'firebase/auth'
import { getFirestore } from 'firebase/firestore'

const firebaseConfig = {
  apiKey: 'AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc',
  authDomain: 'imagine-b1e9c.firebaseapp.com',
  databaseURL: 'https://imagine-b1e9c-default-rtdb.firebaseio.com',
  projectId: 'imagine-b1e9c',
  storageBucket: 'imagine-b1e9c.firebasestorage.app',
  messagingSenderId: '978580126686',
  appId: '1:978580126686:web:df4b17033cc8daca55fb4f',
}

const app = initializeApp(firebaseConfig)
export const auth = getAuth(app)
export const dbf = getFirestore(app)
setPersistence(auth, browserLocalPersistence)
