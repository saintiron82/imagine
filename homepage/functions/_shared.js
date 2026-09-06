/**
 * Cloudflare Pages Functions 공용 헬퍼 (Workers 런타임 — Web Crypto/fetch 기반).
 * - verifyFirebaseIdToken: Firebase REST(accounts:lookup)로 ID 토큰 검증 → {uid,email}
 * - serviceAccountToken: 서비스 계정 JWT(RS256) 서명 → OAuth access token(Firestore 쓰기용)
 * - firestoreSetGroup: groups/{key} 문서 머지 업데이트(라이선스 원장)
 * - stripeCreateCheckout: Stripe Checkout 세션 생성
 * - verifyStripeSignature: 웹훅 서명 검증(HMAC-SHA256)
 */

const FIREBASE_API_KEY = 'AIzaSyDgpwrJbQ8MYkP3NFAOrp-K8R3e8kaWpCc' // 공개 웹 키

export async function verifyFirebaseIdToken(idToken) {
  const r = await fetch(
    `https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${FIREBASE_API_KEY}`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ idToken }) },
  )
  if (!r.ok) throw new Error('Invalid Firebase token')
  const data = await r.json()
  const u = data.users && data.users[0]
  if (!u) throw new Error('Unknown user')
  return { uid: u.localId, email: u.email || '' }
}

// ── Service-account OAuth (for Firestore admin writes) ───────
function b64url(buf) {
  let s = typeof buf === 'string' ? btoa(buf) : btoa(String.fromCharCode(...new Uint8Array(buf)))
  return s.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
function pemToArrayBuffer(pem) {
  const b64 = pem.replace(/-----BEGIN PRIVATE KEY-----/, '').replace(/-----END PRIVATE KEY-----/, '').replace(/\s+/g, '')
  const bin = atob(b64)
  const buf = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i)
  return buf.buffer
}

export async function serviceAccountToken(env) {
  const sa = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT)
  const now = Math.floor(Date.now() / 1000)
  const header = b64url(JSON.stringify({ alg: 'RS256', typ: 'JWT' }))
  const claim = b64url(JSON.stringify({
    iss: sa.client_email, scope: 'https://www.googleapis.com/auth/datastore',
    aud: 'https://oauth2.googleapis.com/token', iat: now, exp: now + 3600,
  }))
  const data = `${header}.${claim}`
  const key = await crypto.subtle.importKey(
    'pkcs8', pemToArrayBuffer(sa.private_key),
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['sign'],
  )
  const sig = await crypto.subtle.sign('RSASSA-PKCS1-v1_5', key, new TextEncoder().encode(data))
  const jwt = `${data}.${b64url(sig)}`
  const r = await fetch('https://oauth2.googleapis.com/token', {
    method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  })
  if (!r.ok) throw new Error('Service account auth failed: ' + (await r.text()))
  return (await r.json()).access_token
}

// Firestore typed-value encoding (subset)
function fsValue(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? { integerValue: String(v) } : { doubleValue: v }
  if (typeof v === 'boolean') return { booleanValue: v }
  return { stringValue: String(v) }
}

export async function firestoreSetGroup(env, token, key, fields) {
  const pid = env.FIREBASE_PROJECT_ID || 'imagine-b1e9c'
  const mask = Object.keys(fields).map(k => `updateMask.fieldPaths=${encodeURIComponent(k)}`).join('&')
  const url = `https://firestore.googleapis.com/v1/projects/${pid}/databases/(default)/documents/groups/${encodeURIComponent(key)}?${mask}`
  const body = { fields: Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, fsValue(v)])) }
  const r = await fetch(url, {
    method: 'PATCH', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error('Firestore write failed: ' + (await r.text()))
  return r.json()
}

// ── Stripe ───────────────────────────────────────────────────
export async function stripeCreateCheckout(env, { priceId, group, uid, email, plan, renew }) {
  const base = env.PUBLIC_BASE_URL || 'https://imagine.app'
  const form = new URLSearchParams()
  form.set('mode', 'subscription')
  form.set('line_items[0][price]', priceId)
  form.set('line_items[0][quantity]', '1')
  form.set('success_url', `${base}/account?ok=1`)
  form.set('cancel_url', `${base}/buy?canceled=1`)
  if (email) form.set('customer_email', email)
  form.set('client_reference_id', group)
  form.set('metadata[group_name]', group)
  form.set('metadata[owner_uid]', uid)
  form.set('metadata[plan]', plan)
  if (renew) form.set('metadata[renew]', '1')
  const r = await fetch('https://api.stripe.com/v1/checkout/sessions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.STRIPE_SECRET_KEY}`, 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form.toString(),
  })
  if (!r.ok) throw new Error('Stripe error: ' + (await r.text()))
  return r.json()
}

export async function verifyStripeSignature(payload, sigHeader, secret) {
  if (!sigHeader) return false
  const parts = Object.fromEntries(sigHeader.split(',').map(p => p.split('=')))
  const signedPayload = `${parts.t}.${payload}`
  const key = await crypto.subtle.importKey(
    'raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'],
  )
  const mac = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(signedPayload))
  const expected = [...new Uint8Array(mac)].map(b => b.toString(16).padStart(2, '0')).join('')
  return expected === parts.v1
}

export const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } })
