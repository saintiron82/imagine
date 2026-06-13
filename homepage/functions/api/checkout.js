/**
 * POST /api/checkout — 결제 시작 또는 체험 발급.
 * body: { plan: 'trial'|'studio'|'team', group_name, id_token, renew? }
 *  - trial: 14일 체험 라이선스를 Firestore 원장에 즉시 기록 → { trial: true }
 *  - paid : Stripe Checkout 세션 생성 → { url }
 *
 * 결제 확정(좌석/만료 부여)은 webhook 에서 일어난다 — 여기서는 체험만 즉시 기록.
 */
import {
  verifyFirebaseIdToken, serviceAccountToken, firestoreSetGroup,
  stripeCreateCheckout, json,
} from '../_shared.js'

const LIMITS = {
  trial: { seats: 3, analyzers: 2, days: 14 },
  studio: { seats: 10, analyzers: 5, priceEnv: 'STRIPE_PRICE_STUDIO' },
  team: { seats: 25, analyzers: 12, priceEnv: 'STRIPE_PRICE_TEAM' },
}

export async function onRequestPost({ request, env }) {
  let body
  try { body = await request.json() } catch { return json({ error: 'Bad JSON' }, 400) }
  const { plan, group_name, id_token, renew } = body || {}
  const limit = LIMITS[plan]
  if (!limit || !group_name || !id_token) return json({ error: 'plan, group_name, id_token required' }, 400)

  let user
  try { user = await verifyFirebaseIdToken(id_token) }
  catch (e) { return json({ error: 'auth: ' + e.message }, 401) }

  // 14일 체험 — 결제 없이 즉시 발급
  if (plan === 'trial') {
    if (!env.FIREBASE_SERVICE_ACCOUNT) return json({ error: 'Server not configured (FIREBASE_SERVICE_ACCOUNT)' }, 501)
    try {
      const token = await serviceAccountToken(env)
      const expires = new Date(Date.now() + limit.days * 86400000).toISOString()
      await firestoreSetGroup(env, token, group_name, {
        plan_id: 'trial', status: 'active', owner_uid: user.uid, owner_email: user.email,
        seat_limit: limit.seats, analyzer_limit: limit.analyzers, expires_at: expires,
      })
      return json({ trial: true, group: group_name, expires_at: expires })
    } catch (e) { return json({ error: e.message }, 500) }
  }

  // 유료 — Stripe Checkout
  const priceId = env[limit.priceEnv]
  if (!env.STRIPE_SECRET_KEY || !priceId) return json({ error: 'Stripe not configured (STRIPE_SECRET_KEY / ' + limit.priceEnv + ')' }, 501)
  try {
    const session = await stripeCreateCheckout(env, {
      priceId, group: group_name, uid: user.uid, email: user.email, plan, renew: !!renew,
    })
    return json({ url: session.url })
  } catch (e) { return json({ error: e.message }, 500) }
}
