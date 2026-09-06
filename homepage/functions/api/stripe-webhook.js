/**
 * POST /api/stripe-webhook — Stripe 결제 확정 시 라이선스 원장에 좌석/만료 부여.
 * checkout.session.completed → metadata{group_name, owner_uid, plan} 로 Firestore
 * groups/{group_name} 에 plan/좌석/분석기 한도 + 만료(+1년) 기록 → 앱이 즉시 읽음.
 *
 * Stripe 대시보드에서 이 엔드포인트를 webhook 으로 등록하고 STRIPE_WEBHOOK_SECRET 설정.
 */
import { serviceAccountToken, firestoreSetGroup, verifyStripeSignature, json } from '../_shared.js'

const LIMITS = {
  studio: { seats: 10, analyzers: 5 },
  team: { seats: 25, analyzers: 12 },
}

export async function onRequestPost({ request, env }) {
  const payload = await request.text()
  const sig = request.headers.get('stripe-signature')
  if (env.STRIPE_WEBHOOK_SECRET) {
    const ok = await verifyStripeSignature(payload, sig, env.STRIPE_WEBHOOK_SECRET).catch(() => false)
    if (!ok) return json({ error: 'bad signature' }, 400)
  }
  let event
  try { event = JSON.parse(payload) } catch { return json({ error: 'bad json' }, 400) }

  if (event.type !== 'checkout.session.completed') return json({ received: true })

  const md = (event.data?.object?.metadata) || {}
  const group = md.group_name
  const plan = md.plan || 'studio'
  const limit = LIMITS[plan] || LIMITS.studio
  if (!group) return json({ received: true, skipped: 'no group' })

  try {
    const token = await serviceAccountToken(env)
    // 연 단위 — 갱신이면 기존 만료에 누적하기보다 결제일 +1년으로 단순화(필요시 누적 로직 추가)
    const expires = new Date(Date.now() + 365 * 86400000).toISOString()
    await firestoreSetGroup(env, token, group, {
      plan_id: plan, status: 'active', owner_uid: md.owner_uid || '',
      seat_limit: limit.seats, analyzer_limit: limit.analyzers, expires_at: expires,
      stripe_customer: event.data.object.customer || '',
    })
    return json({ received: true, group, plan })
  } catch (e) {
    return json({ error: e.message }, 500)
  }
}
