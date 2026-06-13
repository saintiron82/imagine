import { useState } from 'react'
import { useWorkers, useClusterValves, useWorkerControl } from '../api/admin'
import { useConnectionInfo } from '../api/connection'
import { useMembersData, useMemberMutations } from '../api/members'

/**
 * 관리 — 엔진룸. 기술 용어(MC/VV/MV)는 운영자 전용인 이 화면에만 허용된다.
 * 분석기: 병목 진단 + 분석기별 실시간(admin/workers) + 정지/차단/차단해제.
 * 멤버/플랜: 인증 연동(IMGV2-9) 전까지 데모.
 */
export default function AdminScreen() {
  const [tab, setTab] = useState('workers')

  return (
    <section id="scr-admin" className="screen active" style={{ height: '100%' }}>
      <aside className="adm-side">
        {[['workers', '분석기'], ['members', '멤버'], ['tools', '서버 도구']].map(([id, label]) => (
          <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </aside>
      <div className="adm-main">
        {tab === 'workers' && <WorkersPanel />}
        {tab === 'members' && <MembersPanel />}
        {tab === 'tools' && <ToolsPanel />}
      </div>
    </section>
  )
}

const ORIGIN_LABEL = (w) => [w.origin, w.launcher].filter(Boolean).join(' · ') || '—'
const PHASE_KR = { mc: 'MC', vv: 'VV', mv: 'MV', parse: '파싱', dl: 'DL' }

function WorkersPanel() {
  const { isDemo, workers, globalMode } = useWorkers()
  const { valves, bottleneck, total } = useClusterValves()
  const { stop, block, unblock } = useWorkerControl()
  const canControl = !isDemo

  const online = workers.filter(w => w.status === 'online')
  const blocked = workers.filter(w => w.status === 'blocked')

  return (
    <>
      <div className="panel">
        <h4>자동 분석 <span className="hint">서버 전역 정책 · 현재 모드 {globalMode}{isDemo && ' · ● 데모'}</span><span className="toggle" /></h4>
        {bottleneck && bottleneck.pending > 0 ? (
          <div className="bottleneck">
            현재 병목: <b>{bottleneck.label}</b> — 대기 {bottleneck.pending.toLocaleString()}장 · 클러스터 처리력 <b>분당 {bottleneck.rate ?? 0}장</b> (이 단계가 전체 완료 속도를 결정)
            <div className="why">처리력 여유 단계는 이 단계 산출을 기다리는 중 — 병목 단계에 분석기를 더 붙이면 전체가 빨라집니다</div>
          </div>
        ) : (
          <div className="bottleneck" style={{ borderColor: 'var(--emerald-d)' }}>
            병목 없음 — 모든 단계가 따라잡았습니다
          </div>
        )}
        <div className="valves">
          {valves.map(v => {
            const isBn = bottleneck && v.phase === bottleneck.phase && v.pending > 0
            const rateTxt = v.rate != null ? `${v.rate}/분` : (v.done >= v.total ? '완료' : '진행 중')
            return (
              <div className={`valve ${isBn ? 'bn' : ''}`} key={v.phase}>
                <div className="ph">{v.label}{isBn && <span className="bn-tag">병목</span>}</div>
                <div className="n">{v.done.toLocaleString()}/{v.total.toLocaleString()}</div>
                <div className="rate">{rateTxt}</div>
                <span className="sw on">ON</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="panel">
        <h4>분석기 <span className="hint">{online.length}대 온라인 — 정지=이번 세션 종료(재시작 가능) · 차단=재접속 거부{isDemo && ' · ● 데모'}</span></h4>
        <table>
          <thead>
            <tr><th>이름</th><th>상태</th><th>지금 처리 중</th><th>처리율</th><th>누적</th><th style={{ textAlign: 'right' }}>제어</th></tr>
          </thead>
          <tbody>
            {online.map(w => (
              <tr key={w.id}>
                <td><b>{w.worker_name}</b><div style={{ fontSize: 10, color: 'var(--faint)' }}>{ORIGIN_LABEL(w)}</div></td>
                <td><span className="badge b-ok">온라인</span></td>
                <td>{w.current_file
                  ? <span>{w.current_file} {w.current_phase && <span style={{ color: 'var(--cyan)', fontSize: 10 }}>{PHASE_KR[w.current_phase] || w.current_phase}</span>}</span>
                  : <span style={{ color: 'var(--faint)' }}>대기</span>}</td>
                <td className="mono">{w.throughput != null ? `${Number(w.throughput).toFixed(Number(w.throughput) < 10 ? 1 : 0)}/분` : '—'}{w.throughput_mode && <span style={{ color: 'var(--faint)', fontSize: 10 }}> {PHASE_KR[w.throughput_mode] || w.throughput_mode}</span>}</td>
                <td className="mono">{(w.jobs_completed || 0).toLocaleString()}</td>
                <td><div className="row-act">
                  <button disabled={!canControl || stop.isPending} onClick={() => stop.mutate(w.id)}>정지</button>
                  <button className="danger" disabled={!canControl || block.isPending} onClick={() => block.mutate(w.id)}>차단</button>
                </div></td>
              </tr>
            ))}
            {online.length === 0 && (
              <tr><td colSpan={6} style={{ color: 'var(--faint)' }}>온라인 분석기가 없습니다</td></tr>
            )}
          </tbody>
        </table>
        {blocked.map(w => (
          <div className="blocked" key={w.id}>
            ⛔ <b>{w.worker_name}</b> <span className="faint">— 차단됨 (재접속 불가)</span>
            <button className="unb" disabled={!canControl || unblock.isPending} onClick={() => unblock.mutate(w.id)}>차단 해제</button>
          </div>
        ))}
      </div>
    </>
  )
}

function MembersPanel() {
  const { isDemo, members, invites, usage } = useMembersData()
  const { invite, revoke, remove, deactivate } = useMemberMutations()
  const [emails, setEmails] = useState('')
  const canMutate = !isDemo

  const sendInvites = () => {
    const list = emails.split(',').map(s => s.trim()).filter(Boolean)
    if (!list.length) return
    invite.mutate({ emails: list }, { onSuccess: () => setEmails('') })
  }
  const inviteMsg = invite.data
    ? `${invite.data.results.filter(r => r.ok).length}건 처리${invite.data.smtp_configured ? ' · 메일 발송' : ' · 링크 생성(메일 미설정)'}`
    : invite.isPending ? '초대 중…' : null

  return (
    <>
      <div className="panel">
        <h4>플랜 <span className="hint">서버 단위 라이선스 — 한도는 여기서 옴{isDemo && ' · ● 데모'}</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: usage.expired ? 'var(--red)' : '#93c5fd' }}>{usage.expired ? '만료됨' : '활성'}</div><div className="k">{usage.expires_at ? `${usage.expires_at} 만료` : '만료 없음'}</div></div>
          <div className="kpi"><div className="v">{usage.seats_used} <span className="faint" style={{ fontSize: 12 }}>/ {usage.seat_limit || '∞'}</span></div><div className="k">좌석 — 멤버 {usage.members} + 대기 초대 {usage.pending_invites}</div></div>
          <div className="kpi"><div className="v" style={{ color: usage.smtp_configured ? 'var(--emerald)' : 'var(--faint)' }}>{usage.smtp_configured ? '메일 ON' : '메일 OFF'}</div><div className="k">초대 발송 방식</div></div>
          <div className="kpi" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '7px 16px', borderRadius: 6 }}>갱신 / 업그레이드</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 6 }}>만료 시 이 서버 접속이 차단됩니다 (로그인은 되어도 입장 불가) · 데이터는 보존되며 갱신 즉시 복귀</div>
      </div>

      <div className="panel">
        <h4>초대 <span className="hint">이메일로 초대 → 링크 수락 → 좌석 점유 · 그 이메일로만 수락 가능{!usage.smtp_configured && ' · 메일 미설정 시 링크 수동 공유'}</span></h4>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <input placeholder="이메일 주소 — 쉼표로 여러 명" value={emails} onChange={e => setEmails(e.target.value)}
            style={{ flex: 1, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 12px', color: 'var(--text)', fontSize: 12 }} />
          <button onClick={sendInvites} disabled={!canMutate || invite.isPending || !emails.trim()}
            style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '0 18px', borderRadius: 6, opacity: canMutate ? 1 : .5 }}>초대 보내기</button>
        </div>
        {inviteMsg && <div style={{ fontSize: 10.5, color: 'var(--cyan)', marginBottom: 8 }}>{inviteMsg}</div>}
        {isDemo && <div style={{ fontSize: 9.5, color: 'var(--faint)', marginBottom: 8 }}>서버 연결 시 실제 초대가 발송됩니다</div>}
        <table>
          <thead><tr><th>대기 중 초대</th><th>역할</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {invites.map(iv => (
              <tr key={iv.id}>
                <td>{iv.email} <span style={{ fontSize: 9, color: 'var(--amber)' }}>좌석 예약 중</span></td>
                <td className="mono">{iv.role}</td>
                <td><div className="row-act"><button className="danger" disabled={!canMutate} onClick={() => revoke.mutate(iv.id)}>취소</button></div></td>
              </tr>
            ))}
            {invites.length === 0 && <tr><td colSpan={3} style={{ color: 'var(--faint)' }}>대기 중 초대 없음</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h4>멤버 <span className="hint">제거 = 좌석 반환 · 비활성 = 접속만 차단(좌석 유지)</span></h4>
        <table>
          <thead><tr><th>이름</th><th>역할</th><th>가입 경로</th><th>마지막 접속</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {members.map(mb => (
              <tr key={mb.id}>
                <td>{mb.username}<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>{mb.email}</div></td>
                <td>{mb.role === 'admin'
                  ? <span className="badge" style={{ background: 'rgba(192,132,252,.15)', color: 'var(--purple)' }}>운영자</span>
                  : <span className="badge b-ok">{mb.is_active ? '사용자' : '비활성'}</span>}</td>
                <td className="mono" style={{ fontSize: 10 }}>{mb.via === 'firebase' ? '초대/Firebase' : '비밀번호'}</td>
                <td className="mono">{mb.last_login_at || '—'}</td>
                <td>{mb.role !== 'admin' && (
                  <div className="row-act">
                    <button disabled={!canMutate} onClick={() => deactivate.mutate(mb.id)}>비활성</button>
                    <button className="danger" disabled={!canMutate} onClick={() => remove.mutate(mb.id)}>제거</button>
                  </div>
                )}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function ToolsPanel() {
  const { isDemo, external, lan } = useConnectionInfo()
  const extDesc = external?.available
    ? `● 연결됨 — ${external.url || '터널'} · 주소 공유 불필요`
    : '○ 외부 접속 꺼짐 — 터널 미연결'
  const lanDesc = lan?.available ? `${lan.url} · 빠른 경로 (자동 발견)` : 'LAN 경로 없음'

  const rows = [
    ['외부 접속 (Cloudflare 터널)', extDesc, ['재연결'], external?.available ? 'var(--emerald)' : 'var(--faint)'],
    ['같은 네트워크 (LAN)', lanDesc, []],
    ['정합성 감사', '미완성/잔여 작업 검사·복구', ['실행']],
    ['해시 백필', '파생물 캐시 자격 부여 (~16KB/파일)', ['실행']],
    ['DB 내보내기 / 가져오기', '', ['내보내기', '가져오기']],
    ['DB 초기화', '비밀번호 확인 필요', ['초기화…'], 'var(--red)', true],
  ]
  return (
    <div className="panel">
      <h4>서버 도구 <span className="hint">DB·정합성·연결 — 구 헤더 DB 메뉴가 여기로{isDemo && ' · ● 데모'}</span></h4>
      <table>
        <tbody>
          {rows.map(([name, desc, actions, color, danger]) => (
            <tr key={name}>
              <td style={danger ? { color: 'var(--red)' } : undefined}>{name}</td>
              <td style={{ color: color || 'var(--faint)', fontSize: 11 }}>{desc}</td>
              <td><div className="row-act">{actions.map(a => <button key={a} className={danger ? 'danger' : ''}>{a}</button>)}</div></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
