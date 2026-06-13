import { useState } from 'react'
import { useWorkers, useClusterValves, useWorkerControl } from '../api/admin'
import { useConnectionInfo } from '../api/connection'

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
  return (
    <>
      <div style={{ margin: '0 0 12px', padding: '8px 12px', borderRadius: 6, fontSize: 11.5, background: 'rgba(251,191,36,.10)', border: '1px solid rgba(251,191,36,.25)', color: 'var(--amber)' }}>
        ● 데모 — 멤버·플랜·초대는 인증 연동(IMGV2-9) 후 실데이터로 연결됩니다
      </div>
      <div className="panel">
        <h4>플랜 <span className="hint">서버 단위 라이선스 — 한도는 여기서 옴</span></h4>
        <div className="kpis">
          <div className="kpi"><div className="v" style={{ color: '#93c5fd' }}>스튜디오</div><div className="k">연 라이선스 · 2027-06-12 만료</div></div>
          <div className="kpi"><div className="v">5 <span className="faint" style={{ fontSize: 12 }}>/ 10</span></div><div className="k">좌석 — 멤버 3 + 대기 초대 2</div></div>
          <div className="kpi"><div className="v">3 <span className="faint" style={{ fontSize: 12 }}>/ 5</span></div><div className="k">분석기</div></div>
          <div className="kpi" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '7px 16px', borderRadius: 6 }}>갱신 / 업그레이드</button>
          </div>
        </div>
        <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 8 }}>사용 현황 보고: 멤버·분석기 수, 월 분석량 — 수치만 전송되며 파일 내용·이름은 절대 포함되지 않습니다</div>
        <div style={{ fontSize: 10, color: 'var(--amber)', marginTop: 4 }}>만료 시 이 서버 접속이 차단됩니다 (로그인은 되어도 입장 불가) · 데이터는 보존되며 갱신 즉시 복귀</div>
      </div>

      <div className="panel">
        <h4>초대 <span className="hint">이메일로 초대 → 메일의 링크로 수락 → 좌석 점유 · 초대는 그 이메일로만 수락 가능</span></h4>
        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <input placeholder="이메일 주소 — 쉼표로 여러 명" style={{ flex: 1, background: 'var(--panel2)', border: '1px solid var(--line)', borderRadius: 6, padding: '8px 12px', color: 'var(--text)', fontSize: 12 }} />
          <button style={{ background: 'var(--blue-d)', color: '#fff', fontWeight: 600, fontSize: 11.5, padding: '0 18px', borderRadius: 6 }}>초대 보내기</button>
        </div>
        <table>
          <thead><tr><th>대기 중 초대</th><th>보냄</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            {[['minsu@studio.kr', '2일 전'], ['art-extern@partner.co', '5시간 전']].map(([email, when]) => (
              <tr key={email}>
                <td>{email} <span style={{ fontSize: 9, color: 'var(--amber)' }}>좌석 예약 중</span></td>
                <td className="mono">{when}</td>
                <td><div className="row-act"><button>재발송</button><button className="danger">취소</button></div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h4>멤버 <span className="hint">제거 = 좌석 반환 · 비활성 = 접속만 차단(좌석 유지)</span></h4>
        <table>
          <thead><tr><th>이름</th><th>역할</th><th>가입 경로</th><th>마지막 접속</th><th style={{ textAlign: 'right' }} /></tr></thead>
          <tbody>
            <tr>
              <td>성철<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>saintiron82@gmail.com</div></td>
              <td><span className="badge" style={{ background: 'rgba(192,132,252,.15)', color: 'var(--purple)' }}>운영자</span></td>
              <td className="mono" style={{ fontSize: 10 }}>서버 생성자</td><td className="mono">방금</td><td />
            </tr>
            <tr>
              <td>지민<div style={{ fontSize: 9.5, color: 'var(--faint)' }}>jimin@studio.kr</div></td>
              <td><span className="badge b-ok">사용자</span></td>
              <td className="mono" style={{ fontSize: 10 }}>초대 수락</td><td className="mono">2시간 전</td>
              <td><div className="row-act"><button>역할</button><button className="danger">비활성</button></div></td>
            </tr>
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
