import { Link } from 'react-router-dom'

/** 4장: 다운로드 — 앱 받기 + 서버 만들기 안내(키만 있으면 홈페이지 없이 동작). */
const BUILDS = [
  { os: 'macOS', note: 'Apple Silicon · Intel', href: '#' },
  { os: 'Windows', note: 'x64', href: '#' },
  { os: 'Linux', note: 'AppImage · headless 분석기', href: '#' },
]
export default function Download() {
  return (
    <div className="wrap">
      <div className="section-title">다운로드</div>
      <p className="muted">앱은 키만 있으면 홈페이지 없이 동작합니다. 설치 후 [이 컴퓨터를 서버로 만들기]에서 구매한 그룹 이름으로 서버를 켜세요.</p>
      <div className="plans" style={{ gridTemplateColumns: 'repeat(3,1fr)' }}>
        {BUILDS.map(b => (
          <div className="plan" key={b.os}>
            <div className="nm">{b.os}</div>
            <div className="bl">{b.note}</div>
            <div className="pick"><a className="btn" style={{ width: '100%', textAlign: 'center' }} href={b.href}>받기</a></div>
          </div>
        ))}
      </div>
      <div className="card">
        <b>처음이신가요?</b>
        <ol style={{ margin: '10px 0 0 18px', color: 'var(--dim)', fontSize: 14, lineHeight: 1.9 }}>
          <li><Link to="/buy" style={{ color: 'var(--blue)' }}>플랜 구매 또는 14일 체험</Link> — 서버(그룹) 이름을 정합니다</li>
          <li>앱 설치 → 로그인 → [이 컴퓨터를 서버로 만들기] → 같은 그룹 이름 선택</li>
          <li>팀원을 이메일로 초대 — 받은 사람이 그 이메일로 로그인하면 좌석을 점유합니다</li>
        </ol>
      </div>
    </div>
  )
}
