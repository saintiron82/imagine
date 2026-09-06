import { useParams, Link } from 'react-router-dom'

/** 5장: join/<code> 랜딩 — 초대 딥링크 + 설치 유도.
 * 초대는 이메일 바인딩 — 코드 자체보다 '초대받은 이메일로 로그인'이 수락의 핵심.
 */
export default function Join() {
  const { code } = useParams()
  return (
    <div className="wrap">
      <section className="hero" style={{ paddingBottom: 30 }}>
        <div className="tagline">팀 초대</div>
        <h1>Imagine 라이브러리에<br />초대되었습니다</h1>
        <p>아래 순서대로 하면 좌석을 점유하고 팀의 검색 라이브러리에 합류합니다.</p>
      </section>
      <div className="card">
        <div className="kv"><span className="k">초대 코드</span><span className="mono">{code}</span></div>
        <ol style={{ margin: '14px 0 0 18px', color: 'var(--dim)', fontSize: 14.5, lineHeight: 2 }}>
          <li><Link to="/download" style={{ color: 'var(--blue)' }}>앱 다운로드·설치</Link></li>
          <li><b>초대받은 바로 그 이메일</b>로 로그인 — 다른 이메일로는 수락되지 않습니다</li>
          <li>로그인하면 서버가 자동으로 나타나고 좌석을 점유합니다</li>
        </ol>
        <div className="banner warn" style={{ marginTop: 16 }}>초대는 보낸 이메일로만 수락 가능합니다. 메일 주소가 다르면 운영자에게 재발송을 요청하세요.</div>
      </div>
    </div>
  )
}
