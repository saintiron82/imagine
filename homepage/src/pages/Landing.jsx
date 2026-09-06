import { Link } from 'react-router-dom'

/** 1장: 소개 — 무엇을 파는가 + 시작 경로. */
export default function Landing() {
  return (
    <div className="wrap">
      <section className="hero">
        <div className="tagline">로컬 우선 멀티모달 자산 검색</div>
        <h1>내 에셋을, 구글 이미지처럼<br />자연어로 찾는다</h1>
        <p>PSD·PNG·JPG를 단순 파일이 아니라 검색 가능한 분석 단위로. 비슷하게 생긴 레퍼런스부터
          "푸른 갑옷 기사 일러스트" 같은 의미 검색까지 — 내 서버 안에서.</p>
        <div className="cta">
          <Link to="/buy" className="btn lg">플랜 보기</Link>
          <Link to="/download" className="btn ghost lg">다운로드</Link>
        </div>
      </section>

      <section className="feats">
        <div className="feat"><h3>세 축 검색</h3><p>시각 유사도(VV)·의미(MV)·정확 키워드(FTS)를 합쳐 한 문장으로 찾습니다.</p></div>
        <div className="feat"><h3>내 서버 안에서</h3><p>원본 파일은 서버 밖으로 나가지 않습니다. 외부 접속은 주소 공유 없이 터널로.</p></div>
        <div className="feat"><h3>팀으로</h3><p>이메일로 초대하면 좌석을 점유합니다. 여러 대의 컴퓨터를 분석기로 묶어 가속.</p></div>
      </section>
    </div>
  )
}
