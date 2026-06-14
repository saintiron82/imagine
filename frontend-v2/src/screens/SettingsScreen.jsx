import { useNavigate } from 'react-router-dom'
import { useApp } from '../state/AppContext'
import { useLocale } from '../i18n'

/**
 * 설정 — 실제로 동작하는 항목만 둔다(목업/미연동 토글 없음).
 * 추가 설정은 백엔드 엔드포인트가 붙는 대로 확장.
 */
export default function SettingsScreen() {
  const { isOperator } = useApp()
  const { locale, setLocale, availableLocales } = useLocale()
  const navigate = useNavigate()
  const localeLabel = { 'ko-KR': '한국어', 'en-US': 'English' }

  return (
    <section id="scr-settings" className="screen active">
      <div className="wrap">
        <div className="scope">
          <h3>나 <span className="scope-tag t-me">나에게만 적용</span></h3>
          <p>이 계정의 화면 표시 방식</p>
          <div className="srow">
            <div className="lab">언어</div>
            <select value={locale} onChange={e => setLocale(e.target.value)}>
              {availableLocales.map(l => <option key={l} value={l}>{localeLabel[l] || l}</option>)}
            </select>
          </div>
        </div>

        {isOperator && (
          <div className="scope">
            <h3>서버 <span className="scope-tag t-srv">전원에게 적용 · 운영자</span></h3>
            <p>이 라이브러리를 쓰는 모두에게 영향을 주는 정책</p>
            <div className="srow">
              <div className="lab">자동 분석 · 분석기<div className="d">병목·워커 제어는 관리 화면에서</div></div>
              <div className="row-act"><button onClick={() => navigate('/admin')}>관리로 이동 →</button></div>
            </div>
            <div className="srow">
              <div className="lab">멤버 · 초대 · 좌석<div className="d">초대 발송·좌석 관리</div></div>
              <div className="row-act"><button onClick={() => navigate('/admin')}>관리로 이동 →</button></div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
