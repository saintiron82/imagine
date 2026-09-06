import { useNavigate } from 'react-router-dom'
import { useApp } from '../state/AppContext'
import { useLocale } from '../i18n'

/**
 * 설정 — 실제로 동작하는 항목만 둔다(목업/미연동 토글 없음).
 * 추가 설정은 백엔드 엔드포인트가 붙는 대로 확장.
 */
export default function SettingsScreen() {
  const { isOperator } = useApp()
  const { locale, setLocale, availableLocales, t } = useLocale()
  const navigate = useNavigate()
  const localeLabel = { 'ko-KR': '한국어', 'en-US': 'English' }

  return (
    <section id="scr-settings" className="screen active">
      <div className="wrap">
        <div className="scope">
          <h3>{t('v2.settings.scope_me')} <span className="scope-tag t-me">{t('v2.settings.scope_me_tag')}</span></h3>
          <p>{t('v2.settings.scope_me_desc')}</p>
          <div className="srow">
            <div className="lab">{t('v2.settings.language')}</div>
            <select value={locale} onChange={e => setLocale(e.target.value)}>
              {availableLocales.map(l => <option key={l} value={l}>{localeLabel[l] || l}</option>)}
            </select>
          </div>
        </div>

        {isOperator && (
          <div className="scope">
            <h3>{t('v2.settings.scope_server')} <span className="scope-tag t-srv">{t('v2.settings.scope_server_tag')}</span></h3>
            <p>{t('v2.settings.scope_server_desc')}</p>
            <div className="srow">
              <div className="lab">{t('v2.settings.auto_analysis')}<div className="d">{t('v2.settings.auto_analysis_d')}</div></div>
              <div className="row-act"><button onClick={() => navigate('/admin')}>{t('v2.settings.goto_admin')}</button></div>
            </div>
            <div className="srow">
              <div className="lab">{t('v2.settings.members')}<div className="d">{t('v2.settings.members_d')}</div></div>
              <div className="row-act"><button onClick={() => navigate('/admin')}>{t('v2.settings.goto_admin')}</button></div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
