import { useNavigate } from 'react-router-dom'
import SearchPanel from '../components/SearchPanel'

/**
 * 검색 — 구 앱의 SearchPanel 을 그대로 이식(재작성 금지).
 * v2 셸은 자체 상단바·라우팅을 쓰므로, 패널 내부의 "설정/폴더로 이동" 콜백만
 * v2 라우터에 접속한다. 검색 로직·API·결과 UI 는 원본 그대로다.
 *
 * 레이아웃: SearchPanel 루트가 `flex h-full` 이므로 height:100% 컨테이너가 필요.
 * main{flex:1} > .screen{height:100%} 가 그 높이를 공급한다.
 */
export default function SearchScreen() {
  const navigate = useNavigate()
  return (
    <section id="scr-search" className="screen active" style={{ height: '100%', overflow: 'hidden' }}>
      <SearchPanel
        onOpenSettings={() => navigate('/settings')}
        onNavigateToFolder={() => navigate('/folders')}
      />
    </section>
  )
}
