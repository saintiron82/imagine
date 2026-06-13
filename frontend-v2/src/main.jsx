import React, { lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import { createHashRouter, RouterProvider, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './styles/index.css'
import { LocaleProvider } from './i18n'
import { AuthProvider } from './state/AuthContext'
import { AppProvider } from './state/AppContext'
import AppShell from './shell/AppShell'

// 코드 스플리팅: 화면을 라우트별 지연 로드한다. 가장 무거운 검색면
// (SearchPanel/FileGrid + react-virtual)을 초기 청크에서 분리 — 검색 진입 시에만 로드.
const StartScreen = lazy(() => import('./screens/StartScreen'))
const SearchScreen = lazy(() => import('./screens/SearchScreen'))
const FoldersScreen = lazy(() => import('./screens/FoldersScreen'))
const AnalysisScreen = lazy(() => import('./screens/AnalysisScreen'))
const AdminScreen = lazy(() => import('./screens/AdminScreen'))
const SettingsScreen = lazy(() => import('./screens/SettingsScreen'))

// 폴링 통합: 모든 서버 상태는 이 QueryClient 하나를 지난다 (구 앱의 인터벌 4개+ 대체)
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchInterval: 5000, staleTime: 4000, retry: 1 } },
})

const Fallback = () => <div style={{ padding: 24, color: 'var(--faint)', fontSize: 12 }}>불러오는 중…</div>
const lazy_ = (El) => <Suspense fallback={<Fallback />}><El /></Suspense>

const router = createHashRouter([
  { path: '/start', element: lazy_(StartScreen) },
  {
    element: <AppShell />,
    children: [
      { path: '/', element: <Navigate to="/search" replace /> },
      { path: '/search', element: lazy_(SearchScreen) },
      { path: '/folders', element: lazy_(FoldersScreen) },
      { path: '/analysis', element: lazy_(AnalysisScreen) },
      { path: '/admin', element: lazy_(AdminScreen) },
      { path: '/settings', element: lazy_(SettingsScreen) },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <LocaleProvider>
        <AuthProvider>
          <AppProvider>
            <RouterProvider router={router} />
          </AppProvider>
        </AuthProvider>
      </LocaleProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
