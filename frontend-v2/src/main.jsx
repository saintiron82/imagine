import React from 'react'
import ReactDOM from 'react-dom/client'
import { createHashRouter, RouterProvider, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './styles/app.css'
import { AppProvider } from './state/AppContext'
import AppShell from './shell/AppShell'
import StartScreen from './screens/StartScreen'
import SearchScreen from './screens/SearchScreen'
import FoldersScreen from './screens/FoldersScreen'
import AnalysisScreen from './screens/AnalysisScreen'
import AdminScreen from './screens/AdminScreen'
import SettingsScreen from './screens/SettingsScreen'

// 폴링 통합: 모든 서버 상태는 이 QueryClient 하나를 지난다 (구 앱의 인터벌 4개+ 대체)
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchInterval: 5000, staleTime: 4000, retry: 1 } },
})

const router = createHashRouter([
  { path: '/start', element: <StartScreen /> },
  {
    element: <AppShell />,
    children: [
      { path: '/', element: <Navigate to="/search" replace /> },
      { path: '/search', element: <SearchScreen /> },
      { path: '/folders', element: <FoldersScreen /> },
      { path: '/analysis', element: <AnalysisScreen /> },
      { path: '/admin', element: <AdminScreen /> },
      { path: '/settings', element: <SettingsScreen /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <AppProvider>
        <RouterProvider router={router} />
      </AppProvider>
    </QueryClientProvider>
  </React.StrictMode>,
)
