import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider, NavLink, Outlet, Link } from 'react-router-dom'
import './styles.css'
import Landing from './pages/Landing'
import Buy from './pages/Buy'
import Account from './pages/Account'
import Download from './pages/Download'
import Join from './pages/Join'

function Layout() {
  const tab = ({ isActive }) => (isActive ? 'active' : '')
  return (
    <>
      <header className="nav wrap">
        <Link to="/" className="logo"><span className="dot" />Imagine</Link>
        <span className="sp" />
        <NavLink to="/buy" className={tab}>구매</NavLink>
        <NavLink to="/download" className={tab}>다운로드</NavLink>
        <NavLink to="/account" className={tab}>내 라이선스</NavLink>
        <Link to="/buy" className="btn" style={{ marginLeft: 8 }}>시작하기</Link>
      </header>
      <main><Outlet /></main>
      <footer className="foot">© Imagine · 서버 단위 라이선스 · 파일 내용·이름은 절대 전송되지 않습니다</footer>
    </>
  )
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <Landing /> },
      { path: '/buy', element: <Buy /> },
      { path: '/account', element: <Account /> },
      { path: '/download', element: <Download /> },
      { path: '/join/:code', element: <Join /> },
      { path: '*', element: <Landing /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><RouterProvider router={router} /></React.StrictMode>,
)
