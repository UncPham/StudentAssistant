"use client"

import { Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import Sidebar from "./components/Sidebar/Sidebar.jsx"
import Chat from "./pages/Chat/Chat.jsx"
import Translation from "./pages/Translation/Translation.jsx"
import ChatPDF from "./pages/ChatPDF/ChatPDF.jsx"
import "./App.css"

function App() {
  const [activeTab, setActiveTab] = useState("chat")
  const navigate = useNavigate()
  const location = useLocation()

  // Set default route to /chat if on root path
  useEffect(() => {
    if (location.pathname === "/") {
      navigate("/chat")
    }
  }, [location.pathname, navigate])

  // Determine active tab based on current path
  const getActiveTab = () => {
    const path = location.pathname
    if (path.includes("chat") && !path.includes("chatpdf")) return "chat"
    if (path.includes("translation")) return "translation"
    if (path.includes("chatpdf")) return "chatpdf"
    return "chat"
  }

  const [user, setUser] = useState(null)

  const handleLoginSuccess = (userData) => {
    setUser(userData)
  }

  const handleLogout = () => {
    setUser(null)
  }

  return (
    <>
      {/* {!user ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : (  */}
      <div className="app-container">
          <Routes>
            <Route path="/chat" element={<Chat />} />
            <Route path="/translation" element={<Translation />} />
            <Route path="/chatpdf" element={<ChatPDF />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
          <Sidebar activeTab={getActiveTab()} setActiveTab={setActiveTab} />
      </div>
      {/* )} */}
    </>
  )
}

export default App


