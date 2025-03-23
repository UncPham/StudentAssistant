"use client"

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import Login from './pages/Login/Login';
import Sidebar from "./components/Sidebar/Sidebar.jsx"
import Chat from "./pages/Chat/Chat.jsx"
import Translation from "./pages/Translation/Translation.jsx"
// import MessagesPage from "./pages/MessagesPage"
// import FilesPage from "./pages/FilesPage"
// import HelpPage from "./pages/HelpPage"
import "./App.css"

function App() {
  const [darkMode, setDarkMode] = useState(true)

  const [user, setUser] = useState(null)

  const handleLoginSuccess = (userData) => {
    setUser(userData)
  }

  const handleLogout = () => {
    setUser(null)
  }

  return (
    <Router>
      {/* {!user ? (
        <Login onLoginSuccess={handleLoginSuccess} />
      ) : ( */}
      <div className={`app ${darkMode ? "dark" : ""}`}>
        <div className="app-container">
          <main className="main-content">
            <Routes>
              <Route path="/chat" element={<Chat />} /> 
              <Route path="/translation" element={<Translation />} />
              {/* <Route path="/messages" element={<MessagesPage />} />
              <Route path="/files" element={<FilesPage />} />
              <Route path="/help" element={<HelpPage />} />
              <Route path="*" element={<Navigate to="/chat" replace />} /> */}
            </Routes>
          </main>
          <Sidebar />
        </div>
      </div>
      {/* )} */}
    </Router>
  )
}

export default App


