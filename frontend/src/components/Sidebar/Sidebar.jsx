"use client"

import { useState } from "react"
import { useNavigate, useLocation } from "react-router-dom"
import { MessageSquare, FileText, MessageCircle, File, HelpCircle, Search } from "lucide-react"

const Sidebar = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const [userAvatar, setUserAvatar] = useState("/placeholder.svg?height=50&width=50")

  const navItems = [
    { icon: MessageSquare, path: "/chat", label: "Chat" },
    { icon: FileText, path: "/notes", label: "Notes" },
    { icon: MessageCircle, path: "/messages", label: "Messages" },
    { icon: File, path: "/files", label: "Files" },
    { icon: HelpCircle, path: "/help", label: "Help" },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <div className="nav-items">
          {navItems.map((item, index) => (
            <button
              key={index}
              className={`nav-item ${location.pathname === item.path ? "active" : ""}`}
              onClick={() => navigate(item.path)}
              aria-label={item.label}
            >
              <item.icon size={24} />
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <button className="nav-item" aria-label="Search">
            <Search size={24} />
          </button>
          <div className="avatar-container">
            <img src={userAvatar || "/placeholder.svg"} alt="User avatar" className="user-avatar" />
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar

