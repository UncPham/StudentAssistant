"use client"

import { MessageSquare, Languages, FileText } from "lucide-react"
import { Link, useNavigate } from 'react-router-dom';
import "./Sidebar.css"

const Sidebar = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: "chat", icon: MessageSquare, label: "Chat", path: "/chat" },
    { id: "translation", icon: Languages, label: "Translation", path: "/translation" },
    { id: "chatpdf", icon: FileText, label: "ChatPDF", path: "/chatpdf" },
  ]

  return (
    <div className="sidebar">
      {tabs.map((tab) => (
        <Link to={tab.path} key={tab.id}>
          <button
            className={`sidebar-button ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
            aria-label={tab.label}
            title={tab.label}
          >
            <tab.icon size={24} />
          </button>
        </Link>
      ))}
    </div>
  )
}

export default Sidebar

