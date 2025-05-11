"use client"

import { ExternalLink, Book } from "lucide-react"
import { useState, useEffect } from "react"
import "./HistorySidebar.css"

const HistorySidebar = ({ context }) => {
  const [activeTab, setActiveTab] = useState("rag")
  const [ragDocuments, setRagDocuments] = useState([])
  const [webDocuments, setWebDocuments] = useState([])

  useEffect(() => {
    if (context) {
      console.log("web_documents:", context.web_documents) // Debug
      const rag = context.rag_documents || []
      setRagDocuments(rag)
      const web = context.web_documents || []
      setWebDocuments(web)
      if (rag.length > 0) {
        setActiveTab("rag")
      } else if (web.length > 0) {
        setActiveTab("web")
      }
    }
  }, [context])

  const renderRagDocuments = () => {
    if (ragDocuments.length === 0) {
      return (
        <div className="empty-state">
          <Book size={24} className="empty-icon" />
          <p>No document references found</p>
        </div>
      )
    }

    return ragDocuments.map((doc, index) => (
      <div key={index} className="reference-item">
        <div className="reference-number">{index + 1}</div>
        <div className="reference-content">
          <p>{doc.document}</p>
        </div>
      </div>
    ))}

  const renderWebDocuments = () => {
    if (webDocuments.length === 0) {
      return (
        <div className="empty-state">
          <ExternalLink size={24} className="empty-icon" />
          <p>No web references found</p>
        </div>
      )
    }

    return webDocuments.map((doc, index) => {
      const url = doc.url && typeof doc.url === "object" && doc.url.url ? doc.url.url : "#"
      return (
        <div key={url || index} className="reference-item">
          <div className="reference-number">{index + 1}</div>
          <div className="reference-content">
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="reference-link"
            >
              <ExternalLink size={14} />
              {url}
            </a>
            <p className="reference-text">{doc.document}</p>
          </div>
        </div>
      )
    })
  }

  return (
    <div className="pdf-sidebar history-sidebar">
      <div className="pdf-sidebar-header">
        <h3 className="sidebar-title">References</h3>
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === "rag" ? "active" : ""}`}
            onClick={() => setActiveTab("rag")}
          >
            <Book size={16} />
            Document
            {ragDocuments.length > 0 && <span className="tab-count">{ragDocuments.length}</span>}
          </button>
          <button
            className={`tab-button ${activeTab === "web" ? "active" : ""}`}
            onClick={() => setActiveTab("web")}
          >
            <ExternalLink size={16} />
            Web
            {webDocuments.length > 0 && <span className="tab-count">{webDocuments.length}</span>}
          </button>
        </div>
      </div>
      <div className="pdf-sidebar-content references-container">
        {activeTab === "rag" ? renderRagDocuments() : renderWebDocuments()}
      </div>
    </div>
  )
}

export default HistorySidebar