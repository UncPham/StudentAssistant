import { Plus, FileText } from "lucide-react"
import { useState, useEffect } from "react"

const HistorySidebar = () => {
  const [searchQuery, setSearchQuery] = useState("")
  const [files, setFiles] = useState([])
  const filteredFiles = files.filter((file) => file.name.toLowerCase().includes(searchQuery.toLowerCase()))

  const renderFileList = () => {
    if (filteredFiles.length === 0) {
      return (
        <div className="empty-state">
          <FileText size={24} className="empty-icon" />
          <p>No chat sessions found</p>
          <p>Upload a PDF to get started</p>
        </div>
      )
    }

    return filteredFiles.map((file) => (
      <div
        key={file.id}
        className={`file-item ${activeFile && activeFile.id === file.id ? "active" : ""}`}
        onClick={() => setActiveFile(file)}
      >
        <FileText size={16} />
        <div className="file-item-details">
          <div className="file-item-name">{file.name}</div>
          <div className="file-item-timestamp">{file.timestamp}</div>
        </div>
      </div>
    ))
  }

  return (
    <div className="pdf-sidebar">
        <div className="pdf-sidebar-header">
        <button className="new-chat-button" onClick={() => document.getElementById("pdf-upload").click()}>
            <Plus size={16} />
            New Chat
        </button>
        <div className="search-container">
            <input
            type="text"
            className="search-input"
            placeholder="Search files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            />
        </div>
        </div>
        <div className="pdf-sidebar-content">{renderFileList()}</div>
    </div>
  )
}

export default HistorySidebar
