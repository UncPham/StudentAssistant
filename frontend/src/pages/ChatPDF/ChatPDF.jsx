"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Upload, Plus, Send, FileText, ZoomIn, ZoomOut } from "lucide-react"
import { Viewer, Worker } from '@react-pdf-viewer/core';
import { highlightPlugin } from "@react-pdf-viewer/highlight"
import { pageNavigationPlugin } from "@react-pdf-viewer/page-navigation"
import '@react-pdf-viewer/core/lib/styles/index.css';
import "@react-pdf-viewer/highlight/lib/styles/index.css"
import "@react-pdf-viewer/page-navigation/lib/styles/index.css"
import { useNavigate, useParams } from "react-router-dom"
import HistorySidebar from "../../components/HistorySidebar/HistorySidebar";
import axios from 'axios'

const ChatPDF = () => {
  const navigate = useNavigate()
  const [files, setFiles] = useState([])
  const [activeFile, setActiveFile] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const [zoomLevel, setZoomLevel] = useState(100)
  const pdfViewerRef = useRef(null)
  const [currentPage, setCurrentPage] = useState(1)

  const url = "http://localhost:8000"

  // Initialize plugins
  const highlightPluginInstance = highlightPlugin()
  const pageNavigationPluginInstance = pageNavigationPlugin()
  const { jumpToPage } = pageNavigationPluginInstance

  // Store plugin instances in refs to access them in event handlers
  const jumpToPageRef = useRef(null)
  const jumpToHighlightRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      // Reset height to auto to get the correct scrollHeight
      textareaRef.current.style.height = "auto"
      // Set the height to scrollHeight + 2px for border
      const newHeight = Math.min(textareaRef.current.scrollHeight, 120) // Max height of 120px
      textareaRef.current.style.height = `${newHeight}px`
    }
  }, [input])

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (file && file.type === "application/pdf") {
      const fileUrl = URL.createObjectURL(file)
      const newFile = {
        id: Date.now(),
        name: file.name,
        size: file.size,
        file: file,
        fileUrl: fileUrl,
        timestamp: new Date().toLocaleString(),
      }
      setFiles([...files, newFile])
      setActiveFile(newFile)
      
      // Upload file to server
      // setIsThinking(true)
      // const formData = new FormData();
      // formData.append('file', file);

      // const response = await axios.post(`${url}/chatpdf/embeddings`, formData);
      // if (response.data) {
      //   console.log(response.data.message)
      // }
      // setIsThinking(false)

      // Reset messages for new file
      setMessages([
        {
          id: 1,
          sender: "ai",
          content: `I've loaded "${file.name}". What would you like to know about this document?`,
        },
      ])
    } else {
      alert("Please upload a PDF file")
    }
  }

  const handleDrop = async (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type === "application/pdf") {
      const newFile = {
        id: Date.now(),
        name: file.name,
        size: file.size,
        file: file,
        timestamp: new Date().toLocaleString(),
      }
      setFiles([...files, newFile])
      setActiveFile(newFile)

      // Upload file to server
      // setIsThinking(true)
      // const formData = new FormData();
      // formData.append('file', file);

      // const response = await axios.post(`${url}/chatpdf/embeddings`, formData);
      // if (response.data) {
      //   console.log(response.data.message)
      // }
      // setIsThinking(false)
      // Reset messages for new file
      setMessages([
        {
          id: 1,
          sender: "ai",
          content: `I've loaded "${file.name}". What would you like to know about this document?`,
        },
      ])
    } else {
      alert("Please upload a PDF file")
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleSend = async () => {
    try {
      if (!input.trim() || !activeFile) return

      // Add user message
      const userMessage = {
        id: messages.length + 1,
        sender: "user",
        content: input,
      }

      setMessages([...messages, userMessage])
      setInput("")

      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
      }

      // Show thinking effect
      setIsThinking(true)

      const response = await axios.post(`${url}/chatpdf/query`, {
          query: input,
          file: activeFile.name
        }, {
          headers: {
            "Content-Type": "application/json"
          }
        })
      if (response.data?.response) {
          const aiMessage = {
            id: messages.length + 2,
            sender: "ai",
            content: response.data.response,
            context: response.data.context,
          }
          setMessages((prev) => [...prev, aiMessage])
          setIsThinking(false)
        }
      
    } catch (error) {
      console.error("Error Answer:", error)
      setIsThinking(false) 
    } finally {
      setIsThinking(false);
    }
  }

  // Handle reference click to navigate to specific page and highlight text
  const handleReferenceClick = useCallback((refNumber, pageNumber, originalText) => {
    // Set the current page to navigate to
    setCurrentPage(pageNumber)

    // Use the page navigation plugin to jump to the page
    if (jumpToPageRef.current) {
      // Page index is 0-based, so subtract 1
      jumpToPageRef.current(pageNumber - 1)

      // Use the highlight plugin to highlight text
      if (jumpToHighlightRef.current) {
        const normalizedText = originalText.trim().replace(/\s+/g, " ");

        setTimeout(() => {
          jumpToHighlightRef.current({
            keyword: normalizedText,
            matchCase: false,
          })
        }, 300) // Small delay to ensure page is loaded
      }
    }

    console.log(`Navigating to reference [${refNumber}] on page ${pageNumber}`)
    console.log(`Highlighting text: "${originalText}"`)
  }, [])

  // Add event listener for reference clicks
  useEffect(() => {
    const handleLinkClick = (e) => {
      if (e.target.classList.contains("reference-link")) {
        e.preventDefault()
        const refNumber = e.target.getAttribute("data-ref")
        const pageNumber = Number.parseInt(e.target.getAttribute("data-page"))
        const originalText = decodeURIComponent(e.target.getAttribute("data-text"))
        handleReferenceClick(refNumber, pageNumber, originalText)
      }
    }

    document.addEventListener("click", handleLinkClick)
    return () => document.removeEventListener("click", handleLinkClick)
  }, [handleReferenceClick])


  // Render message content with clickable references
  const renderMessageContent = (message) => {
    if (message.sender !== "ai" || !message.context) return message.content

    // Replace references like [1, 2, 5] or [1] with clickable links
    const contentWithLinks = message.content.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (match, refNumbers) => {
      // Split the reference numbers and create links for each
      const numbers = refNumbers.split(",").map((num) => num.trim())

      return numbers
        .map((refNumber, index) => {
          const docIndex = Number.parseInt(refNumber) - 1
          const contextDoc = message.context?.[docIndex]

          if (contextDoc) {
            // Encode the original text to safely include it in the data attribute
            const encodedText = encodeURIComponent(contextDoc.original_text)
            return `<a href="#" class="reference-link" data-ref="${refNumber}" data-page="${contextDoc.page_number}" data-text="${encodedText}">[${refNumber}]</a>${index < numbers.length - 1 ? ", " : ""}`
          }
          return `[${refNumber}]${index < numbers.length - 1 ? ", " : ""}`
        })
        .join("")
    })

    return <span dangerouslySetInnerHTML={{ __html: contentWithLinks }} />
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const filteredFiles = files.filter((file) => file.name.toLowerCase().includes(searchQuery.toLowerCase()))

  const handleZoomIn = () => {
    // Increase zoom by 10%, max 200%
    setZoomLevel((prevZoom) => Math.min(prevZoom + 10, 200))
  }

  const handleZoomOut = () => {
    // Decrease zoom by 10%, min 50%
    setZoomLevel((prevZoom) => Math.max(prevZoom - 10, 50))
  }

  const renderDocumentViewer = () => {
    if (!activeFile) {
      return (
        <div className="upload-document-section">
          <h3>Upload a new document</h3>
          <div className="upload-container" onDrop={handleDrop} onDragOver={handleDragOver}>
            <Upload size={48} className="upload-icon" />
            <p className="upload-text">Drag and drop your PDF file here</p>
            <button className="button" onClick={() => document.getElementById("pdf-upload").click()}>Choose File</button>
            <input type="file" className="file-input" accept=".pdf" onChange={handleFileUpload} id="pdf-upload" />
            <p className="upload-text">PDF files only, up to 10MB</p>
          </div>
        </div>
      )
    }

    return (
      <>
        <div className="document-viewer-header">
          <div className="document-title">{activeFile.name}</div>
          <div className="document-controls">
            <button className="document-control-button" title="Zoom In" onClick={handleZoomIn}>
              <ZoomIn size={18} />
            </button>
            <button className="document-control-button" title="Zoom Out" onClick={handleZoomOut}>
              <ZoomOut size={18} />
            </button>
          </div>
        </div>
        <div className="document-content">
          {/* Placeholder for PDF viewer */}
          <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
            <div 
              className="pdf-placeholder"
              style={{
                transform: `scale(${zoomLevel / 100})`,
                transformOrigin: "center top",
                transition: "transform 0.2s ease-in-out",
              }}
            >
              {activeFile.fileUrl ? (
                  <Viewer 
                    fileUrl={activeFile.fileUrl}
                    plugins={[highlightPluginInstance, pageNavigationPluginInstance]}
                    onDocumentLoad={(e) => {
                      // Store the plugin methods in refs for later use
                      jumpToPageRef.current = jumpToPage
                      jumpToHighlightRef.current = highlightPluginInstance.jumpToHighlightArea

                      // Update current page
                      setCurrentPage(1)
                    }}
                  />
              ) : (
                  <p>Loading document...</p>
              )}
            </div>
          </Worker>
        </div>
      </>
    )
  }

  return (
    <div className="page-container">
      <div className="pdf-container">
        <HistorySidebar/>

        {!activeFile ? (
          <div className="pdf-main-empty">{renderDocumentViewer()}</div>
        ) : (
          <div className="pdf-main-with-chat">
            <div className="pdf-viewer-container">{renderDocumentViewer()}</div>
            <div className="pdf-chat-container">
              <div className="messages-container">
                {messages.map((message) => (
                  <div key={message.id} className={`message ${message.sender}`}>
                    <div className="message-container">
                      {/* {message.sender === "ai" && <div className="message-avatar">AI</div>} */}
                      <div className="message-content">{renderMessageContent(message)}</div>
                    </div>
                  </div>
                ))}

                {isThinking && (
                  <div className="message ai">
                    <div className="message-container">
                      {/* <div className="message-avatar">AI</div> */}
                      <div className="message-content thinking">
                        <div className="thinking-dots">
                          <span></span>
                          <span></span>
                          <span></span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
              <div className="input-container">
                <textarea
                  ref={textareaRef}
                  className="text-input expandable"
                  placeholder="Ask questions about the document..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  disabled={isThinking}
                />
                <button className="send-button" onClick={handleSend} disabled={isThinking}>
                  <Send size={20} />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatPDF

