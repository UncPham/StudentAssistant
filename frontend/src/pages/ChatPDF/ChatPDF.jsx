"use client"

import { useState, useRef, useEffect, useCallback, useMemo  } from "react"
import { Upload, Plus, Send, FileText, ZoomIn, ZoomOut } from "lucide-react"
import { Viewer, Worker, SpecialZoomLevel } from '@react-pdf-viewer/core';
import { highlightPlugin } from "@react-pdf-viewer/highlight"
import { pageNavigationPlugin } from "@react-pdf-viewer/page-navigation"
import '@react-pdf-viewer/core/lib/styles/index.css';
import "@react-pdf-viewer/highlight/lib/styles/index.css"
import "@react-pdf-viewer/page-navigation/lib/styles/index.css"
import { useNavigate, useParams } from "react-router-dom"
import { searchPlugin } from "@react-pdf-viewer/search"
import HistorySidebar from "../../components/HistorySidebar/HistorySidebar";
import axios from 'axios'
import PDFPolygonOverlay from "../../components/PDFPolygonOverlay/PDFPolygonOverlay"
// import docData from "../../data/doc.json"

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
  const [currentPage, setCurrentPage] = useState(1)
  const [highlightStatus, setHighlightStatus] = useState(null) // Track highlight status
  const [selectedElement, setSelectedElement] = useState(null)
  const pdfContainerRef = useRef(null)
  const [pdfScale, setPdfScale] = useState(1)
  const [pageHeight, setPageHeight] = useState(792)
  const [docData, setDocData] = useState(null)

  const url = "http://localhost:8000"

  // Initialize plugins
  const highlightPluginInstance = highlightPlugin()
  const pageNavigationPluginInstance = pageNavigationPlugin()
  const { jumpToPage } = pageNavigationPluginInstance

  const jumpToPageRef = useRef(null)

  // Create refs for plugin instances
  const viewerRef = useRef(null)

  // Store plugin instances in refs to access them in event handlers
  // Initialize search plugin
  const searchPluginInstance = searchPlugin({
    enableShortcuts: true,
  })

  // --- Function to measure and update page height ---
  const measureAndUpdatePageHeight = useCallback((pageIndex) => {
    // Use a timeout to ensure the page element is rendered and measurable
    setTimeout(() => {
      if (pdfContainerRef.current) {
        // Find the specific page layer div using the data-page-index attribute
        // @react-pdf-viewer uses 0-based index internally
        const pageLayer = pdfContainerRef.current.querySelector(
          `.rpv-core__page-layer[data-page-index="${pageIndex}"]`
        );
        if (pageLayer) {
          // Get the offsetHeight, which includes padding and borders and reflects layout size
          const measuredHeight = pageLayer.offsetHeight;
          if (measuredHeight > 0) { // Ensure we got a valid height
              setPageHeight(measuredHeight);
              // console.log(`Measured height for page index ${pageIndex}:`, measuredHeight); // For debugging
          } else {
            console.warn(`Measured height for page index ${pageIndex} is 0. Retrying or keeping previous.`);
            // Optionally try again or keep the last known height
          }
        } else {
          console.warn(`Could not find page layer for page index ${pageIndex} to measure height.`);
          // Fallback: Maybe keep the existing height or set a default?
          // setPageHeight(800); // Or keep previous state
        }
      }
    }, 150); // Delay in ms, adjust if needed (might need longer for complex PDFs or slower machines)
  }, [pdfContainerRef]); // Dependency: the container ref

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

    // Update scale and measure page height when zoomLevel changes
  useEffect(() => {
    const newScale = zoomLevel / 100;
    setPdfScale(newScale);
    // Re-measure height when zoom changes, as layout might adjust (e.g., PageFit)
    measureAndUpdatePageHeight(currentPage - 1); // Use current page's 0-based index
  }, [zoomLevel, currentPage, measureAndUpdatePageHeight]);

  useEffect(() => {
    const fetchPolygons = async () => {
      try {
        if (!activeFile) return;
        const response = await axios.post(`${url}/chatpdf/polygon`, {
          file: activeFile.name
        }, {
          headers: {
            "Content-Type": "application/json"
          }
        });
  
        if (response.data) {
          setDocData(response.data.data);
        }
      } catch (error) {
        console.error("Error Fetching Polygons:", error);
      }
    };
  
    fetchPolygons();
  }, [activeFile]);
  

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
        if (!input.trim() || !activeFile) return;

        // Add user message
        const userMessage = {
            id: messages.length + 1,
            sender: "user",
            content: input,
        };

        setMessages([...messages, userMessage]);
        setInput("");

        // Reset textarea height
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }

        // Show thinking effect
        setIsThinking(true);

        // Log the request data
        console.log("Sending query:", {
            query: input,
            file: activeFile.name,
            doc_id: selectedElement?.id
        });

        const response = await axios.post(`${url}/chatpdf/query`, {
            query: input,
            file: activeFile.name,
            doc_id: selectedElement?.id
        }, {
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (response.data?.response) {
            const aiMessage = {
                id: messages.length + 2,
                sender: "ai",
                content: response.data.response,
                context: response.data.context,
            };
            console.log(aiMessage);
            setMessages((prev) => [...prev, aiMessage]);
            setIsThinking(false);
        }

    } catch (error) {
        console.error("Error Answer:", error);
        setIsThinking(false);
    } finally {
        setIsThinking(false);
    }
}

  // Handle reference click to navigate to specific page and activate polygon
  const handleReferenceClick = useCallback((refNumber, elementId) => {
    const match = elementId.match(/^\/page\/(\d+)\//);
    const pageNumber = match && match[1] ? parseInt(match[1]) : null;
  
    if (!pageNumber) {
      console.error(`Invalid elementId format: ${elementId}`);
      return;
    }
  
    setCurrentPage(pageNumber);
    
    // Set the selected element to activate polygon in PDFPolygonOverlay
    // setSelectedElement({ id: elementId })
    if (jumpToPageRef.current) {
      console.log(jumpToPageRef.current)
      // Page index is 0-based, so subtract 1
      jumpToPageRef.current(pageNumber, () => {
          // Callback này được gọi SAU KHI trang đã sẵn sàng
          console.log(`Page ${pageNumber} navigation complete. Setting selected element.`);
          setSelectedElement({ id: elementId });
      });
    }

    console.log(`Navigating to reference [${refNumber}] on page ${pageNumber}`);
    console.log(`Activating element ID: ${elementId}`);
  }, []);

  // Add event listener for reference clicks
  useEffect(() => {
    const handleLinkClick = (e) => {
      if (e.target.classList.contains("reference-link")) {
        e.preventDefault()
        const refNumber = e.target.getAttribute("data-ref")
        const elementId = e.target.getAttribute("data-id")
        handleReferenceClick(refNumber, elementId)
      }
    }

    document.addEventListener("click", handleLinkClick)
    return () => document.removeEventListener("click", handleLinkClick)
  }, [handleReferenceClick])

  const handlePolygonClick = (element) => {
      setSelectedElement(element);
      
      // Add the element info to the chat
      if (element) {
          const promptMessage = {
              id: messages.length + 1,
              sender: "ai",
              content: `What would you like to know about this context?`,
          };
          setMessages((prev) => [...prev, promptMessage]);
      }
  }


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
            return `<a href="#" class="reference-link" data-ref="${refNumber}" data-id="${contextDoc.doc_id}" data-text="${encodedText}">[${refNumber}]</a>${index < numbers.length - 1 ? ", " : ""}`
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
    const newZoom = Math.min(zoomLevel + 10, 200)
    setZoomLevel(newZoom)
    setPdfScale(newZoom / 100)
  }

  const handleZoomOut = () => {
    // Decrease zoom by 10%, min 50%
    const newZoom = Math.max(zoomLevel - 10, 50)
    setZoomLevel(newZoom)
    setPdfScale(newZoom / 100)
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
          {/* Real PDF viewer */}
          <div
            className="pdf-viewer-container"
            style={{
              width: "100%",
              height: "100%",
              transform: `scale(${zoomLevel / 100})`,
              transformOrigin: "center top",
              transition: "transform 0.2s ease-in-out",
            }}
            ref={pdfContainerRef}
          >
            <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.4.120/build/pdf.worker.min.js">
              {activeFile.fileUrl ? (
                <div className="pdf-viewer-wrapper">
                  {/* Show highlight status notification if available */}
                  {highlightStatus && (
                    <div className={`highlight-status ${highlightStatus.success ? "success" : "error"}`}>
                      {highlightStatus.message}
                    </div>
                  )}

                  <Viewer
                    fileUrl={activeFile.fileUrl}
                    plugins={[pageNavigationPluginInstance]}
                    defaultScale={SpecialZoomLevel.PageFit}
                    onDocumentLoad={(e) => {
                      jumpToPageRef.current = jumpToPage
                      setCurrentPage(1); // Start at page 1 (1-based)
                      setPdfScale(1); // Reset scale on new doc load (zoomLevel is reset elsewhere)
                      // measureAndUpdatePageHeight(0); // Measure height of first page (index 0)
                    }}
                    onPageChange={(e) => {
                      // The viewer uses 0-based index, e.currentPage is the new page index
                      const newPageIndex = e.currentPage;
                      setCurrentPage(newPageIndex)
                      // Re-measure height whenever the page changes
                      // measureAndUpdatePageHeight(newPageIndex);
                    }}
                  />

                  {/* Add the polygon overlay */}
                  <PDFPolygonOverlay
                    docData={docData && typeof docData === "object" ? docData : {}}
                    currentPage={currentPage}
                    scale={pdfScale}
                    containerRef={pdfContainerRef}
                    onPolygonClick={handlePolygonClick}
                    zoomLevel={zoomLevel}
                    pageHeight={pageHeight}
                    activeElementId={selectedElement?.id}
                  />
                </div>
              ) : (
                <div className="pdf-placeholder">
                  <p>Loading document...</p>
                </div>
              )}
            </Worker>
          </div>
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

