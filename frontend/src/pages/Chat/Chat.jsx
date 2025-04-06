// const Chat = () => {
//   const [inputValue, setInputValue] = useState("")

//   const url = "http://localhost:8000"
//   const [messages, setMessages] = useState([])

//   const handleSubmit = async (e) => {
//     try {
//       e.preventDefault()
//       if (inputValue.trim()) {
//         // Call the API to send the message to the chatbot
//         const response = await axios.post(`${url}/chat`, {
//           messages: [inputValue]
//         }, {
//           headers: {
//             "Content-Type": "application/json"
//           }
//         })
//         if (response.data?.messages) {
//           setMessages([...messages, ...response.data.messages])
//         }
//         setInputValue("")
//       }
//     } 
//     catch (error) {
//       console.error("Failed to send the message:", error)
//     }
//   }

//   return (
//     <div className="chat-page">
//       <div className="chat-container">
//         <div className="chat-content">
//           {messages.length === 0 ? (
//               <h1 className="chat-title">What can I help with?</h1>
//           ) : (
//             <div className="message-list">
//               {messages.map((msg, index) => (
//                 <div key={index} className={`message ${msg.type}`}>
//                   {msg.type === 'human' && <div className="user-message">{msg.content}</div>}
//                   {msg.type === 'ai' && <div className="system-message">{msg.content}</div>}
//                 </div>
//               ))}
//             </div>
//           )}
//           <form onSubmit={handleSubmit} className="chat-input-container">
//             <input
//               type="text"
//               value={inputValue}
//               onChange={(e) => setInputValue(e.target.value)}
//               placeholder="Ask anything"
//               className="chat-input"
//             />
//             <div className="input-actions">
//               <button type="button" className="action-button">
//                 <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
//                   <path
//                     d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                   />
//                   <path
//                     d="M8 12H16"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                   />
//                   <path
//                     d="M12 8V16"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                   />
//                 </svg>
//               </button>
//               <button type="submit" className="action-button">
//                 <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
//                   <path
//                     d="M22 2L11 13"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                   />
//                   <path
//                     d="M22 2L15 22L11 13L2 9L22 2Z"
//                     stroke="currentColor"
//                     strokeWidth="2"
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                   />
//                 </svg>
//               </button>
//             </div>
//           </form>
//         </div>
//       </div>
//     </div>
//   )
// }

// export default Chat

"use client"

import { useState, useRef, useEffect } from "react"
import { Send } from "lucide-react"
import axios from 'axios'

const Chat = () => {
  const url = "http://localhost:8000"
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: "ai",
      content: "Hello! I'm your Student Assistant. How can I help you with your studies today?",
    },
  ])
  const [input, setInput] = useState("")
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

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

  const handleSend = async () => {
    if (!input.trim()) return

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

    // Simulate AI response (in a real app, this would call an API)
    const response = await axios.post(`${url}/chat/chat`, {
      messages: [input]
    }, {
      headers: {
        "Content-Type": "application/json"
      }
    })
    if (response.data?.messages) {
      const aiMessage = {
        id: messages.length + 2,
        sender: "ai",
        content: response.data.messages[1]["content"],
      }
      setMessages((prev) => [...prev, aiMessage])
      setIsThinking(false)
    }
    
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="page-container">
      <div className="page-header">Student Assistant Chat</div>
      <div className="chat-container">
        <div className="messages-container">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.sender}`}>
              <div className="message-container">
                {message.sender === "ai" && <div className="message-avatar">AI</div>}
                <div className="message-content">{message.content}</div>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="message ai">
              <div className="message-container">
                <div className="message-avatar">AI</div>
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
            placeholder="Ask me anything about your studies..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
          />
          <button className="send-button" onClick={handleSend} disabled={isThinking}>
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  )
}

export default Chat



