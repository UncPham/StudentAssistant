"use client"

import { useState } from "react"
import axios from 'axios'

const Chat = () => {
  const [inputValue, setInputValue] = useState("")

  const url = "http://localhost:8000"
  const [messages, setMessages] = useState([])

  const handleSubmit = async (e) => {
    try {
      e.preventDefault()
      if (inputValue.trim()) {
        // Call the API to send the message to the chatbot
        const response = await axios.post(`${url}/chat`, {
          messages: [inputValue]
        }, {
          headers: {
            "Content-Type": "application/json"
          }
        })
        if (response.data?.messages) {
          setMessages([...messages, ...response.data.messages])
        }
        setInputValue("")
      }
    } 
    catch (error) {
      console.error("Failed to send the message:", error)
    }
  }

  return (
    <div className="chat-page">
      <div className="chat-container">
        <div className="chat-content">
          <h1 className="chat-title">Tôi có thể giúp gì cho bạn?</h1>
          <div className="message-list">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                {msg.type === 'human' && <div className="user-message">{msg.content}</div>}
                {msg.type === 'ai' && <div className="ai-message">{msg.content}</div>}
              </div>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="chat-input-container">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Hỏi bất cứ điều gì"
              className="chat-input"
            />
            <div className="input-actions">
              <button type="button" className="action-button">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M8 12H16"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M12 8V16"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <button type="submit" className="action-button">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M22 2L11 13"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M22 2L15 22L11 13L2 9L22 2Z"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Chat

