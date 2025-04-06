"use client"

import { useEffect } from "react"
import "./Login.css"
import GoogleLogo from "../../assets/google-logo.svg"
import axios from 'axios'

// const saveUserCookie = (user) => {
//   const expires = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toUTCString()
//   document.cookie = `user=${encodeURIComponent(JSON.stringify(user))}; expires=${expires}; path=/`
// }

const Login = ({ onLoginSuccess = () => {} }) => {

  const url = "http://localhost:8000"

  const handleGoogleLogin = async () => {
    try {
      // Open the login URL in a popup window
      const popup = window.open(
        `${url}/login`,
        'Google Login',
        'width=500,height=600,left=400,top=100'
      );

    } catch (error) {
      console.error("Failed to login:", error);
    }
  }

  return (
    <div className="login-container">
      <h1 className="login-title">Đăng nhập</h1>
      <button className="google-login-button" onClick={handleGoogleLogin}>
        <img src={GoogleLogo} alt="Google" className="google-icon" />
        Tiếp tục với Google
      </button>
    </div>
  )
}

export default Login

