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
  const clientId = '751670405477-v2bpuvr1o3jgudu0pcr5fs9l87n7rs20.apps.googleusercontent.com';
  const redirectUri = chrome.identity.getRedirectURL();
  const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&response_type=token&redirect_uri=${encodeURIComponent(redirectUri)}&scope=https://www.googleapis.com/auth/userinfo.email&prompt=select_account`;


  const handleGoogleLogin = async () => {
    // try {
    //   const response = await axios.get(`${url}/login`, 
    //     // { withCredentials: true }
    //   );
    //   if (response) {
    //     console.log("DItme")
    //   }
    // } catch (error) {
    //   console.error("Failed to login:", error)
    // }
    // window.location.href = `${url}/login`;
    
    chrome.identity.launchWebAuthFlow(
      {
        url: authUrl,
        interactive: true,
      },
      function (redirectUrl) {
        if (chrome.runtime.lastError) {
          console.error(chrome.runtime.lastError);
          return;
        }
        // Xử lý redirectUrl để lấy access token từ URL fragment (với response_type=token)
        console.log('Redirect URL:', redirectUrl);
      }
    );
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

