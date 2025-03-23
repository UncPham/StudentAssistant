import React from "react"
import ReactDOM from "react-dom/client"
import App from './App.jsx'
import StoreContextProvider from './context/StoreContext.jsx'
// import { GoogleOAuthProvider } from "@react-oauth/google";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

ReactDOM.createRoot(document.getElementById('root')).render(
  <StoreContextProvider>
    <React.StrictMode>
      {/* <GoogleOAuthProvider  clientId={CLIENT_ID}> */}
        <App />
      {/* </GoogleOAuthProvider > */}
    </React.StrictMode>
  </StoreContextProvider>
)
