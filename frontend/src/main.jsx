import React from "react"
import ReactDOM from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import App from './App.jsx'
import StoreContextProvider from './context/StoreContext.jsx'
import "@radix-ui/themes/styles.css";
// import { GoogleOAuthProvider } from "@react-oauth/google";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <StoreContextProvider>
      <React.StrictMode>
        {/* <GoogleOAuthProvider  clientId={CLIENT_ID}> */}
        {/* <Theme> */}
          <App />
        {/* </Theme> */}
        {/* </GoogleOAuthProvider > */}
      </React.StrictMode>
    </StoreContextProvider>
  </BrowserRouter>
)
