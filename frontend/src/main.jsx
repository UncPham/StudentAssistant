import React from "react"
import ReactDOM from "react-dom/client"
import App from './App.jsx'
import StoreContextProvider from './context/StoreContext.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <StoreContextProvider>
    <React.StrictMode>
      <App />
    </React.StrictMode>
  </StoreContextProvider>
)
