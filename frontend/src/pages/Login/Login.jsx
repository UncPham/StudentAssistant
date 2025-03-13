import React from 'react'
import { GoogleLogin } from 'react-google-login'

const Login = () => {
    const onSuccess = async (res) => {
        const idToken = res.tokenId
        console.log("idtoken = " + idToken)

        try {
            await fetch("http://localhost:8000/login", 
                {
                    method: "POST",
                    headers: { Authorization: `Bearer ${idToken}` },
                    credentials: 'include'
                }
            )
        }
        catch (error) {
            console.log(error)
        }

        window.location = "/account"
    }

    const onFailure = (res) => {
        console.log(res)
    }

    

  return (
    <div>
      <GoogleLogin
        clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}
        responseType={"id_token"}
        cookiePolicy={"single_host_origin"}
        onSuccess={onSuccess}
        onFailure={onFailure}
        theme={"dark"}
      />
    </div>
  )
}

export default Login
