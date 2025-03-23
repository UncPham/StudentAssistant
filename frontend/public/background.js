chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick  : true })
    .catch((error) => console.error(error));

// function signInWithGoogle() {
//     chrome.identity.getAuthToken({ interactive: true }, function (token) {
//         if (chrome.runtime.lastError) {
//             console.error(chrome.runtime.lastError);
//             return;
//         }
    
//         console.log("Access Token:", token);
//         fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
//             headers: { Authorization: `Bearer ${token}` }
//         })
//         .then((response) => response.json())
//         .then((userInfo) => {
//             console.log("User Info:", userInfo);
//         })
//         .catch((error) => console.error(error));
//     });
// }

      