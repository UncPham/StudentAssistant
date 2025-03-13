import { createContext } from "react";
export const StoreContext = createContext(null);

// Tạo context để chia sẻ trạng thái giữa các component
const StoreContextProvider = (props) => {

    const url = "http://localhost:8000";

    const contextValue = {
        url
    }

    return (
        <StoreContext.Provider value={contextValue} >
            {props.children}
        </StoreContext.Provider >
    )
}

export default StoreContextProvider;