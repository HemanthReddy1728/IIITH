import { createContext, useReducer } from 'react'

export const AnalysisContext = createContext()

export const analysisReducer = (state, action) => {
    switch (action.type) {
        case 'SET_ANALYSIS':
            return {
                analysis_list: action.payload
            }
        case 'CREATE_ANALYSIS':
            return {
                analysis_list: [action.payload, ...state.analysis_list]
            }
        case 'DELETE_ANALYSIS':
            return {
                analysis_list: state.analysis_list.filter((w) => w._id !== action.payload._id)
            }
        default:
            return state
    }
}

export const AnalysisContextProvider = ({ children }) => {
    const [state, dispatch] = useReducer(analysisReducer, {
        analysis_list: null
    })

    return (
        <AnalysisContext.Provider value={{ ...state, dispatch }}>
            {children}
        </AnalysisContext.Provider>
    )
}