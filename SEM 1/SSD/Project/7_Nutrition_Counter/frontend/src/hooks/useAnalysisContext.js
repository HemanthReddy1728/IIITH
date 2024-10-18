import { AnalysisContext } from '../context/AnalysisContext'
import { useContext } from 'react'

export const useAnalysisContext = () => {
    const context = useContext(AnalysisContext)

    if (!context) {
        throw Error('useAnalysisContext must be used inside an AnalysisContextProvider')
    }

    return context
}