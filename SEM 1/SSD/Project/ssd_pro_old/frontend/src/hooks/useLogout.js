import { useAuthContext } from './useAuthContext'
import { useAnalysisContext } from './useAnalysisContext'

export const useLogout = () => {
    const { dispatch } = useAuthContext()
    const { dispatch: dispatchAnalysis } = useAnalysisContext()

    const logout = () => {
        // remove user from storage
        localStorage.removeItem('user')

        // dispatch logout action
        dispatch({ type: 'LOGOUT' })
        dispatchAnalysis({ type: 'SET_ANALYSIS', payload: null })
    }

    return { logout }
}