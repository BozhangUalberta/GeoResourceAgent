import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

import { UserProvider } from './context/UserContext';

// MSW mock API
async function enableMocking() {
  if (import.meta.env.VITE_USE_MOCK !== "true") {
    return
  }
  const { worker } = await import('./mocks/browser')

  return worker.start()
}
 
enableMocking().then(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <UserProvider>
        <App />
      </UserProvider>
    </StrictMode>,
  )
})