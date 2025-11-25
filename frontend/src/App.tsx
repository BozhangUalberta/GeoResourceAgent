import { useState } from 'react'

import './App.css'
import { CurrentConversationProvider } from "./context/CurrentConversationContext";
import MainWindow from './components/MainWindow/MainWindow'
import Sidebar from './components/SideBar/Sidebar'


function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Component Props
  const titleBarProps = { 
    sidebarOpen,
    onSidebarOpen: () => setSidebarOpen(true),
  };


  return (
    <div className="flex h-screen w-screen flex-col">
      <div className="relative flex h-full w-full flex-row">
        <CurrentConversationProvider>
          {/* Left sidebar */}
          <div className={`
            h-full
            transition-all duration-300 overflow-hidden
            ${sidebarOpen ? "w-[270px]" : "w-0"}
          `}>
            <Sidebar onSidebarClose={() => setSidebarOpen(false)} />
          </div>
          
          {/* Main window area */}
          <div className="relative flex-1 bg-grey-100 h-full max-w-full flex-col">
            <MainWindow titleBarProps={titleBarProps} />
          </div>
        </CurrentConversationProvider>
      </div>
    </div>
  )
}

export default App
