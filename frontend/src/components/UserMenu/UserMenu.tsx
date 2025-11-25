import { useState } from "react";
import { useUser } from "../../context/UserContext";
import { registerUser } from "../../api/users";

import AuthModal from "./AuthModal";

function UserMenu({ onClose }) {
  const { user, isAuthenticated, login, logout } = useUser();

  const [mode, setMode] = useState(null)
  const [modalOpen, setModalOpen] = useState(false);

  async function handleLogin({ email, password }) {
    await login({ email, password });
    setModalOpen(false);
  }
  async function handleRegister({ email, password }) {
    await registerUser({ email, password });
    await login({ email, password });          
    setModalOpen(false);
  }


  return (
    <div
      className="h-full w-full min-w-[12rem] py-2 px-2 shadow-lg ring-1 ring-black/5 text-sm"
      tabIndex={-1}
    > 
      {isAuthenticated ? (
        <div>
          <div className="px-4 py-2 font-medium text-gray-400">
            {user.email}
          </div>
          <hr className="my-1 border-gray-200" />
          <button
            className="block w-full text-left px-4 py-2 rounded-2xl hover:bg-gray-100 hover:rounded-2xl transition"
            onClick={onClose}
          >
            Profile
          </button>
          <button
            className="block w-full text-left px-4 py-2 rounded-2xl hover:bg-gray-100 hover:rounded-2xl transition"
            onClick={onClose}
          >
            Settings
          </button>
          <hr className="my-1 border-gray-200" />
          <button
            className="block w-full text-left px-4 py-2 text-red-500 rounded-2xl hover:bg-gray-100 hover:text-red-600 hover:rounded-2xl transition"
            onClick={() => {
              logout();
              onClose();
            }}
          >
            Logout
          </button>
        </div>
      ) : (
        <div>
          <button
            className="block w-full text-left px-4 py-2 rounded-2xl hover:bg-gray-100 hover:rounded-xl transition"
            onClick={() => {
              setMode('login');
              setModalOpen(true);
              // onClose();
            }}
          >
            Login
          </button>
          <button
            className="block w-full text-left px-4 py-2 rounded-2xl hover:bg-gray-100 hover:rounded-xl transition"
            onClick={() => {
              setMode('register');
              setModalOpen(true);
              // onClose();
            }}
          >
            Register
          </button>

          <AuthModal
            mode={mode}
            onSetMode={setMode}
            open={modalOpen}
            onClose={() => setModalOpen(false)}
            onLogin={handleLogin}
            onRegister={handleRegister}
          />
        </div>
      )}
    </div>
  );
}

export default UserMenu;
