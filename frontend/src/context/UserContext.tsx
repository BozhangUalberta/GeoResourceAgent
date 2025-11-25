import React, { createContext, useContext, useEffect, useState } from "react";

import { loginUser, fetchMe } from "../api/users";

const UserContext = createContext(null);

export function useUser() {
  return useContext(UserContext);
}

export function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  // Check local persistant (localStorage)
  useEffect(() => {
    const storedToken = localStorage.getItem("access_token");
    const storedUser = localStorage.getItem("user");
    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
  }, []);


  async function login({ email, password }) {
    const { access_token } = await loginUser({ email, password });
    setToken(access_token);

    const userInfo = await fetchMe(access_token);
    setUser(userInfo);

    localStorage.setItem("access_token", access_token);
    localStorage.setItem("user", JSON.stringify(userInfo));
  }

  function logout() {
    setUser(null);
    setToken(null);
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
  }

  const isAuthenticated = !!token && !!user;

  return (
    <UserContext.Provider value={{
      user,
      token,
      isAuthenticated,
      login,
      logout,
      setUser,
      setToken,
    }}>
      {children}
    </UserContext.Provider>
  );
}