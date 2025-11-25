export const API_URL = import.meta.env.VITE_API_URL;

export async function loginUser({ email, password }) {
  const res = await fetch(`${API_URL}/api/v1/users/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  return res.json(); // { access_token, token_type }
}

export async function logoutUser(token) {
  const res = await fetch(`${API_URL}/api/v1/users/logout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    }
  });
  if (!res.ok) throw new Error("Logout failed");
  return res.json();
}

export async function fetchMe(token) {
  const res = await fetch(`${API_URL}/api/v1/users/me`, {
    method: "GET",
    headers: {
      "Authorization": `Bearer ${token}`,
    },
  });
  if (!res.ok) throw new Error("Fetch me failed");
  return res.json(); // { id, email, full_name }
}

export async function registerUser({ email, password }) {
  const res = await fetch(`${API_URL}/api/v1/users/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    let errMsg = "Register failed";
    try {
      const data = await res.json();
      errMsg = data.detail || errMsg;
    } catch {}
    throw new Error(errMsg);
  }
  return res.json(); // {id, email, full_name}
}