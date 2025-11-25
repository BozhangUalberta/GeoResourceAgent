import { useState, useEffect } from "react";

function AuthModal({ mode, onSetMode, open, onClose, onLogin, onRegister }) {
  const setMode = onSetMode
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (open) {
      setEmail("");
      setPassword("");
      setErr("");
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function handleEsc(e) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [open, onClose]);

  if (!open) return null;

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setErr("");
    try {
      if (mode === "login") {
        await onLogin({ email, password });
      } else {
        await onRegister({ email, password });
      }
      onClose();
    } catch (error) {
      setErr(error.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-gray-500/50" onClick={onClose} />
      <form
        onSubmit={handleSubmit}
        className="relative z-10 bg-white rounded-2xl shadow-xl p-8 w-full max-w-xs flex flex-col gap-4"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold text-center mb-2">
          {mode === "login" ? "Login" : "Register"}
        </h2>
        <input
          type="email"
          className="border rounded-lg p-2 w-full"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          autoFocus
          disabled={loading}
        />
        <input
          type="password"
          className="border rounded-lg p-2 w-full"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          disabled={loading}
        />
        {err && <div className="text-red-500 text-sm">{err}</div>}
        <button
          type="submit"
          className="py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
          disabled={loading}
        >
          {loading
            ? (mode === "login" ? "Logging in..." : "Registering...")
            : (mode === "login" ? "Login" : "Register")}
        </button>
        <button
          type="button"
          className="text-sm text-blue-600 hover:underline mt-2"
          onClick={() => setMode(mode === "login" ? "register" : "login")}
          disabled={loading}
        >
          {mode === "login"
            ? "No account? Register"
            : "Already have an account? Login"}
        </button>
      </form>
    </div>
  );
}

export default AuthModal;