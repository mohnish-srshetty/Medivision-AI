import React, { createContext, useState, useEffect, useContext } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (token) {
      fetchUser(token);
    } else {
      setLoading(false);
    }
  }, [token]);

  const fetchUser = async (authToken) => {
    console.log("Fetching user with token:", authToken);
    try {
      const response = await fetch('/auth/me', {
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });
      console.log("Fetch user response status:", response.status);
      if (response.ok) {
        const userData = await response.json();
        console.log("User data fetched:", userData);
        setUser(userData);
      } else {
        console.error("Fetch user failed");
        logout();
      }
    } catch (error) {
      console.error("Failed to fetch user", error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    console.log("Attempting login for:", email);
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    try {
      const response = await fetch('/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData,
      });
      
      console.log("Login response status:", response.status);

      const text = await response.text();
      console.log("Raw response:", text);

      let data;
      try {
        data = text ? JSON.parse(text) : {};
      } catch (err) {
        console.error("Failed to parse JSON:", err);
        return { success: false, error: "Server returned invalid response: " + text.substring(0, 50) };
      }

      if (response.ok) {
        console.log("Login successful, token:", data.access_token);
        localStorage.setItem('token', data.access_token);
        setToken(data.access_token);
        return { success: true };
      } else {
        return { success: false, error: data.detail || "Login failed" };
      }
    } catch (e) {
      console.error("Login error:", e);
      return { success: false, error: e.message };
    }
  };

  const signup = async (email, password, fullName) => {
    const response = await fetch('/auth/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, password, full_name: fullName }),
    });

    if (response.ok) {
      const data = await response.json();
      localStorage.setItem('token', data.access_token);
      setToken(data.access_token);
      return { success: true };
    }
    const errorData = await response.json();
    return { success: false, error: errorData.detail || "Signup failed" };
  };

  const logout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, signup, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
