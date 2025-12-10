import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from "./ui/button"
import { ModeToggle } from './ui/mode-toggle';
import { Activity, FileText, Home, LogOut, User, Bot } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu"

const Header = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white/80 backdrop-blur-sm dark:bg-zinc-950/80 backdrop:blur-lg">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        <Link to="/" className="flex items-center gap-2 text-lg font-semibold transition-colors hover:text-blue-600">
          <Activity className="h-6 w-6 text-blue-600" />
          <span>MediVision AI</span>
        </Link>
        
        <nav className="hidden md:flex items-center gap-6">
          <Link to="/" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <Home className="h-4 w-4" />
            Home
          </Link>
          <Link to="/upload" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Analysis
          </Link>
          <Link to="/search-doctor" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Find Doctor
          </Link>
          <Link to="/chat" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Chat
          </Link>
          {user && (
            <Link to="/history" className="text-sm font-medium transition-colors hover:text-blue-600 flex items-center gap-2">
              <FileText className="h-4 w-4" />
              History
            </Link>
          )}
        </nav>
        
        <div className="flex items-center gap-4">
          <ModeToggle />
          
          {user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                  <div className="flex h-full w-full items-center justify-center rounded-full bg-blue-100 text-blue-600 font-bold">
                    {user.full_name ? user.full_name[0].toUpperCase() : 'U'}
                  </div>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{user.full_name}</p>
                    <p className="text-xs leading-none text-muted-foreground">
                      {user.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link to="/history">Patient History</Link>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleLogout} className="text-red-600">
                  <LogOut className="mr-2 h-4 w-4" />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div className="flex items-center gap-2">
              <Button asChild variant="ghost" size="sm">
                <Link to="/login">Login</Link>
              </Button>
              <Button asChild size="sm" className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white">
                <Link to="/signup">Sign Up</Link>
              </Button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;