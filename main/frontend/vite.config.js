import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tsconfigPaths from "vite-tsconfig-paths"

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  server: {
    proxy: {
      '/auth': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/history': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/predict': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/generate-report': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/get-latest-report': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/get_latest_results': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/preview': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/chat_with_report': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
      '/public-chat': { target: 'http://127.0.0.1:8000', changeOrigin: true, secure: false },
    }
  }
})
