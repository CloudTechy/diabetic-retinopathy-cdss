import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    watch: {
      usePolling: true,
    },
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://backend:8000',
        changeOrigin: true,
        secure: false,
      },
      '/health': {
        target: process.env.VITE_API_PROXY_TARGET || 'http://backend:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
